#!/usr/bin/env python3
"""Recompress the ISAAC parquet corpus from SNAPPY to ZSTD for the HF mirror.

Row-group boundaries, schema and every value are preserved exactly; only the
page compression codec changes. Each output file is verified row group by row
group against its source (pyarrow Table.equals) before being marked done, and
progress is journalled so the run is resumable.

Point --dst at a different physical drive from --src where possible: reading
and writing one spindle roughly halves throughput.
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

CATEGORIES = ["ability", "age", "race", "sexuality", "skin_tone", "weight"]
LEVEL = 9


def recompress_one(args):
    """Recompress one file and verify it. Never leaks a file handle: every
    ParquetFile is closed in a finally, because a handle kept alive by a failed
    task's traceback locks the source file against the retry pass."""
    src, dst, level = Path(args[0]), Path(args[1]), args[2]
    t0 = time.time()
    tmp = dst.with_suffix(".parquet.tmp")
    err = None
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        # --- write: one output row group per input row group ---
        pf = pq.ParquetFile(src)
        try:
            schema = pf.schema_arrow
            src_rg_rows = [pf.metadata.row_group(i).num_rows
                           for i in range(pf.metadata.num_row_groups)]
            with pq.ParquetWriter(tmp, schema, compression="zstd",
                                  compression_level=level, version="2.6",
                                  use_dictionary=True) as w:
                for i in range(len(src_rg_rows)):
                    w.write_table(pf.read_row_group(i), row_group_size=src_rg_rows[i])
        finally:
            pf.close()

        # --- verify: schema, layout, and every value, one row group at a time ---
        s = pq.ParquetFile(src)
        try:
            d = pq.ParquetFile(tmp)
            try:
                if not s.schema_arrow.equals(d.schema_arrow):
                    raise AssertionError("schema differs")
                if s.metadata.num_rows != d.metadata.num_rows:
                    raise AssertionError(
                        f"row count {s.metadata.num_rows} != {d.metadata.num_rows}")
                dst_rg_rows = [d.metadata.row_group(i).num_rows
                               for i in range(d.metadata.num_row_groups)]
                if src_rg_rows != dst_rg_rows:
                    raise AssertionError(
                        f"row-group layout differs: {src_rg_rows} != {dst_rg_rows}")
                for i in range(len(src_rg_rows)):
                    a = s.read_row_group(i)
                    b = d.read_row_group(i)
                    same = a.equals(b)
                    del a, b
                    if not same:
                        raise AssertionError(f"row group {i} differs")
                rows = s.metadata.num_rows
            finally:
                d.close()
        finally:
            s.close()

        os.replace(tmp, dst)
        return dict(src=str(src), dst=str(dst), ok=True, rows=rows,
                    src_bytes=src.stat().st_size, dst_bytes=dst.stat().st_size,
                    secs=round(time.time() - t0, 1))
    except Exception as e:
        # Keep only strings: holding the exception (or its traceback) alive would
        # pin the frames, and with them the file handles we just tried to free.
        err = (f"{type(e).__name__}: {e}", traceback.format_exc()[-1500:])
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    return dict(src=str(src), dst=str(dst), ok=False,
                error=err[0], tb=err[1], secs=round(time.time() - t0, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="source tree: <src>/<category>/ALL_*.parquet")
    ap.add_argument("--dst", required=True,
                    help="output tree; put it on a different physical drive "
                         "from --src if you can")
    ap.add_argument("--level", type=int, default=LEVEL)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--journal", default="recompress_journal.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N pending files (trial run)")
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    journal = Path(args.journal)

    done = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("ok"):
                    done.add(r["src"])
            except json.JSONDecodeError:
                pass

    tasks = []
    for c in args.categories.split(","):
        for f in sorted((src_root / c).glob("ALL_*.parquet")):
            if str(f) in done:
                continue
            tasks.append((str(f), str(dst_root / c / f.name), args.level))

    total_pending_bytes = sum(os.path.getsize(t[0]) for t in tasks)
    print(f"src {src_root} -> dst {dst_root}  zstd-{args.level}  workers={args.workers}")
    print(f"already verified: {len(done)}   pending: {len(tasks)} files, "
          f"{total_pending_bytes/1e9:.1f} GB", flush=True)
    if args.limit:
        tasks = tasks[:args.limit]
        print(f"TRIAL: limiting to {len(tasks)} files", flush=True)
    if not tasks:
        print("nothing to do")
        return 0

    t_start = time.time()
    n_ok = n_fail = 0
    b_src = b_dst = 0
    with journal.open("a", encoding="utf-8") as jf, \
         ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(recompress_one, t): t for t in tasks}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            jf.write(json.dumps(r) + "\n"); jf.flush()
            if r["ok"]:
                n_ok += 1; b_src += r["src_bytes"]; b_dst += r["dst_bytes"]
                el = time.time() - t_start
                rate = b_src / el / 1e6
                eta = (total_pending_bytes - b_src) / (b_src / el) if b_src else 0
                print(f"[{n}/{len(tasks)}] OK {Path(r['src']).parent.name}/{Path(r['src']).name} "
                      f"{r['src_bytes']/1e6:.0f}->{r['dst_bytes']/1e6:.0f}MB "
                      f"({100*r['dst_bytes']/r['src_bytes']:.0f}%) {r['secs']}s "
                      f"| {rate:.0f} MB/s in, ETA {eta/3600:.1f}h", flush=True)
            else:
                n_fail += 1
                print(f"[{n}/{len(tasks)}] FAIL {r['src']}: {r['error']}", flush=True)

    el = time.time() - t_start
    print(f"\ndone in {el/3600:.2f}h  ok={n_ok} fail={n_fail}")
    if b_src:
        print(f"{b_src/1e9:.1f} GB -> {b_dst/1e9:.1f} GB  ({100*b_dst/b_src:.1f}%, "
              f"saved {(b_src-b_dst)/1e9:.1f} GB)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
