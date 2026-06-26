#!/usr/bin/env python
"""Self-contained validation of the writer-shard -> canonical merge.

Runs entirely on throwaway temp dirs (never touches the real cache). Exercises
the two properties the --shard-cache design relies on:
  * idempotent, no-double-count merge of immutable (author, raw_file) rows, and
  * resume: shards from a "failed" run still contribute, and a later re-run that
    re-banks overlapping files plus new ones folds in correctly.

Run within the ISAAC env (needs zstandard):
    python code/test_merge_location_cache.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (  # noqa: E402
    cache_put_author_file_counts_sharded,
    cache_get_author_file_counts_sharded,
    merge_author_file_counts_shards,
)

RTYPE = "comments"


def _shard(shards_root, run_id, rows):
    """Simulate one task writing its private shard (same call label_location uses)."""
    cache_put_author_file_counts_sharded(os.path.join(shards_root, run_id), RTYPE, rows)


def main() -> int:
    with tempfile.TemporaryDirectory() as canonical:
        shards_root = os.path.join(canonical, "shards")

        # Run A = a wall-time-killed run that banked two files for u1.
        _shard(shards_root, "100_1", [
            ("u1", "RC_2020-01", {"ca": 2}, 5),
            ("u1", "RC_2020-02", {"cb": 1}, 3),
        ])
        # Run B = the re-run: re-banks RC_2020-01 (overlap) + a new file, plus u2.
        _shard(shards_root, "200_1", [
            ("u1", "RC_2020-01", {"ca": 2}, 5),   # overlaps run A
            ("u1", "RC_2020-03", {"cc": 4}, 7),
            ("u2", "RC_2021-05", {"cd": 1}, 2),
        ])

        stats = merge_author_file_counts_shards(canonical, RTYPE, archive=True)
        assert stats["run_dirs"] == 2, stats

        res = cache_get_author_file_counts_sharded(canonical, RTYPE, ["2020", "2021"], {"u1", "u2"})
        u1_files, u1_counts, u1_seen = res["u1"]
        assert u1_files == {"RC_2020-01", "RC_2020-02", "RC_2020-03"}, u1_files
        assert u1_seen == 15, u1_seen            # 5+3+7; RC_2020-01 counted ONCE (no double)
        assert u1_counts == {"ca": 2, "cb": 1, "cc": 4}, u1_counts
        u2_files, u2_counts, u2_seen = res["u2"]
        assert u2_files == {"RC_2021-05"} and u2_seen == 2 and u2_counts == {"cd": 1}, res["u2"]
        print("[ok] failed-run + re-run shards merged to the union, no double-count")

        # Idempotent: drained dirs are archived, so a second merge does nothing.
        stats2 = merge_author_file_counts_shards(canonical, RTYPE, archive=True)
        assert stats2["run_dirs"] == 0, stats2
        again = cache_get_author_file_counts_sharded(canonical, RTYPE, ["2020", "2021"], {"u1"})
        assert again["u1"][2] == 15, again["u1"]
        print("[ok] re-merge is a no-op; canonical unchanged")

        # Resume: a later run re-banks an already-canonical file (dup) + a new one.
        _shard(shards_root, "300_1", [
            ("u1", "RC_2020-01", {"ca": 2}, 5),   # duplicate of canonical -> ignored
            ("u1", "RC_2020-04", {"ce": 9}, 1),   # new
        ])
        stats3 = merge_author_file_counts_shards(canonical, RTYPE, archive=True)
        assert stats3["run_dirs"] == 1, stats3
        final = cache_get_author_file_counts_sharded(canonical, RTYPE, ["2020", "2021"], {"u1"})
        f_files, f_counts, f_seen = final["u1"]
        assert f_files == {"RC_2020-01", "RC_2020-02", "RC_2020-03", "RC_2020-04"}, f_files
        assert f_seen == 16, f_seen              # 15 + 1; the duplicate RC_2020-01 added nothing
        assert f_counts == {"ca": 2, "cb": 1, "cc": 4, "ce": 9}, f_counts
        print("[ok] resume folds new files in; duplicate file contributes nothing")

        # Year routing sanity: 2020 and 2021 landed in separate canonical DBs.
        assert os.path.exists(os.path.join(canonical, f"author_file_counts_{RTYPE}_2020.sqlite"))
        assert os.path.exists(os.path.join(canonical, f"author_file_counts_{RTYPE}_2021.sqlite"))
        assert os.path.isdir(os.path.join(shards_root, "merged", "100_1"))
        print("[ok] rows routed to per-year canonical DBs; drained run dirs archived")

    print("ALL MERGE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
