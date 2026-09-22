"""Smoke tests against the live ISAAC endpoint (small files only).

Run: pytest -q   (requires network access to isaac.psychology.illinois.edu)

Uses an isolated config dir + ISAAC_ACCEPT_AGREEMENT so the Data-Use-Agreement gate
doesn't prompt, and so the gate's behavior can be tested in isolation.
"""
import os
import tempfile

# Isolate the acceptance record to a temp dir for the whole test session.
_CFG = tempfile.mkdtemp(prefix="isaac-cfg-")
os.environ["ISAAC_DATA_CONFIG"] = _CFG

import isaac_data as isaac  # noqa: E402
import isaac_data.agreement as isaac_data_agreement  # noqa: E402
from isaac_data import AgreementNotAccepted  # noqa: E402


def test_catalog_and_files_need_no_acceptance():
    cat = isaac.catalog(refresh=True)
    assert len(cat) > 2000
    assert set(isaac.CATEGORIES).issubset(set(cat["category"].unique()))
    race = isaac.files("race", "2018-01", "2018-12", fmt="parquet")
    assert len(race) == 12 and (race["format"] == "parquet").all()


def test_gate_blocks_without_acceptance(monkeypatch):
    # fresh config dir, no env, no TTY -> data access must raise
    fresh = tempfile.mkdtemp(prefix="isaac-cfg-block-")
    monkeypatch.setenv("ISAAC_DATA_CONFIG", fresh)
    monkeypatch.delenv("ISAAC_ACCEPT_AGREEMENT", raising=False)
    monkeypatch.delenv("ISAAC_ACCEPT_TERMS", raising=False)  # legacy alias
    assert isaac.is_accepted() is False
    try:
        isaac.load("ability", "2007-01", "2007-01", columns=["score"])
        assert False, "expected AgreementNotAccepted"
    except AgreementNotAccepted:
        pass


def test_after_acceptance_load_and_download(monkeypatch):
    monkeypatch.setenv("ISAAC_ACCEPT_AGREEMENT", "1")
    df = isaac.load("ability", "2007-01", "2007-01", columns=["text", "score"])
    assert list(df.columns) == ["_category", "_month", "text", "score"] and len(df) > 0
    assert isaac.is_accepted() is True
    with tempfile.TemporaryDirectory() as d:
        paths = isaac.download("ability", "2007-01", "2007-01", fmt="parquet", dest=d)
        assert len(paths) == 1 and paths[0].stat().st_size > 0


def test_sampling_stratified_total_and_reproducible(monkeypatch):
    monkeypatch.setenv("ISAAC_ACCEPT_AGREEMENT", "1")
    df = isaac.load("ability", "2007-01", "2007-04", columns=["text"], n=20, seed=1)
    assert len(df) == 20                       # n is a TOTAL budget, not per-file
    counts = df["_month"].value_counts()
    assert set(counts.index) == {"2007-01", "2007-02", "2007-03", "2007-04"}
    assert counts.max() - counts.min() <= 1    # equal per month (±1 for rounding)
    # reproducible with the same seed
    df2 = isaac.load("ability", "2007-01", "2007-04", columns=["text"], n=20, seed=1)
    assert df.equals(df2)


def test_legacy_terms_aliases_still_work(monkeypatch):
    """Pre-2026-07-25 "Terms of Use" names must keep importing and behaving."""
    import isaac_data.terms as legacy
    assert legacy.TermsNotAccepted is isaac.AgreementNotAccepted
    assert legacy.accept_terms is isaac_data_agreement.accept_agreement
    assert isaac.accept_terms is isaac.accept_agreement
    assert legacy.TERMS_PAGE.endswith("Data_Use_Agreement.md")
    # the legacy env var still opts in
    fresh = tempfile.mkdtemp(prefix="isaac-cfg-legacy-")
    monkeypatch.setenv("ISAAC_DATA_CONFIG", fresh)
    monkeypatch.delenv("ISAAC_ACCEPT_AGREEMENT", raising=False)
    monkeypatch.setenv("ISAAC_ACCEPT_TERMS", "1")
    df = isaac.load("ability", "2007-01", "2007-01", columns=["score"])
    assert len(df) > 0


if __name__ == "__main__":
    os.environ["ISAAC_ACCEPT_AGREEMENT"] = "1"
    test_catalog_and_files_need_no_acceptance()
    df = isaac.load("ability", "2007-01", "2007-01", columns=["text", "score"])
    print("load ok:", df.shape, "| accepted:", isaac.is_accepted())
    print("all smoke checks passed")


# --------------------------------------------------------------------------- #
# Server-side consent record
# --------------------------------------------------------------------------- #
# The POST at acceptance time is best-effort and never blocks data access, so a
# user who accepts while offline (or during a server outage) ends up with a
# valid local acceptance the project never heard about. Nothing else re-posts
# it, so without a retry that record would stay local forever -- which would
# make "acceptance is recorded server-side" untrue in exactly the case it
# matters.
import json
import time

import pytest


@pytest.fixture
def isolated_record(tmp_path, monkeypatch):
    monkeypatch.setattr(isaac_data_agreement, "_record_file",
                        lambda: tmp_path / "accepted.json")
    return tmp_path / "accepted.json"


def _unacked(**over):
    rec = {"accepted": True, "email": "pytest@example.invalid",
           "client_id": "pypi:test", "accepted_at_utc": "2026-09-22T00:00:00Z",
           "server_ack": False}
    rec.update(over)
    return rec


def test_unacknowledged_consent_is_reposted(isolated_record, monkeypatch):
    seen = []
    monkeypatch.setattr(isaac_data_agreement, "_post_consent",
                        lambda rec, timeout=10: (seen.append(timeout), True)[1])
    rec = _unacked()
    isolated_record.write_text(json.dumps(rec))

    isaac_data_agreement._retry_unacknowledged_consent(rec)

    assert len(seen) == 1, "the unacknowledged acceptance was never re-posted"
    assert seen[0] <= 5, "retry must use a short timeout; it gates data access"
    assert rec["server_ack"] is True
    assert json.loads(isolated_record.read_text())["server_ack"] is True


def test_repost_is_rate_limited(isolated_record, monkeypatch):
    seen = []
    monkeypatch.setattr(isaac_data_agreement, "_post_consent",
                        lambda rec, timeout=10: (seen.append(1), False)[1])
    recent = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = _unacked(last_consent_post_utc=recent)
    isolated_record.write_text(json.dumps(rec))

    isaac_data_agreement._retry_unacknowledged_consent(rec)

    assert seen == [], "a just-attempted post must not be retried immediately"


def test_repost_resumes_after_the_retry_window(isolated_record, monkeypatch):
    seen = []
    monkeypatch.setattr(isaac_data_agreement, "_post_consent",
                        lambda rec, timeout=10: (seen.append(1), True)[1])
    stale = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() - isaac_data_agreement._CONSENT_RETRY_TTL_SECONDS - 60),
    )
    rec = _unacked(last_consent_post_utc=stale)
    isolated_record.write_text(json.dumps(rec))

    isaac_data_agreement._retry_unacknowledged_consent(rec)

    assert len(seen) == 1
    assert rec["server_ack"] is True


def test_repost_failure_never_raises(isolated_record, monkeypatch):
    def boom(rec, timeout=10):
        raise RuntimeError("network down")

    monkeypatch.setattr(isaac_data_agreement, "_post_consent",
                        lambda rec, timeout=10: False)
    rec = _unacked()
    isolated_record.write_text(json.dumps(rec))

    isaac_data_agreement._retry_unacknowledged_consent(rec)  # must not raise

    assert rec["server_ack"] is False
    assert "last_consent_post_utc" in rec, "a failed attempt must still be stamped"


def test_acknowledged_record_is_not_reposted(isolated_record, monkeypatch):
    seen = []
    monkeypatch.setattr(isaac_data_agreement, "_post_consent",
                        lambda rec, timeout=10: (seen.append(1), True)[1])
    rec = _unacked(server_ack=True)
    isaac_data_agreement._retry_unacknowledged_consent(rec)
    assert seen == []


def test_acceptance_message_names_both_copies():
    acked = isaac_data_agreement._where_recorded({"server_ack": True})
    assert "ISAAC project" in acked and "cached locally" in acked

    local_only = isaac_data_agreement._where_recorded({"server_ack": False})
    assert "not been recorded there yet" in local_only
    assert "retried automatically" in local_only
