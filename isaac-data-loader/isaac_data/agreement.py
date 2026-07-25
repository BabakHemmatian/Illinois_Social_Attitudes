"""First-run Data Use Agreement acceptance (recorded locally).

Data access (`load`, `download`, remote `read_parquet`) requires a one-time
acceptance of the ISAAC Data Use Agreement. Acceptance is recorded ONLY on the
local machine (OS-native config dir). Browsing the catalog (`catalog`, `files`)
does not require acceptance.

Non-interactive use (CI, notebooks without a TTY) must either accept beforehand
via `isaac-data accept-agreement` or set the environment variable
``ISAAC_ACCEPT_AGREEMENT=1``; otherwise data access raises
``AgreementNotAccepted``.

Renamed 2026-07-25: the document was previously called the "Terms of Use". The
old names (module ``isaac_data.terms``, ``TermsNotAccepted``, ``accept_terms``,
``ISAAC_ACCEPT_TERMS``, ``isaac-data accept-terms``) still work as aliases.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

AGREEMENT_PAGE = "https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md"
AGREEMENT_RAW = "https://raw.githubusercontent.com/BabakHemmatian/Illinois_Social_Attitudes/main/Data_Use_Agreement.md"
_ENV = "ISAAC_ACCEPT_AGREEMENT"
_ENV_LEGACY = "ISAAC_ACCEPT_TERMS"  # pre-2026-07-25 name, still honored


class AgreementNotAccepted(RuntimeError):
    """Raised when the ISAAC Data Use Agreement has not been accepted."""


def _env_opt_in() -> bool:
    for name in (_ENV, _ENV_LEGACY):
        if os.environ.get(name, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _config_dir() -> Path:
    override = os.environ.get("ISAAC_DATA_CONFIG")
    if override:
        d = Path(override).expanduser()
    else:
        import platformdirs
        d = Path(platformdirs.user_config_dir("isaac-data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_file() -> Path:
    return _config_dir() / "accepted.json"


def is_accepted() -> bool:
    return _record_file().exists()


def status() -> Optional[dict]:
    """Return the local acceptance record, or None if not yet accepted."""
    f = _record_file()
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return {"accepted": True, "record": str(f)}


def fetch_agreement(timeout: int = 15) -> Optional[str]:
    """Best-effort fetch of the current Data Use Agreement text (None if unavailable)."""
    import requests
    try:
        r = requests.get(AGREEMENT_RAW, timeout=timeout)
        if r.ok and r.text.strip():
            return r.text
    except Exception:
        pass
    return None


def withdraw() -> bool:
    """Delete the local acceptance record. Returns True if one existed."""
    f = _record_file()
    if f.exists():
        f.unlink()
        return True
    return False


def _write_record(text: Optional[str]) -> dict:
    from . import __version__
    rec = {
        "accepted": True,
        "accepted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agreement_url": AGREEMENT_PAGE,
        "agreement_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
        "package_version": __version__,
    }
    _record_file().write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def accept_agreement(assume_yes: bool = False) -> dict:
    """Show the Data Use Agreement and record acceptance locally.

    assume_yes : skip the prompt (equivalent to ``ISAAC_ACCEPT_AGREEMENT=1``).
    Raises AgreementNotAccepted if the user declines or no terminal is available.
    """
    out = sys.stderr
    text = fetch_agreement()
    print("\n" + "=" * 72, file=out)
    print("ISAAC dataset — Data Use Agreement", file=out)
    print("=" * 72, file=out)
    if text:
        print(text.strip(), file=out)
        print("-" * 72, file=out)
    else:
        print("By using the ISAAC dataset and this package you agree to the ISAAC", file=out)
        print(f"Data Use Agreement:\n  {AGREEMENT_PAGE}", file=out)
        print("-" * 72, file=out)

    if assume_yes or _env_opt_in():
        rec = _write_record(text)
        print(f"Agreement accepted (recorded at {_record_file()}).", file=out)
        return rec

    if not (sys.stdin and sys.stdin.isatty()):
        raise AgreementNotAccepted(
            "ISAAC Data Use Agreement not accepted and no interactive terminal is available.\n"
            f"Read {AGREEMENT_PAGE}, then run `isaac-data accept-agreement` or set {_ENV}=1."
        )

    resp = input("Do you accept the ISAAC Data Use Agreement? [y/N] ").strip().lower()
    if resp in ("y", "yes"):
        rec = _write_record(text)
        print(f"Thank you — acceptance recorded at {_record_file()}.", file=out)
        return rec
    raise AgreementNotAccepted("ISAAC Data Use Agreement was not accepted; aborting.")


def require_acceptance() -> None:
    """Fast gate called before data access. No-op once accepted."""
    if is_accepted():
        return
    if _env_opt_in():
        _write_record(fetch_agreement())
        return
    accept_agreement()


# ---- Pre-2026-07-25 aliases ("Terms of Use" era). Kept for compatibility. ----
TermsNotAccepted = AgreementNotAccepted
TERMS_PAGE = AGREEMENT_PAGE
TERMS_RAW = AGREEMENT_RAW
fetch_terms = fetch_agreement
accept_terms = accept_agreement
