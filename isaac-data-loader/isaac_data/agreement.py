"""First-run Data Use Agreement acceptance.

Data access (`load`, `download`, remote `read_parquet`) requires acceptance of
the ISAAC Data Use Agreement. Browsing the catalog (`catalog`, `files`) does not.

Accepting requires an email address. It is recorded locally (OS-native config
dir) and posted to the ISAAC server, so the project has one record across access
surfaces and a way to reach users about agreement changes and corpus errata. The
POST itself is best-effort: a network failure never blocks data access, and the
local record notes whether the server acknowledged it.

The agreement text is fetched from the ISAAC server's /dua endpoint, which
serves the live document out of the corpus repo along with a SHA-256 of the
exact bytes. That hash is what identifies the version, so:

  * the hash recorded here matches the one the website records, and
  * if the agreement changes, `require_acceptance` notices and re-prompts
    rather than honoring a stale acceptance forever.

Non-interactive use (CI, notebooks without a TTY) must either accept beforehand
via `isaac-data accept-agreement` or set ``ISAAC_ACCEPT_AGREEMENT=1``. Either way
``ISAAC_AGREEMENT_EMAIL`` (or ``--email``) must supply the address, since there is
no prompt to fall back on. Otherwise data access raises ``AgreementNotAccepted``.

Renamed 2026-07-25: the document was previously called the "Terms of Use". The
old names (module ``isaac_data.terms``, ``TermsNotAccepted``, ``accept_terms``,
``ISAAC_ACCEPT_TERMS``, ``isaac-data accept-terms``) still work as aliases.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

AGREEMENT_PAGE = "https://github.com/BabakHemmatian/Illinois_Social_Attitudes/blob/main/Data_Use_Agreement.md"
AGREEMENT_RAW = "https://raw.githubusercontent.com/BabakHemmatian/Illinois_Social_Attitudes/main/Data_Use_Agreement.md"
_ENV = "ISAAC_ACCEPT_AGREEMENT"
_ENV_LEGACY = "ISAAC_ACCEPT_TERMS"  # pre-2026-07-25 name, still honored
_ENV_EMAIL = "ISAAC_AGREEMENT_EMAIL"

# Why we ask for an email. Keep this in sync with the wording on the website and
# the HuggingFace dataset card — and keep it accurate: do not promise uses we do
# not actually carry out.
EMAIL_PURPOSE = (
    "We ask for your email so we can notify you of changes to the Data Use\n"
    "Agreement and of corrections or errata affecting the corpus, and to keep a\n"
    "record of your acceptance. We do not share it, and we don't use it for\n"
    "anything else."
)

# How often to re-verify that the accepted agreement is still the current one.
# Matches the manifest cache policy in core.py; a miss costs one small GET.
_VERSION_CHECK_TTL_SECONDS = 24 * 3600

# Deliberately permissive: this is a typo guard, not identity verification.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AgreementNotAccepted(RuntimeError):
    """Raised when the ISAAC Data Use Agreement has not been accepted."""


def _env_opt_in() -> bool:
    for name in (_ENV, _ENV_LEGACY):
        if os.environ.get(name, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _env_email() -> Optional[str]:
    val = os.environ.get(_ENV_EMAIL, "").strip()
    return val if val and _EMAIL_RE.match(val) else None


def _base_url() -> str:
    # Read at call time (not import time) so ISAAC_BASE_URL can be set in-process.
    return os.environ.get("ISAAC_BASE_URL", "https://isaac.psychology.illinois.edu").rstrip("/")


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


def _read_record() -> Optional[dict]:
    f = _record_file()
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        # Unreadable record: treat as "accepted, version unknown" rather than
        # forcing a re-prompt because of a corrupt file.
        return {"accepted": True, "record": str(f)}


def is_accepted() -> bool:
    """True if this machine has an acceptance record. Cheap; never hits network.

    Does not check whether the accepted version is still current —
    `require_acceptance` does that, so this stays usable as a fast predicate.
    """
    return _record_file().exists()


def status() -> Optional[dict]:
    """Return the local acceptance record, or None if not yet accepted."""
    return _read_record()


def fetch_agreement_record(timeout: int = 15) -> Optional[dict]:
    """Fetch the current agreement text plus its version identifiers.

    Prefers the ISAAC server's /dua endpoint so the SHA-256 recorded here is the
    same one the website records. Falls back to raw GitHub (hashing the bytes
    ourselves) if the server is unreachable. Returns None if both fail.
    """
    import requests
    try:
        r = requests.get(f"{_base_url()}/dua", timeout=timeout,
                         headers={"Accept": "application/json"})
        if r.ok:
            d = r.json()
            if d.get("markdown"):
                return {
                    "text": d["markdown"],
                    "sha256": d.get("sha256"),
                    "commit": d.get("commit"),
                    "version": d.get("version"),
                }
    except Exception:
        pass
    try:
        r = requests.get(AGREEMENT_RAW, timeout=timeout)
        if r.ok and r.text.strip():
            return {
                "text": r.text,
                "sha256": hashlib.sha256(r.content).hexdigest(),
                "commit": None,
                "version": None,
            }
    except Exception:
        pass
    return None


def fetch_agreement(timeout: int = 15) -> Optional[str]:
    """Best-effort fetch of the current Data Use Agreement text (None if unavailable)."""
    rec = fetch_agreement_record(timeout=timeout)
    return rec["text"] if rec else None


def withdraw() -> bool:
    """Delete the local acceptance record. Returns True if one existed."""
    f = _record_file()
    if f.exists():
        f.unlink()
        return True
    return False


def _client_id() -> str:
    """Stable per-machine id, so repeat acceptances are recognisable as one user.

    Not an identity claim — just a correlation handle for the consent log.
    """
    prev = _read_record() or {}
    return prev.get("client_id") or f"pypi:{uuid.uuid4()}"


def _post_consent(rec: dict, timeout: int = 10) -> bool:
    """Best-effort POST of the acceptance to the ISAAC server. Never raises."""
    if not rec.get("email"):
        return False
    import requests
    from . import __version__
    try:
        r = requests.post(
            f"{_base_url()}/record_consent",
            json={
                "uid": rec["client_id"],
                "email": rec["email"],
                "agreement_version": rec.get("agreement_version") or "unknown",
                "agreement_sha256": rec.get("agreement_sha256"),
                "agreement_commit": rec.get("agreement_commit"),
                "accepted_at": rec["accepted_at_utc"],
                "source": "pypi",
                "package_version": __version__,
            },
            timeout=timeout,
        )
        return bool(r.ok)
    except Exception:
        return False


def _prompt_email(out) -> str:
    """Ask for an email address. Required — raises if one is not supplied.

    An address is part of accepting the agreement (it is how we reach you about
    changes and errata), so there is no skip option here. Ctrl-C still aborts.
    """
    print(EMAIL_PURPOSE, file=out)
    print(file=out)
    for _ in range(5):
        try:
            resp = input("Your email address: ").strip()
        except EOFError:
            break
        if _EMAIL_RE.match(resp):
            return resp
        if resp:
            print("That doesn't look like an email address — please try again.", file=out)
        else:
            print("An email address is required to accept the agreement.", file=out)
    raise AgreementNotAccepted(
        "An email address is required to accept the ISAAC Data Use Agreement; aborting."
    )


def _require_email(email: Optional[str]) -> str:
    """Validate a non-interactively supplied address, with an actionable error."""
    if email and _EMAIL_RE.match(email):
        return email
    raise AgreementNotAccepted(
        "An email address is required to accept the ISAAC Data Use Agreement.\n"
        f"Pass `isaac-data accept-agreement --email you@example.edu` or set {_ENV_EMAIL}."
        + ("" if not email else f"\n(Got an address that doesn't parse: {email!r})")
    )


def _write_record(agreement: Optional[dict], email: Optional[str], via: str) -> dict:
    from . import __version__
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = {
        "accepted": True,
        "accepted_at_utc": now,
        "agreement_url": AGREEMENT_PAGE,
        "agreement_sha256": (agreement or {}).get("sha256"),
        "agreement_commit": (agreement or {}).get("commit"),
        "agreement_version": (agreement or {}).get("version"),
        "email": email,
        "client_id": _client_id(),
        "accepted_via": via,          # "prompt" | "env" | "assume_yes"
        "package_version": __version__,
        "last_version_check_utc": now,
    }
    rec["server_ack"] = _post_consent(rec)
    _record_file().write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def _print_agreement(agreement: Optional[dict], out) -> None:
    print("\n" + "=" * 72, file=out)
    print("ISAAC dataset — Data Use Agreement", file=out)
    print("=" * 72, file=out)
    if agreement and agreement.get("text"):
        print(agreement["text"].strip(), file=out)
        if agreement.get("version"):
            print(f"\n[version {agreement['version']}]", file=out)
    else:
        print("By using the ISAAC dataset and this package you agree to the ISAAC", file=out)
        print(f"Data Use Agreement:\n  {AGREEMENT_PAGE}", file=out)
    print("-" * 72, file=out)


def accept_agreement(assume_yes: bool = False, email: Optional[str] = None) -> dict:
    """Show the Data Use Agreement and record acceptance.

    assume_yes : skip the prompt (equivalent to ``ISAAC_ACCEPT_AGREEMENT=1``).
    email      : contact address to record; falls back to ``ISAAC_AGREEMENT_EMAIL``,
                 then to an interactive prompt. Required either way.

    Raises AgreementNotAccepted if the user declines, if no email is supplied, or
    if no terminal is available to ask on.
    """
    out = sys.stderr
    agreement = fetch_agreement_record()
    _print_agreement(agreement, out)

    if assume_yes or _env_opt_in():
        rec = _write_record(agreement, _require_email(email or _env_email()),
                            via="assume_yes" if assume_yes else "env")
        print(f"Agreement accepted (recorded at {_record_file()}).", file=out)
        return rec

    if not (sys.stdin and sys.stdin.isatty()):
        raise AgreementNotAccepted(
            "ISAAC Data Use Agreement not accepted and no interactive terminal is available.\n"
            f"Read {AGREEMENT_PAGE}, then run `isaac-data accept-agreement` or set {_ENV}=1."
        )

    resp = input("Do you accept the ISAAC Data Use Agreement? [y/N] ").strip().lower()
    if resp not in ("y", "yes"):
        raise AgreementNotAccepted("ISAAC Data Use Agreement was not accepted; aborting.")

    if email is None:
        email = _env_email() or _prompt_email(out)

    rec = _write_record(agreement, email, via="prompt")
    print(f"Thank you — acceptance recorded at {_record_file()}.", file=out)
    return rec


def _needs_version_recheck(rec: dict) -> bool:
    """True if enough time has passed to re-verify the accepted version."""
    last = rec.get("last_version_check_utc")
    if not last:
        return True
    try:
        import calendar
        elapsed = time.time() - calendar.timegm(time.strptime(last, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return True
    return elapsed > _VERSION_CHECK_TTL_SECONDS


def _touch_version_check(rec: dict) -> None:
    rec["last_version_check_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        _record_file().write_text(json.dumps(rec, indent=2) + "\n")
    except Exception:
        pass


def require_acceptance() -> None:
    """Gate called before data access. Cheap once accepted and up to date.

    Re-prompts if the agreement text has changed since it was accepted, checking
    at most once per `_VERSION_CHECK_TTL_SECONDS`. If the check cannot reach the
    network it proceeds on the existing acceptance — being offline must not
    block a user who has already agreed.
    """
    rec = _read_record()
    if rec is None:
        if _env_opt_in():
            _write_record(fetch_agreement_record(), _require_email(_env_email()), via="env")
            return
        accept_agreement()
        return

    accepted_sha = rec.get("agreement_sha256")
    if not accepted_sha or not _needs_version_recheck(rec):
        return

    current = fetch_agreement_record()
    if current is None or not current.get("sha256"):
        return  # offline or server down: honor the existing acceptance
    if current["sha256"] == accepted_sha:
        _touch_version_check(rec)
        return

    # The agreement changed; acceptance of the old text does not carry over.
    print("\nThe ISAAC Data Use Agreement has changed since you accepted it.",
          file=sys.stderr)
    if _env_opt_in():
        # Carry the address forward from the prior acceptance where we have one.
        _write_record(current, _require_email(rec.get("email") or _env_email()), via="env")
        return
    if not (sys.stdin and sys.stdin.isatty()):
        raise AgreementNotAccepted(
            "The ISAAC Data Use Agreement has changed and the new version has not "
            "been accepted, and no interactive terminal is available.\n"
            f"Review it at {AGREEMENT_PAGE}, then run `isaac-data accept-agreement` "
            f"or set {_ENV}=1."
        )
    _print_agreement(current, sys.stderr)
    resp = input("Do you accept the updated ISAAC Data Use Agreement? [y/N] ").strip().lower()
    if resp not in ("y", "yes"):
        raise AgreementNotAccepted(
            "The updated ISAAC Data Use Agreement was not accepted; aborting."
        )
    _write_record(current, rec.get("email") or _env_email(), via="prompt")


# ---- Pre-2026-07-25 aliases ("Terms of Use" era). Kept for compatibility. ----
TermsNotAccepted = AgreementNotAccepted
TERMS_PAGE = AGREEMENT_PAGE
TERMS_RAW = AGREEMENT_RAW
fetch_terms = fetch_agreement
accept_terms = accept_agreement
