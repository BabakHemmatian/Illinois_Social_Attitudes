"""Deprecated alias for :mod:`isaac_data.agreement`.

The ISAAC "Terms of Use" was renamed to the "Data Use Agreement" on 2026-07-25.
This module re-exports the new API under the old names so existing imports keep
working; new code should use ``isaac_data.agreement``.
"""
from __future__ import annotations

from .agreement import (  # noqa: F401
    AGREEMENT_PAGE,
    AGREEMENT_RAW,
    TERMS_PAGE,
    TERMS_RAW,
    AgreementNotAccepted,
    TermsNotAccepted,
    accept_agreement,
    accept_terms,
    fetch_agreement,
    fetch_terms,
    is_accepted,
    require_acceptance,
    status,
    withdraw,
)

__all__ = [
    "AGREEMENT_PAGE", "AGREEMENT_RAW", "TERMS_PAGE", "TERMS_RAW",
    "AgreementNotAccepted", "TermsNotAccepted",
    "accept_agreement", "accept_terms", "fetch_agreement", "fetch_terms",
    "is_accepted", "require_acceptance", "status", "withdraw",
]
