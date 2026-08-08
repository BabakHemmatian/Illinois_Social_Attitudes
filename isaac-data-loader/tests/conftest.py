"""Shared test setup for the Data-Use-Agreement gate.

Two things every test in this suite needs:

1. An email address. Accepting the agreement requires one (see
   `isaac_data.agreement`), and tests opt in non-interactively via
   ``ISAAC_ACCEPT_AGREEMENT``, where there is no prompt to fall back on.

2. No live consent POSTs. Acceptance is normally posted to the ISAAC server,
   and ``ISAAC_BASE_URL`` defaults to production — so without this the suite
   would write junk rows into the real consent log on every run. The transport
   is stubbed rather than pointed at a fake host because the same base URL is
   used to fetch the manifest, which tests do want to reach.
"""
import pytest

import isaac_data.agreement as agreement

# Reserved by RFC 2606, so it can never collide with a real address.
TEST_EMAIL = "pytest@example.invalid"


@pytest.fixture(autouse=True)
def agreement_test_env(monkeypatch):
    monkeypatch.setenv("ISAAC_AGREEMENT_EMAIL", TEST_EMAIL)
    monkeypatch.setattr(agreement, "_post_consent", lambda rec, timeout=10: False)
