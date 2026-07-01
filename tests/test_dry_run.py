"""The deploy dry-run harness must pass in mock mode (no API key / network), so it can
gate a deploy in CI. The live path is exercised manually via `python -m cloud.fc_deploy.dry_run`.
"""

from cloud.fc_deploy.dry_run import run


def test_dry_run_mock_passes():
    # mock=True skips the Qwen call but exercises the real handler + CloudClient path.
    assert run(mock=True) == 0
