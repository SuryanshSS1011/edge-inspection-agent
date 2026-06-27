"""M8: the scripted demo runs all five beats, including the cut -> queue -> reconnect ->
reconcile sequence, with zero PII egress."""

from demo.demo_runner import run_demo


def test_demo_runs_end_to_end(capsys):
    result = run_demo()
    assert result["pii_bytes_out"] == 0           # the measured privacy claim
    assert result["reconciled"] == 1              # the deferred item synced on reconnect
    assert len(result["events"]) == 3

    out = capsys.readouterr().out
    # The killer beat must be visible on screen.
    assert "NETWORK CUT" in out
    assert "NETWORK RESTORED" in out
    assert "queued" in out
    assert "back-filled" in out
