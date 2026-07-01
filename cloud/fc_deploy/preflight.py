"""Deploy preflight: check everything that can block `s deploy` BEFORE running it.

Each check corresponds to a real failure we hit (or would hit) during deployment, so a
green preflight means the only remaining variable is the Alibaba account state (service
activation / risk hold), which no local check can clear.

    python -m cloud.fc_deploy.preflight

Exits non-zero if any hard check fails. Account-side items (FC activated, order not
suspended) are reported as reminders, not failures — they can't be checked locally.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _ok(msg):
    print(f"  ok    {msg}")


def _fail(msg):
    print(f"  FAIL  {msg}")


def _warn(msg):
    print(f"  note  {msg}")


def check_cli() -> bool:
    if shutil.which("s"):
        try:
            v = subprocess.run(["s", "--version"], capture_output=True, text=True, timeout=15)
            _ok(f"Serverless Devs CLI present ({v.stdout.strip().splitlines()[0] if v.stdout else 's'})")
            return True
        except Exception:
            _ok("Serverless Devs CLI present")
            return True
    _fail("`s` (Serverless Devs) not found — install: npm i -g @serverless-devs/s")
    return False


def check_syaml() -> bool:
    syaml = HERE / "s.yaml"
    if not syaml.is_file():
        _fail(f"s.yaml missing at {syaml}")
        return False
    text = syaml.read_text()
    ok = True
    # `code`, not the old `codeUri` (the exact bug that gave "Code config is empty").
    if "\n      code:" in text or text.strip().startswith("code:") or "\n  code:" in text or "      code: " in text:
        _ok("s.yaml uses the `code` field (fc3), not codeUri")
    elif "codeUri" in text:
        _fail("s.yaml uses `codeUri` — fc3 needs `code`; deploy fails with 'Code config is empty'")
        ok = False
    else:
        _warn("could not confirm the `code` field in s.yaml — verify manually")
    for key in ("region:", "runtime:", "customRuntimeConfig", "triggers"):
        if key not in text:
            _fail(f"s.yaml missing `{key}`")
            ok = False
    if "access:" in text:
        access = next((l.split(":", 1)[1].strip() for l in text.splitlines()
                       if l.strip().startswith("access:")), "?")
        _ok(f"s.yaml access profile = '{access}' (must exist in `s config get`)")
    return ok


def check_package() -> bool:
    """The code package must exclude the datasets, or the upload is ~1.5 GB."""
    fcignore = REPO / ".fcignore"
    ok = True
    if fcignore.is_file() and "data" in fcignore.read_text():
        _ok(".fcignore excludes data/ (keeps the package small)")
    else:
        _fail(".fcignore missing or does not exclude data/ — package would balloon to ~1.5 GB")
        ok = False
    data_dir = REPO / "data"
    if data_dir.is_dir():
        _warn("data/ exists locally (~1.5 GB) — relies on .fcignore to stay out of the zip")
    # the runtime entrypoints must be present
    for f in ("cloud/fc_deploy/handler.py", "cloud/fc_deploy/bootstrap",
              "cloud/mcp_server.py", "cloud/qwen_reason.py"):
        if (REPO / f).is_file():
            _ok(f"present: {f}")
        else:
            _fail(f"missing runtime file: {f}")
            ok = False
    boot = REPO / "cloud/fc_deploy/bootstrap"
    if boot.is_file() and os.access(boot, os.X_OK):
        _ok("bootstrap is executable")
    elif boot.is_file():
        _warn("bootstrap is not executable (chmod +x cloud/fc_deploy/bootstrap)")
    return ok


def check_env() -> bool:
    from edge.dotenv import load_dotenv
    load_dotenv()
    if os.environ.get("DASHSCOPE_API_KEY"):
        _ok("DASHSCOPE_API_KEY is set (function will receive its Qwen key)")
        return True
    _fail("DASHSCOPE_API_KEY not set — export it or put it in .env before `s deploy`")
    return False


def main() -> None:
    print("Deploy preflight (local checks):")
    hard = [check_cli(), check_syaml(), check_package(), check_env()]

    print("\nAccount-side (cannot be checked locally — verify in the Alibaba console):")
    _warn("Function Compute is ACTIVATED in the target region (ap-southeast-1)")
    _warn("the account order is NOT suspended (risk hold cleared)")
    _warn("the RAM user has AliyunFCFullAccess + AliyunOSSFullAccess")

    if all(hard):
        print("\nPREFLIGHT PASSED — local config is deploy-ready. Remaining blockers are "
              "account-side only.\nDeploy:  cd cloud/fc_deploy && set -a && source ../../.env "
              "&& set +a && s deploy")
        sys.exit(0)
    print("\nPREFLIGHT FAILED — fix the FAIL items above before deploying.")
    sys.exit(1)


if __name__ == "__main__":
    main()
