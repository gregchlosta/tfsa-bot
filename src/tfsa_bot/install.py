"""Install helpers: macOS launchd schedule and Claude Desktop MCP registration."""

import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LABEL = "com.tfsabot.run"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
RUNNER = REPO / "bin" / "run-cycle.sh"
LOG = REPO / "data" / "run.log"
DESKTOP_CFG = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
TIMES = [(9, 45), (12, 30), (16, 45)]  # local time, Mon-Fri


def install_schedule() -> None:
    uv = shutil.which("uv") or "/opt/homebrew/bin/uv"
    RUNNER.parent.mkdir(exist_ok=True)
    LOG.parent.mkdir(exist_ok=True)
    # The API key lives in ~/.zshrc; launchd doesn't load it, so pull just that one export line.
    RUNNER.write_text(f"""#!/bin/zsh
eval "$(grep -E '^export ANTHROPIC_API_KEY=' "$HOME/.zshrc" 2>/dev/null)"
cd "{REPO}"
echo "=== $(date) ===" >> "{LOG}"
"{uv}" run tfsa-bot run >> "{LOG}" 2>&1
""")
    RUNNER.chmod(0o755)
    intervals = "".join(
        f"<dict><key>Weekday</key><integer>{d}</integer><key>Hour</key><integer>{h}</integer>"
        f"<key>Minute</key><integer>{m}</integer></dict>"
        for d in range(1, 6) for h, m in TIMES
    )
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    PLIST.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>{LABEL}</string>
<key>ProgramArguments</key><array><string>{RUNNER}</string></array>
<key>StartCalendarInterval</key><array>{intervals}</array>
<key>StandardErrorPath</key><string>{LOG}</string>
</dict></plist>
""")
    subprocess.run(["launchctl", "unload", str(PLIST)], capture_output=True)
    subprocess.run(["launchctl", "load", str(PLIST)], check=True)
    print(f"Installed {PLIST}\nRuns Mon-Fri at {', '.join(f'{h}:{m:02d}' for h, m in TIMES)}; log: {LOG}")


def uninstall_schedule() -> None:
    subprocess.run(["launchctl", "unload", str(PLIST)], capture_output=True)
    PLIST.unlink(missing_ok=True)
    print("Removed schedule.")


def install_desktop() -> None:
    uv = shutil.which("uv") or "/opt/homebrew/bin/uv"
    cfg = json.loads(DESKTOP_CFG.read_text()) if DESKTOP_CFG.exists() else {}
    backup = DESKTOP_CFG.with_suffix(".json.bak-tfsabot")
    if DESKTOP_CFG.exists() and not backup.exists():
        shutil.copy2(DESKTOP_CFG, backup)
    cfg.setdefault("mcpServers", {})["tfsa-bot"] = {
        "command": uv, "args": ["--directory", str(REPO), "run", "tfsa-bot", "mcp"],
    }
    DESKTOP_CFG.write_text(json.dumps(cfg, indent=2))
    print(f"Registered 'tfsa-bot' in {DESKTOP_CFG} (backup: {backup}). Restart Claude Desktop.")
