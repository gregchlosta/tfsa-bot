"""Notifications: macOS banner always; phone push via ntfy.sh if `ntfy_topic` is set in data/config.json.

ntfy topics are public-by-obscurity: pick a long random topic name (e.g. tfsa-<random 20 chars>)
and subscribe to it in the ntfy phone app.
"""

import subprocess
import urllib.request

from . import db


def send(title: str, body: str) -> None:
    try:
        script = f"display notification {_q(body)} with title {_q('TFSA bot: ' + title)}"
        subprocess.run(["osascript", "-e", script], check=False, timeout=10, capture_output=True)
    except Exception:
        pass
    topic = db.config().get("ntfy_topic")
    if topic:
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{topic}", data=body.encode(), headers={"Title": f"TFSA bot: {title}"}
            )
            urllib.request.urlopen(req, timeout=15).read()
        except Exception:
            pass


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
