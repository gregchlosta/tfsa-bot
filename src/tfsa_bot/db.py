"""SQLite store: decisions per track, news assessments, run log. No account data: the bot only gives signals."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .data import DATA_DIR

DB_PATH = DATA_DIR / "tfsa.db"
CONFIG_PATH = DATA_DIR / "config.json"

DEFAULT_CONFIG = {
    "follow": "core",  # "core" (backtested) or "overlay" (core + news check, once it has proven itself)
    "use_llm": True,
    "ntfy_topic": None,  # set to a private random string to get phone pushes via ntfy.sh
    "btc_share": 0.0,  # share of the portfolio in the Bitcoin pot (btc.py); 0 = off
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
  track TEXT, as_of TEXT, created_at TEXT, weights TEXT, notes TEXT,
  PRIMARY KEY (track, as_of));
CREATE TABLE IF NOT EXISTS assessments (
  id INTEGER PRIMARY KEY, created_at TEXT, as_of TEXT, stress TEXT, llm TEXT, headlines TEXT, model TEXT);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY, created_at TEXT, summary TEXT);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    else:
        save_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# --- decisions -------------------------------------------------------------

def save_decision(track: str, as_of: str, weights: dict, notes: list[str] | None = None) -> None:
    with connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?)",
            (track, as_of, now(), json.dumps(weights), json.dumps(notes or [])),
        )


def decisions(track: str) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM decisions WHERE track=? ORDER BY as_of", (track,)).fetchall()
    return [{**dict(r), "weights": json.loads(r["weights"]), "notes": json.loads(r["notes"])} for r in rows]


# --- assessments -----------------------------------------------------------

def save_assessment(as_of: str, stress: dict, llm: dict | None, headlines: list, model: str | None) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO assessments (created_at, as_of, stress, llm, headlines, model) VALUES (?,?,?,?,?,?)",
            (now(), as_of, json.dumps(stress), json.dumps(llm), json.dumps(headlines), model),
        )


def latest_assessment(with_llm: bool = True) -> dict | None:
    q = "SELECT * FROM assessments " + ("WHERE llm != 'null' " if with_llm else "") + "ORDER BY id DESC LIMIT 1"
    with connect() as con:
        r = con.execute(q).fetchone()
    if not r:
        return None
    return {**dict(r), **{k: json.loads(r[k]) for k in ("stress", "llm", "headlines")}}


def save_run(summary: dict) -> None:
    with connect() as con:
        con.execute("INSERT INTO runs (created_at, summary) VALUES (?,?)", (now(), json.dumps(summary, default=str)))


def last_run() -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    return json.loads(r["summary"]) if r else None
