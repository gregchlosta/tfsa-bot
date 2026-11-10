import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(prog="tfsa-bot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("research", help="full backtest report")
    rn = sub.add_parser("run", help="one full cycle: data, news check, decisions, notify (what the scheduler runs)")
    rn.add_argument("--no-llm", action="store_true")
    rn.add_argument("--quiet", action="store_true", help="no notifications")
    sub.add_parser("brief", help="show the signal using cached data and the latest news check")
    sub.add_parser("performance", help="forward-test results")
    sub.add_parser("btc", help="Bitcoin trend signal: weekly IN/OUT, flip price, indicator panel, cycle clock")
    sg = sub.add_parser("signal", help="trend signal for one asset: a universe ETF or any Yahoo Finance ticker")
    sg.add_argument("ticker")
    sub.add_parser("mcp", help="run the MCP server (stdio)")
    sub.add_parser("install-schedule", help="install macOS launchd job (weekdays 09:45, 12:30, 16:45)")
    sub.add_parser("uninstall-schedule", help="remove the launchd job")
    sub.add_parser("install-desktop", help="register the MCP server in Claude Desktop")
    args = ap.parse_args()

    if args.cmd == "research":
        from .research import main as research
        research()
    elif args.cmd in ("run", "brief"):
        from . import engine, report
        if args.cmd == "run":
            s = engine.run_cycle(refresh=True, use_llm=False if args.no_llm else None, send=not args.quiet)
        else:
            s = engine.run_cycle(refresh=False, with_news=False, send=False)
        print(report.brief_text(s))
    elif args.cmd == "performance":
        from .data import build_returns
        from .paper import report
        print(json.dumps(report(build_returns()), indent=2))
    elif args.cmd == "btc":
        from . import engine, report
        m = engine.load_market(refresh=True)
        print(report.btc_text(engine.bitcoin_report(m, engine.Params())))
    elif args.cmd == "signal":
        from . import engine, report
        print(report.signal_text(engine.asset_report(engine.load_market(), args.ticker)))
    elif args.cmd == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()
    elif args.cmd in ("install-schedule", "uninstall-schedule", "install-desktop"):
        from . import install
        getattr(install, args.cmd.replace("-", "_"))()
