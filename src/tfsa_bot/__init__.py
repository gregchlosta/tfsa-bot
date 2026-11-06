import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(prog="tfsa-bot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("research", help="full backtest report")
    rn = sub.add_parser("run", help="one full cycle: data, news check, decisions, notify (what the scheduler runs)")
    rn.add_argument("--no-llm", action="store_true")
    rn.add_argument("--quiet", action="store_true", help="no notifications")
    sub.add_parser("brief", help="show recommendation using cached data and the latest news check")
    sub.add_parser("performance", help="forward-test results")
    sub.add_parser("btc", help="Bitcoin pot: weekly signal, live score, flip price, trend")
    bs = sub.add_parser("btc-share", help="set the Bitcoin pot's share of the account (0 = off)")
    bs.add_argument("share", type=float, help="e.g. 0.5 for 50%%")
    lt = sub.add_parser("log-trade", help="record a fill")
    lt.add_argument("side", choices=["BUY", "SELL", "buy", "sell"])
    lt.add_argument("ticker")
    lt.add_argument("shares", type=float)
    lt.add_argument("price", type=float)
    lt.add_argument("--date")
    dp = sub.add_parser("deposit", help="record cash added to the TFSA")
    dp.add_argument("amount", type=float)
    dp.add_argument("--date")
    sub.add_parser("mcp", help="run the MCP server (stdio)")
    sub.add_parser("install-schedule", help="install macOS launchd job (weekdays 09:45, 12:30, 16:45)")
    sub.add_parser("uninstall-schedule", help="remove the launchd job")
    sub.add_parser("install-desktop", help="register the MCP server in Claude Desktop")
    args = ap.parse_args()

    if args.cmd == "research":
        from .research import main as research
        research()
    elif args.cmd in ("run", "brief"):
        from . import engine
        if args.cmd == "run":
            s = engine.run_cycle(refresh=True, use_llm=False if args.no_llm else None, send=not args.quiet)
        else:
            s = engine.run_cycle(refresh=False, with_news=False, send=False)
        print(engine.brief_text(s))
    elif args.cmd == "performance":
        from .data import build_returns
        from .paper import report
        print(json.dumps(report(build_returns()), indent=2))
    elif args.cmd == "btc":
        from . import engine
        s = engine.run_cycle(refresh=True, with_news=False, send=False)
        print(engine.btc_text(s["btc"]) if s["btc"] else "Bitcoin pot is off. Turn it on: tfsa-bot btc-share 0.5")
    elif args.cmd == "btc-share":
        from . import db
        if not 0 <= args.share <= 1:
            raise SystemExit("share must be between 0 and 1")
        cfg = db.config()
        cfg["btc_share"] = args.share
        db.save_config(cfg)
        print(f"Bitcoin pot set to {args.share:.0%} of the account. Run `tfsa-bot brief` for the trades.")
    elif args.cmd == "log-trade":
        from . import portfolio
        tk = portfolio.record_trade(args.ticker, args.side, args.shares, args.price, args.date)
        print(f"logged {args.side.upper()} {args.shares} {tk} @ {args.price}")
    elif args.cmd == "deposit":
        from . import db, portfolio
        db.add_cashflow(args.date or portfolio.today(), args.amount)
        print(f"logged deposit {args.amount:,.2f}")
    elif args.cmd == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()
    elif args.cmd in ("install-schedule", "uninstall-schedule", "install-desktop"):
        from . import install
        getattr(install, args.cmd.replace("-", "_"))()
