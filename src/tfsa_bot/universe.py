"""What the strategy can buy — everything tradeable on Wealthsimple (TSX + US exchanges + crypto ETFs).

CAD-listed sleeves trade commission-free with no FX. US-listed sleeves (`usd=True`) cost
~1.5% FX on each buy and each sell in a CAD TFSA, unless you hold a USD account — the
backtest charges this per trade so US assets have to earn their keep.

Most Canadian ETFs are young, so for backtesting we splice in history from a proxy
(converted to CAD when it's USD-priced) before the ETF existed. `proxy_drag` is an annual
haircut on proxy history only, approximating the MER + foreign withholding tax the
Canadian ETF would have paid.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Sleeve:
    key: str
    ticker: str  # what you actually buy on Wealthsimple
    name: str
    kind: str  # equity | sector | bond | real_estate | commodity | crypto | cash
    usd: bool = False  # ticker itself is US-listed (priced in USD, FX fee to trade)
    proxy: str | None = None  # history before `ticker` existed
    proxy_fx: bool = False  # proxy is priced in USD -> convert to CAD
    proxy_drag: float = 0.0  # annual drag applied to proxy returns
    max_weight: float = 1.0

    @property
    def market(self) -> str:
        return "US" if self.usd else "CA"


SLEEVES: list[Sleeve] = [
    # --- broad equity (CAD-listed) ---
    Sleeve("ca_eq", "XIC.TO", "Canada (TSX Composite)", "equity"),
    Sleeve("us_eq", "VFV.TO", "US S&P 500 (unhedged)", "equity", proxy="SPY", proxy_fx=True, proxy_drag=0.003),
    Sleeve("nasdaq", "QQC.TO", "Nasdaq-100 (unhedged)", "equity", proxy="QQQ", proxy_fx=True, proxy_drag=0.003),
    Sleeve("intl", "XEF.TO", "International developed", "equity", proxy="EFA", proxy_fx=True, proxy_drag=0.004),
    Sleeve("em", "XEC.TO", "Emerging markets", "equity", proxy="EEM", proxy_fx=True, proxy_drag=0.004),
    # --- Canadian sectors ---
    Sleeve("ca_banks", "ZEB.TO", "Canadian banks", "sector", proxy="XFN.TO"),
    Sleeve("ca_energy", "XEG.TO", "Canadian energy", "sector"),
    Sleeve("ca_tech", "XIT.TO", "Canadian tech", "sector"),
    Sleeve("ca_reits", "ZRE.TO", "Canadian REITs", "real_estate", proxy="XRE.TO"),
    # --- defensive (CAD-listed) ---
    Sleeve("bonds", "ZAG.TO", "Canadian bonds (aggregate)", "bond", proxy="XBB.TO"),
    Sleeve("gold", "CGL-C.TO", "Gold (unhedged)", "commodity", proxy="GLD", proxy_fx=True, proxy_drag=0.004),
    # --- crypto ETFs (CAD-listed) ---
    Sleeve("btc", "FBTC.TO", "Bitcoin", "crypto", proxy="BTC-USD", proxy_fx=True, proxy_drag=0.004, max_weight=0.15),
    Sleeve("eth", "ETHX-B.TO", "Ether", "crypto", proxy="ETH-USD", proxy_fx=True, proxy_drag=0.004, max_weight=0.10),
    # --- US-listed (FX fee applies) ---
    Sleeve("us_small", "IWM", "US small caps (Russell 2000)", "equity", usd=True),
    Sleeve("semis", "SMH", "Semiconductors", "sector", usd=True),
    Sleeve("us_tsy", "TLT", "US long Treasuries (20y+)", "bond", usd=True),
    Sleeve("commod", "DBC", "Broad commodities", "commodity", usd=True),
    Sleeve("silver", "SLV", "Silver", "commodity", usd=True),
    Sleeve("uranium", "URA", "Uranium & nuclear", "sector", usd=True, max_weight=0.15),
]

CASH = Sleeve("cash", "CASH.TO", "High-interest savings ETF", "cash", proxy="FRED:IR3TIB01CAM156N")

FX_TICKER = "CAD=X"  # CAD per 1 USD

# Market-stress inputs (not tradeable here): volatility index, junk vs treasury bonds, S&P 500.
STRESS_TICKERS = ["^VIX", "HYG", "IEF", "SPY"]

# The backtest starts when all of these have history (~late 2004); others join as they list.
CORE_KEYS = ["ca_eq", "us_eq", "intl", "em", "bonds", "gold", "cash"]

BY_KEY = {s.key: s for s in SLEEVES + [CASH]}


# ---- watchlist: scanned for signals, never part of the model portfolio ----------------------------
#
# Everything here is CAD-listed (no FX fee on Wealthsimple). The scan gives each one the same weekly
# trend signal as the ETFs above, as information: the strategy was only backtested on SLEEVES, and
# `strategy.target_weights` never sees the watchlist. Single stocks are far noisier than ETFs.
#
# CDRs (Canadian Depositary Receipts, Cboe Canada) are CAD-hedged slices of US stocks. Yahoo has no
# usable history for them, so the signal is computed on the US stock in USD (which is what a hedged
# CDR tracks) less `CDR_DRAG` a year for the hedge.

CDR_DRAG = 0.006


@dataclass(frozen=True)
class Watch:
    ticker: str  # what you buy on Wealthsimple
    name: str
    kind: str  # equity | sector | bond | commodity | crypto | stock | cdr | custom
    source: str | None = None  # Yahoo ticker that carries the price history, when it isn't `ticker`
    drag: float = 0.0  # annual drag applied to the source's returns

    @property
    def history(self) -> str:
        return self.source or self.ticker


def _cdr(us: str, name: str, ticker: str | None = None) -> Watch:
    return Watch(ticker or f"{us}.NE", f"{name} (CDR, CAD-hedged)", "cdr", source=us, drag=CDR_DRAG)


WATCHLIST: list[Watch] = [
    # --- broad equity and regions ---
    Watch("XEQT.TO", "All-equity portfolio (the buy-and-hold benchmark)", "equity"),
    Watch("XQQ.TO", "Nasdaq-100 (CAD-hedged)", "equity"),
    Watch("XSU.TO", "US small caps (CAD-hedged)", "equity"),
    Watch("XMD.TO", "Canadian mid caps", "equity"),
    Watch("XCS.TO", "Canadian small caps", "equity"),
    Watch("ZLB.TO", "Canadian low-volatility stocks", "equity"),
    Watch("VDY.TO", "Canadian high-dividend stocks", "equity"),
    Watch("XEU.TO", "Europe", "equity"),
    Watch("ZJPN.TO", "Japan", "equity"),
    Watch("ZID.TO", "India", "equity"),
    Watch("ZCH.TO", "China", "equity"),
    # --- sectors and themes ---
    Watch("XFN.TO", "Canadian financials", "sector"),
    Watch("XMA.TO", "Canadian materials", "sector"),
    Watch("XGD.TO", "Gold miners", "sector"),
    Watch("ZUT.TO", "Canadian utilities", "sector"),
    Watch("XST.TO", "Canadian consumer staples", "sector"),
    Watch("ZIN.TO", "Canadian industrials", "sector"),
    Watch("XHC.TO", "Global health care", "sector"),
    Watch("ZUH.TO", "US health care (CAD-hedged)", "sector"),
    Watch("ZUB.TO", "US banks (CAD-hedged)", "sector"),
    Watch("TEC.TO", "Global technology leaders", "sector"),
    Watch("CHPS.TO", "Semiconductors", "sector"),
    Watch("CYBR.TO", "Cybersecurity", "sector"),
    Watch("HURA.TO", "Uranium", "sector"),
    Watch("COPP.TO", "Copper miners", "sector"),
    Watch("ZCLN.TO", "Clean energy", "sector"),
    Watch("ZGI.TO", "Global infrastructure", "sector"),
    # --- bonds ---
    Watch("XSB.TO", "Canadian short-term bonds", "bond"),
    Watch("ZFL.TO", "Canadian long federal bonds", "bond"),
    Watch("XRB.TO", "Canadian real-return bonds", "bond"),
    Watch("ZTL.NE", "US long Treasuries (20y+)", "bond"),
    Watch("ZHY.TO", "US high-yield bonds (CAD-hedged)", "bond"),
    Watch("ZPR.TO", "Canadian preferred shares", "bond"),
    # --- commodities and crypto ---
    Watch("SVR.TO", "Silver (CAD-hedged)", "commodity"),
    Watch("HUC.TO", "Crude oil (futures)", "commodity"),
    Watch("SOLQ.TO", "Solana", "crypto"),
    # --- large Canadian stocks ---
    Watch("RY.TO", "Royal Bank", "stock"),
    Watch("TD.TO", "TD Bank", "stock"),
    Watch("BMO.TO", "Bank of Montreal", "stock"),
    Watch("MFC.TO", "Manulife", "stock"),
    Watch("BN.TO", "Brookfield", "stock"),
    Watch("SHOP.TO", "Shopify", "stock"),
    Watch("CSU.TO", "Constellation Software", "stock"),
    Watch("TRI.TO", "Thomson Reuters", "stock"),
    Watch("CNQ.TO", "Canadian Natural Resources", "stock"),
    Watch("SU.TO", "Suncor", "stock"),
    Watch("ENB.TO", "Enbridge", "stock"),
    Watch("CNR.TO", "CN Rail", "stock"),
    Watch("CP.TO", "CPKC Rail", "stock"),
    Watch("WCN.TO", "Waste Connections", "stock"),
    Watch("ATD.TO", "Couche-Tard", "stock"),
    Watch("DOL.TO", "Dollarama", "stock"),
    Watch("L.TO", "Loblaw", "stock"),
    Watch("AEM.TO", "Agnico Eagle", "stock"),
    Watch("CCO.TO", "Cameco", "stock"),
    Watch("NTR.TO", "Nutrien", "stock"),
    # --- US stocks as CDRs ---
    _cdr("NVDA", "Nvidia"),
    _cdr("AAPL", "Apple"),
    _cdr("MSFT", "Microsoft"),
    _cdr("AMZN", "Amazon"),
    _cdr("GOOG", "Alphabet"),
    _cdr("META", "Meta"),
    _cdr("TSLA", "Tesla"),
    _cdr("AVGO", "Broadcom"),
    _cdr("AMD", "AMD"),
    _cdr("NFLX", "Netflix"),
    _cdr("PLTR", "Palantir"),
    _cdr("UBER", "Uber"),
    _cdr("COST", "Costco"),
    _cdr("WMT", "Walmart"),
    _cdr("DIS", "Disney"),
    _cdr("LLY", "Eli Lilly"),
    _cdr("JPM", "JPMorgan"),
    _cdr("BRK-B", "Berkshire Hathaway", "BRK.NE"),
    _cdr("V", "Visa", "VISA.NE"),
    _cdr("MA", "Mastercard"),
]
