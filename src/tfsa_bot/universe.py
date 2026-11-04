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
