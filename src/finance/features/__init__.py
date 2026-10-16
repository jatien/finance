"""Capa 2: features d'estat de mercat i panell sectorial, point-in-time.

    inputs.py       MarketData + càrrega des del lake (preus, macro PIT, amplitud S&P 500)
    transforms.py   transformacions pures i causals (SMA, vol, momentum, correlació...)
    market.py       catàleg i càlcul de les features de mercat
    sectors.py      panell per ETF de sector/indústria amb percentil transversal
    diagnostics.py  cobertura, correlació de Spearman, redundàncies, informe HTML
    job.py          job `features` (escriu al lake) i lectors per a les capes superiors
"""

from finance.features.config import FeatureConfig
from finance.features.inputs import MarketData, load_market_data
from finance.features.market import CATALOG, compute_market_features, percentile_ranks
from finance.features.sectors import compute_sector_panel

__all__ = ["CATALOG", "FeatureConfig", "MarketData", "compute_market_features",
           "compute_sector_panel", "load_market_data", "percentile_ranks"]
