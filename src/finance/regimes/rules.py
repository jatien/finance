"""Classificador de règims per regles: dos eixos amb histèresi + sobreescriptura de crisi.

    tendència   bull / bear       mercat vs SMA200, banda ±2 %, 3 sessions de confirmació
    volatilitat calm / volatile   percentil expansiu del VIX, entra a 0,75 i surt a 0,60
    crisi       sí / no           corba del VIX invertida + crèdit deteriorant-se

    règim = crisi si n'hi ha; si no, "<tendència>_<volatilitat>"

Tot és causal: l'estat de la data t només depèn de les features <= t (que ja són
point-in-time). L'exposició decidida al tancament de t s'aplica al retorn de t+1; aquest
desfasament el fa l'avaluació (`evaluation.overlay_returns`), mai aquí.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finance.regimes.config import REGIMES, RegimeConfig


def hysteresis(enter: pd.Series, exit_: pd.Series, valid: pd.Series, init: pd.Series,
               confirm_enter: int = 1, confirm_exit: int = 1) -> pd.Series:
    """Màquina d'estats binària amb confirmació. Retorna 1.0 (actiu), 0.0 o NaN (sense dades).

    - Inactiu → actiu quan `enter` és cert `confirm_enter` sessions seguides.
    - Actiu → inactiu quan `exit_` és cert `confirm_exit` sessions seguides.
    - Sessions amb `valid` fals: es manté l'estat i els comptadors (no hi ha informació nova).
    - Primera sessió vàlida: l'estat és `init`.
    """
    en = enter.fillna(False).to_numpy(dtype=bool)
    ex = exit_.fillna(False).to_numpy(dtype=bool)
    ok = valid.fillna(False).to_numpy(dtype=bool)
    ini = init.fillna(False).to_numpy(dtype=bool)
    out = np.full(len(en), np.nan)
    state: bool | None = None
    c_on = c_off = 0
    for t in range(len(en)):
        if not ok[t]:
            if state is not None:
                out[t] = float(state)
            continue
        if state is None:
            state = bool(ini[t])
        elif not state:
            c_on = c_on + 1 if en[t] else 0
            if c_on >= confirm_enter:
                state, c_on, c_off = True, 0, 0
        else:
            c_off = c_off + 1 if ex[t] else 0
            if c_off >= confirm_exit:
                state, c_on, c_off = False, 0, 0
        out[t] = float(state)
    return pd.Series(out, index=enter.index)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df else pd.Series(np.nan, index=df.index, dtype=float)


def trend_state(values: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
    """1.0 = bear, 0.0 = bull."""
    d = _col(values, "mkt_dist_sma200")
    return hysteresis(d < -cfg.trend_band, d > cfg.trend_band, d.notna(), d < 0,
                      cfg.confirm_days, cfg.confirm_days)


def vol_state(pct: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
    """1.0 = volàtil, 0.0 = calma."""
    vp = _col(pct, "vix")
    return hysteresis(vp >= cfg.vol_enter_pct, vp < cfg.vol_exit_pct, vp.notna(),
                      vp >= cfg.vol_enter_pct, cfg.confirm_days, cfg.confirm_days)


def crisis_state(values: pd.DataFrame, pct: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
    """1.0 = crisi. Entrada immediata, sortida amb `crisis_confirm_exit` sessions.

    Sense VIX3M (abans de 2006) no hi ha crisi, tret que `crisis_fallback` estigui actiu.
    """
    term = _col(values, "vix_term")
    vixp = _col(pct, "vix")
    credit = _col(pct, "baa_spread_chg_3m")
    has_term = term.notna()
    if cfg.crisis_fallback:
        acute = np.where(has_term, term >= cfg.crisis_term_enter, vixp >= cfg.crisis_vix_pct_fallback)
        calm = np.where(has_term, term < cfg.crisis_term_exit, vixp < cfg.crisis_vix_pct_exit_fallback)
    else:
        acute = has_term & (term >= cfg.crisis_term_enter)
        calm = ~has_term | (term < cfg.crisis_term_exit)
    acute = pd.Series(acute, index=values.index)
    calm = pd.Series(calm, index=values.index)
    enter = acute & (credit >= cfg.crisis_credit_pct)
    valid = vixp.notna() & credit.notna()
    return hysteresis(enter, calm, valid, pd.Series(False, index=values.index),
                      1, cfg.crisis_confirm_exit)


def classify(values: pd.DataFrame, pct: pd.DataFrame, cfg: RegimeConfig | None = None) -> pd.DataFrame:
    """Règim diari a partir de les features (valors i percentils expansius).

    Columnes: trend, vol, crisis, regime, equity_exposure, days_in_regime, action.
    `action` compara l'exposició amb la del dia anterior: add | reduce | hold | init.
    """
    cfg = cfg or RegimeConfig()
    idx = values.index
    pct = pct.reindex(idx)
    tr = trend_state(values, cfg)
    vo = vol_state(pct, cfg) if cfg.use_vol_axis else pd.Series(0.0, index=idx).where(tr.notna())
    cr = crisis_state(values, pct, cfg) if cfg.use_crisis else pd.Series(0.0, index=idx)

    defined = tr.notna() & vo.notna()
    trend = np.where(tr == 1, "bear", "bull")
    vol = np.where(vo == 1, "volatile", "calm")
    base = pd.Series(np.char.add(np.char.add(trend.astype(str), "_"), vol.astype(str)), index=idx)
    regime = base.where(cr.fillna(0) != 1, "crisis").where(defined)

    exposure = regime.map(cfg.exposure).astype(float)
    # Durada del règim actual (sessions, comptant la d'avui)
    change = regime.ne(regime.shift()) | regime.isna()
    run_id = change.cumsum()
    days = regime.groupby(run_id).cumcount() + 1
    prev = exposure.shift()
    action = np.select(
        [exposure.isna(), prev.isna(), exposure > prev + 1e-12, exposure < prev - 1e-12],
        ["", "init", "add", "reduce"], default="hold")

    out = pd.DataFrame({
        "trend": pd.Series(trend, index=idx).where(tr.notna()),
        "vol": pd.Series(vol, index=idx).where(vo.notna()),
        "crisis": (cr == 1).where(cr.notna()),
        "regime": regime,
        "equity_exposure": exposure,
        "days_in_regime": days.where(defined),
        "action": pd.Series(action, index=idx).replace("", np.nan),
    }, index=idx)
    unknown = set(out["regime"].dropna()) - set(REGIMES)
    if unknown:  # pragma: no cover - protecció davant canvis de noms
        raise ValueError(f"règims desconeguts: {unknown}")
    return out
