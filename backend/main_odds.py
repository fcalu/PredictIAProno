from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from datetime import date, timedelta
import asyncio
import logging

import httpx
import numpy as np
from scipy.stats import poisson

# -------------------------------------------------------------------
# CONFIGURACIÓN BÁSICA
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictia-odds")

app = FastAPI(title="PredictIA Odds · ESPN + Poisson")

# CORS abierto para desarrollo (ajusta en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

URL_BASE_ESPN = "http://site.api.espn.com/apis/site/v2/sports"
HOME = "home"
AWAY = "away"

# Ligas que queremos usar
LIGAS_IMPORTANTES: Dict[str, str] = {
    "Premier League": "eng.1",
    "LaLiga": "esp.1",
    "Liga MX": "mex.1",
    "Serie A": "ita.1",
    "UEFA Champions League": "uefa.champions",
    "Bundesliga": "ger.1",
    "Ligue 1": "fra.1",
    "Eredivisie": "ned.1",
}

# -------------------------------------------------------------------
# MODELOS DE RESPUESTA
# -------------------------------------------------------------------

class FutureMatchOut(BaseModel):
    id: int
    league: str
    kickoff: str
    home_team: str
    away_team: str
    home_logo: Optional[str] = None
    away_logo: Optional[str] = None

    odds_home: float
    odds_draw: float
    odds_away: float

    # probabilidades del modelo (Poisson)
    home_win_prob: float
    draw_prob: float
    away_win_prob: float

    # otras métricas del modelo
    over25_prob: float
    btts_prob: float
    model_goals: float
    model_corners: float

    # valor vs “book”
    edge_home: float
    edge_draw: float
    edge_away: float
    edge_best: float
    best_market: str  # "home" | "draw" | "away"

    # perfil de riesgo sugerido
    risk_level: str   # "low" | "medium" | "high"


# -------------------------------------------------------------------
# FUNCIONES AUXILIARES ESPN
# -------------------------------------------------------------------

def get_team_logo(team_obj: Dict[str, Any]) -> Optional[str]:
    """
    Extrae un logo usable desde el JSON de ESPN.
    """
    team = team_obj.get("team", team_obj)
    # formato 1: 'logo'
    if "logo" in team and isinstance(team["logo"], str):
        return team["logo"]
    # formato 2: 'logos': [{ href: ... }]
    logos = team.get("logos")
    if isinstance(logos, list) and logos:
        href = logos[0].get("href")
        if isinstance(href, str):
            return href
    return None


async def get_future_matches_data(
    league_slug: str,
    days_forward: int = 3,
) -> List[Dict[str, Any]]:
    """
    Obtiene próximos partidos de ESPN (estado = '1' = no empezado)
    para la liga indicada.
    """
    sport_slug = "soccer"
    hoy = date.today()

    result: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=6) as client:
        for offset in range(1, days_forward + 1):
            fecha = hoy + timedelta(days=offset)
            url = (
                f"{URL_BASE_ESPN}/{sport_slug}/{league_slug}/scoreboard"
                f"?dates={fecha.strftime('%Y%m%d')}"
            )

            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"[future] Error {e} en {url}")
                await asyncio.sleep(0.05)
                continue

            for event in data.get("events", []):
                comp = (event.get("competitions") or [{}])[0]
                status_id = comp.get("status", {}).get("type", {}).get("id")
                # 1 = not_started; 2 = in progress; 3 = final
                if status_id != "1":
                    continue

                competitors = comp.get("competitors", [])
                home_c = next((c for c in competitors if c.get("homeAway") == HOME), None)
                away_c = next((c for c in competitors if c.get("homeAway") == AWAY), None)
                if not home_c or not away_c:
                    continue

                try:
                    home_id = str(home_c["team"]["id"])
                    away_id = str(away_c["team"]["id"])
                except Exception:
                    continue

                result.append(
                    {
                        "event_id": event.get("id"),
                        "kickoff": event.get("date"),
                        "league_slug": league_slug,
                        "home_team": home_c["team"]["displayName"],
                        "away_team": away_c["team"]["displayName"],
                        "home_id": home_id,
                        "away_id": away_id,
                        "home_logo": get_team_logo(home_c),
                        "away_logo": get_team_logo(away_c),
                    }
                )

            await asyncio.sleep(0.05)

    return result


async def build_league_recent_stats(
    league_slug: str,
    days_back: int = 45,
) -> Dict[str, Dict[str, float]]:
    """
    Recorre los últimos X días de la liga en ESPN y calcula
    goles a favor/en contra por partido (total, casa, fuera) para cada equipo.
    """
    sport_slug = "soccer"
    hoy = date.today()

    team_stats: Dict[str, Dict[str, Any]] = {}
    total_goals = 0
    total_matches = 0

    async with httpx.AsyncClient(timeout=6) as client:
        for delta in range(1, days_back + 1):
            fecha = hoy - timedelta(days=delta)
            url = (
                f"{URL_BASE_ESPN}/{sport_slug}/{league_slug}/scoreboard"
                f"?dates={fecha.strftime('%Y%m%d')}"
            )

            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"[stats] Error {e} en {url}")
                await asyncio.sleep(0.05)
                continue

            for event in data.get("events", []):
                comp = (event.get("competitions") or [{}])[0]
                status_state = comp.get("status", {}).get("type", {}).get("state")
                # Solo partidos terminados
                if status_state not in ("post", "final", "status_final"):
                    continue

                competitors = comp.get("competitors", [])
                if len(competitors) != 2:
                    continue

                home_c = next((c for c in competitors if c.get("homeAway") == HOME), None)
                away_c = next((c for c in competitors if c.get("homeAway") == AWAY), None)
                if not home_c or not away_c:
                    continue

                try:
                    home_id = str(home_c["team"]["id"])
                    away_id = str(away_c["team"]["id"])
                    gh = int(home_c["score"])
                    ga = int(away_c["score"])
                except Exception:
                    continue

                total_goals += gh + ga
                total_matches += 1

                for team_id, gf, gc, is_home, comp_team in [
                    (home_id, gh, ga, True, home_c),
                    (away_id, ga, gh, False, away_c),
                ]:
                    ts = team_stats.setdefault(
                        team_id,
                        {
                            "name": comp_team["team"]["displayName"],
                            "matches": 0,
                            "gf": 0,
                            "ga": 0,
                            "matches_home": 0,
                            "gf_home": 0,
                            "ga_home": 0,
                            "matches_away": 0,
                            "gf_away": 0,
                            "ga_away": 0,
                        },
                    )
                    ts["matches"] += 1
                    ts["gf"] += gf
                    ts["ga"] += gc
                    if is_home:
                        ts["matches_home"] += 1
                        ts["gf_home"] += gf
                        ts["ga_home"] += gc
                    else:
                        ts["matches_away"] += 1
                        ts["gf_away"] += gf
                        ts["ga_away"] += gc

            await asyncio.sleep(0.05)

    if total_matches == 0:
        league_avg_gf = 1.35  # fallback razonable
    else:
        league_avg_gf = total_goals / (2 * total_matches)

    result: Dict[str, Dict[str, float]] = {}

    alpha = 0.7  # peso de los datos propios vs media liga

    for team_id, ts in team_stats.items():
        matches = max(ts["matches"], 1)
        gf_per = ts["gf"] / matches
        ga_per = ts["ga"] / matches

        mh = max(ts["matches_home"], 1)
        ma = max(ts["matches_away"], 1)

        gf_home = (ts["gf_home"] / mh) if ts["matches_home"] > 0 else gf_per
        ga_home = (ts["ga_home"] / mh) if ts["matches_home"] > 0 else ga_per
        gf_away = (ts["gf_away"] / ma) if ts["matches_away"] > 0 else gf_per
        ga_away = (ts["ga_away"] / ma) if ts["matches_away"] > 0 else ga_per

        result[team_id] = {
            "name": ts["name"],
            "gf_home": alpha * gf_home + (1 - alpha) * league_avg_gf,
            "ga_home": alpha * ga_home + (1 - alpha) * league_avg_gf,
            "gf_away": alpha * gf_away + (1 - alpha) * league_avg_gf,
            "ga_away": alpha * ga_away + (1 - alpha) * league_avg_gf,
            "league_avg_gf": league_avg_gf,
        }

    return result


# -------------------------------------------------------------------
# POISSON Y MODELO SIMPLE
# -------------------------------------------------------------------

def estimate_lambdas_from_stats(
    home_stats: Dict[str, float],
    away_stats: Dict[str, float],
) -> Tuple[float, float]:
    """
    Calcula λ_local y λ_visitante combinando ataque/defensa de ambos
    con la media de goles de la liga.
    """
    league_avg = home_stats.get("league_avg_gf", 1.35)

    home_attack = home_stats["gf_home"] / league_avg
    home_defense = league_avg / max(home_stats["ga_home"], 0.2)

    away_attack = away_stats["gf_away"] / league_avg
    away_defense = league_avg / max(away_stats["ga_away"], 0.2)

    lambda_home = league_avg * home_attack * away_defense
    lambda_away = league_avg * away_attack * home_defense

    # Ventaja de local ligera
    lambda_home *= 1.10
    lambda_away *= 0.95

    lambda_home = float(np.clip(lambda_home, 0.2, 3.5))
    lambda_away = float(np.clip(lambda_away, 0.2, 3.5))

    return lambda_home, lambda_away


def poisson_1x2_from_lambdas(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = 7,
) -> Tuple[float, float, float, float, float]:
    """
    P(H), P(D), P(A), Prob Over 2.5, Prob BTTS usando dos Poisson independientes.
    """
    p_home = p_draw = p_away = 0.0
    p_over25 = 0.0
    p_btts = 0.0

    for i in range(0, max_goals + 1):
        for j in range(0, max_goals + 1):
            p = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

            if i + j >= 3:
                p_over25 += p
            if i > 0 and j > 0:
                p_btts += p

    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    return p_home, p_draw, p_away, p_over25, p_btts


def make_book_odds_from_probs(
    p_home: float, p_draw: float, p_away: float, margin: float = 0.06
) -> Tuple[float, float, float, float, float, float]:
    """
    A partir de las probabilidades "justas" del modelo construye
    cuotas de la casa con un margen total aproximado.
    Devuelve (odds_home, odds_draw, odds_away, book_home, book_draw, book_away).
    """
    probs = np.array([p_home, p_draw, p_away], dtype=float)
    probs = np.maximum(probs, 0.01)
    probs /= probs.sum()

    # La casa aplica margen: reduce las probabilidades reales
    book_probs = probs * (1 - margin)
    book_probs /= book_probs.sum()

    odds = 1.0 / np.maximum(book_probs, 0.02)

    return (
        float(odds[0]),
        float(odds[1]),
        float(odds[2]),
        float(book_probs[0]),
        float(book_probs[1]),
        float(book_probs[2]),
    )


# -------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/future-matches", response_model=List[FutureMatchOut])
async def future_matches(days: int = 6):
    """
    Devuelve partidos de los próximos 'days' días de varias ligas,
    con probabilidades y métricas calculadas con Poisson a partir
    del historial reciente obtenido en ESPN.
    """
    all_matches: List[FutureMatchOut] = []
    match_id = 1

    for league_name, league_slug in LIGAS_IMPORTANTES.items():
        logger.info(f"Construyendo stats recientes para {league_name}...")
        stats = await build_league_recent_stats(league_slug, days_back=45)

        logger.info(f"Buscando partidos futuros para {league_name}...")
        futuros = await get_future_matches_data(league_slug, days_forward=days)

        for m in futuros:
            home_id = m["home_id"]
            away_id = m["away_id"]

            home_stats = stats.get(
                home_id,
                {
                    "gf_home": 1.3,
                    "ga_home": 1.3,
                    "gf_away": 1.3,
                    "ga_away": 1.3,
                    "league_avg_gf": 1.35,
                },
            )
            away_stats = stats.get(
                away_id,
                {
                    "gf_home": 1.3,
                    "ga_home": 1.3,
                    "gf_away": 1.3,
                    "ga_away": 1.3,
                    "league_avg_gf": 1.35,
                },
            )

            lambda_home, lambda_away = estimate_lambdas_from_stats(
                home_stats, away_stats
            )

            (
                p_home,
                p_draw,
                p_away,
                p_over25,
                p_btts,
            ) = poisson_1x2_from_lambdas(lambda_home, lambda_away)

            (
                odds_home,
                odds_draw,
                odds_away,
                book_home,
                book_draw,
                book_away,
            ) = make_book_odds_from_probs(p_home, p_draw, p_away)

            edge_home = p_home - book_home
            edge_draw = p_draw - book_draw
            edge_away = p_away - book_away

            # Mejor valor
            best_market_idx = int(
                np.argmax([edge_home, edge_draw, edge_away])
            )
            if best_market_idx == 0:
                best_market = "home"
                best_edge = edge_home
            elif best_market_idx == 1:
                best_market = "draw"
                best_edge = edge_draw
            else:
                best_market = "away"
                best_edge = edge_away

            # Nivel de riesgo: edges muy grandes = más riesgo
            if best_edge > 0.12:
                risk_level = "high"
            elif best_edge > 0.06:
                risk_level = "medium"
            else:
                risk_level = "low"

            total_goals = lambda_home + lambda_away

            # corners modelo muy simple: escalamos desde una base de 9.5
            base_corners = 9.5
            model_corners = base_corners * (total_goals / 2.7)

            all_matches.append(
                FutureMatchOut(
                    id=match_id,
                    league=league_name,
                    kickoff=m["kickoff"],
                    home_team=m["home_team"],
                    away_team=m["away_team"],
                    home_logo=m["home_logo"],
                    away_logo=m["away_logo"],
                    odds_home=round(odds_home, 2),
                    odds_draw=round(odds_draw, 2),
                    odds_away=round(odds_away, 2),
                    home_win_prob=p_home,
                    draw_prob=p_draw,
                    away_win_prob=p_away,
                    over25_prob=p_over25,
                    btts_prob=p_btts,
                    model_goals=total_goals,
                    model_corners=model_corners,
                    edge_home=edge_home,
                    edge_draw=edge_draw,
                    edge_away=edge_away,
                    edge_best=best_edge,
                    best_market=best_market,
                    risk_level=risk_level,
                )
            )
            match_id += 1

    # ordenar por fecha
    all_matches.sort(key=lambda x: x.kickoff)
    return all_matches
