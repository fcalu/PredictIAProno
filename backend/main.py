from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, List

# ==========================
# CONFIGURACIÓN APP
# ==========================

app = FastAPI(
    title="API de Predicciones de Fútbol",
    description="API con modelo RandomForest entrenado con cuotas Bet365",
    version="0.3.0",
)

# Frontend (Vite) corre aquí
origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchInput(BaseModel):
    home_team: str
    away_team: str
    odds_home: float
    odds_draw: float
    odds_away: float


# ==========================
# CARGAR MODELO ENTRENADO
# ==========================

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_odds_rf.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"No se encontró el modelo en {MODEL_PATH}. "
        f"Primero ejecuta 'python train_model_odds.py'."
    )

model_obj = joblib.load(MODEL_PATH)
model = model_obj["model"]
feature_cols = model_obj["feature_cols"]  # ['B365H', 'B365D', 'B365A']


# ==========================
# LÓGICA DE PREDICCIÓN
# ==========================

def compute_prediction_from_odds(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
):
    """
    Calcula:
    - prob_book: probabilidades implícitas de las cuotas
    - prob_model: probabilidades del modelo
    - edge: diferencia modelo - casa
    - edge_best: máximo edge
    - best_market: 'home' | 'draw' | 'away'
    - risk_level: 'low' | 'medium' | 'high'
    """

    # Prob implícitas de la casa
    p_home_book = 1.0 / odds_home
    p_draw_book = 1.0 / odds_draw
    p_away_book = 1.0 / odds_away

    s = p_home_book + p_draw_book + p_away_book
    if s == 0:
        raise ValueError("Suma de probabilidades del book es 0")

    p_home_book /= s
    p_draw_book /= s
    p_away_book /= s

    prob_book = {
        "home": float(p_home_book),
        "draw": float(p_draw_book),
        "away": float(p_away_book),
    }

    # Vector de features en el mismo orden de entrenamiento
    odds_map = {
        "B365H": odds_home,
        "B365D": odds_draw,
        "B365A": odds_away,
    }
    x = np.array([odds_map[col] for col in feature_cols]).reshape(1, -1)

    probas = model.predict_proba(x)[0]
    class_to_index = {cls: i for i, cls in enumerate(model.classes_)}

    p_home_model = float(probas[class_to_index["H"]])
    p_draw_model = float(probas[class_to_index["D"]])
    p_away_model = float(probas[class_to_index["A"]])

    prob_model = {
        "home": p_home_model,
        "draw": p_draw_model,
        "away": p_away_model,
    }

    edge = {
        "home": p_home_model - p_home_book,
        "draw": p_draw_model - p_draw_book,
        "away": p_away_model - p_away_book,
    }

    edge_best = max(edge.values())
    best_market = max(edge, key=edge.get)

    # Nivel de riesgo según edge máximo
    if edge_best >= 0.08:
        risk_level = "high"
    elif edge_best >= 0.04:
        risk_level = "medium"
    else:
        risk_level = "low"

    return prob_book, prob_model, edge, float(edge_best), best_market, risk_level


# ==========================
# ENDPOINTS BÁSICOS
# ==========================

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "API de predicciones funcionando con RandomForest entrenado 🚀",
    }


@app.post("/prediccion_demo")
def prediccion_demo(data: MatchInput):
    """
    Endpoint para un partido individual (lo que ya usabas).
    """
    prob_book, prob_model, edge, edge_best, best_market, risk_level = compute_prediction_from_odds(
        data.odds_home,
        data.odds_draw,
        data.odds_away,
    )

    return {
        "partido": f"{data.home_team} vs {data.away_team}",
        "prob_book": {k: round(v, 3) for k, v in prob_book.items()},
        "prob_model": {k: round(v, 3) for k, v in prob_model.items()},
        "edge": {k: round(v, 3) for k, v in edge.items()},
        "edge_best": round(edge_best, 3),
        "best_market": best_market,
        "risk_level": risk_level,
    }


@app.get("/predicciones_demo")
def predicciones_demo():
    """
    Lista de partidos DEMO ya evaluados por el modelo.
    Sirve como respaldo aunque luego uses /premier_proximos.
    """
    demo_matches = [
        {
            "id": 1,
            "home_team": "Arsenal",
            "away_team": "Manchester City",
            "kickoff": "2024-12-01 14:00",
            "odds_home": 2.40,
            "odds_draw": 3.60,
            "odds_away": 2.80,
        },
        {
            "id": 2,
            "home_team": "Liverpool",
            "away_team": "Chelsea",
            "kickoff": "2024-12-01 16:30",
            "odds_home": 1.95,
            "odds_draw": 3.80,
            "odds_away": 3.60,
        },
    ]

    results = []
    for m in demo_matches:
        prob_book, prob_model, edge, edge_best, best_market, risk_level = compute_prediction_from_odds(
            m["odds_home"],
            m["odds_draw"],
            m["odds_away"],
        )

        results.append(
            {
                "id": m["id"],
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "kickoff": m["kickoff"],
                "odds_home": m["odds_home"],
                "odds_draw": m["odds_draw"],
                "odds_away": m["odds_away"],
                "prob_book": prob_book,
                "prob_model": prob_model,
                "edge": edge,
                "edge_best": edge_best,
                "best_market": best_market,
                "risk_level": risk_level,
            }
        )

    return results

# ==========================
# ESPN: SOCCER (multi-liga)
# ==========================

SOCCER_SCOREBOARD_BASE = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
)

# 8 ligas principales (puedes editar nombres y códigos cuando quieras)
SOCCER_LEAGUES_DEFAULT: Dict[str, str] = {
    "eng.1": "Premier League (ING)",
    "esp.1": "LaLiga (ESP)",
    "ita.1": "Serie A (ITA)",
    "ger.1": "Bundesliga (ALE)",
    "fra.1": "Ligue 1 (FRA)",
    "ned.1": "Eredivisie (HOL)",
    "usa.1": "MLS (USA)",
    "mex.1": "Liga MX (MEX)",
}


def american_to_decimal(moneyline: float) -> Optional[float]:
    """
    Convierte una cuota americana (moneyline) a cuota decimal aproximada.
      +150 -> 2.5
      -200 -> 1.5
    """
    try:
        ml = float(moneyline)
    except (TypeError, ValueError):
        return None

    if ml == 0:
        return None

    if ml > 0:
        return 1.0 + (ml / 100.0)
    else:
        return 1.0 + (100.0 / abs(ml))


def extract_odds_from_comp(comp: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """
    Intenta extraer cuotas home/draw/away de la estructura de odds de ESPN.
    Si no podemos interpretarlas, devolvemos None.
    """
    odds_list = comp.get("odds") or []
    if not odds_list:
        return None

    for o in odds_list:
        # Casos típicos para soccer:
        home_obj = o.get("homeTeamOdds") or o.get("homeOdds")
        away_obj = o.get("awayTeamOdds") or o.get("awayOdds")
        draw_obj = o.get("drawOdds") or o.get("drawWinOdds")

        if not (home_obj and away_obj and draw_obj):
            continue

        def get_decimal_from_obj(obj: Dict[str, Any]) -> Optional[float]:
            if obj is None:
                return None

            # 1) Intentar campo decimal explícito
            for key in ("decimalOdds", "decimal", "dec"):
                if key in obj and obj[key] is not None:
                    try:
                        return float(obj[key])
                    except (TypeError, ValueError):
                        pass

            # 2) Moneyline americano
            ml = obj.get("moneyLine")
            if ml is not None:
                return american_to_decimal(ml)

            return None

        home_dec = get_decimal_from_obj(home_obj)
        away_dec = get_decimal_from_obj(away_obj)
        draw_dec = get_decimal_from_obj(draw_obj)

        if home_dec and draw_dec and away_dec:
            return float(home_dec), float(draw_dec), float(away_dec)

    return None


def fetch_fixtures_for_league(
    league_code: str,
    date_str: str,
) -> List[Dict[str, Any]]:
    """
    Llama al scoreboard de ESPN para una liga (eng.1, esp.1, etc.)
    y devuelve lista de partidos por jugar con cuotas.
    """
    url = SOCCER_SCOREBOARD_BASE.format(league=league_code)
    params: Dict[str, Any] = {"limit": 300, "dates": date_str}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error llamando a ESPN para {league_code}: {e}")
        return []

    data = resp.json()
    events = data.get("events", [])
    fixtures: List[Dict[str, Any]] = []

    for ev in events:
        competitions = ev.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]

        status = comp.get("status", {}).get("type", {})
        state = status.get("state")
        # Solo partidos que aún no empiezan
        if state != "pre":
            continue

        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        odds_tuple = extract_odds_from_comp(comp)
        if not odds_tuple:
            # Si no podemos interpretar las cuotas, saltamos el partido
            continue

        odds_home, odds_draw, odds_away = odds_tuple

        fixtures.append(
            {
                "event_id": ev.get("id"),
                "league_code": league_code,
                "home_team": home.get("team", {}).get("name"),
                "away_team": away.get("team", {}).get("name"),
                "kickoff": ev.get("date"),  # ISO 8601
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
            }
        )

    return fixtures


def enrich_with_model(fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pasa cada fixture por el modelo y añade prob_book, prob_model, edge, etc.
    """
    results: List[Dict[str, Any]] = []

    for m in fixtures:
        try:
            prob_book, prob_model, edge, edge_best, best_market, risk_level = compute_prediction_from_odds(
                m["odds_home"],
                m["odds_draw"],
                m["odds_away"],
            )
        except Exception as e:
            print(f"Error en compute_prediction_from_odds para {m.get('event_id')}: {e}")
            continue

        results.append(
            {
                "id": m["event_id"],
                "league_code": m["league_code"],
                "league_name": SOCCER_LEAGUES_DEFAULT.get(
                    m["league_code"], m["league_code"]
                ),
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "kickoff": m["kickoff"],
                "odds_home": m["odds_home"],
                "odds_draw": m["odds_draw"],
                "odds_away": m["odds_away"],
                "prob_book": prob_book,
                "prob_model": prob_model,
                "edge": edge,
                "edge_best": edge_best,
                "best_market": best_market,
                "risk_level": risk_level,
            }
        )

    return results


@app.get("/soccer_proximos")
def soccer_proximos(
    days: int = 3,
    leagues: Optional[str] = None,
    use_demo_if_empty: bool = True,
):
    """
    Devuelve partidos próximos de las ligas de soccer indicadas,
    entre hoy y los próximos `days` días (incluido hoy).

    Parámetros:
    - days: número de días a partir de hoy (>=1, por defecto 3).
    - leagues: lista de códigos separada por coma (ej: eng.1,esp.1,ita.1).
               Si es None, usa las 8 ligas por defecto.
    - use_demo_if_empty: si True, devuelve partidos demo si no se encuentra nada.
    """
    if days < 1:
        days = 1
    if days > 7:
        days = 7  # por si acaso

    if leagues:
        league_codes = [c.strip() for c in leagues.split(",") if c.strip()]
    else:
        league_codes = list(SOCCER_LEAGUES_DEFAULT.keys())

    all_fixtures: List[Dict[str, Any]] = []

    today = datetime.now(timezone.utc).date()

    for offset in range(days):
        date = today + timedelta(days=offset)
        date_str = date.strftime("%Y%m%d")

        for lc in league_codes:
            fixtures = fetch_fixtures_for_league(lc, date_str)
            all_fixtures.extend(fixtures)

    if not all_fixtures and use_demo_if_empty:
        # Reutilizamos la lógica de predicciones_demo como fallback
        from fastapi.encoders import jsonable_encoder

        demo_resp = predicciones_demo()
        return jsonable_encoder(demo_resp)

    enriched = enrich_with_model(all_fixtures)

    # Ordenar por mayor edge
    enriched.sort(key=lambda x: x["edge_best"], reverse=True)

    return enriched


# Mantengo un endpoint /premier_proximos para compatibilidad.
@app.get("/premier_proximos")
def premier_proximos(
    days: int = 1,
    use_demo_if_empty: bool = True,
):
    """
    Alias de soccer_proximos solo para Premier League (eng.1).
    """
    return soccer_proximos(
        days=days,
        leagues="eng.1",
        use_demo_if_empty=use_demo_if_empty,
    )
