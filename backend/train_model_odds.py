import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Ruta del CSV (ajusta el nombre si el archivo se llama distinto)
# Ejemplos:
#   "england-premier-league-matches-2024-to-2025-stats (1).csv"
#   "EPL_5_seasons_combined.csv"
CSV_PATH = os.path.join("data", "epl_2024_2025.csv")


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # NOMBRES REALES DE TU ARCHIVO (Football-Data):
    # - FTHG: Full Time Home Goals
    # - FTAG: Full Time Away Goals
    # - B365H, B365D, B365A: cuotas 1X2 de Bet365
    home_goals_col = "FTHG"
    away_goals_col = "FTAG"
    odd_home_col = "B365H"
    odd_draw_col = "B365D"
    odd_away_col = "B365A"

    # Comprobar que las columnas existen
    required_cols = [
        home_goals_col,
        away_goals_col,
        odd_home_col,
        odd_draw_col,
        odd_away_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    # Crear target 1X2 a partir de los goles
    def result_row(row):
        if row[home_goals_col] > row[away_goals_col]:
            return "H"  # gana local
        elif row[home_goals_col] < row[away_goals_col]:
            return "A"  # gana visitante
        else:
            return "D"  # empate

    df["result_1x2"] = df.apply(result_row, axis=1)

    # Features: cuotas (podríamos usar 1/odds también, pero de momento usamos odds brutas)
    X = df[[odd_home_col, odd_draw_col, odd_away_col]].copy()
    y = df["result_1x2"]

    # Quitar filas con NaN por si acaso
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]

    return X, y


def train_and_save_model():
    X, y = load_data(CSV_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy en test (solo usando cuotas): {acc:.3f}")

    # Guardar modelo y nombres de features
    model_obj = {
        "model": model,
        "feature_cols": list(X.columns),  # ['B365H', 'B365D', 'B365A']
    }

    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "model_odds_rf.pkl")
    joblib.dump(model_obj, out_path)
    print(f"Modelo guardado en: {out_path}")


if __name__ == "__main__":
    train_and_save_model()
