import { useEffect, useState } from "react";
import "./App.css";

type RiskProfile = "conservative" | "balanced" | "aggressive";

interface MatchFromApi {
  id: number;
  league: string;
  kickoff: string;
  home_team: string;
  away_team: string;
  home_logo?: string;
  away_logo?: string;
  odds_home: number;
  odds_draw: number;
  odds_away: number;
  home_win_prob: number; // modelo
  draw_prob: number;     // modelo
  away_win_prob: number; // modelo
  over25_prob: number;
  btts_prob: number;
  model_goals: number;    // goles totales modelo
  model_corners: number;  // córners totales modelo
  edge_home: number;
  edge_draw: number;
  edge_away: number;
  edge_best: number;
  best_market: "home" | "draw" | "away";
  risk_level: "low" | "medium" | "high";
}

interface MatchEnriched extends MatchFromApi {
  prob_book: {
    home: number;
    draw: number;
    away: number;
  };
  prob_model: {
    home: number;
    draw: number;
    away: number;
  };
  edge: {
    home: number;
    draw: number;
    away: number;
    best: number;
  };
  main_pick: "home" | "draw" | "away";
}

const RISK_CONFIG: Record<
  RiskProfile,
  { label: string; description: string; minEdge: number; riskLabel: string }
> = {
  conservative: {
    label: "Conservador",
    description: "Valor controlado, picks más estables.",
    minEdge: 0.05,
    riskLabel: "Baja",
  },
  balanced: {
    label: "Balanceado",
    description: "Equilibrio entre seguridad y edge.",
    minEdge: 0.03,
    riskLabel: "Media",
  },
  aggressive: {
    label: "Agresivo",
    description: "Buscando edges grandes y más volatilidad.",
    minEdge: 0.0,
    riskLabel: "Alta",
  },
};

function formatPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function formatEdgePct(edge: number): string {
  const pct = edge * 100;
  if (Math.abs(pct) < 0.1) return "<0.1%";
  return `${pct.toFixed(1)}%`;
}

function formatKickoff(kickoff: string, league: string): string {
  const d = new Date(kickoff);
  if (Number.isNaN(d.getTime())) return league;
  const days = ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"];
  const months = [
    "ene",
    "feb",
    "mar",
    "abr",
    "may",
    "jun",
    "jul",
    "ago",
    "sep",
    "oct",
    "nov",
    "dic",
  ];

  const dayName = days[d.getDay()];
  const day = String(d.getDate()).padStart(2, "0");
  const month = months[d.getMonth()];

  let hours = d.getHours();
  const minutes = String(d.getMinutes()).padStart(2, "0");
  const suffix = hours >= 12 ? "p.m." : "a.m.";
  if (hours === 0) hours = 12;
  else if (hours > 12) hours -= 12;

  return `${dayName} ${day} de ${month}, ${hours}:${minutes} ${suffix} · ${league}`;
}

function enrichMatch(m: MatchFromApi): MatchEnriched {
  // Probabilidades implícitas de cuotas 1X2
  const invH = 1 / m.odds_home;
  const invD = 1 / m.odds_draw;
  const invA = 1 / m.odds_away;
  const k = invH + invD + invA;

  const bookHome = invH / k;
  const bookDraw = invD / k;
  const bookAway = invA / k;

  const modelHome = m.home_win_prob;
  const modelDraw = m.draw_prob;
  const modelAway = m.away_win_prob;

  const edgeHome = modelHome - bookHome;
  const edgeDraw = modelDraw - bookDraw;
  const edgeAway = modelAway - bookAway;

  const edgeMap: Record<"home" | "draw" | "away", number> = {
    home: edgeHome,
    draw: edgeDraw,
    away: edgeAway,
  };

  // pick principal según mejor edge
  let main_pick: "home" | "draw" | "away" = "home";
  let bestEdge = edgeHome;

  (["draw", "away"] as const).forEach((k) => {
    if (edgeMap[k] > bestEdge) {
      bestEdge = edgeMap[k];
      main_pick = k;
    }
  });

  return {
    ...m,
    prob_book: {
      home: bookHome,
      draw: bookDraw,
      away: bookAway,
    },
    prob_model: {
      home: modelHome,
      draw: modelDraw,
      away: modelAway,
    },
    edge: {
      home: edgeHome,
      draw: edgeDraw,
      away: edgeAway,
      best: bestEdge,
    },
    main_pick,
  };
}

function getRiskClass(risk: MatchFromApi["risk_level"]): string {
  if (risk === "low") return "risk-low";
  if (risk === "medium") return "risk-medium";
  return "risk-high";
}

function getRiskLabel(risk: MatchFromApi["risk_level"]): string {
  if (risk === "low") return "Baja";
  if (risk === "medium") return "Media";
  return "Alta";
}

function getPickLabel(pick: "home" | "draw" | "away"): string {
  if (pick === "home") return "Local";
  if (pick === "draw") return "Empate";
  return "Visitante";
}

function getOver25Text(p: number): string {
  if (p >= 0.7) return "Partido muy propenso al Over 2.5 goles.";
  if (p >= 0.6) return "Ligera inclinación al Over 2.5 goles.";
  if (p >= 0.45) return "Escenario equilibrado en la línea de 2.5 goles.";
  return "Tendencia a partido cerrado (Under 2.5 más probable).";
}

function getBttsText(p: number): string {
  if (p >= 0.7) return "Alta probabilidad de que marquen ambos equipos (BTTS).";
  if (p >= 0.55) return "Ligera ventaja para el BTTS (ambos marcan).";
  if (p >= 0.45) return "BTTS muy 50/50 según el modelo.";
  return "Más probable que NO marquen ambos equipos.";
}

function App() {
  const [riskProfile, setRiskProfile] = useState<RiskProfile>("balanced");
  const [matches, setMatches] = useState<MatchEnriched[]>([]);
  const [selectedMatch, setSelectedMatch] = useState<MatchEnriched | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  async function fetchMatches() {
    try {
      setLoading(true);
      setErrorMsg(null);

      const res = await fetch("http://localhost:9100/future-matches?days=3");

      if (!res.ok) {
        throw new Error(
          `Backend respondió ${res.status}. Revisa que esté levantado en el puerto 9100.`
        );
      }

      const raw: MatchFromApi[] = await res.json();
      const enriched = raw.map(enrichMatch);
      setMatches(enriched);
    } catch (err) {
      console.error(err);
      setErrorMsg(
        "No se pudieron cargar los partidos. Revisa que el backend esté en el puerto 9100 (/future-matches)."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchMatches();
  }, []);

  const riskCfg = RISK_CONFIG[riskProfile];

  const filteredMatches = matches.filter((m) => m.edge.best >= riskCfg.minEdge);
  const listToShow = riskProfile === "aggressive" ? matches : filteredMatches;

  // Pick destacado: mejor edge entre todos los partidos
  const topPick: MatchEnriched | null =
    matches.length > 0
      ? matches.reduce((best, m) =>
          m.edge.best > best.edge.best ? m : best
        )
      : null;

  // Info del pick destacado
  let topOdds = 0;
  let topPickLabel = "";
  let topRiskLabel = "";
  let topRiskClass = "";

  if (topPick) {
    if (topPick.main_pick === "home") topOdds = topPick.odds_home;
    else if (topPick.main_pick === "draw") topOdds = topPick.odds_draw;
    else topOdds = topPick.odds_away;

    topPickLabel = getPickLabel(topPick.main_pick);
    topRiskLabel = getRiskLabel(topPick.risk_level);
    topRiskClass = getRiskClass(topPick.risk_level);
  }

  return (
    <div className="app-root">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-avatar">IA</div>
          <div>
            <h1 className="brand-title">PredictIA Fútbol</h1>
            <p className="brand-subtitle">
              Motor probabilístico de valor en partidos.
            </p>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button className="nav-item active">
            <span>Radar de valor</span>
          </button>
          <button className="nav-item">
            <span>Parleys & combinadas</span>
            <span className="nav-pill">Próximamente</span>
          </button>
          <button className="nav-item">
            <span>Zona premium</span>
            <span className="nav-pill">Próximamente</span>
          </button>
          <button className="nav-item">
            <span>Configuración</span>
          </button>
        </nav>
      </aside>

      {/* Main layout */}
      <div className="main-layout">
        <header className="main-header">
          <div>
            <h2>Opciones de valor</h2>
            <p>
              Explora combinadas y herramientas para monetizar mejor tus picks
              del modelo.
            </p>
          </div>
        </header>

        {/* PICK DESTACADO */}
        <section className="section hero-section">
          <div className="hero-header">
            <h3>Pick destacado del modelo</h3>
          </div>

          {topPick ? (
            <div className="hero-card">
              <div className="hero-main">
                <p className="hero-kickoff">
                  {formatKickoff(topPick.kickoff, topPick.league)}
                </p>

                <div className="hero-teams">
                  <div className="team-block big">
                    {topPick.home_logo && (
                      <img
                        src={topPick.home_logo}
                        alt={topPick.home_team}
                        className="team-logo"
                      />
                    )}
                    <span>{topPick.home_team}</span>
                  </div>
                  <span className="vs-label">vs</span>
                  <div className="team-block big">
                    {topPick.away_logo && (
                      <img
                        src={topPick.away_logo}
                        alt={topPick.away_team}
                        className="team-logo"
                      />
                    )}
                    <span>{topPick.away_team}</span>
                  </div>
                </div>

                <p className="hero-pick">
                  Ganador 1X2: <strong>{topPickLabel}</strong>
                </p>
                <p className="hero-note">
                  Basado en el modelo de probabilidad interna (Poisson + ajustes
                  de mercado).
                </p>
              </div>

              <div className="hero-odds">
                <p className="hero-odds-label">Cuota</p>
                <p className="hero-odds-value">{topOdds.toFixed(2)}</p>
                <p className="hero-odds-sub">
                  Prob. modelo:{" "}
                  <strong>
                    {formatPct(
                      topPick.prob_model[topPick.main_pick] as number
                    )}
                  </strong>{" "}
                  · Prob. casa:{" "}
                  <strong>
                    {formatPct(
                      topPick.prob_book[topPick.main_pick] as number
                    )}
                  </strong>
                </p>
                <p className="hero-odds-sub">
                  Over 2.5:{" "}
                  <strong>{formatPct(topPick.over25_prob)}</strong> · BTTS:{" "}
                  <strong>{formatPct(topPick.btts_prob)}</strong>
                </p>
              </div>

              <div className="hero-side">
                <span className={`risk-badge ${topRiskClass}`}>
                  Riesgo: {topRiskLabel}
                </span>
                <p className="hero-edge">
                  Edge modelo:{" "}
                  <strong>{formatEdgePct(topPick.edge.best)}</strong>
                </p>
                <button
                  className="btn primary"
                  onClick={() => setSelectedMatch(topPick)}
                >
                  Ver explicación detallada
                </button>
                <p className="hero-tip">
                  Click para abrir el análisis completo del partido.
                </p>
              </div>
            </div>
          ) : (
            <p className="info-msg">
              Cargando pick destacado… espera mientras se obtienen los
              partidos.
            </p>
          )}
        </section>

        {/* OPCIONES DE PARLEY (burbuja blanca) */}
        <section className="section parley-section">
          <div className="parley-grid">
            <div className="parley-bubble">
              <h4>Parley selección doble</h4>
              <p>2 picks de alto edge en un solo ticket.</p>
              <span className="parley-tag">Próximamente</span>
            </div>
            <div className="parley-bubble">
              <h4>Parley mayor (3+)</h4>
              <p>Combinadas de 3 o más partidos con filtros de probabilidad.</p>
              <span className="parley-tag">Próximamente</span>
            </div>
            <div className="parley-bubble">
              <h4>Parley goles & BTTS</h4>
              <p>Solo Over 2.5 y ambos marcan, basado en el modelo.</p>
              <span className="parley-tag">Próximamente</span>
            </div>
            <div className="parley-bubble">
              <h4>Zona premium</h4>
              <p>Ideas para picks exclusivos, stakes y gestión de banca.</p>
              <span className="parley-tag">Próximamente</span>
            </div>
          </div>
        </section>

        {/* RADAR COMPLETO */}
        <section className="section">
          <h3>Todos los partidos con valor</h3>
          <p className="section-subtitle">
            Vista completa del radar de valor. Selecciona tu perfil de riesgo y
            explora los picks del modelo.
          </p>

          <div className="risk-selector">
            {(["conservative", "balanced", "aggressive"] as RiskProfile[]).map(
              (rp) => (
                <button
                  key={rp}
                  className={
                    "risk-btn" + (riskProfile === rp ? " risk-btn-active" : "")
                  }
                  onClick={() => setRiskProfile(rp)}
                >
                  <span className="risk-label">{RISK_CONFIG[rp].label}</span>
                  <span className="risk-desc">
                    {RISK_CONFIG[rp].description}
                  </span>
                </button>
              )
            )}
          </div>

          <p className="risk-summary">
            Perfil seleccionado:{" "}
            <strong>{RISK_CONFIG[riskProfile].label}</strong> · Riesgo{" "}
            <strong>{RISK_CONFIG[riskProfile].riskLabel}</strong> · Edge mínimo
            mostrado:{" "}
            <strong>{formatEdgePct(RISK_CONFIG[riskProfile].minEdge)}</strong>.
          </p>

          {loading && (
            <p className="info-msg">Cargando partidos con valor…</p>
          )}

          {!loading && errorMsg && <p className="error-msg">{errorMsg}</p>}

          {!loading && !errorMsg && listToShow.length === 0 && (
            <p className="info-msg">
              No hay partidos que cumplan el filtro de edge para este perfil de
              riesgo. Prueba con un perfil más agresivo o revisa los parámetros
              del modelo.
            </p>
          )}

          {!loading && !errorMsg && listToShow.length > 0 && (
            <div className="cards-grid">
              {listToShow.map((m) => {
                const riskClass = getRiskClass(m.risk_level);
                const riskLabel = getRiskLabel(m.risk_level);
                const pickLabel = getPickLabel(m.main_pick);

                return (
                  <article
                    key={m.id}
                    className="card match-card"
                    onClick={() => setSelectedMatch(m)}
                  >
                    <div className="match-header">
                      <div>
                        <div className="teams-row">
                          <div className="team-block">
                            {m.home_logo && (
                              <img
                                src={m.home_logo}
                                alt={m.home_team}
                                className="team-logo"
                              />
                            )}
                            <span>{m.home_team}</span>
                          </div>
                          <span className="vs-label">vs</span>
                          <div className="team-block">
                            {m.away_logo && (
                              <img
                                src={m.away_logo}
                                alt={m.away_team}
                                className="team-logo"
                              />
                            )}
                            <span>{m.away_team}</span>
                          </div>
                        </div>
                        <p className="match-kickoff">
                          {formatKickoff(m.kickoff, m.league)}
                        </p>
                      </div>

                      <span className={`risk-badge ${riskClass}`}>
                        Riesgo: {riskLabel}
                      </span>
                    </div>

                    <div className="match-body">
                      <div className="match-col">
                        <h4>Prob. casa</h4>
                        <p>Local: {formatPct(m.prob_book.home)}</p>
                        <p>Empate: {formatPct(m.prob_book.draw)}</p>
                        <p>Visita: {formatPct(m.prob_book.away)}</p>
                      </div>

                      <div className="match-col">
                        <h4>Prob. modelo</h4>
                        <p>Local: {formatPct(m.prob_model.home)}</p>
                        <p>Empate: {formatPct(m.prob_model.draw)}</p>
                        <p>Visita: {formatPct(m.prob_model.away)}</p>
                      </div>

                      <div className="match-col highlight-col">
                        <p className="best-market">
                          🧠 Valor principal: {pickLabel} · Edge{" "}
                          {formatEdgePct(m.edge.best)}
                        </p>
                        <p className="odds-line">
                          Cuotas 1X2: {m.odds_home.toFixed(2)} /{" "}
                          {m.odds_draw.toFixed(2)} / {m.odds_away.toFixed(2)}
                        </p>
                        <p className="odds-line small-metrics">
                          Over 2.5 (modelo): {formatPct(m.over25_prob)} · BTTS:{" "}
                          {formatPct(m.btts_prob)}
                        </p>
                        <p className="odds-line small-metrics">
                          Goles totales modelo: {m.model_goals.toFixed(2)} ·
                          {"  "}Córners totales: {m.model_corners.toFixed(1)}
                        </p>
                        <p className="cta-text">
                          Click para ver explicación detallada
                        </p>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <footer className="footer">
          Modelo Poisson simplificado para goles/córners. No hay resultados
          garantizados.
        </footer>
      </div>

      {/* MODAL */}
      {selectedMatch && (
        <div
          className="modal-overlay"
          onClick={() => setSelectedMatch(null)}
        >
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h2>
                  {selectedMatch.home_team} vs {selectedMatch.away_team}
                </h2>
                <p className="match-kickoff">
                  {formatKickoff(selectedMatch.kickoff, selectedMatch.league)}
                </p>
              </div>
              <button
                className="close-btn"
                onClick={() => setSelectedMatch(null)}
              >
                ×
              </button>
            </div>

            <div className="modal-section">
              <h3>✅ Pick principal del modelo</h3>
              <p className="modal-main-pick">
                {`Pick: ${getPickLabel(
                  selectedMatch.main_pick
                )} con probabilidad del modelo ${formatPct(
                  selectedMatch.prob_model[selectedMatch.main_pick] as number
                )} frente a ${formatPct(
                  selectedMatch.prob_book[selectedMatch.main_pick] as number
                )} implícita en las cuotas.`}
              </p>
              <p>
                Edge estimado:{" "}
                <strong>{formatEdgePct(selectedMatch.edge.best)}</strong> sobre
                la probabilidad implícita de la casa.
              </p>
            </div>

            <div className="modal-section">
              <h3>📊 Comparación completa</h3>
              <div className="modal-grid">
                <div>
                  <h4>Prob. casa (cuotas)</h4>
                  <p>Local: {formatPct(selectedMatch.prob_book.home)}</p>
                  <p>Empate: {formatPct(selectedMatch.prob_book.draw)}</p>
                  <p>Visita: {formatPct(selectedMatch.prob_book.away)}</p>
                </div>
                <div>
                  <h4>Prob. modelo</h4>
                  <p>Local: {formatPct(selectedMatch.prob_model.home)}</p>
                  <p>Empate: {formatPct(selectedMatch.prob_model.draw)}</p>
                  <p>Visita: {formatPct(selectedMatch.prob_model.away)}</p>
                </div>
                <div>
                  <h4>Edge por resultado</h4>
                  <p>Local: {formatEdgePct(selectedMatch.edge.home)}</p>
                  <p>Empate: {formatEdgePct(selectedMatch.edge.draw)}</p>
                  <p>Visita: {formatEdgePct(selectedMatch.edge.away)}</p>
                </div>
              </div>
            </div>

            <div className="modal-section">
              <h3>⚽ Proyección de goles & corners</h3>
              <p>
                Goles totales esperados por el modelo:{" "}
                <strong>{selectedMatch.model_goals.toFixed(2)}</strong> goles.
              </p>
              <p>
                Córners totales esperados:{" "}
                <strong>{selectedMatch.model_corners.toFixed(1)}</strong>.
              </p>
              <p>
                Over 2.5 (modelo):{" "}
                <strong>{formatPct(selectedMatch.over25_prob)}</strong> · BTTS:{" "}
                <strong>{formatPct(selectedMatch.btts_prob)}</strong>.
              </p>
              <p>
                {getOver25Text(selectedMatch.over25_prob)}{" "}
                {getBttsText(selectedMatch.btts_prob)}
              </p>
            </div>

            <div className="modal-section">
              <h3>⚠️ Nota sobre riesgo</h3>
              <p>
                Este pick está clasificado como riesgo{" "}
                <strong>{getRiskLabel(selectedMatch.risk_level)}</strong>.
                Ajusta tu stake según tu tolerancia al riesgo y banca total.
                Esto es una ayuda cuantitativa, no una garantía de resultados.
              </p>
            </div>

            <div className="modal-footer">
              <button
                className="btn"
                onClick={() => setSelectedMatch(null)}
              >
                Cerrar
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
