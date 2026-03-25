import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const RISK_LABELS = { high: "Élevé", medium: "Moyen", low: "Faible", unknown: "—" };
const RISK_COLORS = { high: "#ef4444", medium: "#f97316", low: "#22c55e", unknown: "#888" };

const MODEL_LABELS = {
  logistic:      "Rég. Log.",
  xgboost:       "XGBoost",
  randomforest:  "Rand. Forest",
  gradientboost: "Grad. Boost",
  lightgbm:      "LightGBM",
  decisiontree:  "Dec. Tree",
  adaboost:      "AdaBoost",
  extratrees:    "Extra Trees",
  bagging:       "Bagging",
};
const MODEL_KEYS = [
  "logistic", "xgboost", "randomforest", "gradientboost",
  "lightgbm", "decisiontree", "adaboost", "extratrees", "bagging",
];

function RiskBadge({ risk }) {
  return (
    <span className={`risk-badge risk-${risk}`}>
      {RISK_LABELS[risk] ?? "—"}
    </span>
  );
}

function MiniBar({ score, risk }) {
  const segments = 20;
  const filled   = Math.round((score / 100) * segments);
  const color    = RISK_COLORS[risk];
  return (
    <div className="mini-bar-wrap" title={`${score}%`}>
      {Array.from({ length: segments }).map((_, i) => (
        <div
          key={i}
          className="mini-bar-seg"
          style={{
            background: i < filled
              ? color
              : undefined,
            opacity: i < filled ? (0.55 + (i / segments) * 0.45) : undefined,
          }}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SHAP Modal  — fetches on open, caches result
// ---------------------------------------------------------------------------
function ShapModal({ client, modelKey, onClose }) {
  const [shap,    setShap]    = useState(null);   // null = loading
  const [error,   setError]   = useState(null);
  const isFallback = modelKey === "adaboost" || modelKey === "bagging";

  useEffect(() => {
    let cancelled = false;
    setShap(null);
    setError(null);
    axios.get(`http://localhost:8000/shap/${client.client_id}/${modelKey}`)
      .then(r => { if (!cancelled) setShap(r.data.shap ?? []); })
      .catch(() => { if (!cancelled) setError("Erreur lors du chargement."); });
    return () => { cancelled = true; };
  }, [client.client_id, modelKey]);

  const maxAbs = shap ? Math.max(...shap.map((s) => Math.abs(s.shap_value)), 0.001) : 1;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box shap-box" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Explications SHAP</h2>
            <p className="modal-sub">
              {MODEL_LABELS[modelKey]} — Client #{client.client_id}
              {isFallback && <span className="modal-sub-note"> · importances de variables</span>}
            </p>
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        {error ? (
          <p className="shap-empty">{error}</p>
        ) : shap === null ? (
          <p className="shap-loading">Chargement des explications…</p>
        ) : shap.length === 0 ? (
          <p className="shap-empty">Aucune donnée disponible pour ce client.</p>
        ) : (
          <>
            <div className="shap-list">
              {shap.map((item, i) => {
                const pct = (Math.abs(item.shap_value) / maxAbs) * 100;
                const pos = item.shap_value >= 0;
                return (
                  <div key={i} className="shap-row">
                    <div className="shap-feature" title={item.feature}>{item.feature}</div>
                    <div className="shap-bar-track">
                      <div
                        className={`shap-bar-fill ${pos ? "shap-pos" : "shap-neg"}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <div className={`shap-val ${pos ? "shap-val-pos" : "shap-val-neg"}`}>
                      {pos ? "+" : ""}{item.shap_value.toFixed(3)}
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="shap-legend">
              <span className="shap-pos-dot" /> Augmente le risque &nbsp;|&nbsp;
              <span className="shap-neg-dot" /> Réduit le risque
            </p>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Details Modal
// ---------------------------------------------------------------------------
function DetailsModal({ client, onClose, onOpenShap }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box details-box" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Détails — Client #{client.client_id}</h2>
            <p className="modal-sub">Analyse multi-modèles</p>
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        {/* Final risk */}
        <div className="detail-final">
          <span className="detail-final-label">Décision finale :</span>
          <RiskBadge risk={client.final_risk} />
          {client.avg_score != null && (
            <span className="detail-avg-score">
              Score moyen : <strong style={{ color: RISK_COLORS[client.final_risk] }}>
                {client.avg_score}%
              </strong>
              &nbsp;sur {Object.values(client.models).filter(m => m.score != null).length} modèles
            </span>
          )}
        </div>

        {/* Per-model blocks */}
        <div className="detail-models">
          {MODEL_KEYS.map((key) => {
            const m = client.models[key];
            if (!m || m.score === null) return null;
            return (
              <div key={key} className="detail-model-block">

                {/* Score row */}
                <div className="detail-model-row">
                  <div className="detail-model-name">{MODEL_LABELS[key]}</div>
                  <div className="detail-model-score" style={{ color: RISK_COLORS[m.risk] }}>
                    {m.score}%
                  </div>
                  <MiniBar score={m.score} risk={m.risk} />
                  <RiskBadge risk={m.risk} />
                  <button
                    className="btn-expl"
                    onClick={() => onOpenShap(key)}
                    title={`Voir les valeurs SHAP pour ${MODEL_LABELS[key]}`}
                  >
                    📊 Voir explication
                  </button>
                </div>

                {/* Factors grid */}
                {(m.facteurs_risque?.length > 0 || m.facteurs_protecteurs?.length > 0) && (
                  <div className="detail-factors-grid">
                    {m.facteurs_risque?.length > 0 && (
                      <div className="detail-factors-col">
                        <div className="factors-title risk-title">⚠️ Facteurs de risque</div>
                        <ul>
                          {m.facteurs_risque.map((f, i) => <li key={i}>{f}</li>)}
                        </ul>
                      </div>
                    )}
                    {m.facteurs_protecteurs?.length > 0 && (
                      <div className="detail-factors-col">
                        <div className="factors-title protect-title">✅ Facteurs protecteurs</div>
                        <ul>
                          {m.facteurs_protecteurs.map((f, i) => <li key={i}>{f}</li>)}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {/* Recommendations */}
                {m.recommandations?.length > 0 && (
                  <div className="detail-model-reco">
                    <div className="detail-reco-title">💡 Pour améliorer les chances d'approbation</div>
                    <ul>
                      {m.recommandations.map((r, i) => <li key={i}>{r}</li>)}
                    </ul>
                  </div>
                )}

              </div>
            );
          })}
        </div>

        {/* Bar chart */}
        <div className="detail-chart">
          <div className="chart-title">Comparaison des modèles</div>
          <div className="chart-columns">
            {MODEL_KEYS.map((key) => {
              const m = client.models[key];
              if (!m || m.score === null) return null;
              return (
                <div key={key} className="chart-col">
                  <div className="chart-col-bar-wrap">
                    <div
                      className="chart-col-bar-fill"
                      style={{ height: `${m.score}%`, background: RISK_COLORS[m.risk] }}
                    />
                  </div>
                  <div className="chart-col-pct">{m.score}%</div>
                  <div className="chart-col-label">{MODEL_LABELS[key]}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------
function App() {
  const [clientId,   setClientId]   = useState("");
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);

  const [data,       setData]       = useState(null);

  const [clients,    setClients]    = useState([]);

  const [selected,   setSelected]   = useState(null);
  const [shapModel,  setShapModel]  = useState(null);

  const [filterRisk, setFilterRisk] = useState("all");
  const [sortBy,     setSortBy]     = useState("final");
  const [searchQ,    setSearchQ]    = useState("");

  // -------------------------------------------------------------------------
  const handleAnalyze = async () => {
    if (!clientId) return;
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await axios.get(`http://127.0.0.1:8000/predict_all/${clientId}`);
      const d   = res.data;

      if (d.error) {
        setError("Client ID introuvable.");
      } else {
        setData(d.card);
        setClients((prev) => {
          const entry = { client_id: d.client_id, models: d.models,
                          final_risk: d.final_risk, avg_score: d.avg_score };
          const idx = prev.findIndex((c) => c.client_id === d.client_id);
          if (idx >= 0) {
            const copy = [...prev];
            copy[idx] = entry;
            return copy;
          }
          return [entry, ...prev];
        });
      }
    } catch {
      setError("Erreur lors de la récupération des données.");
    }

    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleAnalyze();
  };

  // -------------------------------------------------------------------------
  const SORT_ORDER = { high: 2, medium: 1, low: 0, unknown: -1 };

  const displayed = clients
    .filter((c) => {
      if (filterRisk !== "all" && c.final_risk !== filterRisk) return false;
      if (searchQ && !String(c.client_id).includes(searchQ.trim())) return false;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === "final") {
        return SORT_ORDER[b.final_risk] - SORT_ORDER[a.final_risk];
      }
      const sa = a.models[sortBy]?.score ?? -1;
      const sb = b.models[sortBy]?.score ?? -1;
      return sb - sa;
    });

  // -------------------------------------------------------------------------
  const openDetails = (client) => {
    setSelected(client);
    setShapModel(null);
  };

  const openShap = (modelKey) => setShapModel(modelKey);

  const closeDetails = () => {
    setSelected(null);
    setShapModel(null);
  };

  const closeShap = () => setShapModel(null);

  // -------------------------------------------------------------------------
  return (
    <div className="container">
      <div className={`wrapper ${clients.length > 0 ? "wide" : data ? "card" : ""}`}>

        {/* Header */}
        <div className="header">
          <h1>Credit Risk</h1>
          <p>Analyse décisionnelle par Machine Learning</p>
        </div>

        {/* Search bar */}
        <div className="search">
          <input
            type="number"
            placeholder="Entrer un Client ID"
            value={clientId}
            onChange={(e) => setClientId(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button onClick={handleAnalyze} disabled={loading}>
            {loading ? "Analyse…" : "Analyser"}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        {/* ── Original Logistic Regression result card ── */}
        {data && (
          <div className="result">
            <div className={`decision ${data.solvable ? "approved" : "rejected"}`}>
              {data.solvable ? "Approuvé" : "Refusé"}
            </div>

            <div className="score-label">Score de risque — Régression Logistique</div>
            <div className="score-value">{data.score_percent}%</div>

            <div className="bar-container">
              <div className="bar" style={{ width: `${data.score_percent}%` }} />
            </div>

            <div className="explanation">
              <div className="explanation-block">
                <div className="explanation-title">Principaux facteurs de risque</div>
                <ul>
                  {data.facteurs_risque?.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>
              <div className="explanation-block">
                <div className="explanation-title">Facteurs protecteurs</div>
                <ul>
                  {data.facteurs_protecteurs?.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>
            </div>

            {!data.solvable && (
              <div className="suggestions">
                <div className="explanation-title">
                  Pour améliorer vos chances d'approbation :
                </div>
                <ul>
                  {data.recommandations?.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* ── Multi-model table (addition) ── */}
        {clients.length > 0 && <div className="section-divider" />}

        {/* Filters + table */}
        {clients.length > 0 && (
          <>
            {/* Filter bar */}
            <div className="filters-row">
              <div className="filter-group">
                <span className="filter-label">Risque :</span>
                {[
                  ["all",    "Tous"],
                  ["high",   "🔴 Élevé"],
                  ["medium", "🟠 Moyen"],
                  ["low",    "🟢 Faible"],
                ].map(([val, label]) => (
                  <button
                    key={val}
                    className={`filter-btn filter-${val} ${filterRisk === val ? "active" : ""}`}
                    onClick={() => setFilterRisk(val)}
                    title={`Afficher uniquement les clients à risque ${label.replace(/[🔴🟠🟢]/g, "").trim().toLowerCase()}`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="filter-group">
                <span className="filter-label">Trier :</span>
                {[["final", "Risque final"], ...MODEL_KEYS.map(k => [k, MODEL_LABELS[k]])].map(([val, label]) => (
                  <button
                    key={val}
                    className={`filter-btn ${sortBy === val ? "active" : ""}`}
                    onClick={() => setSortBy(val)}
                    title={`Trier par ${label}`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="filter-search">
                <input
                  type="text"
                  placeholder="🔍 Filtrer par ID…"
                  value={searchQ}
                  onChange={(e) => setSearchQ(e.target.value)}
                />
              </div>
            </div>

            {/* Clients table */}
            <div className="table-wrap">
              <table className="clients-table">
                <thead>
                  <tr>
                    <th>Client ID</th>
                    {MODEL_KEYS.map(k => <th key={k}>{MODEL_LABELS[k]}</th>)}
                    <th>Score moy.</th>
                    <th>Risque Final</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {displayed.length === 0 ? (
                    <tr>
                      <td colSpan={13} className="table-empty">
                        Aucun client ne correspond aux filtres sélectionnés.
                      </td>
                    </tr>
                  ) : (
                    displayed.map((c) => (
                      <tr key={c.client_id} className={`row-risk-${c.final_risk}`}>
                        <td className="td-id">#{c.client_id}</td>
                        {MODEL_KEYS.map((key) => {
                          const m = c.models[key];
                          return (
                            <td key={key}>
                              {m?.score != null ? (
                                <span
                                  className="score-pill"
                                  style={{ color: RISK_COLORS[m.risk] }}
                                >
                                  {m.score}%
                                </span>
                              ) : (
                                <span className="score-na">N/A</span>
                              )}
                            </td>
                          );
                        })}
                        <td>
                          <span className="score-pill" style={{
                            color: RISK_COLORS[c.final_risk]
                          }}>
                            {c.avg_score != null ? `${c.avg_score}%` : "—"}
                          </span>
                        </td>
                        <td>
                          <RiskBadge risk={c.final_risk} />
                        </td>
                        <td>
                          <button
                            className="btn-details"
                            onClick={() => openDetails(c)}
                            title="Voir les détails de prédiction et les explications SHAP"
                          >
                            🔍 Voir détails
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <div className="table-footer">
              {displayed.length} client{displayed.length !== 1 ? "s" : ""} affiché{displayed.length !== 1 ? "s" : ""}
              {filterRisk !== "all" || searchQ ? ` (filtrés sur ${clients.length})` : ""}
            </div>
          </>
        )}
      </div>

      {/* Details Modal */}
      {selected && !shapModel && (
        <DetailsModal
          client={selected}
          onClose={closeDetails}
          onOpenShap={openShap}
        />
      )}

      {/* SHAP Modal */}
      {selected && shapModel && (
        <ShapModal
          client={selected}
          modelKey={shapModel}
          onClose={closeShap}
        />
      )}
    </div>
  );
}

export default App;
