import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const RISK_LABELS = { high: "High", medium: "Medium", low: "Low", unknown: "—" };
const RISK_COLORS = { high: "#ef4444", medium: "#f97316", low: "#22c55e", unknown: "#888" };

const MODEL_LABELS = {
  xgboost: "XGBoost",
  randomforest: "Random Forest",
  lightgbm: "LightGBM",
};
const MODEL_KEYS = ["xgboost", "randomforest", "lightgbm"];

const SORT_LABELS = {
  final: "Final risk",
  xgboost: "XGBoost score",
  randomforest: "Random Forest score",
  lightgbm: "LightGBM score",
};

function RiskBadge({ risk }) {
  return (
    <span className={`risk-badge risk-${risk}`}>
      {RISK_LABELS[risk] ?? "—"}
    </span>
  );
}

function MiniBar({ score, risk }) {
  const segments = 20;
  const filled = Math.round((score / 100) * segments);
  const color = RISK_COLORS[risk];
  return (
    <div className="mini-bar-wrap" title={`${score}%`}>
      {Array.from({ length: segments }).map((_, i) => (
        <div
          key={i}
          className="mini-bar-seg"
          style={{
            background: i < filled ? color : undefined,
            opacity: i < filled ? 0.55 + (i / segments) * 0.45 : undefined,
          }}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SHAP Modal
// ---------------------------------------------------------------------------
function ShapModal({ client, modelKey, onClose }) {
  const [shap, setShap] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadShap() {
      try {
        setShap(null);
        setError(null);

        const res = await axios.get(
          `http://127.0.0.1:8000/shap/${client.client_id}/${modelKey}`
        );

        if (!cancelled) {
          const rows = Array.isArray(res.data?.shap) ? res.data.shap : [];
          setShap(rows);
        }
      } catch (err) {
        if (!cancelled) {
          const msg =
            err?.response?.data?.detail ||
            err?.response?.data?.error ||
            "Error while loading SHAP explanations.";
          setError(msg);
        }
      }
    }

    loadShap();

    return () => {
      cancelled = true;
    };
  }, [client.client_id, modelKey]);

  const maxAbs =
    shap && shap.length > 0
      ? Math.max(...shap.map((s) => Math.abs(Number(s.shap_value) || 0)))
      : 1;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box shap-box" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>SHAP Explanations</h2>
            <p className="modal-sub">
              {MODEL_LABELS[modelKey]} — Client #{client.client_id}
            </p>
          </div>

          <button className="modal-close" onClick={onClose}>
            ×
          </button>
        </div>

        {error ? (
          <p className="shap-empty">{error}</p>
        ) : shap === null ? (
          <p className="shap-loading">Loading explanations…</p>
        ) : shap.length === 0 ? (
          <p className="shap-empty">No SHAP data available.</p>
        ) : (
          <>
            <div className="shap-list">
              {shap.map((item, i) => {
                const value = Number(item.shap_value) || 0;
                const pct = maxAbs > 0 ? (Math.abs(value) / maxAbs) * 100 : 0;
                const isPositive = value >= 0;

                return (
                  <div key={i} className="shap-row">
                    <div className="shap-feature" title={item.feature}>
                      {item.feature}
                    </div>

                    <div className="shap-bar-track">
                      <div
                        className={`shap-bar-fill ${isPositive ? "shap-pos" : "shap-neg"}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>

                    <div
                      className={`shap-val ${isPositive ? "shap-val-pos" : "shap-val-neg"}`}
                    >
                      {isPositive ? "+" : ""}
                      {value.toFixed(3)}
                    </div>
                  </div>
                );
              })}
            </div>

            <p className="shap-legend">
              <span className="shap-pos-dot" /> Increases risk &nbsp;|&nbsp;
              <span className="shap-neg-dot" /> Reduces risk
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
            <h2>Details — Client #{client.client_id}</h2>
            <p className="modal-sub">Multi-model analysis</p>
          </div>
          <button className="modal-close" onClick={onClose}>
            ×
          </button>
        </div>

        <div className="detail-final">
          <span className="detail-final-label">Final decision:</span>
          <RiskBadge risk={client.final_risk} />
          {client.avg_score != null && (
            <span className="detail-avg-score">
              Average score:{" "}
              <strong style={{ color: RISK_COLORS[client.final_risk] }}>{client.avg_score}%</strong>
              &nbsp;across {Object.values(client.models).filter((m) => m.score != null).length} models
            </span>
          )}
        </div>

        <div className="detail-models">
          {MODEL_KEYS.map((key) => {
            const m = client.models[key];
            if (!m || m.score === null) return null;
            return (
              <div key={key} className="detail-model-block">
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
                    title={`View SHAP values for ${MODEL_LABELS[key]}`}
                  >
                    📊 View explanation
                  </button>
                </div>

                {(m.risk_factors?.length > 0 || m.protective_factors?.length > 0) && (
                  <div className="detail-factors-grid">
                    {m.risk_factors?.length > 0 && (
                      <div className="detail-factors-col">
                        <div className="factors-title risk-title">Risk factors</div>
                        <ul>
                          {m.risk_factors.map((f, i) => (
                            <li key={i}>{f}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {m.protective_factors?.length > 0 && (
                      <div className="detail-factors-col">
                        <div className="factors-title protect-title">Protective factors</div>
                        <ul>
                          {m.protective_factors.map((f, i) => (
                            <li key={i}>{f}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {m.recommendations?.length > 0 && (
                  <div className="detail-model-reco">
                    <div className="detail-reco-title">
                      Suggestions to improve approval chances
                    </div>
                    <ul>
                      {m.recommendations.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="detail-chart">
          <div className="chart-title">Model comparison</div>
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
  const [clientId, setClientId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [data, setData] = useState(null);
  const [clients, setClients] = useState([]);

  const [selected, setSelected] = useState(null);
  const [shapModel, setShapModel] = useState(null);

  const [filterRisk, setFilterRisk] = useState("all");
  const [filterModel, setFilterModel] = useState("all");
  const [sortBy, setSortBy] = useState("final");
  const [searchQ, setSearchQ] = useState("");

  const handleAnalyze = async () => {
    if (!clientId) return;
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await axios.get(`http://127.0.0.1:8000/predict_all/${clientId}`);
      const d = res.data;

      if (d.error) {
        setError("Client ID not found.");
      } else {
        const cid = d.client_id ?? d.card?.client_id;
        setData(d.card);
        setClients((prev) => {
          const entry = {
            client_id: cid,
            models: d.models,
            final_risk: d.final_risk,
            avg_score: d.avg_score,
          };
          const idx = prev.findIndex((c) => c.client_id === cid);
          if (idx >= 0) {
            const copy = [...prev];
            copy[idx] = entry;
            return copy;
          }
          return [entry, ...prev];
        });
      }
    } catch {
      setError("Error while fetching data.");
    }

    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleAnalyze();
  };

  const SORT_ORDER = { high: 2, medium: 1, low: 0, unknown: -1 };
  const activeSortedModel = sortBy !== "final" ? sortBy : null;
  const visibleModelKeys = filterModel === "all" ? MODEL_KEYS : [filterModel];
  const searchTrim = searchQ.trim();

  const displayed = clients
    .filter((c) => {
      if (searchTrim && !String(c.client_id).includes(searchTrim)) return false;
      if (filterRisk !== "all" && c.final_risk !== filterRisk) return false;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === "final") {
        const d = SORT_ORDER[b.final_risk] - SORT_ORDER[a.final_risk];
        if (d !== 0) return d;
      } else {
        const sa = a.models[sortBy]?.score ?? -1;
        const sb = b.models[sortBy]?.score ?? -1;
        if (sb !== sa) return sb - sa;
      }
      return Number(b.client_id) - Number(a.client_id);
    });

  const handleSortChange = (nextSort) => {
    setSortBy(nextSort);
    if (nextSort !== "final") {
      setFilterModel(nextSort);
    }
  };

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

  return (
    <div className="container">
      <div className={`wrapper ${clients.length > 0 ? "wide" : data ? "card" : ""}`}>
        <div className="header">
          <h1>Credit Risk</h1>
          <p>Machine learning decision support</p>
        </div>

        <div className="search">
          <input
            type="number"
            placeholder="Enter a Client ID"
            value={clientId}
            onChange={(e) => setClientId(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button onClick={handleAnalyze} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        {data && (
          <div className="result">
            <div className={`decision ${data.approvable ? "approved" : "rejected"}`}>
              {data.approvable ? "Approved" : "Rejected"}
            </div>

            <div className="score-label">Risk score — Main score</div>
            <div className="score-value">{data.score_percent}%</div>

            <div className="bar-container">
              <div className="bar" style={{ width: `${data.score_percent}%` }} />
            </div>

            <div className="explanation">
              <div className="explanation-block">
                <div className="explanation-title">Main risk factors</div>
                <ul>
                  {data.risk_factors?.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </div>
              <div className="explanation-block">
                <div className="explanation-title">Protective factors</div>
                <ul>
                  {data.protective_factors?.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </div>
            </div>

            {!data.approvable && (
              <div className="suggestions">
                <div className="explanation-title">To improve approval chances:</div>
                <ul>
                  {data.recommendations?.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {clients.length > 0 && <div className="section-divider" />}

        {clients.length > 0 && (
          <>
            <div className="filters-panel">
              <div className="filters-row">
                <div className="filter-group">
                  <span className="filter-label">Risk:</span>
                  {[
                    ["all", "All"],
                    ["high", "High"],
                    ["medium", "Medium"],
                    ["low", "Low"],
                  ].map(([val, label]) => (
                    <button
                      key={val}
                      className={`filter-btn filter-${val} ${filterRisk === val ? "active" : ""}`}
                      onClick={() => setFilterRisk(val)}
                      title={`Show only ${label.toLowerCase()} risk clients`}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                <div className="filter-group">
                  <span className="filter-label">Model:</span>
                  {[["all", "All"], ...MODEL_KEYS.map((k) => [k, MODEL_LABELS[k]])].map(
                    ([val, label]) => (
                      <button
                        key={val}
                        className={`filter-btn ${filterModel === val ? "active" : ""}`}
                        onClick={() => setFilterModel(val)}
                        title={val === "all" ? "Show all models" : `Show only ${label}`}
                      >
                        {label}
                      </button>
                    )
                  )}
                </div>

                <div className="filter-group">
                  <span className="filter-label">Sort:</span>
                  {[["final", SORT_LABELS.final], ...MODEL_KEYS.map((k) => [k, SORT_LABELS[k]])].map(
                    ([val, label]) => (
                      <button
                        key={val}
                        className={`filter-btn ${sortBy === val ? "active" : ""}`}
                        onClick={() => handleSortChange(val)}
                        title={
                          val === "final"
                            ? "Sort by overall risk level (highest to lowest)"
                            : `Sort by ${MODEL_LABELS[val]} risk percentage (highest to lowest)`
                        }
                      >
                        {label}
                      </button>
                    )
                  )}
                </div>

                <div className="filter-search">
                  <input
                    type="text"
                    placeholder="Search analyzed ID..."
                    value={searchQ}
                    onChange={(e) => setSearchQ(e.target.value)}
                    title="Filter rows whose ID contains this value"
                  />
                </div>
              </div>

              <p className="filters-hint">
                <strong>Risk</strong> and <strong>ID</strong> can be combined. <strong>Sort</strong>{" "}
                orders rows and automatically highlights the selected model column.{" "}
                <strong>Model</strong> changes visible columns manually.
              </p>
            </div>

            <div className="table-wrap">
              <table className="clients-table">
                <thead>
                  <tr>
                    <th>Client ID</th>
                    {visibleModelKeys.map((k) => (
                      <th key={k} className={activeSortedModel === k ? "sorted-col-header" : ""}>
                        {MODEL_LABELS[k]}
                        {activeSortedModel === k ? " ↓" : ""}
                      </th>
                    ))}
                    <th>Avg. score</th>
                    <th>Final risk</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {displayed.length === 0 ? (
                    <tr>
                      <td colSpan={1 + visibleModelKeys.length + 3} className="table-empty">
                        {searchTrim && filterRisk !== "all"
                          ? "No client matches both this ID and this risk level. Try 'All' for risk or clear the ID field."
                          : searchTrim
                          ? "No analyzed client matches this ID."
                          : filterRisk !== "all"
                          ? "No client matches the selected risk level. Try 'All' or analyze another client."
                          : "No client matches the selected filters."}
                      </td>
                    </tr>
                  ) : (
                    displayed.map((c) => (
                      <tr key={c.client_id} className={`row-risk-${c.final_risk}`}>
                        <td className="td-id">#{c.client_id}</td>
                        {visibleModelKeys.map((key) => {
                          const m = c.models[key];
                          return (
                            <td key={key} className={activeSortedModel === key ? "sorted-col-cell" : ""}>
                              {m?.score != null ? (
                                <span className="score-pill" style={{ color: RISK_COLORS[m.risk] }}>
                                  {m.score}%
                                </span>
                              ) : (
                                <span className="score-na">N/A</span>
                              )}
                            </td>
                          );
                        })}
                        <td>
                          <span className="score-pill" style={{ color: RISK_COLORS[c.final_risk] }}>
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
                            title="View prediction details and SHAP explanations"
                          >
                            View details
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <div className="table-footer">
              <div className="table-footer-line">
                {displayed.length} client{displayed.length !== 1 ? "s" : ""} shown out of {clients.length}
                {filterRisk !== "all" && ` · Risk: ${RISK_LABELS[filterRisk]}`}
                {searchTrim && ` · ID contains "${searchTrim}"`}
              </div>
              <div className="table-footer-line table-footer-meta">
                Active sort: {SORT_LABELS[sortBy]}
                {sortBy === "final" ? " (high → low)" : " (descending % score)"}
                {clients.length === 1 && " — with only one row, the order does not visibly change."}
              </div>
            </div>
          </>
        )}
      </div>

      {selected && !shapModel && (
        <DetailsModal client={selected} onClose={closeDetails} onOpenShap={openShap} />
      )}

      {selected && shapModel && (
        <ShapModal client={selected} modelKey={shapModel} onClose={closeShap} />
      )}
    </div>
  );
}

export default App;
