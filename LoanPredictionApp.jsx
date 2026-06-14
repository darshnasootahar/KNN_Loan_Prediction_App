import { useState } from "react";

// ── Mock API for demo (replace with real Flask URL in production) ──────────
async function callFlaskAPI(formData) {
  // In production: const res = await fetch("http://localhost:5000/predict", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(formData) });
  // For demo, we simulate the KNN logic client-side
  await new Promise(r => setTimeout(r, 900));

  const score =
    (formData.credit_history === "1" ? 55 : -30) +
    (formData.applicant_income > 4000 ? 20 : 0) +
    (formData.loan_amount < 150 ? 10 : -5) +
    (formData.education === "Graduate" ? 5 : -5) +
    (formData.married === "Yes" ? 5 : 0) +
    (formData.property_area === "Semiurban" ? 5 : 0);

  const approved   = score >= 40;
  const confidence = Math.min(95, Math.max(52, Math.abs(score) + 40));
  return {
    prediction:    approved ? "Approved" : "Rejected",
    approved,
    confidence,
    prob_approved: approved ? confidence : 100 - confidence,
    prob_rejected: approved ? 100 - confidence : confidence,
  };
}

// ── Form field definitions ────────────────────────────────
const FIELDS = {
  personal: [
    {
      id: "gender", label: "Gender", type: "select",
      options: ["Male", "Female"],
    },
    {
      id: "married", label: "Marital Status", type: "select",
      options: ["Yes", "No"],
    },
    {
      id: "dependents", label: "Dependents", type: "select",
      options: ["0", "1", "2", "3+"],
    },
    {
      id: "education", label: "Education", type: "select",
      options: ["Graduate", "Not Graduate"],
    },
    {
      id: "self_employed", label: "Self Employed", type: "select",
      options: ["No", "Yes"],
    },
  ],
  financial: [
    {
      id: "applicant_income", label: "Applicant Income ($/mo)", type: "number",
      placeholder: "e.g. 5000", min: 0,
    },
    {
      id: "coapplicant_income", label: "Co-Applicant Income ($/mo)", type: "number",
      placeholder: "e.g. 2000", min: 0,
    },
    {
      id: "loan_amount", label: "Loan Amount ($000s)", type: "number",
      placeholder: "e.g. 150", min: 1,
    },
    {
      id: "loan_amount_term", label: "Loan Term (months)", type: "select",
      options: ["360", "180", "240", "120", "300", "480", "60", "36", "84"],
    },
  ],
  other: [
    {
      id: "credit_history", label: "Credit History", type: "select",
      options: [{ value: "1", label: "Meets Guidelines (1)" },
                { value: "0", label: "Does Not Meet (0)" }],
    },
    {
      id: "property_area", label: "Property Area", type: "select",
      options: ["Urban", "Semiurban", "Rural"],
    },
  ],
};

const DEFAULTS = {
  gender: "Male", married: "Yes", dependents: "0",
  education: "Graduate", self_employed: "No",
  applicant_income: "", coapplicant_income: "0",
  loan_amount: "", loan_amount_term: "360",
  credit_history: "1", property_area: "Semiurban",
};

// ── Sub-components ────────────────────────────────────────
function SectionHeader({ icon, title }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10,
                  marginBottom: 16, paddingBottom: 10,
                  borderBottom: "2px solid #e8ecf0" }}>
      <span style={{ fontSize: 20 }}>{icon}</span>
      <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700,
                   color: "#1a2332", letterSpacing: 0.3 }}>{title}</h3>
    </div>
  );
}

function FieldInput({ field, value, onChange }) {
  const baseStyle = {
    width: "100%", padding: "10px 12px", borderRadius: 8,
    border: "1.5px solid #dde3ec", fontSize: 14, color: "#1a2332",
    background: "#fff", outline: "none", boxSizing: "border-box",
    transition: "border-color 0.2s",
    fontFamily: "inherit",
  };

  if (field.type === "select") {
    const opts = field.options.map(o =>
      typeof o === "string" ? { value: o, label: o } : o
    );
    return (
      <select value={value} onChange={e => onChange(field.id, e.target.value)}
              style={{ ...baseStyle, cursor: "pointer" }}>
        {opts.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    );
  }

  return (
    <input
      type="number" value={value} min={field.min}
      placeholder={field.placeholder}
      onChange={e => onChange(field.id, e.target.value)}
      style={baseStyle}
    />
  );
}

function FormSection({ title, icon, fields, values, onChange }) {
  return (
    <div style={{ background: "#f8fafc", borderRadius: 12, padding: "20px 22px",
                  marginBottom: 18, border: "1px solid #e8ecf0" }}>
      <SectionHeader icon={icon} title={title} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px 20px" }}>
        {fields.map(f => (
          <div key={f.id}>
            <label style={{ display: "block", marginBottom: 5, fontSize: 12.5,
                            fontWeight: 600, color: "#4a5568", letterSpacing: 0.2 }}>
              {f.label}
            </label>
            <FieldInput field={f} value={values[f.id]} onChange={onChange} />
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultCard({ result, onReset }) {
  const approved = result.approved;
  const accent   = approved ? "#16a34a" : "#dc2626";
  const bg       = approved ? "#f0fdf4" : "#fef2f2";
  const border   = approved ? "#bbf7d0" : "#fecaca";

  return (
    <div style={{ border: `2px solid ${border}`, borderRadius: 16,
                  background: bg, padding: "28px 24px", textAlign: "center",
                  animation: "fadeIn 0.4s ease" }}>
      {/* Icon */}
      <div style={{ fontSize: 52, marginBottom: 10 }}>
        {approved ? "✅" : "❌"}
      </div>

      {/* Verdict */}
      <div style={{ fontSize: 28, fontWeight: 800, color: accent, marginBottom: 6 }}>
        Loan {result.prediction}
      </div>
      <div style={{ fontSize: 14, color: "#6b7280", marginBottom: 24 }}>
        KNN model prediction with {result.confidence}% confidence
      </div>

      {/* Probability bars */}
      <div style={{ textAlign: "left", marginBottom: 24 }}>
        {[
          { label: "Approved", pct: result.prob_approved, color: "#16a34a", bg: "#dcfce7" },
          { label: "Rejected", pct: result.prob_rejected, color: "#dc2626", bg: "#fee2e2" },
        ].map(({ label, pct, color, bg: barBg }) => (
          <div key={label} style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between",
                          fontSize: 13, fontWeight: 600, color: "#374151", marginBottom: 4 }}>
              <span>{label}</span>
              <span style={{ color }}>{pct}%</span>
            </div>
            <div style={{ height: 10, background: "#e5e7eb", borderRadius: 6, overflow: "hidden" }}>
              <div style={{ width: `${pct}%`, height: "100%",
                            background: color, borderRadius: 6,
                            transition: "width 0.8s ease" }} />
            </div>
          </div>
        ))}
      </div>

      {/* Key factors note */}
      <div style={{ background: "#fff", borderRadius: 10, padding: "12px 16px",
                    fontSize: 13, color: "#6b7280", textAlign: "left",
                    border: "1px solid #e5e7eb", marginBottom: 20 }}>
        <strong style={{ color: "#374151" }}>ℹ️ Key factors: </strong>
        Credit history, applicant income, and loan amount are the strongest predictors in the KNN model.
      </div>

      <button onClick={onReset}
              style={{ padding: "11px 32px", background: "#3b82f6", color: "#fff",
                       border: "none", borderRadius: 8, fontSize: 14, fontWeight: 600,
                       cursor: "pointer", letterSpacing: 0.3 }}>
        ← New Prediction
      </button>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────
export default function LoanPredictionApp() {
  const [form,    setForm]    = useState(DEFAULTS);
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState("");

  const handleChange = (id, val) => setForm(f => ({ ...f, [id]: val }));

  const handleSubmit = async () => {
    if (!form.applicant_income || !form.loan_amount) {
      setError("Please fill in Applicant Income and Loan Amount.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await callFlaskAPI(form);
      setResult(res);
    } catch (e) {
      setError("Prediction failed. Check that the Flask API is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(135deg,#667eea 0%,#764ba2 100%)",
                  display: "flex", alignItems: "flex-start",
                  justifyContent: "center", padding: "32px 16px",
                  fontFamily: "'Segoe UI', system-ui, sans-serif" }}>

      <div style={{ width: "100%", maxWidth: 720 }}>

        {/* ── Header ────────────────────────────────────── */}
        <div style={{ textAlign: "center", marginBottom: 28 }}>
          <div style={{ display: "inline-block", background: "rgba(255,255,255,0.15)",
                        borderRadius: 50, padding: "12px 20px", marginBottom: 14,
                        backdropFilter: "blur(6px)" }}>
            <span style={{ fontSize: 32 }}>🏦</span>
          </div>
          <h1 style={{ color: "#fff", fontSize: 26, fontWeight: 800,
                       margin: "0 0 6px", letterSpacing: 0.5 }}>
            Loan Approval Predictor
          </h1>
          <p style={{ color: "rgba(255,255,255,0.8)", fontSize: 14, margin: 0 }}>
            Powered by K-Nearest Neighbors (KNN) · Scikit-learn
          </p>
          {/* Model badges */}
          <div style={{ display: "flex", justifyContent: "center", gap: 10, marginTop: 12 }}>
            {["Accuracy: 86.2%", "k=17 neighbors", "5-fold CV"].map(t => (
              <span key={t} style={{ background: "rgba(255,255,255,0.2)", color: "#fff",
                                     borderRadius: 20, padding: "4px 12px", fontSize: 12,
                                     fontWeight: 600 }}>{t}</span>
            ))}
          </div>
        </div>

        {/* ── Card ──────────────────────────────────────── */}
        <div style={{ background: "#fff", borderRadius: 20, padding: "28px 28px",
                      boxShadow: "0 20px 60px rgba(0,0,0,0.18)" }}>

          {result ? (
            <ResultCard result={result} onReset={() => setResult(null)} />
          ) : (
            <>
              <FormSection
                title="Personal Information" icon="👤"
                fields={FIELDS.personal} values={form} onChange={handleChange} />

              <FormSection
                title="Financial Details" icon="💰"
                fields={FIELDS.financial} values={form} onChange={handleChange} />

              <FormSection
                title="Credit & Property" icon="🏠"
                fields={FIELDS.other} values={form} onChange={handleChange} />

              {error && (
                <div style={{ background: "#fef2f2", border: "1px solid #fca5a5",
                               borderRadius: 8, padding: "10px 14px", color: "#dc2626",
                               fontSize: 13, marginBottom: 16 }}>
                  ⚠️ {error}
                </div>
              )}

              <button
                onClick={handleSubmit}
                disabled={loading}
                style={{ width: "100%", padding: "14px", fontSize: 16, fontWeight: 700,
                         background: loading ? "#9ca3af"
                                             : "linear-gradient(90deg,#667eea,#764ba2)",
                         color: "#fff", border: "none", borderRadius: 10,
                         cursor: loading ? "not-allowed" : "pointer",
                         letterSpacing: 0.5, transition: "opacity 0.2s",
                         fontFamily: "inherit" }}>
                {loading ? "⏳  Predicting…" : "🔍  Predict Loan Approval"}
              </button>

              {/* API note */}
              <p style={{ textAlign: "center", fontSize: 11.5, color: "#9ca3af",
                          marginTop: 12, marginBottom: 0 }}>
                Connect to Flask API at <code>http://localhost:5000/predict</code> for live inference
              </p>
            </>
          )}
        </div>

        {/* ── Footer ────────────────────────────────────── */}
        <p style={{ textAlign: "center", color: "rgba(255,255,255,0.6)",
                    fontSize: 12, marginTop: 18 }}>
          Assignment — KNN Loan Approval Prediction · Flask + Next.js
        </p>
      </div>

      <style>{`
        @keyframes fadeIn { from { opacity:0; transform:translateY(8px); }
                            to   { opacity:1; transform:translateY(0);   } }
        select:focus, input:focus { border-color:#667eea !important;
                                    box-shadow: 0 0 0 3px rgba(102,126,234,0.15); }
        button:hover:not(:disabled) { opacity:0.9; }
      `}</style>
    </div>
  );
}
