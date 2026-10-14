import { useEffect, useState } from "react";
import { ApiError, api, type FeatureSpec, type Prediction, type PredictSchema } from "../api";
import { fmt, label, pct } from "../format";
import { Card, EmptyState, ErrorState, Loading, Notice } from "../components/ui";

/** Human-readable prompts for the BRFSS codes. Ranges come from the backend schema, not from here. */
const PROMPTS: Record<string, string> = {
  HighBP: "High blood pressure",
  HighChol: "High cholesterol",
  CholCheck: "Cholesterol check in past 5 years",
  BMI: "Body mass index",
  Smoker: "Smoked ≥ 100 cigarettes in lifetime",
  Stroke: "Ever had a stroke",
  HeartDiseaseorAttack: "Coronary heart disease or heart attack",
  PhysActivity: "Physical activity in past 30 days",
  Fruits: "Eats fruit ≥ 1 per day",
  Veggies: "Eats vegetables ≥ 1 per day",
  HvyAlcoholConsump: "Heavy alcohol consumption",
  AnyHealthcare: "Has any health-care coverage",
  NoDocbcCost: "Could not see a doctor because of cost",
  GenHlth: "General health (1 excellent … 5 poor)",
  MentHlth: "Days of poor mental health (past 30)",
  PhysHlth: "Days of poor physical health (past 30)",
  DiffWalk: "Serious difficulty walking",
  Sex: "Sex (0 female, 1 male)",
  Age: "Age group (1 = 18–24 … 13 = 80+)",
  Education: "Education level (1–6)",
  Income: "Income level (1–8)",
};

/** Client-side validation mirroring the backend rules; the backend re-validates everything. */
export function validateField(spec: FeatureSpec, raw: string): string | null {
  if (raw.trim() === "") return "Required";
  const v = Number(raw);
  if (!Number.isFinite(v)) return "Must be a number";
  if (v < spec.min || v > spec.max) return `Must be between ${spec.min} and ${spec.max}`;
  if (spec.kind !== "continuous" && !Number.isInteger(v)) return "Must be a whole number";
  return null;
}

export default function Demo() {
  const [schema, setSchema] = useState<PredictSchema | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [model, setModel] = useState<string>("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [result, setResult] = useState<Prediction | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .predictSchema()
      .then((s) => {
        setSchema(s);
        setModel(s.models[0] ?? "");
        // Start from empty fields: the user must enter every value; nothing is pre-filled from data.
        setValues(Object.fromEntries(s.features.map((f) => [f.name, f.kind === "binary" ? "0" : ""])));
      })
      .catch((e: unknown) => setLoadError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  if (loadError) return <ErrorState message={loadError} />;
  if (!schema) return <Loading />;
  if (!schema.models.length)
    return <EmptyState message="No trained model is available for the demo. Run `python -m ml.run centralized` first." />;

  const errors = Object.fromEntries(schema.features.map((f) => [f.name, validateField(f, values[f.name] ?? "")]));
  const hasErrors = Object.values(errors).some(Boolean);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setTouched(Object.fromEntries(schema.features.map((f) => [f.name, true])));
    setResult(null);
    setSubmitError(null);
    if (hasErrors) return;
    setBusy(true);
    try {
      const features = Object.fromEntries(schema.features.map((f) => [f.name, Number(values[f.name])]));
      setResult(await api.predict(model, features));
    } catch (err) {
      setSubmitError(err instanceof ApiError ? `${err.status ? `HTTP ${err.status}: ` : ""}${err.message}` : String(err));
    } finally {
      setBusy(false);
    }
  };

  const order = ["binary", "ordinal", "count", "continuous"];
  const sorted = [...schema.features].sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind));

  return (
    <>
      <header>
        <h1 className="text-xl font-semibold">Demo: diabetes indicator classification</h1>
        <p className="text-sm text-slate-600">
          Enter the 21 BRFSS indicator codes and score them with a saved model. Inputs are sent only to the local backend and are not
          stored.
        </p>
      </header>
      <Notice>{schema.disclaimer}</Notice>
      <form onSubmit={submit} noValidate className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <Card title="Indicators">
          <label className="mb-3 flex items-center gap-2 text-sm">
            Model
            <select className="input w-auto" value={model} onChange={(e) => setModel(e.target.value)}>
              {schema.models.map((m) => (
                <option key={m} value={m}>
                  {label(m)}
                </option>
              ))}
            </select>
          </label>
          <div className="grid gap-x-4 gap-y-2 sm:grid-cols-2 lg:grid-cols-3">
            {sorted.map((f) => {
              const err = touched[f.name] ? errors[f.name] : null;
              const id = `f-${f.name}`;
              return (
                <div key={f.name}>
                  <label htmlFor={id} className="block text-xs font-medium text-slate-700">
                    {PROMPTS[f.name] ?? f.name} <span className="text-slate-400">({f.name})</span>
                  </label>
                  {f.kind === "binary" ? (
                    <select
                      id={id}
                      className={`input ${err ? "input-error" : ""}`}
                      value={values[f.name] ?? ""}
                      onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                      onBlur={() => setTouched({ ...touched, [f.name]: true })}
                    >
                      <option value="0">0 – no</option>
                      <option value="1">1 – yes</option>
                    </select>
                  ) : (
                    <input
                      id={id}
                      className={`input ${err ? "input-error" : ""}`}
                      inputMode="decimal"
                      placeholder={`${f.min}–${f.max}`}
                      value={values[f.name] ?? ""}
                      onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                      onBlur={() => setTouched({ ...touched, [f.name]: true })}
                      aria-invalid={!!err}
                      aria-describedby={err ? `${id}-err` : undefined}
                    />
                  )}
                  {err && (
                    <p id={`${id}-err`} className="text-xs text-rose-700">
                      {err}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
          <div className="mt-3 flex items-center gap-3">
            <button type="submit" className="btn" disabled={busy}>
              {busy ? "Scoring…" : "Score with model"}
            </button>
            {Object.values(touched).some(Boolean) && hasErrors && (
              <span className="text-xs text-rose-700" role="alert">
                Fix the highlighted fields before submitting.
              </span>
            )}
          </div>
        </Card>
        <Card title="Model output">
          {submitError && <ErrorState message={submitError} />}
          {!result && !submitError && <p className="text-sm text-slate-500">No result yet.</p>}
          {result && (
            <div className="space-y-2 text-sm" aria-live="polite">
              <p className="text-3xl font-semibold tabular-nums">{pct(result.probability, 1)}</p>
              <p>
                Model-estimated probability of the <code>Diabetes_binary = 1</code> pattern from <span className="font-medium">{label(result.model)}</span>.
              </p>
              <p>
                Decision at threshold {fmt(result.threshold, 3)} ({result.threshold_source}):{" "}
                <span className="font-medium">{result.predicted_label}</span>
              </p>
              <Notice>{result.disclaimer}</Notice>
            </div>
          )}
        </Card>
      </form>
    </>
  );
}
