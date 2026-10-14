import { useEffect, useState } from "react";
import { ApiError, api, type FeatureSpec, type Prediction, type PredictSchema } from "../api";
import { fmt, label, pct } from "../format";
import { Card, EmptyState, ErrorState, Loading, PageHeader } from "../components/ui";

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
    return (
      <EmptyState
        title="No scoring model available"
        message="Train a model to enable patient risk scoring."
      />
    );

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

  const loadSampleProfile = () => {
    setValues({
      HighBP: "1",
      HighChol: "1",
      CholCheck: "1",
      BMI: "28.5",
      Smoker: "0",
      Stroke: "0",
      HeartDiseaseorAttack: "0",
      PhysActivity: "1",
      Fruits: "1",
      Veggies: "1",
      HvyAlcoholConsump: "0",
      AnyHealthcare: "1",
      NoDocbcCost: "0",
      GenHlth: "3",
      MentHlth: "2",
      PhysHlth: "4",
      DiffWalk: "0",
      Sex: "1",
      Age: "8",
      Education: "5",
      Income: "6",
    });
    setTouched({});
    setSubmitError(null);
  };

  const order = ["binary", "ordinal", "count", "continuous"];
  const sorted = [...schema.features].sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind));

  return (
    <div className="space-y-6">
      <PageHeader
        title="Patient risk analysis"
        description="Enter clinical indicators to evaluate diabetes risk probability using calibrated models. All inference is processed locally."
        actions={
          <button type="button" onClick={loadSampleProfile} className="btn-secondary text-xs">
            Load sample clinical profile
          </button>
        }
      />
      <form onSubmit={submit} noValidate className="grid gap-6 lg:grid-cols-[1.7fr_1fr]">
        <Card title="Clinical indicators" subtitle="Complete every field, then score with the selected model.">
          <label className="mb-5 flex items-center gap-3 text-sm font-medium text-slate-700">
            Model
            <select className="input w-auto min-w-[220px]" value={model} onChange={(e) => setModel(e.target.value)}>
              {schema.models.map((m) => (
                <option key={m} value={m}>
                  {label(m)}
                </option>
              ))}
            </select>
          </label>
          <div className="grid gap-x-4 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
            {sorted.map((f) => {
              const err = touched[f.name] ? errors[f.name] : null;
              const id = `f-${f.name}`;
              return (
                <div key={f.name}>
                  <label htmlFor={id} className="mb-1 block text-sm font-medium text-slate-700">
                    {PROMPTS[f.name] ?? f.name} <span className="font-normal text-slate-400">({f.name})</span>
                  </label>
                  {f.kind === "binary" ? (
                    <select
                      id={id}
                      className={`input ${err ? "input-error" : ""}`}
                      value={values[f.name] ?? ""}
                      onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                      onBlur={() => setTouched({ ...touched, [f.name]: true })}
                    >
                      <option value="0">No</option>
                      <option value="1">Yes</option>
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
          <div className="mt-6 flex items-center gap-3">
            <button type="submit" className="btn-teal" disabled={busy}>
              {busy ? "Scoring…" : "Score with model"}
            </button>
            {Object.values(touched).some(Boolean) && hasErrors && (
              <span className="text-sm text-rose-700" role="alert">
                Complete the highlighted fields before scoring.
              </span>
            )}
          </div>
        </Card>
        <Card title="Risk estimate">
          {submitError && <ErrorState message={submitError} />}
          {!result && !submitError && (
            <div className="flex min-h-[220px] flex-col items-center justify-center text-center">
              <p className="text-sm font-medium text-slate-800">Awaiting input</p>
              <p className="mt-1 max-w-[240px] text-sm text-slate-500">
                Submit the form to see probability, threshold, and predicted class.
              </p>
            </div>
          )}
          {result && (
            <div className="space-y-4 text-sm" aria-live="polite">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">Estimated probability</p>
                <p className="mt-1 text-4xl font-semibold tabular-nums tracking-tight text-slate-900">
                  {pct(result.probability, 1)}
                </p>
              </div>
              <p className="leading-relaxed text-slate-600">
                Probability of a diabetes-positive pattern from{" "}
                <span className="font-medium text-slate-800">{label(result.model)}</span>.
              </p>
              <div className="rounded-xl border border-slate-100 bg-slate-50 px-4 py-3">
                <p className="text-slate-600">
                  Decision at threshold {fmt(result.threshold, 3)}
                  <span className="mx-1.5 text-slate-300">·</span>
                  <span className="font-medium text-slate-900">{result.predicted_label}</span>
                </p>
              </div>
              {result.disclaimer && <p className="text-sm leading-relaxed text-slate-500">{result.disclaimer}</p>}
            </div>
          )}
        </Card>
      </form>
    </div>
  );
}
