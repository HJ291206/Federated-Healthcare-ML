import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Row } from "../api";
import { COLORS, fmt, label, num, rows } from "../format";
import { useSection } from "../hooks";
import { Card, DataTable, Figure, PageHeader, SectionGate } from "../components/ui";

export default function Explainability() {
  const state = useSection("explainability");
  const [model, setModel] = useState<string | null>(null);
  const [caseName, setCaseName] = useState<string | null>(null);
  return (
    <div className="space-y-6">
      <PageHeader
        title="Explainability"
        description="SHAP attributions in probability units for centralized and federated models on a held-out sample."
      />
      <SectionGate state={state}>
        {(s) => {
          const imp = rows(s, "global_importance");
          const models = [...new Set(imp.map((r) => String(r.model)))];
          const features = [...new Set(imp.map((r) => String(r.feature)))];
          const grouped = features
            .map((f) => {
              const row: Row = { feature: f };
              for (const m of models) row[m] = num(imp.find((r) => r.feature === f && r.model === m)?.mean_abs_shap);
              return row;
            })
            .sort((a, b) => models.reduce((acc, m) => acc + (num(b[m]) ?? 0), 0) - models.reduce((acc, m) => acc + (num(a[m]) ?? 0), 0));
          const individual = rows(s, "individual");
          const indModels = [...new Set(individual.map((r) => String(r.model)))];
          const selModel = model ?? indModels[0];
          const cases = [...new Set(individual.filter((r) => r.model === selModel).map((r) => String(r.case)))];
          const selCase = caseName && cases.includes(caseName) ? caseName : cases[0];
          const ind = individual
            .filter((r) => r.model === selModel && r.case === selCase)
            .sort((a, b) => Math.abs(num(b.shap_value) ?? 0) - Math.abs(num(a.shap_value) ?? 0));
          const head = ind[0];
          const method = (s.method ?? {}) as Record<string, unknown>;
          return (
            <>
              <Card title="Global feature importance" subtitle="Mean absolute SHAP — change in predicted probability.">
                <ResponsiveContainer width="100%" height={420}>
                  <BarChart data={grouped} margin={{ bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="feature" angle={-50} textAnchor="end" interval={0} fontSize={11} />
                    <YAxis fontSize={11} />
                    <Tooltip formatter={(v) => fmt(v)} />
                    <Legend formatter={(v) => label(v)} />
                    {models.map((m, i) => (
                      <Bar key={m} dataKey={m} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
                {method.n_explained ? (
                  <p className="mt-2 text-sm text-slate-500">
                    {String(method.n_explained)} explained rows, {String(method.n_background)} background rows
                    {method.shap_version ? `, SHAP ${String(method.shap_version)}` : ""}.
                  </p>
                ) : null}
              </Card>
              {individual.length > 0 && (
                <Card title="Individual explanation">
                  <div className="mb-2 flex flex-wrap items-center gap-3 text-sm">
                    <label className="flex items-center gap-1">
                      Model
                      <select className="input w-auto" value={selModel} onChange={(e) => setModel(e.target.value)}>
                        {indModels.map((m) => (
                          <option key={m} value={m}>
                            {label(m)}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="flex items-center gap-1">
                      Case
                      <select className="input w-auto" value={selCase} onChange={(e) => setCaseName(e.target.value)}>
                        {cases.map((c) => (
                          <option key={c} value={c}>
                            {c.replace(/_/g, " ")}
                          </option>
                        ))}
                      </select>
                    </label>
                    {head && (
                      <span className="text-xs text-slate-600">
                        base rate {fmt(head.base_value, 3)} → predicted P = {fmt(head.predicted_probability, 3)}; true label {String(head.true_label)}
                      </span>
                    )}
                  </div>
                  <ResponsiveContainer width="100%" height={380}>
                    <BarChart data={ind.slice(0, 12)} layout="vertical" margin={{ left: 120 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" fontSize={11} />
                      <YAxis type="category" dataKey="feature" fontSize={11} tickFormatter={(f) => {
                        const r = ind.find((x) => x.feature === f);
                        return `${f} = ${fmt(r?.feature_value, 0)}`;
                      }} width={120} />
                      <Tooltip formatter={(v) => fmt(v)} />
                      <Bar dataKey="shap_value" name="SHAP contribution">
                        {ind.slice(0, 12).map((r, i) => (
                          <Cell key={i} fill={(num(r.shap_value) ?? 0) > 0 ? "#be123c" : "#0369a1"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="mt-2 text-sm text-slate-500">
                    Red raises predicted probability; blue lowers it for this record.
                  </p>
                </Card>
              )}
              <div className="grid gap-6 lg:grid-cols-2">
                <Card title="Rank agreement">
                  <DataTable
                    rows={rows(s, "rank_agreement")}
                    columns={[
                      { key: "model_a", header: "Model A", render: (v) => label(v) },
                      { key: "model_b", header: "Model B", render: (v) => label(String(v).replace(" (permutation importance)", "")) + (String(v).includes("permutation") ? " (permutation importance)" : "") },
                      { key: "spearman_rho", header: "ρ", digits: 3 },
                    ]}
                  />
                </Card>
                <Card title="Top features by model">
                  <DataTable
                    rows={imp.filter((r) => Number(r.rank) <= 5).sort((a, b) => String(a.model).localeCompare(String(b.model)) || Number(a.rank) - Number(b.rank))}
                    columns={[
                      { key: "model", header: "Model", render: (v) => label(v) },
                      { key: "rank", header: "Rank", digits: 0 },
                      { key: "feature", header: "Feature" },
                      { key: "mean_abs_shap", header: "Mean |SHAP|" },
                      { key: "mean_shap", header: "Mean SHAP (signed)" },
                    ]}
                  />
                </Card>
              </div>
              <div className="grid gap-6 lg:grid-cols-2">
                {(s.figures ?? [])
                  .filter((f) => f.includes("global"))
                  .map((f) => (
                    <Figure key={f} rel={f} alt={f} />
                  ))}
              </div>
            </>
          );
        }}
      </SectionGate>
    </div>
  );
}
