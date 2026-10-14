import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Row } from "../api";
import { COLORS, fmt, label, rows } from "../format";
import { useSection } from "../hooks";
import { Card, DataTable, Figure, SectionGate } from "../components/ui";

const METRIC_COLS = [
  { key: "model", header: "Model", render: (v: unknown) => label(v) },
  { key: "seed", header: "Seed", digits: 0 },
  { key: "threshold", header: "Threshold", digits: 3 },
  { key: "accuracy", header: "Accuracy" },
  { key: "precision_class1", header: "Precision (diabetes)" },
  { key: "recall_class1", header: "Recall (diabetes)" },
  { key: "f1_class1", header: "F1 (diabetes)" },
  { key: "roc_auc", header: "ROC-AUC" },
  { key: "pr_auc", header: "PR-AUC" },
  { key: "train_seconds", header: "Train s", digits: 1 },
];

export default function Centralized() {
  const state = useSection("centralized");
  const [op, setOp] = useState("val_max_f1");
  return (
    <>
      <header>
        <h1 className="text-xl font-semibold">Centralized model performance</h1>
        <p className="text-sm text-slate-600">
          Pooled training data; stratified 70/15/15 split. Thresholds are chosen on the validation split only; the test split is
          scored once.
        </p>
      </header>
      <SectionGate state={state}>
        {(s) => {
          const results = rows(s, "results");
          const ops = (s.operating_points as string[]) ?? [];
          const test = results.filter((r) => r.split === "test" && r.operating_point === op);
          const hist = rows(s, "mlp_history");
          const seeds = [...new Set(hist.map((h) => Number(h.seed)))];
          const epochs = [...new Set(hist.map((h) => Number(h.epoch)))].sort((a, b) => a - b);
          const curve = epochs.map((e) => {
            const row: Row = { epoch: e };
            for (const sd of seeds) {
              const h = hist.find((x) => Number(x.epoch) === e && Number(x.seed) === sd);
              row[`seed ${sd}`] = h ? Number(h.val_roc_auc) : null;
            }
            return row;
          });
          const selection = rows(s, "model_selection");
          return (
            <>
              <Card title="Test-split metrics by model and seed">
                <label className="mb-2 flex items-center gap-2 text-sm">
                  Operating point
                  <select className="input w-auto" value={op} onChange={(e) => setOp(e.target.value)}>
                    {ops.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                  <span className="text-xs text-slate-500">(threshold selected on validation data)</span>
                </label>
                <DataTable rows={test} columns={METRIC_COLS} />
              </Card>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="MLP validation ROC-AUC per epoch (early-stopping criterion)">
                  {curve.length ? (
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={curve}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="epoch" fontSize={11} label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 11 }} />
                        <YAxis domain={["auto", "auto"]} fontSize={11} tickFormatter={(v) => Number(v).toFixed(3)} />
                        <Tooltip formatter={(v) => fmt(v)} />
                        <Legend />
                        {seeds.map((sd, i) => (
                          <Line key={sd} type="monotone" dataKey={`seed ${sd}`} stroke={COLORS[i % COLORS.length]} dot={false} connectNulls />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <p className="text-sm text-slate-500">No MLP history saved.</p>
                  )}
                </Card>
                <Card title="Hyperparameter selection (validation ROC-AUC)">
                  <DataTable
                    rows={selection.filter((r) => r.seed === 0 || r.seed === null || r.seed === undefined)}
                    columns={[
                      { key: "model", header: "Model", render: (v) => label(v) },
                      { key: "params", header: "Params" },
                      { key: "val_roc_auc", header: "Val ROC-AUC" },
                      { key: "fit_seconds", header: "Fit s", digits: 1 },
                      { key: "selected", header: "Selected", render: (v) => (v ? "yes" : "") },
                    ]}
                  />
                </Card>
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                {(s.figures ?? [])
                  .filter((f) => /roc|pr_curve|confusion|threshold/.test(f))
                  .map((f) => (
                    <Figure key={f} rel={f} alt={f} />
                  ))}
              </div>
            </>
          );
        }}
      </SectionGate>
    </>
  );
}
