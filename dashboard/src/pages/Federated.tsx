import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Row } from "../api";
import { bytesHuman, fmt, label, num, rows } from "../format";
import { useSection } from "../hooks";
import { Card, DataTable, Disclaimers, Figure, Notice, SectionGate } from "../components/ui";

const FAMILY_COLOR: Record<string, string> = { centralized: "#0369a1", local_only: "#64748b", fedavg: "#ea580c", fedprox: "#15803d", dp: "#7c3aed" };

const pm = (r: Row, m: string) => {
  const mean = num(r[`${m}_mean`]);
  const std = num(r[`${m}_std`]);
  if (mean === null) return "–";
  return Number(r.n_seeds) > 1 && std !== null ? `${mean.toFixed(4)} ± ${std.toFixed(4)}` : mean.toFixed(4);
};

export default function Federated() {
  const state = useSection("federated");
  return (
    <>
      <header>
        <h1 className="text-xl font-semibold">Federated-learning comparison</h1>
        <p className="text-sm text-slate-600">
          Centralized MLP vs local-only clients vs FedAvg vs FedProx on the held-out test split. Mean ± std over seeds where
          more than one seed was run.
        </p>
      </header>
      <SectionGate state={state}>
        {(s) => {
          const final = rows(s, "final_comparison").filter((r) => r.family !== "dp");
          const ci = rows(s, "bootstrap_ci");
          const local = rows(s, "local_only_by_client");
          const chart = final.map((r) => ({
            name: `${label(r.setting)} · ${label(r.method)}`,
            family: String(r.family),
            auc: num(r.roc_auc_mean),
            err: Number(r.n_seeds) > 1 ? num(r.roc_auc_std) ?? 0 : 0,
          }));
          const aucs = chart.map((c) => c.auc ?? 0);
          return (
            <>
              <Disclaimers items={s.disclaimers} keys={["clients", "privacy"]} />
              {!s.final_comparison_available && (
                <Notice>
                  The aggregated comparison table has not been generated yet. Run <code>python -m ml.final_evaluation.report</code>; raw
                  per-run results from <code>results/federated/{String(s.tag)}</code> are shown below.
                </Notice>
              )}
              {final.length > 0 && (
                <Card title="Test ROC-AUC (mean ± std over seeds)">
                  <ResponsiveContainer width="100%" height={340}>
                    <BarChart data={chart} margin={{ bottom: 110 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" angle={-40} textAnchor="end" interval={0} fontSize={10} />
                      <YAxis domain={[Math.min(...aucs) - 0.02, Math.max(...aucs) + 0.01]} fontSize={11} tickFormatter={(v) => Number(v).toFixed(3)} />
                      <Tooltip formatter={(v) => fmt(v)} />
                      <Bar dataKey="auc" name="Test ROC-AUC">
                        {chart.map((c, i) => (
                          <Cell key={i} fill={FAMILY_COLOR[c.family] ?? "#334155"} />
                        ))}
                        <ErrorBar dataKey="err" width={4} strokeWidth={1.5} />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              )}
              {final.length > 0 && (
                <Card title="Final comparison table (test split)">
                  <DataTable
                    rows={final}
                    columns={[
                      { key: "setting", header: "Setting", render: (v) => label(v) },
                      { key: "method", header: "Method", render: (v) => label(v) },
                      { key: "n_seeds", header: "Seeds", digits: 0 },
                      { key: "accuracy", header: "Accuracy", render: (_, r) => pm(r, "accuracy") },
                      { key: "precision_class1", header: "Precision", render: (_, r) => pm(r, "precision_class1") },
                      { key: "recall_class1", header: "Recall", render: (_, r) => pm(r, "recall_class1") },
                      { key: "f1_class1", header: "F1", render: (_, r) => pm(r, "f1_class1") },
                      { key: "roc_auc", header: "ROC-AUC", render: (_, r) => pm(r, "roc_auc") },
                      { key: "pr_auc", header: "PR-AUC", render: (_, r) => pm(r, "pr_auc") },
                      { key: "train_seconds", header: "Train s", render: (_, r) => fmt(r.train_seconds_mean, 1) },
                      { key: "rounds", header: "Rounds", digits: 0 },
                      { key: "total_bytes", header: "Communication", render: (v) => bytesHuman(v) },
                    ]}
                    caption="Precision/recall/F1 refer to the diabetes class at the validation-chosen max-F1 threshold. Local-only rows average the 5 clients within each seed."
                  />
                </Card>
              )}
              {ci.length > 0 && (
                <Card title="Bootstrap 95% confidence intervals (test-set resampling, seed 0)">
                  <DataTable
                    rows={ci}
                    columns={[
                      { key: "model", header: "Model", render: (v) => label(v) },
                      { key: "roc_auc", header: "ROC-AUC" },
                      { key: "roc_auc_ci_low", header: "CI low" },
                      { key: "roc_auc_ci_high", header: "CI high" },
                      { key: "f1_class1", header: "F1" },
                      { key: "f1_ci_low", header: "F1 CI low" },
                      { key: "f1_ci_high", header: "F1 CI high" },
                      { key: "n_bootstrap", header: "Resamples", digits: 0 },
                    ]}
                    caption="CIs measure test-set sampling uncertainty for one trained model; seed std measures training variability."
                  />
                </Card>
              )}
              {local.length > 0 && (
                <Card title="Local-only models per client (test ROC-AUC)">
                  <DataTable
                    rows={local}
                    columns={[
                      { key: "setting", header: "Partition", render: (v) => label(v) },
                      { key: "seed", header: "Seed", digits: 0 },
                      { key: "client", header: "Client", digits: 0 },
                      { key: "roc_auc", header: "ROC-AUC" },
                      { key: "recall_class1", header: "Recall" },
                      { key: "f1_class1", header: "F1" },
                    ]}
                  />
                </Card>
              )}
              {final.length === 0 && (
                <Card title={`Raw per-run test results (results/federated/${String(s.tag)})`}>
                  <DataTable
                    rows={rows(s, "results").filter((r) => r.operating_point === "val_max_f1")}
                    columns={[
                      { key: "partition", header: "Partition", render: (v) => label(v) },
                      { key: "method", header: "Method", render: (v) => label(v) },
                      { key: "seed", header: "Seed", digits: 0 },
                      { key: "client", header: "Client", digits: 0 },
                      { key: "roc_auc", header: "ROC-AUC" },
                      { key: "f1_class1", header: "F1" },
                      { key: "total_bytes", header: "Bytes", render: (v) => bytesHuman(v) },
                    ]}
                  />
                </Card>
              )}
              <div className="grid gap-4 lg:grid-cols-2">
                {(s.figures ?? [])
                  .filter((f) => /final_comparison|confusion/.test(f))
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
