import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { COLORS, fmt, label, num, rows } from "../format";
import { useSection } from "../hooks";
import { Card, DataTable, Disclaimers, Figure, Notice, SectionGate } from "../components/ui";

export default function Privacy() {
  const state = useSection("privacy");
  return (
    <>
      <header>
        <h1 className="text-xl font-semibold">Differential privacy and privacy–utility comparison</h1>
        <p className="text-sm text-slate-600">
          Example-level DP-SGD (Opacus) inside each client's local training, compared with the non-private FedAvg/FedProx runs.
        </p>
      </header>
      <SectionGate state={state}>
        {(s) => {
          const res = rows(s, "results").filter((r) => r.operating_point === "val_max_f1");
          const priv = res.filter((r) => r.private === true);
          const nonpriv = res.filter((r) => r.private !== true);
          const partitions = [...new Set(res.map((r) => String(r.partition)))];
          const points = partitions.map((p) => ({
            name: label(p),
            data: priv
              .filter((r) => String(r.partition) === p)
              .map((r) => ({ eps: num(r.epsilon_max), auc: num(r.roc_auc), sigma: r.noise_multiplier, algo: label(r.algorithm) })),
          }));
          const cfg = (s.config ?? {}) as Record<string, unknown>;
          return (
            <>
              <Disclaimers items={s.disclaimers} keys={["privacy"]} />
              <Notice>
                <span className="font-medium">Privacy unit:</span> {String(s.privacy_unit ?? "one training example of one client (example-level DP)")}.
                This is <span className="font-medium">not</span> client-level DP and does not hide which client participated. ε is the maximum over clients
                at δ = {fmt(cfg.delta, 6)}; clipping norm C = {fmt(cfg.max_grad_norm, 2)}; accountant: {String(cfg.accountant ?? "–")}.
              </Notice>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Privacy–utility: test ROC-AUC vs ε (lower ε = stronger guarantee)">
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart margin={{ bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="eps" name="ε (max over clients)" type="number" fontSize={11} label={{ value: "ε", position: "insideBottom", offset: -10, fontSize: 11 }} />
                      <YAxis dataKey="auc" name="Test ROC-AUC" domain={["auto", "auto"]} fontSize={11} tickFormatter={(v) => Number(v).toFixed(3)} />
                      <ZAxis range={[80, 80]} />
                      <Tooltip
                        cursor={{ strokeDasharray: "3 3" }}
                        content={({ payload }) => {
                          const p = payload?.[0]?.payload as { eps: number; auc: number; sigma: unknown; algo: string } | undefined;
                          return p ? (
                            <div className="rounded border bg-white p-2 text-xs shadow">
                              {p.algo}, σ = {String(p.sigma)}: ε = {fmt(p.eps, 2)}, AUC = {fmt(p.auc)}
                            </div>
                          ) : null;
                        }}
                      />
                      <Legend />
                      {points.map((pt, i) => (
                        <Scatter key={pt.name} name={pt.name} data={pt.data} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                  <p className="mt-1 text-xs text-slate-500">
                    Non-private reference (no formal guarantee):{" "}
                    {nonpriv.map((r) => `${label(r.partition)} ${label(r.algorithm)} ${fmt(r.roc_auc)}`).join("; ") || "–"}
                  </p>
                </Card>
                <Card title="Private vs non-private runs (test split, val_max_f1 threshold)">
                  <DataTable
                    rows={res}
                    columns={[
                      { key: "partition", header: "Partition", render: (v) => label(v) },
                      { key: "algorithm", header: "Algorithm", render: (v) => label(v) },
                      { key: "private", header: "DP", render: (v) => (v ? "yes" : "no") },
                      { key: "noise_multiplier", header: "σ", digits: 1 },
                      { key: "epsilon_max", header: "ε max", digits: 2 },
                      { key: "roc_auc", header: "ROC-AUC" },
                      { key: "recall_class1", header: "Recall" },
                      { key: "f1_class1", header: "F1" },
                      { key: "train_seconds", header: "Seconds", digits: 1 },
                    ]}
                  />
                </Card>
              </div>
              <Card title="Privacy accounting per client">
                <DataTable
                  rows={rows(s, "accounting")}
                  columns={[
                    { key: "partition", header: "Partition", render: (v) => label(v) },
                    { key: "algorithm", header: "Algorithm", render: (v) => label(v) },
                    { key: "noise_multiplier", header: "σ", digits: 1 },
                    { key: "client", header: "Client", digits: 0 },
                    { key: "n_train", header: "Train rows", digits: 0 },
                    { key: "sample_rate", header: "Sample rate", digits: 4 },
                    { key: "steps", header: "Steps", digits: 0 },
                    { key: "epsilon", header: "ε", digits: 3 },
                    { key: "accountant", header: "Accountant" },
                  ]}
                  caption="ε accumulates across all rounds for each client (RDP accountant, Poisson sampling)."
                />
              </Card>
              <div className="grid gap-4 lg:grid-cols-2">
                {(s.figures ?? []).map((f) => (
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
