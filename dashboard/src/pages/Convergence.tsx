import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Row } from "../api";
import { COLORS, bytesHuman, fmt, label, num, rows } from "../format";
import { useSection } from "../hooks";
import { Card, DataTable, PageHeader, SectionGate } from "../components/ui";

/** Mean of a metric per (series, round) across seeds. */
function series(hist: Row[], partition: string, metric: string) {
  const algos = [...new Set(hist.map((h) => String(h.algorithm)))];
  const roundsAll = [...new Set(hist.map((h) => Number(h.round)))].sort((a, b) => a - b);
  const data = roundsAll.map((r) => {
    const row: Row = { round: r };
    for (const a of algos) {
      const vals = hist.filter((h) => String(h.partition) === partition && String(h.algorithm) === a && Number(h.round) === r).map((h) => num(h[metric]));
      const ok = vals.filter((v): v is number => v !== null);
      row[a] = ok.length ? ok.reduce((x, y) => x + y, 0) / ok.length : null;
    }
    return row;
  });
  return { algos, data };
}

export default function Convergence() {
  const state = useSection("convergence");
  const [partition, setPartition] = useState<string | null>(null);
  return (
    <div className="space-y-6">
      <PageHeader
        title="Convergence"
        description="Validation ROC-AUC and mean client loss per round, with the parameter payload exchanged."
      />
      <SectionGate state={state}>
        {(s) => {
          const hist = rows(s, "round_history");
          const partitions = [...new Set(hist.map((h) => String(h.partition)))];
          const part = partition ?? partitions[0];
          const auc = series(hist, part, "val_roc_auc");
          const loss = series(hist, part, "client_train_loss");
          const conv = rows(s, "convergence_summary");
          const cost = rows(s, "cost").filter((c) => Number(c.rounds) > 0);
          const perRound = hist.find((h) => Number(h.round) === 1);
          return (
            <>
              <div className="flex items-center gap-2 text-sm">
                Partition
                <select className="input w-auto" value={part} onChange={(e) => setPartition(e.target.value)}>
                  {partitions.map((p) => (
                    <option key={p} value={p}>
                      {label(p)}
                    </option>
                  ))}
                </select>
                {perRound && (
                  <span className="text-xs text-slate-500">
                    Per round: {bytesHuman(num(perRound.bytes_down))} down + {bytesHuman(num(perRound.bytes_up))} up across {String(perRound.n_clients)} clients.
                  </span>
                )}
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Validation ROC-AUC per round">
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={auc.data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" fontSize={11} />
                      <YAxis domain={["auto", "auto"]} fontSize={11} tickFormatter={(v) => Number(v).toFixed(3)} />
                      <Tooltip formatter={(v) => fmt(v)} />
                      <Legend formatter={(v) => label(v)} />
                      {auc.algos.map((a, i) => (
                        <Line key={a} dataKey={a} stroke={COLORS[i % COLORS.length]} type="monotone" connectNulls />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
                <Card title="Mean client training loss per round">
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={loss.data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" fontSize={11} />
                      <YAxis domain={["auto", "auto"]} fontSize={11} tickFormatter={(v) => Number(v).toFixed(3)} />
                      <Tooltip formatter={(v) => fmt(v)} />
                      <Legend formatter={(v) => label(v)} />
                      {loss.algos.map((a, i) => (
                        <Line key={a} dataKey={a} stroke={COLORS[i % COLORS.length]} type="monotone" connectNulls />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              </div>
              {conv.length > 0 && (
                <Card title="Convergence summary per run">
                  <DataTable
                    rows={conv}
                    columns={[
                      { key: "partition", header: "Partition", render: (v) => label(v) },
                      { key: "algorithm", header: "Algorithm", render: (v) => label(v) },
                      { key: "seed", header: "Seed", digits: 0 },
                      { key: "rounds", header: "Rounds", digits: 0 },
                      { key: "round1_val_roc_auc", header: "Round-1 val AUC" },
                      { key: "final_val_roc_auc", header: "Final val AUC" },
                      { key: "final_test_roc_auc", header: "Final test AUC" },
                      { key: "rounds_to_within_0.005_of_final", header: "Rounds to ±0.005 of final", digits: 0 },
                      { key: "rounds_to_target", header: "Rounds to centralized target", digits: 0 },
                      { key: "total_seconds", header: "Seconds", digits: 1 },
                      { key: "cumulative_bytes", header: "Total communication", render: (v) => bytesHuman(v) },
                    ]}
                    caption="Target = centralized MLP validation ROC-AUC minus the configured tolerance; '–' means not reached within the run."
                  />
                </Card>
              )}
              <Card title="Runtime and communication per federated run">
                <DataTable
                  rows={cost}
                  columns={[
                    { key: "partition", header: "Partition", render: (v) => label(v) },
                    { key: "method", header: "Method", render: (v) => label(v) },
                    { key: "seed", header: "Seed", digits: 0 },
                    { key: "train_seconds", header: "Seconds", digits: 1 },
                    { key: "rounds", header: "Rounds", digits: 0 },
                    { key: "bytes_down", header: "Down", render: (v) => bytesHuman(v) },
                    { key: "bytes_up", header: "Up", render: (v) => bytesHuman(v) },
                    { key: "total_bytes", header: "Total", render: (v) => bytesHuman(v) },
                    { key: "bytes_to_target", header: "Bytes to target", render: (v) => bytesHuman(v) },
                  ]}
                  caption="Bytes count raw float32 parameters (3,521 per model copy) without compression or protocol overhead."
                />
              </Card>
            </>
          );
        }}
      </SectionGate>
    </div>
  );
}
