import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Row } from "../api";
import { COLORS, fmt, label, pct, rows } from "../format";
import { useSection } from "../hooks";
import { Card, DataTable, Disclaimers, Figure, PageHeader, SectionGate } from "../components/ui";

export default function Partitions() {
  const state = useSection("partitions");
  return (
    <div className="space-y-6">
      <PageHeader
        title="Site partitions"
        description="How training rows and positive labels are spread across five sites under IID and Dirichlet partitioning."
      />
      <SectionGate state={state}>
        {(s) => {
          const parts = rows(s, "partitions").filter((r) => r.split === "train");
          const partitions = [...new Set(parts.map((p) => String(p.partition)))];
          const clients = [...new Set(parts.map((p) => Number(p.client)))].sort((a, b) => a - b);
          const build = (key: string) =>
            clients.map((c) => {
              const row: Row = { client: `client ${c}` };
              for (const p of partitions) {
                const r = parts.find((x) => String(x.partition) === p && Number(x.client) === c);
                row[p] = r ? Number(r[key]) : null;
              }
              return row;
            });
          const params = (s.partition_params ?? {}) as Record<string, Record<string, unknown>>;
          return (
            <>
              <Disclaimers items={s.disclaimers} keys={["clients"]} />
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Training rows per client">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={build("n")}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="client" fontSize={11} />
                      <YAxis fontSize={11} />
                      <Tooltip formatter={(v) => fmt(v, 0)} />
                      <Legend formatter={(v) => label(v)} />
                      {partitions.map((p, i) => (
                        <Bar key={p} dataKey={p} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
                <Card title="Positive (diabetes) rate per client">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={build("positive_rate")}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="client" fontSize={11} />
                      <YAxis fontSize={11} tickFormatter={(v) => pct(v, 0)} />
                      <Tooltip formatter={(v) => pct(v, 2)} />
                      <Legend formatter={(v) => label(v)} />
                      {partitions.map((p, i) => (
                        <Bar key={p} dataKey={p} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </div>
              <Card title="Partition parameters">
                <ul className="text-sm">
                  {Object.entries(params).map(([name, p]) => (
                    <li key={name} className="py-0.5">
                      <span className="font-medium">{label(name)}</span>:{" "}
                      {Object.entries(p)
                        .filter(([k]) => k !== "proportions")
                        .map(([k, v]) => `${k} = ${Array.isArray(v) ? v.join(", ") : String(v)}`)
                        .join("; ")}
                    </li>
                  ))}
                </ul>
              </Card>
              <Card title="Per-client table">
                <DataTable
                  rows={rows(s, "partitions")}
                  columns={[
                    { key: "partition", header: "Partition", render: (v) => label(v) },
                    { key: "split", header: "Split" },
                    { key: "client", header: "Client", digits: 0 },
                    { key: "n", header: "Rows", digits: 0 },
                    { key: "n_positive", header: "Positives", digits: 0 },
                    { key: "positive_rate", header: "Positive rate", render: (v) => pct(v, 2) },
                  ]}
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
    </div>
  );
}
