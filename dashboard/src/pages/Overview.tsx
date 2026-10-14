import { ArrowRight, BarChart2, Database, Layers, Network, Shield } from "lucide-react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Row } from "../api";
import { fmt, pct, rows } from "../format";
import { useExperiments, useSection } from "../hooks";
import { Card, DataTable, Figure, PageHeader, SectionGate, Stat, StatusBadge } from "../components/ui";

export default function Overview() {
  const state = useSection("overview");
  const { experiments } = useExperiments();

  return (
    <div className="space-y-6">
      <PageHeader
        title="Overview"
        description="Monitor cohort health, model readiness, and federated training across participating sites."
        actions={
          <>
            <Link to="/patient-risk" className="btn-secondary">
              Score a patient
            </Link>
            <Link to="/federated-learning" className="btn-teal">
              <Network className="h-4 w-4" />
              New experiment
            </Link>
          </>
        }
      />

      <SectionGate state={state}>
        {(s) => {
          const datasets = rows(s, "datasets");
          const primary = datasets.find((d) => d.role === "primary");
          const classes = rows(s, "class_distribution");
          const feats = rows(s, "features");
          const status = (s.status ?? {}) as Record<string, boolean>;
          const smd = [...feats]
            .filter((f) => typeof f.standardized_mean_difference === "number")
            .sort((a, b) => Math.abs(Number(b.standardized_mean_difference)) - Math.abs(Number(a.standardized_mean_difference)))
            .map((f) => ({ feature: f.feature, smd: Number(f.standardized_mean_difference) }));

          const recentExperiments = experiments.slice(0, 3);

          return (
            <>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat
                  label="Cohort size"
                  value={fmt(primary?.n_rows)}
                  hint="Health indicator records"
                  icon={<Database className="h-4 w-4" />}
                />
                <Stat
                  label="Clinical features"
                  value={fmt(primary?.n_features)}
                  hint="Demographic and health factors"
                  icon={<BarChart2 className="h-4 w-4" />}
                />
                <Stat
                  label="Positive prevalence"
                  value={pct(primary?.positive_or_max_class_share)}
                  hint="Diabetes-positive share"
                  icon={<Shield className="h-4 w-4" />}
                />
                <Stat
                  label="Exact duplicates"
                  value={fmt(primary?.n_exact_duplicate_rows)}
                  hint={`${pct(primary?.duplicate_share)} of the cohort`}
                  icon={<Layers className="h-4 w-4" />}
                />
              </div>

              {/* Two Column Section: Target Definition & Platform Artifact Status */}
              <div className="grid gap-6 lg:grid-cols-2">
                <Card title="Outcome definition">
                  <div className="mb-4 space-y-1.5 text-sm">
                    <p>
                      <span className="font-medium text-slate-700">Documented target</span>
                      <span className="mx-2 text-slate-300">·</span>
                      <span className="text-slate-600">{String(primary?.target_definition_documented ?? "–")}</span>
                    </p>
                    <p>
                      <span className="font-medium text-slate-700">Observed in data</span>
                      <span className="mx-2 text-slate-300">·</span>
                      <span className="text-slate-600">{String(primary?.target_definition_observed ?? "–")}</span>
                    </p>
                  </div>
                  <DataTable
                    rows={classes}
                    columns={[
                      { key: "class", header: "Class" },
                      { key: "meaning_observed", header: "Meaning" },
                      { key: "count", header: "Records", digits: 0 },
                      { key: "share", header: "Share", render: (v) => pct(v, 2) },
                    ]}
                  />
                </Card>

                <Card title="Pipeline status">
                  <p className="mb-4 text-sm text-slate-500">Latest artifacts available for review.</p>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(status).map(([k, v]) => (
                      <div
                        key={k}
                        className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50/80 px-3 py-2.5"
                      >
                        <span className="text-sm font-medium capitalize text-slate-700">{k.replace(/_/g, " ")}</span>
                        <span className={v ? "badge-success" : "badge-neutral"}>{v ? "Ready" : "Pending"}</span>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>

              {/* Recent Experiments Widget */}
              {recentExperiments.length > 0 && (
                <Card
                  title="Recent experiments"
                  actions={
                    <Link
                      to="/federated-learning"
                      className="inline-flex items-center gap-1 text-sm font-medium text-teal-700 hover:text-teal-900"
                    >
                      View all
                      <ArrowRight className="h-3.5 w-3.5" />
                    </Link>
                  }
                >
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                    {recentExperiments.map((exp) => (
                      <div
                        key={exp.id}
                        className="flex flex-col justify-between rounded-xl border border-slate-200 bg-slate-50/40 p-4"
                      >
                        <div>
                          <div className="mb-2 flex items-center justify-between">
                            <span className="badge-teal uppercase">{exp.algorithm}</span>
                            <StatusBadge status={exp.status} />
                          </div>
                          <h4 className="line-clamp-1 text-sm font-semibold text-slate-900">{exp.name}</h4>
                          <p className="mt-1 text-xs text-slate-500">
                            {exp.config.partition} · {exp.config.num_clients} sites · {exp.config.rounds} rounds
                          </p>
                        </div>
                        <div className="mt-4 flex items-center justify-between border-t border-slate-200/70 pt-3 text-sm">
                          <span className="text-slate-500">Test ROC-AUC</span>
                          <strong className="tabular-nums text-teal-800">
                            {exp.result ? fmt(exp.result.test_roc_auc, 4) : fmt(exp.progress.val_roc_auc, 4)}
                          </strong>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* Feature Separation: SMD Chart */}
              <Card title="Feature separation" subtitle="Standardized mean difference between diabetes and non-diabetes groups.">
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={smd} margin={{ top: 10, right: 10, left: -10, bottom: 65 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" />
                      <XAxis dataKey="feature" angle={-45} textAnchor="end" interval={0} fontSize={11} stroke="#94a3b8" />
                      <YAxis fontSize={11} stroke="#94a3b8" />
                      <Tooltip formatter={(v) => fmt(v, 3)} />
                      <Bar dataKey="smd" fill="#0f766e" name="Standardized mean difference" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <Card title="Feature dictionary">
                <DataTable
                  rows={feats as Row[]}
                  columns={[
                    { key: "feature", header: "Feature" },
                    { key: "group", header: "Type" },
                    { key: "meaning", header: "Description" },
                    { key: "mean", header: "Mean", digits: 3 },
                    { key: "min", header: "Min", digits: 0 },
                    { key: "max", header: "Max", digits: 0 },
                    { key: "mean_no_diabetes", header: "Mean (no diabetes)", digits: 3 },
                    { key: "mean_diabetes", header: "Mean (diabetes)", digits: 3 },
                    { key: "pearson_r_with_target", header: "Pearson r", digits: 3 },
                  ]}
                />
              </Card>

              {/* Figures */}
              {(s.figures ?? [])
                .filter((f) => f.includes("target_distribution"))
                .map((f) => (
                  <Figure key={f} rel={f} alt="Target distribution artifact" />
                ))}
            </>
          );
        }}
      </SectionGate>
    </div>
  );
}
