import { Activity, ArrowRight, BarChart2, CheckCircle2, Database, Layers, Network, Shield } from "lucide-react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Row } from "../api";
import { fmt, pct, rows } from "../format";
import { useExperiments, useSection } from "../hooks";
import { Card, DataTable, Disclaimers, Figure, SectionGate, Stat, StatusBadge } from "../components/ui";

export default function Overview() {
  const state = useSection("overview");
  const { experiments } = useExperiments();

  return (
    <div className="space-y-6">
      {/* Brand Header */}
      <header className="flex flex-col gap-2 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-slate-900">
              MediFederate AI Overview
            </h1>
            <span className="badge-teal font-semibold">Enterprise Health Edition</span>
          </div>
          <p className="mt-0.5 text-xs text-slate-500">
            Privacy-preserving federated machine learning intelligence for diabetes risk stratifications on heterogeneous client partitions.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link to="/federated-learning" className="btn-teal text-xs py-1.5 px-3">
            <Network className="h-3.5 w-3.5" />
            Launch Experiment
          </Link>
          <Link to="/patient-risk" className="btn-secondary text-xs py-1.5 px-3">
            Patient Risk Scoring
          </Link>
        </div>
      </header>

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
              <Disclaimers items={s.disclaimers} keys={["clients", "privacy"]} />

              {/* High-level KPI Stats */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat
                  label="Total Cohort Records"
                  value={fmt(primary?.n_rows)}
                  hint="CDC BRFSS 2015 Health Survey"
                  icon={<Database className="h-4 w-4" />}
                />
                <Stat
                  label="Clinical Indicators"
                  value={fmt(primary?.n_features)}
                  hint="Self-reported demographic & health factors"
                  icon={<BarChart2 className="h-4 w-4" />}
                />
                <Stat
                  label="Positive Class Share"
                  value={pct(primary?.positive_or_max_class_share)}
                  hint="Diabetes_binary = 1 (imbalanced)"
                  icon={<Shield className="h-4 w-4" />}
                />
                <Stat
                  label="Exact Duplicates"
                  value={fmt(primary?.n_exact_duplicate_rows)}
                  hint={`${pct(primary?.duplicate_share)} documented survey ties`}
                  icon={<Layers className="h-4 w-4" />}
                />
              </div>

              {/* Two Column Section: Target Definition & Platform Artifact Status */}
              <div className="grid gap-6 lg:grid-cols-2">
                <Card
                  title={
                    <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700">
                      <Database className="h-4 w-4 text-teal-600" />
                      Epidemiological Target Definition
                    </div>
                  }
                >
                  <div className="mb-3 space-y-1 text-xs">
                    <p>
                      <span className="font-semibold text-slate-700">Documented Target:</span>{" "}
                      <span className="text-slate-600">{String(primary?.target_definition_documented ?? "–")}</span>
                    </p>
                    <p>
                      <span className="font-semibold text-slate-700">Observed in File:</span>{" "}
                      <span className="text-slate-600">{String(primary?.target_definition_observed ?? "–")}</span>
                    </p>
                  </div>
                  <DataTable
                    rows={classes}
                    columns={[
                      { key: "class", header: "Class Code" },
                      { key: "meaning_observed", header: "Observed Diagnostic Meaning" },
                      { key: "count", header: "Record Count", digits: 0 },
                      { key: "share", header: "Prevalence Share", render: (v) => pct(v, 2) },
                    ]}
                  />
                </Card>

                {/* Model Artifacts & System Health */}
                <Card
                  title={
                    <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700">
                      <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                      Verified Experiment Artifacts on Disk
                    </div>
                  }
                >
                  <p className="text-xs text-slate-500 mb-3">
                    Aggregated outputs available in <code>results/</code> directory:
                  </p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {Object.entries(status).map(([k, v]) => (
                      <div
                        key={k}
                        className="flex items-center justify-between rounded bg-slate-50 p-2 border border-slate-100"
                      >
                        <span className="capitalize text-slate-700 font-medium">{k.replace(/_/g, " ")}</span>
                        <span className={v ? "badge-success text-[10px]" : "badge-neutral text-[10px]"}>
                          {v ? "Verified" : "Pending"}
                        </span>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>

              {/* Recent Experiments Widget */}
              {recentExperiments.length > 0 && (
                <Card
                  title={
                    <div className="flex items-center justify-between w-full">
                      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700">
                        <Activity className="h-4 w-4 text-teal-600" />
                        Recent Federated Learning Experiments
                      </div>
                      <Link
                        to="/federated-learning"
                        className="text-xs text-teal-700 hover:text-teal-900 font-semibold inline-flex items-center gap-1"
                      >
                        Control Center
                        <ArrowRight className="h-3 w-3" />
                      </Link>
                    </div>
                  }
                >
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                    {recentExperiments.map((exp) => (
                      <div
                        key={exp.id}
                        className="rounded border border-slate-200 bg-slate-50/50 p-3 flex flex-col justify-between"
                      >
                        <div>
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="badge-teal font-mono text-[10px] uppercase">{exp.algorithm}</span>
                            <StatusBadge status={exp.status} />
                          </div>
                          <h4 className="text-xs font-bold text-slate-900 line-clamp-1">{exp.name}</h4>
                          <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                            {exp.config.partition} · {exp.config.num_clients} clients · {exp.config.rounds} rounds
                          </p>
                        </div>
                        <div className="mt-3 border-t border-slate-200/60 pt-2 flex items-center justify-between text-xs font-mono">
                          <span className="text-slate-500 text-[10px]">Test ROC-AUC:</span>
                          <strong className="text-teal-700">
                            {exp.result ? fmt(exp.result.test_roc_auc, 4) : fmt(exp.progress.val_roc_auc, 4)}
                          </strong>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* Feature Separation: SMD Chart */}
              <Card
                title={
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700">
                    <BarChart2 className="h-4 w-4 text-teal-600" />
                    Feature Separation: Standardized Mean Difference (Diabetes vs Non-Diabetes)
                  </div>
                }
              >
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={smd} margin={{ top: 10, right: 10, left: -10, bottom: 65 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="feature" angle={-45} textAnchor="end" interval={0} fontSize={10} stroke="#64748b" />
                      <YAxis fontSize={10} stroke="#64748b" />
                      <Tooltip formatter={(v) => fmt(v, 3)} />
                      <Bar dataKey="smd" fill="#0d9488" name="Standardized Mean Difference" radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="mt-2 text-xs text-slate-500">
                  Descriptive standardized association in the survey sample (Cohen&apos;s d equivalent), not an individual causal effect.
                </p>
              </Card>

              {/* Feature Dictionary */}
              <Card
                title={
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700">
                    <Database className="h-4 w-4 text-slate-500" />
                    Clinical Indicator Feature Dictionary
                  </div>
                }
              >
                <DataTable
                  rows={feats as Row[]}
                  columns={[
                    { key: "feature", header: "Feature Identifier" },
                    { key: "group", header: "Data Type" },
                    { key: "meaning", header: "Clinical / Behavioral Indicator" },
                    { key: "mean", header: "Cohort Mean", digits: 3 },
                    { key: "min", header: "Min", digits: 0 },
                    { key: "max", header: "Max", digits: 0 },
                    { key: "mean_no_diabetes", header: "Mean (No Diabetes)", digits: 3 },
                    { key: "mean_diabetes", header: "Mean (Diabetes)", digits: 3 },
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
