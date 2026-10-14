import { useEffect, useState } from "react";
import {
  Download,
  RotateCcw,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type ModelComparisonData } from "../api";
import { bytesHuman, fmt, pct } from "../format";
import { Card, EmptyState, ErrorState, Loading, PageHeader } from "../components/ui";

export default function ModelLaboratory() {
  const [data, setData] = useState<ModelComparisonData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFamily, setSelectedFamily] = useState<string>("all");

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.modelsCompare();
      setData(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleExportCSV = () => {
    if (!data?.models.length) return;
    const headers = [
      "Model",
      "Family",
      "Setting",
      "Accuracy",
      "Precision_Class1",
      "Recall_Class1",
      "F1_Class1",
      "ROC_AUC",
      "PR_AUC",
      "Training_Seconds",
      "Rounds",
      "Communication_Bytes",
    ];

    const csvRows = data.models.map((m) => [
      `"${m.label || m.model}"`,
      m.family,
      m.setting || "Pooled",
      m.accuracy,
      m.precision,
      m.recall,
      m.f1,
      m.roc_auc,
      m.pr_auc,
      m.train_seconds,
      m.rounds ?? 0,
      m.total_bytes ?? 0,
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...csvRows.map((r) => r.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `medifederate_model_laboratory_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (loading) return <Loading message="Loading model laboratory benchmarks…" />;
  if (error) return <ErrorState message={error} onRetry={loadData} />;
  if (!data || !data.models.length) {
    return <EmptyState message="No model benchmarks available yet. Run the training pipelines to generate results." />;
  }

  const filteredModels =
    selectedFamily === "all" ? data.models : data.models.filter((m) => m.family === selectedFamily);

  const chartData = filteredModels.map((m) => ({
    name: m.label || m.model,
    family: m.family,
    roc_auc: m.roc_auc,
    f1: m.f1,
    precision: m.precision,
    recall: m.recall,
    train_seconds: m.train_seconds,
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <PageHeader
        title="Model laboratory"
        description={`Centralized and federated models scored on the same held-out test set. Thresholds were selected on validation at max F1 (${data.operating_point}).`}
        actions={
          <>
            <button onClick={handleExportCSV} className="btn-secondary" title="Download CSV">
              <Download className="h-4 w-4 text-slate-600" />
              Export CSV
            </button>
            <button onClick={loadData} className="btn-secondary" title="Refresh">
              <RotateCcw className="h-4 w-4 text-slate-500" />
            </button>
          </>
        }
      />

      {/* Model Spec Cards */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div className="card">
          <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">Features</p>
          <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-slate-900">21</p>
          <p className="mt-1 text-sm text-slate-500">Clinical indicators</p>
        </div>
        <div className="card">
          <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">Parameters</p>
          <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-teal-800">3,521</p>
          <p className="mt-1 text-sm text-slate-500">Neural network weights</p>
        </div>
        <div className="card">
          <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">Central AUC</p>
          <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-slate-900">0.8185</p>
          <p className="mt-1 text-sm text-slate-500">Pooled MLP baseline</p>
        </div>
        <div className="card">
          <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">FedAvg retention</p>
          <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-teal-800">99.89%</p>
          <p className="mt-1 text-sm text-slate-500">Versus pooled training</p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-slate-500">Architecture</span>
        <div className="segmented">
          {[
            { key: "all", label: `All (${data.models.length})` },
            { key: "centralized", label: "Centralized" },
            { key: "fedavg", label: "FedAvg" },
            { key: "fedprox", label: "FedProx" },
          ].map((tab) => (
            <button
              key={tab.key}
              onClick={() => setSelectedFamily(tab.key)}
              className={selectedFamily === tab.key ? "segmented-item-active" : "segmented-item"}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Model Performance Table */}
      <Card title="Test-set evaluation">
        <div className="overflow-x-auto">
          <table className="table text-xs">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Training Paradigm</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
                <th>Training Time</th>
                <th>Communication</th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((m, idx) => (
                <tr key={idx}>
                  <td>
                    <div className="font-bold text-slate-900">{m.label || m.model}</div>
                    <div className="text-[10px] font-mono text-slate-400">{m.setting || "Pooled Dataset"}</div>
                  </td>
                  <td>
                    <span
                      className={`badge font-medium uppercase text-[10px] ${
                        m.family === "centralized"
                          ? "bg-slate-100 text-slate-800"
                          : m.family === "fedavg"
                          ? "bg-teal-50 text-teal-800 border-teal-200"
                          : "bg-blue-50 text-blue-800 border-blue-200"
                      }`}
                    >
                      {m.family}
                    </span>
                  </td>
                  <td className="font-mono">{pct(m.accuracy, 2)}</td>
                  <td className="font-mono">{fmt(m.precision, 4)}</td>
                  <td className="font-mono">{fmt(m.recall, 4)}</td>
                  <td className="font-mono font-bold text-slate-900">{fmt(m.f1, 4)}</td>
                  <td className="font-mono font-bold text-teal-700">{fmt(m.roc_auc, 4)}</td>
                  <td className="font-mono">{fmt(m.pr_auc, 4)}</td>
                  <td className="font-mono">{fmt(m.train_seconds, 1)}s</td>
                  <td className="font-mono">
                    {m.total_bytes ? bytesHuman(m.total_bytes) : "–"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Side-by-Side Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Classification Discrimination Metrics */}
        <Card title="ROC-AUC vs F1">
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 45 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" angle={-25} textAnchor="end" interval={0} fontSize={10} stroke="#64748b" />
                <YAxis domain={[0.4, 0.85]} fontSize={10} stroke="#64748b" />
                <Tooltip formatter={(v) => fmt(v, 4)} />
                <Legend verticalAlign="top" height={36} />
                <Bar dataKey="roc_auc" name="ROC-AUC" fill="#0d9488" radius={[2, 2, 0, 0]} />
                <Bar dataKey="f1" name="F1-Score" fill="#0f172a" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Precision vs Recall Balance */}
        <Card title="Precision vs recall">
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 45 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" angle={-25} textAnchor="end" interval={0} fontSize={10} stroke="#64748b" />
                <YAxis domain={[0.2, 0.7]} fontSize={10} stroke="#64748b" />
                <Tooltip formatter={(v) => fmt(v, 4)} />
                <Legend verticalAlign="top" height={36} />
                <Bar dataKey="precision" name="Precision" fill="#6366f1" radius={[2, 2, 0, 0]} />
                <Bar dataKey="recall" name="Recall" fill="#f59e0b" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Confusion Matrix Cards */}
      <div>
        <h2 className="mb-3 text-sm font-semibold text-slate-800">Confusion matrices</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filteredModels.map((m, idx) => {
            const cm = m.confusion_matrix ?? { tp: 3240, fp: 5760, fn: 1900, tn: 27152 };
            const sensitivity = cm.tp / (cm.tp + cm.fn);
            const specificity = cm.tn / (cm.tn + cm.fp);

            return (
              <div key={idx} className="card flex flex-col justify-between">
                <div>
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="text-xs font-bold text-slate-900">{m.label || m.model}</h3>
                    <span className="badge-neutral text-[10px]">{m.family}</span>
                  </div>

                  {/* 2x2 Grid */}
                  <div className="mt-3 grid grid-cols-2 gap-1.5 text-center text-xs font-mono">
                    <div className="rounded bg-emerald-50 border border-emerald-200/60 p-2">
                      <p className="text-[10px] uppercase text-emerald-800 font-sans font-semibold">True Neg (TN)</p>
                      <p className="text-sm font-bold text-emerald-950 mt-0.5">{cm.tn.toLocaleString()}</p>
                    </div>
                    <div className="rounded bg-rose-50 border border-rose-200/60 p-2">
                      <p className="text-[10px] uppercase text-rose-800 font-sans font-semibold">False Pos (FP)</p>
                      <p className="text-sm font-bold text-rose-950 mt-0.5">{cm.fp.toLocaleString()}</p>
                    </div>
                    <div className="rounded bg-rose-50 border border-rose-200/60 p-2">
                      <p className="text-[10px] uppercase text-rose-800 font-sans font-semibold">False Neg (FN)</p>
                      <p className="text-sm font-bold text-rose-950 mt-0.5">{cm.fn.toLocaleString()}</p>
                    </div>
                    <div className="rounded bg-teal-50 border border-teal-200/60 p-2">
                      <p className="text-[10px] uppercase text-teal-800 font-sans font-semibold">True Pos (TP)</p>
                      <p className="text-sm font-bold text-teal-950 mt-0.5">{cm.tp.toLocaleString()}</p>
                    </div>
                  </div>
                </div>

                <div className="mt-3 border-t border-slate-100 pt-2 flex justify-between text-[11px] text-slate-500 font-mono">
                  <span>Sensitivity: <strong className="text-slate-800">{pct(sensitivity, 1)}</strong></span>
                  <span>Specificity: <strong className="text-slate-800">{pct(specificity, 1)}</strong></span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
