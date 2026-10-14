import { useEffect, useState } from "react";
import {
  Activity,
  Building2,
  CheckCircle2,
  Clock,
  Play,
  RotateCcw,
  StopCircle,
  Trash2,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CreateExperimentInput } from "../api";
import { bytesHuman, fmt, pct } from "../format";
import { useExperiments } from "../hooks";
import { Card, EmptyState, ErrorState, PageHeader, StatusBadge } from "../components/ui";

export default function FederatedLearning() {
  const { experiments, loading, error, create, cancel, remove, refresh } = useExperiments();

  // Active running or selected experiment
  const runningExp = experiments.find((e) => e.status === "running" || e.status === "queued");
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);

  // Form state
  const [algorithm, setAlgorithm] = useState<"fedavg" | "fedprox">("fedavg");
  const [partition, setPartition] = useState<"iid" | "dirichlet_a0.5" | "dirichlet_a0.1">("iid");
  const [numClients, setNumClients] = useState<number>(5);
  const [rounds, setRounds] = useState<number>(10);
  const [localEpochs, setLocalEpochs] = useState<number>(2);
  const [batchSize, setBatchSize] = useState<number>(64);
  const [learningRate, setLearningRate] = useState<number>(0.01);
  const [mu, setMu] = useState<number>(0.01);
  const [seed, setSeed] = useState<number>(42);
  const [name, setName] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Auto-generate name when parameters change if user hasn't explicitly customized it
  useEffect(() => {
    const dateStr = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    const algoStr = algorithm === "fedavg" ? "FedAvg" : `FedProx-μ${mu}`;
    const partStr = partition === "iid" ? "IID" : "NonIID-a0.5";
    setName(`${algoStr}-${partStr}-${numClients}C-R${rounds}-${dateStr}`);
  }, [algorithm, partition, numClients, rounds, mu]);

  const activeExp = runningExp ?? experiments.find((e) => e.id === selectedExpId) ?? experiments[0];

  const handleStart = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      const input: CreateExperimentInput = {
        name: name.trim(),
        algorithm,
        partition,
        num_clients: numClients,
        rounds,
        local_epochs: localEpochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        mu: algorithm === "fedprox" ? mu : undefined,
        seed,
      };
      const created = await create(input);
      setSelectedExpId(created.id);
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = async (id: string) => {
    try {
      await cancel(id);
    } catch (err: unknown) {
      console.error("Cancel failed:", err);
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm("Delete this experiment record?")) {
      await remove(id);
      if (selectedExpId === id) setSelectedExpId(null);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Federated learning"
        description="Configure a run, train across sites, and watch global metrics converge — without moving patient records."
        actions={
          <button onClick={() => refresh()} className="btn-secondary" title="Refresh status">
            <RotateCcw className="h-4 w-4 text-slate-500" />
            Refresh
          </button>
        }
      />

      {/* Main Grid: Experiment Configuration on left, Live Active Monitor on right */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Left Column: Configuration Form (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <Card title="New experiment">
            <form onSubmit={handleStart} className="space-y-4 text-sm">
              {submitError && (
                <div className="rounded border border-rose-300 bg-rose-50 p-2.5 text-rose-900">
                  {submitError}
                </div>
              )}

              {/* Experiment Name */}
              <div>
                <label className="block font-semibold text-slate-700 mb-1">Experiment Name</label>
                <input
                  type="text"
                  className="input font-mono text-xs"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. FedAvg-IID-R10"
                  required
                />
              </div>

              {/* Algorithm & Partition */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Algorithm</label>
                  <select
                    className="input"
                    value={algorithm}
                    onChange={(e) => setAlgorithm(e.target.value as "fedavg" | "fedprox")}
                  >
                    <option value="fedavg">FedAvg</option>
                    <option value="fedprox">FedProx</option>
                  </select>
                </div>
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Partition</label>
                  <select
                    className="input"
                    value={partition}
                    onChange={(e) => setPartition(e.target.value as "iid" | "dirichlet_a0.5")}
                  >
                    <option value="iid">Stratified IID</option>
                    <option value="dirichlet_a0.5">Non-IID (Dirichlet α=0.5)</option>
                  </select>
                </div>
              </div>

              {/* FedProx Mu Parameter (conditional) */}
              {algorithm === "fedprox" && (
                <div className="rounded border border-teal-200 bg-teal-50/50 p-2.5">
                  <div className="flex items-center justify-between mb-1">
                    <label className="font-semibold text-teal-900">FedProx Proximal Term (μ)</label>
                    <span className="font-mono font-bold text-teal-700">{mu}</span>
                  </div>
                  <input
                    type="range"
                    min="0.001"
                    max="0.5"
                    step="0.005"
                    value={mu}
                    onChange={(e) => setMu(Number(e.target.value))}
                    className="w-full accent-teal-600"
                  />
                  <p className="mt-1 text-[11px] text-teal-700">
                    Constrains local weight drift on heterogeneous client data.
                  </p>
                </div>
              )}

              {/* Number of Clients Slider */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="font-semibold text-slate-700">Participating sites</label>
                  <span className="badge-neutral font-bold">{numClients} Nodes</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="10"
                  step="1"
                  value={numClients}
                  onChange={(e) => setNumClients(Number(e.target.value))}
                  className="w-full accent-teal-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>2 Clients</span>
                  <span>5 Clients</span>
                  <span>10 Clients</span>
                </div>
              </div>

              {/* Rounds, Local Epochs */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Communication Rounds</label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    className="input tabular-nums"
                    value={rounds}
                    onChange={(e) => setRounds(Math.max(1, Math.min(50, Number(e.target.value))))}
                  />
                </div>
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Local Epochs</label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    className="input tabular-nums"
                    value={localEpochs}
                    onChange={(e) => setLocalEpochs(Math.max(1, Math.min(10, Number(e.target.value))))}
                  />
                </div>
              </div>

              {/* Batch Size & Learning Rate */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Batch Size</label>
                  <select
                    className="input"
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                  >
                    <option value="32">32</option>
                    <option value="64">64 (Recommended)</option>
                    <option value="128">128</option>
                    <option value="256">256</option>
                  </select>
                </div>
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Learning Rate</label>
                  <input
                    type="number"
                    step="0.001"
                    min="0.0001"
                    max="0.5"
                    className="input tabular-nums"
                    value={learningRate}
                    onChange={(e) => setLearningRate(Number(e.target.value))}
                  />
                </div>
              </div>

              {/* Random Seed */}
              <div>
                <label className="block font-semibold text-slate-700 mb-1">Random Seed</label>
                <input
                  type="number"
                  className="input tabular-nums"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                />
              </div>

              <div className="pt-2">
                <button
                  type="submit"
                  disabled={isSubmitting || Boolean(runningExp)}
                  className="btn-teal h-10 w-full font-semibold"
                >
                  <Play className="h-4 w-4" />
                  {runningExp ? "Experiment Already Running…" : isSubmitting ? "Initializing…" : "Start Experiment"}
                </button>
                {runningExp && (
                  <p className="mt-1 text-center text-[11px] text-amber-700">
                    Wait for active run to finish or cancel it before starting another.
                  </p>
                )}
              </div>
            </form>
          </Card>
        </div>

        {/* Right Column: Running / Selected Experiment Live Panel (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {activeExp ? (
            <Card
              title={
                <div className="flex items-center justify-between w-full">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-teal-600" />
                    <span className="font-bold text-slate-900">{activeExp.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <StatusBadge status={activeExp.status} />
                    {activeExp.status === "running" && (
                      <button
                        onClick={() => handleCancel(activeExp.id)}
                        className="btn-danger text-xs px-2 py-1"
                        title="Cancel running experiment"
                      >
                        <StopCircle className="h-3.5 w-3.5" />
                        Cancel
                      </button>
                    )}
                  </div>
                </div>
              }
              subtitle={
                <span className="text-slate-500 font-mono text-[11px]">
                  ID: {activeExp.id} · Created: {new Date(activeExp.created_at).toLocaleTimeString()} · Algorithm: {activeExp.algorithm.toUpperCase()} · Partition: {activeExp.config.partition}
                </span>
              }
            >
              {/* Progress Bar & Round Status */}
              <div className="mb-4 rounded bg-slate-50 p-3 border border-slate-200/80">
                <div className="flex items-center justify-between mb-1.5 text-xs">
                  <div className="font-semibold text-slate-700 flex items-center gap-1.5">
                    <Clock className="h-3.5 w-3.5 text-slate-500" />
                    Round {activeExp.progress.current_round} of {activeExp.progress.total_rounds}
                  </div>
                  <div className="font-mono font-bold text-teal-700">
                    {Math.round((activeExp.progress.current_round / activeExp.progress.total_rounds) * 100)}%
                  </div>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full bg-teal-600 transition-all duration-500 rounded-full"
                    style={{
                      width: `${Math.min(100, Math.round((activeExp.progress.current_round / activeExp.progress.total_rounds) * 100))}%`,
                    }}
                  />
                </div>
                <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
                  <span>Elapsed Time: {activeExp.progress.elapsed_seconds}s</span>
                  <span>Payload Exchanged: {bytesHuman(activeExp.progress.current_round * activeExp.config.num_clients * 3521 * 8)}</span>
                </div>
              </div>

              {/* Real-time Metrics Tiles */}
              <div className="grid grid-cols-3 gap-3 mb-4">
                <div className="rounded border border-slate-200 bg-white p-2.5 text-center">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Current Loss</p>
                  <p className="mt-0.5 text-lg font-bold tabular-nums text-slate-800">
                    {fmt(activeExp.progress.loss, 4)}
                  </p>
                </div>
                <div className="rounded border border-slate-200 bg-white p-2.5 text-center">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Global Accuracy</p>
                  <p className="mt-0.5 text-lg font-bold tabular-nums text-teal-700">
                    {pct(activeExp.progress.accuracy, 2)}
                  </p>
                </div>
                <div className="rounded border border-slate-200 bg-white p-2.5 text-center">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Val ROC-AUC</p>
                  <p className="mt-0.5 text-lg font-bold tabular-nums text-sky-700">
                    {fmt(activeExp.progress.val_roc_auc ?? activeExp.result?.test_roc_auc, 4)}
                  </p>
                </div>
              </div>

              {/* Live Metric Trajectory Chart */}
              <div className="mb-4">
                <h3 className="text-xs font-semibold text-slate-700 mb-2 flex items-center gap-1.5">
                  <Activity className="h-3.5 w-3.5 text-teal-600" />
                  Metric Trajectory Across Rounds
                </h3>
                <div className="h-44 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={activeExp.progress.history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="round" fontSize={10} stroke="#94a3b8" />
                      <YAxis domain={["auto", "auto"]} fontSize={10} stroke="#94a3b8" />
                      <Tooltip formatter={(v) => fmt(v, 4)} />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#0d9488"
                        strokeWidth={2}
                        dot={false}
                        name="Accuracy"
                      />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#f43f5e"
                        strokeWidth={1.5}
                        dot={false}
                        name="Loss"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Client Training Nodes Grid */}
              <div>
                <h3 className="text-xs font-semibold text-slate-700 mb-2 flex items-center gap-1.5">
                  <Building2 className="h-3.5 w-3.5 text-slate-500" />
                  Participating sites ({activeExp.progress.clients.length})
                </h3>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {activeExp.progress.clients.map((c) => (
                    <div
                      key={c.client_id}
                      className="rounded border border-slate-200 bg-slate-50/50 p-2.5 text-xs flex items-center justify-between"
                    >
                      <div>
                        <div className="flex items-center gap-1.5">
                          <span className="font-semibold text-slate-800">Client #{c.client_id}</span>
                          <span className="text-[10px] text-slate-400">({c.samples.toLocaleString()} samples)</span>
                        </div>
                        <p className="text-[11px] text-slate-500 truncate max-w-[180px]">{c.name}</p>
                      </div>
                      <div className="text-right">
                        <StatusBadge status={c.status} />
                        <p className="text-[10px] font-mono text-slate-600 mt-1">
                          loss: {fmt(c.loss, 3)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Final Results Banner if Completed */}
              {activeExp.result && (
                <div className="mt-4 rounded border border-emerald-200 bg-emerald-50/70 p-3 text-xs">
                  <div className="flex items-center gap-2 font-semibold text-emerald-950 mb-1">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    Final Evaluation on Held-Out Test Split
                  </div>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 font-mono text-emerald-900 mt-2">
                    <div>ROC-AUC: <span className="font-bold">{fmt(activeExp.result.test_roc_auc, 4)}</span></div>
                    <div>Accuracy: <span className="font-bold">{pct(activeExp.result.test_accuracy, 2)}</span></div>
                    <div>F1-Score: <span className="font-bold">{fmt(activeExp.result.test_f1, 3)}</span></div>
                    <div>Recall: <span className="font-bold">{pct(activeExp.result.test_recall, 1)}</span></div>
                  </div>
                </div>
              )}
            </Card>
          ) : (
            <EmptyState
              title="No experiment selected"
              message="Configure a run on the left, or choose a previous experiment from the history table."
            />
          )}
        </div>
      </div>

      {/* Experiment History Table */}
      <Card
              title={`Experiment history (${experiments.length})`}
      >
        {loading ? (
          <p className="text-xs text-slate-500 py-4 text-center">Loading past runs…</p>
        ) : error ? (
          <ErrorState message={error} />
        ) : experiments.length === 0 ? (
          <EmptyState title="No runs yet" message="Start an experiment to populate this history." />
        ) : (
          <div className="overflow-x-auto">
            <table className="table text-xs">
              <thead>
                <tr>
                  <th>Experiment Name</th>
                  <th>Algorithm</th>
                  <th>Partition</th>
                  <th>Clients</th>
                  <th>Rounds</th>
                  <th>Status</th>
                  <th>Test ROC-AUC</th>
                  <th>Runtime</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp) => (
                  <tr
                    key={exp.id}
                    onClick={() => setSelectedExpId(exp.id)}
                    className={`cursor-pointer transition-colors ${
                      activeExp?.id === exp.id ? "bg-teal-50/60 font-medium" : ""
                    }`}
                  >
                    <td>
                      <div className="font-semibold text-slate-900">{exp.name}</div>
                      <div className="text-[10px] font-mono text-slate-400">{exp.id}</div>
                    </td>
                    <td>
                      <span className="uppercase font-mono text-slate-700">{exp.algorithm}</span>
                    </td>
                    <td>{exp.config.partition === "iid" ? "IID" : "Non-IID Dirichlet"}</td>
                    <td>{exp.config.num_clients}</td>
                    <td>
                      {exp.progress.current_round} / {exp.config.rounds}
                    </td>
                    <td>
                      <StatusBadge status={exp.status} />
                    </td>
                    <td className="font-mono">
                      {exp.result ? fmt(exp.result.test_roc_auc, 4) : exp.progress.val_roc_auc ? fmt(exp.progress.val_roc_auc, 4) : "–"}
                    </td>
                    <td>{exp.progress.elapsed_seconds}s</td>
                    <td>
                      <button
                        onClick={(e) => handleDelete(exp.id, e)}
                        disabled={exp.status === "running"}
                        className="p-1 text-slate-400 hover:text-rose-600 disabled:opacity-20"
                        title="Delete run record"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
