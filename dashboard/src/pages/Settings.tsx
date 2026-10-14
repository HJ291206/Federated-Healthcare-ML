import { useEffect, useState } from "react";
import {
  Check,
  RotateCcw,
  Save,
} from "lucide-react";
import { api, type Health, type SystemSettings } from "../api";
import { bytesHuman } from "../format";
import { Card, ErrorState, Loading, PageHeader } from "../components/ui";

export default function Settings() {
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Form values
  const [defaultAlgorithm, setDefaultAlgorithm] = useState<"fedavg" | "fedprox">("fedavg");
  const [defaultRounds, setDefaultRounds] = useState<number>(10);
  const [defaultClients, setDefaultClients] = useState<number>(5);
  const [defaultBatchSize, setDefaultBatchSize] = useState<number>(64);
  const [defaultLr, setDefaultLr] = useState<number>(0.01);
  const [device, setDevice] = useState<"cpu" | "cuda" | "mps">("cpu");

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [settingsRes, healthRes] = await Promise.allSettled([api.settings.get(), api.health()]);
      if (settingsRes.status === "fulfilled") {
        const s = settingsRes.value;
        setSettings(s);
        setDefaultAlgorithm(s.default_algorithm ?? "fedavg");
        setDefaultRounds(s.default_rounds ?? 8);
        setDefaultClients(s.default_clients ?? 5);
        setDefaultBatchSize(s.default_batch_size ?? 64);
        setDefaultLr(s.default_lr ?? 0.01);
        setDevice(
          s.device === "cuda" || String(s.device).startsWith("cuda")
            ? "cuda"
            : s.device === "mps"
            ? "mps"
            : "cpu"
        );
      } else {
        setError(settingsRes.reason instanceof Error ? settingsRes.reason.message : String(settingsRes.reason));
      }
      if (healthRes.status === "fulfilled") {
        setHealth(healthRes.value);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setSavedSuccess(false);
    try {
      const updated = await api.settings.update({
        default_algorithm: defaultAlgorithm,
        default_rounds: defaultRounds,
        default_clients: defaultClients,
        default_batch_size: defaultBatchSize,
        default_lr: defaultLr,
        device,
      });
      setSettings(updated);
      setSavedSuccess(true);
      setTimeout(() => setSavedSuccess(false), 3000);
    } catch (e: unknown) {
      alert(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <Loading message="Loading system settings and telemetry…" />;
  if (error && !settings) return <ErrorState message={error} onRetry={loadData} />;

  return (
    <div className="space-y-6">
      {/* Header */}
      <PageHeader
        title="Settings"
        description="Defaults for new experiments, compute device, and platform health."
        actions={
          <button onClick={loadData} className="btn-secondary" title="Refresh">
            <RotateCcw className="h-4 w-4 text-slate-500" />
            Refresh
          </button>
        }
      />

      {savedSuccess && (
        <div className="flex items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
          <Check className="h-4 w-4 text-emerald-600" />
          Settings saved.
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Left: Experiment Defaults Form */}
        <div className="space-y-6">
          <Card title="Experiment defaults">
            <form onSubmit={handleSave} className="space-y-4 text-sm">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Default Algorithm</label>
                  <select
                    className="input"
                    value={defaultAlgorithm}
                    onChange={(e) => setDefaultAlgorithm(e.target.value as "fedavg" | "fedprox")}
                  >
                    <option value="fedavg">FedAvg</option>
                    <option value="fedprox">FedProx</option>
                  </select>
                </div>
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Compute Hardware</label>
                  <select
                    className="input"
                    value={device}
                    onChange={(e) => setDevice(e.target.value as "cpu" | "cuda" | "mps")}
                  >
                    <option value="cpu">CPU</option>
                    <option value="cuda">CUDA</option>
                    <option value="mps">Apple Silicon (MPS)</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Communication Rounds</label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    className="input tabular-nums"
                    value={defaultRounds}
                    onChange={(e) => setDefaultRounds(Number(e.target.value))}
                  />
                </div>
                <div>
                  <label className="mb-1 block font-medium text-slate-700">Sites</label>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    className="input tabular-nums"
                    value={defaultClients}
                    onChange={(e) => setDefaultClients(Number(e.target.value))}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Default Batch Size</label>
                  <select
                    className="input"
                    value={defaultBatchSize}
                    onChange={(e) => setDefaultBatchSize(Number(e.target.value))}
                  >
                    <option value="32">32</option>
                    <option value="64">64</option>
                    <option value="128">128</option>
                    <option value="256">256</option>
                  </select>
                </div>
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Default Learning Rate</label>
                  <input
                    type="number"
                    step="0.001"
                    min="0.0001"
                    max="0.5"
                    className="input tabular-nums"
                    value={defaultLr}
                    onChange={(e) => setDefaultLr(Number(e.target.value))}
                  />
                </div>
              </div>

              <div className="pt-2">
                <button type="submit" disabled={saving} className="btn-teal">
                  <Save className="h-4 w-4" />
                  {saving ? "Saving…" : "Save Preferences"}
                </button>
              </div>
            </form>
          </Card>

          {/* Dataset Configuration */}
          <Card title="Dataset">
            <div className="space-y-4 text-sm">
              <div>
                <p className="font-semibold text-slate-700">Source Dataset</p>
                <p className="mt-0.5 font-mono text-xs text-slate-500">
                  {settings?.dataset_name || "CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015"}
                </p>
              </div>
              <div>
                <p className="font-semibold text-slate-700">File Path on Disk</p>
                <p className="mt-1 break-all rounded-lg bg-slate-50 px-2.5 py-2 font-mono text-xs text-slate-800 select-all">
                  {settings?.dataset_path || "data/raw/diabetes_binary_health_indicators_BRFSS2015.csv"}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-2 pt-1 font-mono text-[11px]">
                <div className="rounded border border-slate-200 p-2">
                  <span className="text-slate-400 block text-[10px]">TOTAL ROWS</span>
                  <strong className="text-slate-900">
                    {(settings?.total_samples ?? 253680).toLocaleString()}
                  </strong>
                </div>
                <div className="rounded border border-slate-200 p-2">
                  <span className="text-slate-400 block text-[10px]">FEATURES</span>
                  <strong className="text-slate-900">21 Indicators</strong>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Right: Application Info & Backend Health */}
        <div className="space-y-6">
          {/* Backend Diagnostics */}
          <Card
            title="Platform health"
            actions={
              <span className="badge-success">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-600" />
                {health?.status === "ok" ? "Connected" : "Unknown"}
              </span>
            }
          >
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                <span className="text-slate-500">API endpoint</span>
                <span className="font-mono text-xs text-slate-800">http://127.0.0.1:8000</span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                <span className="text-slate-500">Status</span>
                <span className="font-semibold uppercase text-emerald-700">{health?.status ?? "OK"}</span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                <span className="text-slate-500">Serving models</span>
                <span className="text-slate-800">{health?.demo_models?.length ?? 0} ready</span>
              </div>

              <div>
                <p className="mb-2 font-medium text-slate-700">Artifacts</p>
                <div className="grid grid-cols-2 gap-1.5 text-xs">
                  {Object.entries(health?.experiments ?? {}).map(([key, val]) => (
                    <div
                      key={key}
                      className="flex items-center justify-between rounded bg-slate-50 px-2 py-1 border border-slate-100"
                    >
                      <span className="text-slate-600 capitalize">{key.replace(/_/g, " ")}</span>
                      <span
                        className={`font-semibold ${
                          val ? "text-emerald-700" : "text-slate-400"
                        }`}
                      >
                        {val ? "Available" : "Not Run"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          {/* SQLite Database Telemetry */}
          <Card title="Application database">
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                <span className="text-slate-500">Database Path</span>
                <span className="font-mono text-[11px] text-slate-800">
                  {settings?.db_health?.path || "results/app/medifederate.db"}
                </span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                <span className="text-slate-500">Database Size</span>
                <span className="font-mono text-slate-800">{bytesHuman(settings?.db_health?.size_bytes ?? 0)}</span>
              </div>
              <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                <span className="text-slate-500">Persisted Experiments</span>
                <span className="font-mono font-semibold text-slate-800">{settings?.db_health?.experiments ?? 0}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Generated Reports</span>
                <span className="font-mono font-semibold text-slate-800">{settings?.db_health?.reports ?? 0}</span>
              </div>
            </div>
          </Card>

          {/* Platform Info */}
          <Card title="About">
            <div className="space-y-2.5 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-500">Product</span>
                <strong className="text-slate-900">MediFederate</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Version</span>
                <span className="font-medium tabular-nums text-slate-800">2.4.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Deployment</span>
                <span>On-premises</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
