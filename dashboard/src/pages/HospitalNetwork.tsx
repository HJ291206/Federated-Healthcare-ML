import { useState } from "react";
import {
  Building2,
  Database,
  Lock,
  Server,
} from "lucide-react";
import { fmt, pct, rows } from "../format";
import { useSection } from "../hooks";
import { Card, DataTable, Notice, SectionGate, StatusBadge } from "../components/ui";

const INSTITUTION_PROFILES = [
  {
    name: "St. Jude Research Medical Center",
    type: "Academic Research Hospital",
    region: "Metropolitan North",
    tier: "Tertiary Referral",
  },
  {
    name: "Metro General Hospital & Health",
    type: "Urban Safety-Net Hospital",
    region: "Central Urban Corridor",
    tier: "Level 1 Trauma",
  },
  {
    name: "University Health System Center",
    type: "University Academic Center",
    region: "Eastern Academic District",
    tier: "Quaternary Research",
  },
  {
    name: "Riverside Community Health Clinic",
    type: "Suburban Ambulatory Network",
    region: "Riverside District",
    tier: "Primary Care / Outpatient",
  },
  {
    name: "Valley Memorial Medical Center",
    type: "Regional Community Hospital",
    region: "Southern Valley",
    tier: "Secondary Community",
  },
];

export default function HospitalNetwork() {
  const state = useSection("partitions");
  const [selectedPartition, setSelectedPartition] = useState<string>("iid");

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-1 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Building2 className="h-6 w-6 text-teal-600" />
            Hospital Network Simulation
          </h1>
          <p className="text-xs text-slate-500">
            Multi-institutional client network simulation with physical data locality perimeters and central coordinate aggregation.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="badge-warning font-semibold uppercase tracking-wider text-[10px]">
            Simulated Network
          </span>
        </div>
      </header>

      <Notice variant="warning">
        <span className="font-semibold text-slate-900">SIMULATED ENVIRONMENT DISCLAIMER:</span> The 5 nodes below represent simulated client partitions created from the public CDC BRFSS 2015 survey dataset. They are not real hospitals or healthcare facilities. All training data resides exclusively inside each simulated node; raw data is never transmitted across the network perimeter.
      </Notice>

      <SectionGate state={state}>
        {(s) => {
          const allParts = rows(s, "partitions").filter((r) => r.split === "train");
          const partitionTypes = [...new Set(allParts.map((p) => String(p.partition)))];
          const activeType = partitionTypes.includes(selectedPartition) ? selectedPartition : partitionTypes[0] || "iid";

          const currentPartitionClients = allParts.filter((p) => String(p.partition) === activeType);
          const totalRecords = currentPartitionClients.reduce((acc, c) => acc + Number(c.n ?? 0), 0);
          const totalPositives = currentPartitionClients.reduce((acc, c) => acc + Number(c.n_positive ?? 0), 0);
          const overallPosRate = totalRecords > 0 ? totalPositives / totalRecords : 0.139;

          return (
            <div className="space-y-6">
              {/* Partition Switcher & Network Stats Header */}
              <div className="flex flex-wrap items-center justify-between gap-4 rounded border border-slate-200 bg-white p-3 shadow-xs">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                    Distribution Mode:
                  </span>
                  <div className="flex rounded border border-slate-200 p-0.5 bg-slate-50 text-xs">
                    {partitionTypes.map((pt) => (
                      <button
                        key={pt}
                        onClick={() => setSelectedPartition(pt)}
                        className={`rounded px-3 py-1 font-medium transition-colors ${
                          activeType === pt
                            ? "bg-white text-teal-700 shadow-xs font-semibold"
                            : "text-slate-600 hover:text-slate-900"
                        }`}
                      >
                        {pt === "iid" ? "Stratified IID" : "Non-IID (Dirichlet α=0.5)"}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex items-center gap-4 text-xs font-mono text-slate-600">
                  <span>Total Records: <strong className="text-slate-900">{totalRecords.toLocaleString()}</strong></span>
                  <span>Positive Prevalence: <strong className="text-teal-700">{pct(overallPosRate, 2)}</strong></span>
                  <span>Active Nodes: <strong className="text-slate-900">5 / 5</strong></span>
                </div>
              </div>

              {/* NETWORK ARCHITECTURE DIAGRAM */}
              <div className="rounded border border-slate-200/90 bg-slate-50/60 p-6 shadow-xs relative overflow-hidden">
                {/* Central Server Card */}
                <div className="mx-auto max-w-lg mb-8">
                  <div className="rounded border-2 border-slate-800 bg-slate-900 p-4 text-white shadow-md relative">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2.5">
                        <div className="flex h-9 w-9 items-center justify-center rounded bg-teal-500/20 text-teal-400">
                          <Server className="h-5 w-5" />
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <h2 className="text-sm font-bold tracking-wide">Central Parameter Server</h2>
                            <span className="rounded bg-teal-500/30 px-1.5 py-0.5 text-[10px] font-mono text-teal-300">
                              SIMULATED AGGREGATOR
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400">
                            MediFederate Secure Enclave Aggregation Hub
                          </p>
                        </div>
                      </div>
                      <StatusBadge status="ready" />
                    </div>

                    <div className="mt-3 grid grid-cols-3 gap-2 border-t border-slate-800 pt-2.5 text-[11px] font-mono text-slate-300">
                      <div>
                        <span className="text-slate-500 block text-[10px]">COORDINATION</span>
                        <span>FedAvg / FedProx</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[10px]">WEIGHT MATRIX</span>
                        <span>3,521 Float32</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[10px]">CONNECTED NODES</span>
                        <span>5 Institutions</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Vertical Bus & Distribution Lines */}
                <div className="hidden md:flex flex-col items-center justify-center mb-6">
                  {/* Central vertical stem */}
                  <div className="h-6 w-0.5 bg-teal-600/70" />
                  {/* Horizontal distribution bar */}
                  <div className="w-[84%] h-0.5 bg-teal-600/70 relative">
                    {/* Connection drop pins */}
                    <div className="absolute -top-1 left-0 h-2.5 w-2.5 rounded-full bg-teal-600" />
                    <div className="absolute -top-1 left-[25%] h-2.5 w-2.5 rounded-full bg-teal-600" />
                    <div className="absolute -top-1 left-[50%] -ml-1.5 h-3 w-3 rounded-full bg-teal-500 animate-ping" />
                    <div className="absolute -top-1 left-[50%] -ml-1 h-2.5 w-2.5 rounded-full bg-teal-600" />
                    <div className="absolute -top-1 left-[75%] h-2.5 w-2.5 rounded-full bg-teal-600" />
                    <div className="absolute -top-1 right-0 h-2.5 w-2.5 rounded-full bg-teal-600" />
                  </div>
                  {/* Stems to nodes */}
                  <div className="w-[84%] grid grid-cols-5">
                    <div className="h-6 w-0.5 bg-teal-600/70 mx-auto" />
                    <div className="h-6 w-0.5 bg-teal-600/70 mx-auto" />
                    <div className="h-6 w-0.5 bg-teal-600/70 mx-auto" />
                    <div className="h-6 w-0.5 bg-teal-600/70 mx-auto" />
                    <div className="h-6 w-0.5 bg-teal-600/70 mx-auto" />
                  </div>
                </div>

                {/* 5 HOSPITAL CLIENT CARDS GRID */}
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
                  {Array.from({ length: 5 }, (_, clientIdx) => {
                    const clientRow = currentPartitionClients.find((r) => Number(r.client) === clientIdx);
                    const profile = INSTITUTION_PROFILES[clientIdx];
                    const n = Number(clientRow?.n ?? 0);
                    const nPos = Number(clientRow?.n_positive ?? 0);
                    const nNeg = n - nPos;
                    const posRate = n > 0 ? nPos / n : 0;
                    const shareOfTotal = totalRecords > 0 ? n / totalRecords : 0.2;

                    return (
                      <div
                        key={clientIdx}
                        className="rounded border border-slate-200 bg-white p-3.5 shadow-xs flex flex-col justify-between hover:border-teal-500 transition-colors"
                      >
                        <div>
                          {/* Client Header */}
                          <div className="flex items-start justify-between mb-2">
                            <span className="badge-teal font-mono font-bold text-[10px]">
                              Client #{clientIdx}
                            </span>
                            <span className="rounded bg-slate-100 px-1 py-0.5 text-[9px] font-bold uppercase tracking-wider text-slate-500">
                              SIMULATED
                            </span>
                          </div>

                          <h3 className="text-xs font-bold text-slate-900 leading-snug line-clamp-2">
                            {profile.name}
                          </h3>
                          <p className="text-[10px] text-slate-400 mt-0.5">{profile.type}</p>
                          <p className="text-[10px] text-slate-400">{profile.region}</p>

                          {/* Records Count Metric */}
                          <div className="mt-3 rounded bg-slate-50 p-2 border border-slate-100">
                            <div className="flex justify-between text-[11px]">
                              <span className="text-slate-500">Local Cohort:</span>
                              <strong className="text-slate-900 tabular-nums">{n.toLocaleString()}</strong>
                            </div>
                            <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                              <span>Global Share:</span>
                              <span className="tabular-nums">{pct(shareOfTotal, 1)}</span>
                            </div>
                          </div>

                          {/* Class Distribution Bar */}
                          <div className="mt-3">
                            <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                              <span>Class Balance</span>
                              <span className="font-semibold text-rose-700">{pct(posRate, 1)} positive</span>
                            </div>
                            <div className="h-2 w-full overflow-hidden rounded-full bg-sky-200 flex" title={`Negative: ${nNeg.toLocaleString()} | Positive: ${nPos.toLocaleString()}`}>
                              <div
                                className="h-full bg-sky-600"
                                style={{ width: `${(1 - posRate) * 100}%` }}
                              />
                              <div
                                className="h-full bg-rose-500"
                                style={{ width: `${posRate * 100}%` }}
                              />
                            </div>
                            <div className="mt-1 flex justify-between text-[9px] text-slate-400 font-mono">
                              <span>Non-Diabetes ({fmt(nNeg, 0)})</span>
                              <span>Diabetes ({fmt(nPos, 0)})</span>
                            </div>
                          </div>
                        </div>

                        {/* Privacy & Status Footer */}
                        <div className="mt-4 border-t border-slate-100 pt-2 text-[10px]">
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="text-slate-500">Node Status:</span>
                            <StatusBadge status="ready" />
                          </div>
                          <div className="flex items-center gap-1 text-slate-500">
                            <Lock className="h-3 w-3 text-teal-600 shrink-0" />
                            <span className="truncate">Data stays on premises</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* DETAILED PARTITION DATA TABLE */}
              <Card
                title={
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700">
                    <Database className="h-4 w-4 text-slate-500" />
                    Partition Breakdown Table ({activeType})
                  </div>
                }
              >
                <DataTable
                  rows={currentPartitionClients}
                  columns={[
                    { key: "client", header: "Client ID", render: (v) => `Client #${v}` },
                    {
                      key: "institution",
                      header: "Simulated Institution Name",
                      render: (_, r) => INSTITUTION_PROFILES[Number(r.client)]?.name ?? `Client ${r.client}`,
                    },
                    { key: "n", header: "Training Cohort Rows", digits: 0 },
                    { key: "n_positive", header: "Diabetes Positive Cases", digits: 0 },
                    {
                      key: "positive_rate",
                      header: "Prevalence Rate",
                      render: (v) => pct(v, 2),
                    },
                    {
                      key: "ratio",
                      header: "Share of Total Network",
                      render: (_, r) => pct(Number(r.n) / totalRecords, 1),
                    },
                  ]}
                  caption="Source: CDC BRFSS 2015 stratified training split (70% partition). Positives correspond to Diabetes_binary = 1."
                />
              </Card>
            </div>
          );
        }}
      </SectionGate>
    </div>
  );
}
