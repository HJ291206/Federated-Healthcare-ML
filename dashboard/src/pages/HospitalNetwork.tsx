import { useState } from "react";
import {
  Lock,
  Server,
} from "lucide-react";
import { fmt, pct, rows } from "../format";
import { useSection } from "../hooks";
import { Card, DataTable, PageHeader, SectionGate, StatusBadge } from "../components/ui";

const INSTITUTION_PROFILES = [
  {
    name: "St. Jude Research Medical Center",
    type: "Tertiary medical center",
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
    type: "University health system",
    region: "Eastern district",
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
      <PageHeader
        title="Hospital network"
        description="Five participating sites train locally. The aggregator receives model weights only — never patient records."
        actions={<span className="badge-teal">5 sites connected</span>}
      />

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
              <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-slate-200/90 bg-white px-4 py-3 shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
                <div className="flex items-center gap-3">
                  <span className="text-sm text-slate-500">Partition</span>
                  <div className="segmented">
                    {partitionTypes.map((pt) => (
                      <button
                        key={pt}
                        onClick={() => setSelectedPartition(pt)}
                        className={activeType === pt ? "segmented-item-active" : "segmented-item"}
                      >
                        {pt === "iid" ? "Stratified IID" : "Non-IID (α=0.5)"}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-5 text-sm text-slate-600">
                  <span>
                    Records <strong className="ml-1 tabular-nums text-slate-900">{totalRecords.toLocaleString()}</strong>
                  </span>
                  <span>
                    Prevalence <strong className="ml-1 tabular-nums text-teal-800">{pct(overallPosRate, 2)}</strong>
                  </span>
                  <span>
                    Sites <strong className="ml-1 tabular-nums text-slate-900">5 / 5</strong>
                  </span>
                </div>
              </div>

              {/* NETWORK ARCHITECTURE DIAGRAM */}
              <div className="relative overflow-hidden rounded-2xl border border-slate-200/90 bg-white p-6 shadow-[0_1px_2px_rgba(15,23,42,0.04),0_10px_28px_rgba(15,23,42,0.04)]">
                <div className="mx-auto mb-8 max-w-lg">
                  <div className="relative rounded-2xl bg-[#0c1628] p-5 text-white">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-500/20 text-teal-300">
                          <Server className="h-5 w-5" />
                        </div>
                        <div>
                          <h2 className="text-sm font-semibold tracking-tight">Aggregation server</h2>
                          <p className="mt-0.5 text-xs text-slate-400">Coordinates training without seeing local records</p>
                        </div>
                      </div>
                      <StatusBadge status="ready" />
                    </div>

                    <div className="mt-4 grid grid-cols-3 gap-3 border-t border-white/10 pt-3 text-xs text-slate-300">
                      <div>
                        <span className="mb-0.5 block text-[11px] text-slate-500">Protocols</span>
                        <span>FedAvg / FedProx</span>
                      </div>
                      <div>
                        <span className="mb-0.5 block text-[11px] text-slate-500">Parameters</span>
                        <span>3,521 weights</span>
                      </div>
                      <div>
                        <span className="mb-0.5 block text-[11px] text-slate-500">Connected</span>
                        <span>5 institutions</span>
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
                        className="flex flex-col justify-between rounded-2xl border border-slate-200 bg-white p-4 transition-colors hover:border-teal-400/70"
                      >
                        <div>
                          <div className="mb-3 flex items-start justify-between">
                            <span className="badge-teal">Site {clientIdx + 1}</span>
                            <Lock className="h-3.5 w-3.5 text-teal-700" />
                          </div>

                          <h3 className="line-clamp-2 text-sm font-semibold leading-snug text-slate-900">
                            {profile.name}
                          </h3>
                          <p className="mt-1 text-xs text-slate-500">{profile.type}</p>
                          <p className="text-xs text-slate-400">{profile.region}</p>

                          {/* Records Count Metric */}
                          <div className="mt-3 rounded-xl border border-slate-100 bg-slate-50 p-2.5">
                            <div className="flex justify-between text-sm">
                              <span className="text-slate-500">Local cohort</span>
                              <strong className="tabular-nums text-slate-900">{n.toLocaleString()}</strong>
                            </div>
                            <div className="mt-1 flex justify-between text-xs text-slate-400">
                              <span>Network share</span>
                              <span className="tabular-nums">{pct(shareOfTotal, 1)}</span>
                            </div>
                          </div>

                          <div className="mt-3">
                            <div className="mb-1 flex justify-between text-xs text-slate-500">
                              <span>Class balance</span>
                              <span className="font-medium text-rose-700">{pct(posRate, 1)} positive</span>
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
                            <div className="mt-1.5 flex justify-between text-[11px] text-slate-400">
                              <span>No diabetes ({fmt(nNeg, 0)})</span>
                              <span>Diabetes ({fmt(nPos, 0)})</span>
                            </div>
                          </div>
                        </div>

                        <div className="mt-4 flex items-center justify-between border-t border-slate-100 pt-3 text-xs">
                          <span className="text-slate-500">On-premises data</span>
                          <StatusBadge status="ready" />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* DETAILED PARTITION DATA TABLE */}
              <Card title="Site breakdown">
                <DataTable
                  rows={currentPartitionClients}
                  columns={[
                    { key: "client", header: "Site", render: (v) => `Site ${Number(v) + 1}` },
                    {
                      key: "institution",
                      header: "Institution",
                      render: (_, r) => INSTITUTION_PROFILES[Number(r.client)]?.name ?? `Site ${r.client}`,
                    },
                    { key: "n", header: "Training records", digits: 0 },
                    { key: "n_positive", header: "Positive cases", digits: 0 },
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
                  caption="Training split only. Positives correspond to the diabetes-positive class."
                />
              </Card>
            </div>
          );
        }}
      </SectionGate>
    </div>
  );
}
