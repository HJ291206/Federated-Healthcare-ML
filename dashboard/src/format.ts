import type { Row } from "./api";

export const fmt = (v: unknown, digits = 4): string => {
  if (v === null || v === undefined || v === "") return "–";
  if (typeof v === "number") return Number.isInteger(v) && Math.abs(v) >= 1000 ? v.toLocaleString() : v.toFixed(digits);
  return String(v);
};

export const pct = (v: unknown, digits = 1): string => (typeof v === "number" ? `${(100 * v).toFixed(digits)}%` : "–");

export const bytesHuman = (b: unknown): string => {
  if (typeof b !== "number") return "–";
  if (b >= 1e6) return `${(b / 1e6).toFixed(2)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)} kB`;
  return `${b} B`;
};

export const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function rows<T = Row>(section: Record<string, unknown> | null, key: string): T[] {
  const v = section?.[key];
  return Array.isArray(v) ? (v as T[]) : [];
}

/** Human-readable labels for method names used in the CSVs. */
export const METHOD_LABELS: Record<string, string> = {
  centralized_logistic_regression: "Centralized logistic regression",
  centralized_random_forest: "Centralized random forest",
  centralized_mlp: "Centralized MLP",
  logistic_regression: "Logistic regression",
  random_forest: "Random forest",
  mlp: "MLP (PyTorch)",
  local_only: "Local-only (per client)",
  fedavg: "FedAvg",
  "fedprox_mu0.01": "FedProx (μ = 0.01)",
  "fedprox_mu0.1": "FedProx (μ = 0.1)",
  iid: "IID",
  "dirichlet_a0.5": "Non-IID (Dirichlet α = 0.5)",
  "dirichlet_a0.1": "Non-IID (Dirichlet α = 0.1)",
  "centralized (pooled data)": "Centralized (pooled data)",
};

export const label = (s: unknown): string => {
  const k = String(s ?? "");
  if (METHOD_LABELS[k]) return METHOD_LABELS[k];
  const dp = k.match(/^(.*)_dp_sigma([\d.]+)$/);
  if (dp) return `${label(dp[1])} + DP (σ = ${dp[2]})`;
  return k.replace(/_/g, " ");
};

export const COLORS = ["#0369a1", "#ea580c", "#15803d", "#7c3aed", "#be123c", "#64748b", "#ca8a04", "#0f766e"];
