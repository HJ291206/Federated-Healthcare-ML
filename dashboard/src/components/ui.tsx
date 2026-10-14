import type { ReactNode } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, FileQuestion, Info, Loader2 } from "lucide-react";
import type { Row, Section } from "../api";
import { api } from "../api";
import { fmt } from "../format";
import type { Loaded } from "../hooks";

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = "",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div className="mb-3 flex items-start justify-between gap-3 border-b border-slate-100 pb-2.5">
          <div>
            {typeof title === "string" ? <h2 className="card-title mb-0">{title}</h2> : title}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Notice({
  children,
  variant = "warning",
}: {
  children: ReactNode;
  variant?: "warning" | "info" | "success" | "error";
}) {
  const styles = {
    warning: "border-amber-200 bg-amber-50/90 text-amber-900",
    info: "border-sky-200 bg-sky-50/90 text-sky-900",
    success: "border-emerald-200 bg-emerald-50/90 text-emerald-900",
    error: "border-rose-200 bg-rose-50/90 text-rose-900",
  };

  const icons = {
    warning: <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600 mt-0.5" />,
    info: <Info className="h-4 w-4 shrink-0 text-sky-600 mt-0.5" />,
    success: <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600 mt-0.5" />,
    error: <AlertCircle className="h-4 w-4 shrink-0 text-rose-600 mt-0.5" />,
  };

  return (
    <div className={`notice flex items-start gap-2.5 ${styles[variant]}`} role="status">
      {icons[variant]}
      <div className="flex-1 text-xs leading-relaxed">{children}</div>
    </div>
  );
}

export function EmptyState({
  title = "No results yet",
  message,
  action,
}: {
  title?: string;
  message: string;
  action?: ReactNode;
}) {
  return (
    <div className="card border-dashed py-8 text-center" role="status">
      <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-slate-100 text-slate-400">
        <FileQuestion className="h-5 w-5" />
      </div>
      <p className="text-sm font-semibold text-slate-800">{title}</p>
      <p className="mx-auto mt-1 max-w-md text-xs text-slate-500">{message}</p>
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>
  );
}

export function ErrorState({
  title = "Could not load data from the backend",
  message,
  onRetry,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="rounded border border-rose-300 bg-rose-50 p-4 text-sm text-rose-900" role="alert">
      <div className="flex items-start gap-3">
        <AlertCircle className="h-5 w-5 text-rose-600 shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="font-semibold text-rose-950 text-xs uppercase tracking-wider">{title}</p>
          <p className="mt-1 text-xs text-rose-800 leading-relaxed">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-2.5 inline-flex items-center gap-1 text-xs font-semibold text-rose-900 underline hover:text-rose-950"
            >
              Retry request
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function Loading({ message = "Loading data from backend…" }: { message?: string }) {
  return (
    <div className="flex items-center justify-center p-8 text-slate-500" role="status">
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="h-6 w-6 animate-spin text-teal-600" />
        <p className="text-xs font-medium text-slate-500">{message}</p>
      </div>
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-slate-200/80 ${className}`} />;
}

/** Wraps the three states every page shares: loading, backend error, experiment not run. */
export function SectionGate({
  state,
  children,
}: {
  state: Loaded<Section>;
  children: (s: Section) => ReactNode;
}) {
  if (state.loading) return <Loading />;
  if (state.error) return <ErrorState message={state.error} />;
  if (!state.data) return <ErrorState message="Empty response" />;
  if (!state.data.available) return <EmptyState message={state.data.message} />;
  return <>{children(state.data)}</>;
}

export interface Column {
  key: string;
  header: string;
  render?: (v: unknown, row: Row) => ReactNode;
  digits?: number;
  align?: "left" | "center" | "right";
}

export function DataTable({
  rows,
  columns,
  caption,
  className = "",
}: {
  rows: Row[];
  columns: Column[];
  caption?: string;
  className?: string;
}) {
  if (!rows.length) {
    return (
      <div className="py-6 text-center text-xs text-slate-400">
        No records found matching the active filter.
      </div>
    );
  }

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="table">
        {caption && <caption className="mb-1 text-left text-xs text-slate-500 font-medium">{caption}</caption>}
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                className={c.align === "right" ? "text-right" : c.align === "center" ? "text-center" : "text-left"}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={c.align === "right" ? "text-right" : c.align === "center" ? "text-center" : "text-left"}
                >
                  {c.render ? c.render(r[c.key], r) : fmt(r[c.key], c.digits)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Figure({ rel, alt }: { rel: string; alt: string }) {
  return (
    <figure className="card">
      <img src={api.figureUrl(rel)} alt={alt} className="w-full rounded border border-slate-100" loading="lazy" />
      <figcaption className="mt-2 text-xs text-slate-500 flex items-center justify-between">
        <span>{rel}</span>
        <span className="text-[10px] text-slate-400 uppercase tracking-wider">Research Artifact</span>
      </figcaption>
    </figure>
  );
}

export function Disclaimers({ items, keys }: { items?: Record<string, string>; keys: string[] }) {
  if (!items) return null;
  return (
    <div className="space-y-2">
      {keys
        .filter((k) => items[k])
        .map((k) => (
          <Notice key={k}>{items[k]}</Notice>
        ))}
    </div>
  );
}

export function Stat({
  label,
  value,
  hint,
  icon,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="card flex flex-col justify-between">
      <div className="flex items-center justify-between">
        <p className="text-xs font-bold uppercase tracking-wider text-slate-500">{label}</p>
        {icon && <div className="text-slate-400">{icon}</div>}
      </div>
      <div className="mt-2">
        <p className="text-2xl font-bold tabular-nums text-slate-900">{value}</p>
        {hint && <p className="mt-1 text-xs text-slate-500 leading-tight">{hint}</p>}
      </div>
    </div>
  );
}

export function StatusBadge({
  status,
}: {
  status: "completed" | "failed" | "running" | "cancelled" | "queued" | "ready" | "idle" | "training" | "syncing";
}) {
  switch (status) {
    case "completed":
    case "ready":
      return (
        <span className="badge-success">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-600" />
          {status}
        </span>
      );
    case "running":
    case "training":
    case "syncing":
      return (
        <span className="badge-running">
          <span className="h-1.5 w-1.5 rounded-full bg-teal-600" />
          {status}
        </span>
      );
    case "queued":
    case "idle":
      return (
        <span className="badge-neutral">
          <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
          {status}
        </span>
      );
    case "failed":
      return (
        <span className="badge-error">
          <span className="h-1.5 w-1.5 rounded-full bg-rose-600" />
          {status}
        </span>
      );
    case "cancelled":
      return (
        <span className="badge-warning">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-600" />
          {status}
        </span>
      );
    default:
      return <span className="badge-neutral">{status}</span>;
  }
}
