import type { ReactNode } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, FileQuestion, Info, Loader2 } from "lucide-react";
import type { Row, Section } from "../api";
import { api } from "../api";
import { fmt } from "../format";
import type { Loaded } from "../hooks";

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <h1 className="page-title">{title}</h1>
        {description && <p className="page-subtitle">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

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
        <div className="mb-4 flex items-start justify-between gap-3 border-b border-slate-100 pb-3">
          <div className="min-w-0">
            {typeof title === "string" ? <h2 className="card-title mb-0">{title}</h2> : title}
            {subtitle && <p className="mt-1 text-sm text-slate-500">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Notice({
  children,
  variant = "info",
}: {
  children: ReactNode;
  variant?: "warning" | "info" | "success" | "error";
}) {
  const styles = {
    warning: "border-amber-200 bg-amber-50/90 text-amber-900",
    info: "border-slate-200 bg-slate-50/90 text-slate-700 shadow-xs",
    success: "border-emerald-200 bg-emerald-50/90 text-emerald-900",
    error: "border-rose-200 bg-rose-50/90 text-rose-900",
  };

  const icons = {
    warning: <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600 mt-0.5" />,
    info: <Info className="h-4 w-4 shrink-0 text-teal-700 mt-0.5" />,
    success: <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600 mt-0.5" />,
    error: <AlertCircle className="h-4 w-4 shrink-0 text-rose-600 mt-0.5" />,
  };

  return (
    <div className={`notice flex items-start gap-3 ${styles[variant]}`} role="status">
      {icons[variant]}
      <div className="flex-1 text-sm leading-relaxed">{children}</div>
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
    <div className="card border-dashed py-12 text-center" role="status">
      <div className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-full bg-slate-100 text-slate-400">
        <FileQuestion className="h-5 w-5" />
      </div>
      <p className="text-sm font-semibold text-slate-800">{title}</p>
      <p className="mx-auto mt-1.5 max-w-md text-sm text-slate-500">{message}</p>
      {action && <div className="mt-5 flex justify-center">{action}</div>}
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
    <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900" role="alert">
      <div className="flex items-start gap-3">
        <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" />
        <div className="flex-1">
          <p className="text-sm font-semibold text-rose-950">{title}</p>
          <p className="mt-1 text-sm leading-relaxed text-rose-800">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-rose-900 underline underline-offset-2 hover:text-rose-950"
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
    <div className="flex items-center justify-center p-12 text-slate-500" role="status">
      <div className="flex flex-col items-center gap-3">
        <Loader2 className="h-6 w-6 animate-spin text-teal-700" />
        <p className="text-sm font-medium text-slate-500">{message}</p>
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
      <div className="py-8 text-center text-sm text-slate-400">
        No records found matching the active filter.
      </div>
    );
  }

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="table">
        {caption && <caption className="mb-2 text-left text-sm text-slate-500">{caption}</caption>}
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
    <figure className="card overflow-hidden p-3">
      <img src={api.figureUrl(rel)} alt={alt} className="w-full rounded-xl border border-slate-100" loading="lazy" />
      <figcaption className="mt-3 px-1 text-sm text-slate-500">{rel}</figcaption>
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
        <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">{label}</p>
        {icon && <div className="text-slate-400">{icon}</div>}
      </div>
      <div className="mt-3">
        <p className="text-2xl font-semibold tabular-nums tracking-tight text-slate-900">{value}</p>
        {hint && <p className="mt-1 text-sm leading-snug text-slate-500">{hint}</p>}
      </div>
    </div>
  );
}

export function StatusBadge({
  status,
}: {
  status: "completed" | "failed" | "running" | "cancelled" | "queued" | "ready" | "idle" | "training" | "syncing";
}) {
  const label = status.charAt(0).toUpperCase() + status.slice(1);
  switch (status) {
    case "completed":
    case "ready":
      return (
        <span className="badge-success">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-600" />
          {label}
        </span>
      );
    case "running":
    case "training":
    case "syncing":
      return (
        <span className="badge-running">
          <span className="h-1.5 w-1.5 rounded-full bg-teal-600" />
          {label}
        </span>
      );
    case "queued":
    case "idle":
      return (
        <span className="badge-neutral">
          <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
          {label}
        </span>
      );
    case "failed":
      return (
        <span className="badge-error">
          <span className="h-1.5 w-1.5 rounded-full bg-rose-600" />
          {label}
        </span>
      );
    case "cancelled":
      return (
        <span className="badge-warning">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-600" />
          {label}
        </span>
      );
    default:
      return <span className="badge-neutral">{label}</span>;
  }
}
