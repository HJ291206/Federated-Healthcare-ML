import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import {
  LayoutDashboard,
  UserCheck,
  Network,
  Building2,
  FlaskConical,
  Brain,
  Shield,
  FileText,
  Settings as SettingsIcon,
  Activity,
} from "lucide-react";
import type { ReactNode } from "react";

import Overview from "./pages/Overview";
import Demo from "./pages/Demo";
import FederatedLearning from "./pages/FederatedLearning";
import HospitalNetwork from "./pages/HospitalNetwork";
import ModelLaboratory from "./pages/ModelLaboratory";
import Explainability from "./pages/Explainability";
import Privacy from "./pages/Privacy";
import Reports from "./pages/Reports";
import SettingsPage from "./pages/Settings";

/* Legacy pages kept as sub-routes for backward compatibility */
import Centralized from "./pages/Centralized";
import Federated from "./pages/Federated";
import Partitions from "./pages/Partitions";
import Convergence from "./pages/Convergence";

interface PageDef {
  path: string;
  label: string;
  icon: ReactNode;
  element: ReactNode;
}

const PAGES: PageDef[] = [
  { path: "/", label: "Overview", icon: <LayoutDashboard className="h-4 w-4" />, element: <Overview /> },
  { path: "/patient-risk", label: "Patient Risk Analysis", icon: <UserCheck className="h-4 w-4" />, element: <Demo /> },
  { path: "/federated-learning", label: "Federated Learning", icon: <Network className="h-4 w-4" />, element: <FederatedLearning /> },
  { path: "/hospital-network", label: "Hospital Network", icon: <Building2 className="h-4 w-4" />, element: <HospitalNetwork /> },
  { path: "/model-lab", label: "Model Laboratory", icon: <FlaskConical className="h-4 w-4" />, element: <ModelLaboratory /> },
  { path: "/explainability", label: "Explainable AI", icon: <Brain className="h-4 w-4" />, element: <Explainability /> },
  { path: "/privacy", label: "Privacy & Security", icon: <Shield className="h-4 w-4" />, element: <Privacy /> },
  { path: "/reports", label: "Reports", icon: <FileText className="h-4 w-4" />, element: <Reports /> },
  { path: "/settings", label: "Settings", icon: <SettingsIcon className="h-4 w-4" />, element: <SettingsPage /> },
];

export default function App() {
  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      {/* ── Sidebar ── */}
      <aside className="flex w-full shrink-0 flex-col border-b border-slate-700 bg-[#1e293b] text-white md:w-64 md:border-b-0 md:border-r md:border-slate-700">
        {/* Brand header */}
        <div className="border-b border-slate-600/50 px-4 py-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-600">
              <Activity className="h-4 w-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight text-white">MediFederate AI</h1>
              <p className="text-[10px] font-medium tracking-wide text-teal-400">
                Privacy-Preserving Healthcare Intelligence
              </p>
            </div>
          </div>
          <div className="mt-2.5 inline-flex items-center gap-1.5 rounded-full border border-slate-600 px-2 py-0.5 text-[10px] text-slate-400">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Local Instance
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex flex-1 flex-wrap gap-0.5 overflow-y-auto px-2 py-2 md:block md:space-y-0.5" aria-label="Main navigation">
          {PAGES.map((p) => (
            <NavLink
              key={p.path}
              to={p.path}
              end={p.path === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2.5 rounded px-2.5 py-2 text-[13px] transition-colors ${
                  isActive
                    ? "border-l-2 border-teal-400 bg-slate-700/60 font-medium text-white"
                    : "border-l-2 border-transparent text-slate-400 hover:bg-slate-700/40 hover:text-slate-200"
                }`
              }
            >
              {p.icon}
              {p.label}
            </NavLink>
          ))}
        </nav>

        {/* Footer disclaimer */}
        <div className="border-t border-slate-600/50 px-4 py-3">
          <p className="text-[10px] leading-relaxed text-slate-500">
            Academic research prototype. Simulated institutions from CDC BRFSS 2015 survey data.
            Model outputs are educational, not clinical diagnoses.
          </p>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="min-w-0 flex-1 bg-[#f8fafc]">
        <div className="mx-auto max-w-7xl space-y-4 p-4 md:p-6">
          <Routes>
            {PAGES.map((p) => (
              <Route key={p.path} path={p.path} element={p.element} />
            ))}
            {/* Legacy routes redirect to new pages */}
            <Route path="/demo" element={<Navigate to="/patient-risk" replace />} />
            <Route path="/centralized" element={<Centralized />} />
            <Route path="/federated" element={<Federated />} />
            <Route path="/partitions" element={<Partitions />} />
            <Route path="/convergence" element={<Convergence />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
