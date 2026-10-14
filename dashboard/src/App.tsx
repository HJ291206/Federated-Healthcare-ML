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
  { path: "/patient-risk", label: "Patient Risk", icon: <UserCheck className="h-4 w-4" />, element: <Demo /> },
  { path: "/federated-learning", label: "Federated Learning", icon: <Network className="h-4 w-4" />, element: <FederatedLearning /> },
  { path: "/hospital-network", label: "Hospital Network", icon: <Building2 className="h-4 w-4" />, element: <HospitalNetwork /> },
  { path: "/model-lab", label: "Model Laboratory", icon: <FlaskConical className="h-4 w-4" />, element: <ModelLaboratory /> },
  { path: "/explainability", label: "Explainability", icon: <Brain className="h-4 w-4" />, element: <Explainability /> },
  { path: "/privacy", label: "Privacy & Security", icon: <Shield className="h-4 w-4" />, element: <Privacy /> },
  { path: "/reports", label: "Reports", icon: <FileText className="h-4 w-4" />, element: <Reports /> },
  { path: "/settings", label: "Settings", icon: <SettingsIcon className="h-4 w-4" />, element: <SettingsPage /> },
];

export default function App() {
  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <aside className="flex w-full shrink-0 flex-col bg-[#0c1628] text-white md:w-[248px] md:border-r md:border-white/5">
        <div className="px-5 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-teal-400 to-teal-700 shadow-[0_8px_20px_rgba(13,148,136,0.35)]">
              <Activity className="h-4 w-4 text-white" />
            </div>
            <div>
              <h1 className="text-[15px] font-semibold tracking-tight text-white">MediFederate AI</h1>
              <p className="text-[11px] font-medium tracking-wide text-teal-300/80">Clinical Intelligence</p>
            </div>
          </div>
          <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-medium text-emerald-300">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
            Clinical Network Live
          </div>
        </div>

        <nav className="flex flex-1 flex-wrap gap-1 overflow-y-auto px-3 pb-4 md:block md:space-y-0.5" aria-label="Main navigation">
          {PAGES.map((p) => (
            <NavLink
              key={p.path}
              to={p.path}
              end={p.path === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13.5px] transition-colors ${
                  isActive
                    ? "bg-white/10 font-medium text-white shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]"
                    : "text-slate-400 hover:bg-white/5 hover:text-slate-100"
                }`
              }
            >
              {p.icon}
              {p.label}
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-white/8 px-5 py-4">
          <p className="text-[11px] leading-relaxed text-slate-400">
            Enterprise Deployment
            <span className="mx-1.5 text-slate-600">·</span>
            Zero Data Egress
          </p>
          <p className="mt-1 text-[10px] text-slate-500 font-mono">v2.4.0 • HIPAA Compliant</p>
        </div>
      </aside>

      <main className="min-w-0 flex-1 bg-[#f4f6f9]">
        <div className="mx-auto max-w-[1200px] space-y-6 px-5 py-7 md:px-8 md:py-8">
          <Routes>
            {PAGES.map((p) => (
              <Route key={p.path} path={p.path} element={p.element} />
            ))}
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
