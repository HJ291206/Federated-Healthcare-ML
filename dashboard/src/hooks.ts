import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, type CreateExperimentInput, type Experiment, type Section } from "./api";

export interface Loaded<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Fetch one backend section; errors are surfaced to the page instead of thrown. */
export function useSection(name: string): Loaded<Section> {
  const [state, setState] = useState<Loaded<Section>>({ data: null, error: null, loading: true });
  useEffect(() => {
    let active = true;
    setState({ data: null, error: null, loading: true });
    api
      .section(name)
      .then((data) => active && setState({ data, error: null, loading: false }))
      .catch((e: unknown) =>
        active && setState({ data: null, error: e instanceof ApiError ? e.message : String(e), loading: false }),
      );
    return () => {
      active = false;
    };
  }, [name]);
  return state;
}

/**
 * Periodically invokes an async function when enabled is true.
 * Cleans up timer on unmount or when dependencies change.
 */
export function usePolling(fn: () => void | Promise<unknown>, intervalMs = 3000, enabled = true): void {
  const savedCallback = useRef(fn);

  useEffect(() => {
    savedCallback.current = fn;
  }, [fn]);

  useEffect(() => {
    if (!enabled || intervalMs <= 0) return;
    const tick = async () => {
      try {
        await savedCallback.current();
      } catch {
        // Polling errors handled within callback
      }
    };

    tick();
    const id = setInterval(tick, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, enabled]);
}

/**
 * Fetches the list of all federated learning experiments,
 * automatically polling if any experiment is active.
 */
export function useExperiments() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await api.experiments.list();
      setExperiments(list);
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const hasRunning = experiments.some((e) => e.status === "running" || e.status === "queued");
  usePolling(refresh, 2500, hasRunning);

  const create = async (input: CreateExperimentInput) => {
    const created = await api.experiments.create(input);
    await refresh();
    return created;
  };

  const cancel = async (id: string) => {
    const updated = await api.experiments.cancel(id);
    await refresh();
    return updated;
  };

  const remove = async (id: string) => {
    const res = await api.experiments.delete(id);
    await refresh();
    return res;
  };

  return {
    experiments,
    loading,
    error,
    refresh,
    create,
    cancel,
    remove,
  };
}

/**
 * Fetches a single experiment by ID with real-time polling while in running state.
 */
export function useExperiment(id: string | null) {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [loading, setLoading] = useState(Boolean(id));
  const [error, setError] = useState<string | null>(null);

  const fetchOne = useCallback(async () => {
    if (!id) {
      setExperiment(null);
      setLoading(false);
      return;
    }
    try {
      const data = await api.experiments.get(id);
      setExperiment(data);
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    setLoading(Boolean(id));
    fetchOne();
  }, [fetchOne, id]);

  const isRunning = experiment?.status === "running" || experiment?.status === "queued";
  usePolling(fetchOne, 2000, isRunning);

  const cancel = async () => {
    if (!id) return;
    const res = await api.experiments.cancel(id);
    setExperiment(res);
  };

  return {
    experiment,
    loading,
    error,
    refresh: fetchOne,
    cancel,
  };
}
