import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import Federated from "../pages/Federated";
import Demo, { validateField } from "../pages/Demo";
import { ApiError } from "../api";

function mockFetch(handler: (url: string, init?: RequestInit) => { status?: number; body: unknown }) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const { status = 200, body } = handler(String(input), init);
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  });
}

const SCHEMA = {
  features: [
    { name: "HighBP", kind: "binary", min: 0, max: 1 },
    { name: "BMI", kind: "continuous", min: 10, max: 100 },
    { name: "GenHlth", kind: "ordinal", min: 1, max: 5 },
  ],
  models: ["centralized_mlp"],
  disclaimer: "Educational output, not a medical diagnosis.",
};

describe("validateField", () => {
  it("enforces ranges, integers and required values", () => {
    const bmi = SCHEMA.features[1] as never;
    const gen = SCHEMA.features[2] as never;
    expect(validateField(bmi, "")).toBe("Required");
    expect(validateField(bmi, "abc")).toBe("Must be a number");
    expect(validateField(bmi, "250")).toMatch(/between 10 and 100/);
    expect(validateField(bmi, "27.5")).toBeNull();
    expect(validateField(gen, "2.5")).toBe("Must be a whole number");
    expect(validateField(gen, "3")).toBeNull();
  });
});

describe("Federated page", () => {
  it("shows the empty state when the experiment has not run", async () => {
    mockFetch(() => ({ body: { available: false, message: "The federated experiment has not been run yet. Run `python -m ml.run federated`." } }));
    render(<Federated />, { wrapper: MemoryRouter });
    expect(await screen.findByText(/has not been run yet/)).toBeInTheDocument();
    expect(screen.getByText("No results yet")).toBeInTheDocument();
  });

  it("renders values from the backend rather than hardcoded numbers", async () => {
    mockFetch(() => ({
      body: {
        available: true,
        message: "",
        tag: "final",
        final_comparison_available: true,
        disclaimers: { clients: "Simulated clients, not hospitals." },
        final_comparison: [
          { setting: "iid", method: "fedavg", family: "fedavg", n_seeds: 3, roc_auc_mean: 0.8176, roc_auc_std: 0.0009, accuracy_mean: 0.7941, accuracy_std: 0.003, precision_class1_mean: 0.36, precision_class1_std: 0.004, recall_class1_mean: 0.63, recall_class1_std: 0.007, f1_class1_mean: 0.459, f1_class1_std: 0.002, pr_auc_mean: 0.388, pr_auc_std: 0.001, train_seconds_mean: 2.97, rounds: 8, total_bytes: 1126720 },
        ],
        bootstrap_ci: [],
        local_only_by_client: [],
        results: [],
        figures: [],
      },
    }));
    render(<Federated />, { wrapper: MemoryRouter });
    expect(await screen.findByText("0.8176 ± 0.0009")).toBeInTheDocument();
    expect(screen.getByText("1.13 MB")).toBeInTheDocument();
    expect(screen.getByText(/Simulated clients, not hospitals/)).toBeInTheDocument();
  });

  it("shows an error state when the backend is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    render(<Federated />, { wrapper: MemoryRouter });
    expect(await screen.findByRole("alert")).toHaveTextContent(/Cannot reach the backend/);
  });
});

describe("Demo form", () => {
  it("blocks submission on invalid input and shows field errors", async () => {
    const calls: string[] = [];
    mockFetch((url) => {
      calls.push(url);
      return { body: SCHEMA };
    });
    render(<Demo />);
    const bmi = await screen.findByLabelText(/Body mass index/);
    await userEvent.type(bmi, "250");
    await userEvent.click(screen.getByRole("button", { name: /score/i }));
    expect(await screen.findByText(/between 10 and 100/)).toBeInTheDocument();
    expect(screen.getByText("Required")).toBeInTheDocument(); // GenHlth left empty
    expect(calls.filter((u) => u.endsWith("/api/predict"))).toHaveLength(0);
  });

  it("submits valid input and displays the prediction with its disclaimer", async () => {
    mockFetch((url, init) => {
      if (url.endsWith("/api/predict")) {
        const body = JSON.parse(String(init?.body));
        expect(body.features).toEqual({ HighBP: 1, BMI: 31, GenHlth: 3 });
        return {
          body: { model: "centralized_mlp", probability: 0.42, threshold: 0.66, threshold_source: "val_max_f1", predicted_class: 0, predicted_label: "model output: lower-risk pattern", disclaimer: "Not a medical diagnosis." },
        };
      }
      return { body: SCHEMA };
    });
    render(<Demo />);
    await userEvent.selectOptions(await screen.findByLabelText(/High blood pressure/), "1");
    await userEvent.type(screen.getByLabelText(/Body mass index/), "31");
    await userEvent.type(screen.getByLabelText(/General health/), "3");
    await userEvent.click(screen.getByRole("button", { name: /score/i }));
    expect(await screen.findByText("42.0%")).toBeInTheDocument();
    expect(screen.getByText("Not a medical diagnosis.")).toBeInTheDocument();
  });

  it("surfaces backend validation errors", async () => {
    mockFetch((url) => (url.endsWith("/api/predict") ? { status: 422, body: { detail: "BMI outside allowed range" } } : { body: SCHEMA }));
    render(<Demo />);
    await userEvent.type(await screen.findByLabelText(/Body mass index/), "31");
    await userEvent.type(screen.getByLabelText(/General health/), "3");
    await userEvent.click(screen.getByRole("button", { name: /score/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/HTTP 422: BMI outside allowed range/);
  });

  it("ApiError carries status", () => {
    expect(new ApiError(503, "x").status).toBe(503);
  });
});
