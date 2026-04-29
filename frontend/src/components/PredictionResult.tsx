import { Activity, Gauge } from "lucide-react";
import type { PredictionResponse } from "@/lib/api";

type PredictionResultProps = {
  result: PredictionResponse;
};

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function PredictionResult({ result }: PredictionResultProps) {
  const entries = Object.entries(result.probabilities);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-soft">
      <div className="mb-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.12em] text-clinical">
            Prediction
          </p>
          <h2 className="mt-1 text-2xl font-bold text-ink">{result.class_name}</h2>
          <p className="mt-1 text-sm text-slate-600">Class {result.predicted_class}</p>
        </div>
        <div className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
          <Gauge className="h-5 w-5 text-clinical" aria-hidden="true" />
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
              Confidence
            </p>
            <p className="text-xl font-bold text-ink">{formatPercent(result.confidence)}</p>
          </div>
        </div>
      </div>

      <div className="grid gap-3">
        {entries.map(([className, probability]) => (
          <div
            key={className}
            className="rounded-lg border border-slate-200 bg-slate-50 p-3"
          >
            <div className="mb-2 flex items-center justify-between gap-4">
              <span className="flex min-w-0 items-center gap-2 text-sm font-medium text-ink">
                <Activity className="h-4 w-4 shrink-0 text-signal" aria-hidden="true" />
                <span className="truncate">{className}</span>
              </span>
              <span className="text-sm font-semibold text-slate-700">
                {formatPercent(probability)}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-white">
              <div
                className="h-full rounded-full bg-clinical"
                style={{ width: `${Math.max(probability * 100, 1)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

