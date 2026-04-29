"use client";

import { Cpu, Database, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { API_BASE_URL, fetchModelInfo, type ModelInfo as ModelInfoType } from "@/lib/api";

export default function ModelInfo() {
  const [info, setInfo] = useState<ModelInfoType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    fetchModelInfo()
      .then((modelInfo) => {
        if (isMounted) {
          setInfo(modelInfo);
          setError(null);
        }
      })
      .catch((err: Error) => {
        if (isMounted) {
          setError(err.message);
        }
      })
      .finally(() => {
        if (isMounted) {
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-soft">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.12em] text-clinical">
            Backend
          </p>
          <h2 className="mt-1 text-xl font-bold text-ink">Model Info</h2>
        </div>
        {isLoading ? (
          <RefreshCw className="h-5 w-5 animate-spin text-slate-400" aria-hidden="true" />
        ) : (
          <Cpu className="h-5 w-5 text-clinical" aria-hidden="true" />
        )}
      </div>

      <div className="space-y-3 text-sm">
        <div className="rounded-lg bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
            API
          </p>
          <p className="mt-1 break-all text-slate-700">{API_BASE_URL}</p>
        </div>

        {error ? (
          <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-red-700">
            {error}
          </p>
        ) : null}

        {info ? (
          <>
            <div className="flex items-center justify-between gap-3 rounded-lg bg-slate-50 p-3">
              <span className="text-slate-600">Model</span>
              <span className="font-semibold text-ink">{info.model_name}</span>
            </div>
            <div className="flex items-center justify-between gap-3 rounded-lg bg-slate-50 p-3">
              <span className="text-slate-600">Device</span>
              <span className="font-semibold text-ink">{info.device}</span>
            </div>
            <div className="flex items-center justify-between gap-3 rounded-lg bg-slate-50 p-3">
              <span className="flex items-center gap-2 text-slate-600">
                <Database className="h-4 w-4" aria-hidden="true" />
                Checkpoint
              </span>
              <span
                className={
                  info.checkpoint_exists
                    ? "font-semibold text-clinical"
                    : "font-semibold text-red-600"
                }
              >
                {info.checkpoint_exists ? "Ready" : "Missing"}
              </span>
            </div>
          </>
        ) : null}
      </div>
    </section>
  );
}

