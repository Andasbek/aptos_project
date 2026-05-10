"use client";

import { Cpu, Database, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import {
  API_BASE_URL,
  fetchModelInfo,
  type ModelInfo as ModelInfoType,
} from "@/lib/api";
import { useLanguage } from "@/lib/LanguageContext";

const MODEL_LABELS: Record<string, string> = {
  resnet50: "ResNet50",
  efficientnet_b0: "EfficientNet-B0",
  mobilenet_v2: "MobileNetV2",
  cnn: "Custom CNN",
};

export default function ModelInfo() {
  const { t } = useLanguage();
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
            {t("modelInfo.eyebrow")}
          </p>
          <h2 className="mt-1 text-xl font-bold text-ink">
            {t("modelInfo.title")}
          </h2>
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
            {t("modelInfo.api")}
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
              <span className="text-slate-600">{t("modelInfo.device")}</span>
              <span className="font-semibold text-ink">{info.device}</span>
            </div>
            <div className="flex items-center justify-between gap-3 rounded-lg bg-slate-50 p-3">
              <span className="text-slate-600">{t("modelInfo.llm")}</span>
              <span
                className={
                  info.llm_enabled
                    ? "font-semibold text-clinical"
                    : "font-semibold text-amber-600"
                }
              >
                {info.llm_enabled
                  ? t("modelInfo.llm.ready")
                  : t("modelInfo.llm.noKey")}
              </span>
            </div>

            <div className="rounded-lg bg-slate-50 p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
                {t("modelInfo.availableModels")}
              </p>
              <div className="space-y-2">
                {info.available_models.map((model) => (
                  <div
                    key={model.name}
                    className="flex items-center justify-between gap-3"
                  >
                    <span className="flex items-center gap-2 text-slate-700">
                      <Database className="h-4 w-4" aria-hidden="true" />
                      {MODEL_LABELS[model.name] ?? model.name}
                      {model.is_default ? (
                        <span className="rounded bg-clinical/10 px-1.5 py-0.5 text-xs font-semibold text-clinical">
                          {t("modelInfo.default")}
                        </span>
                      ) : null}
                    </span>
                    <span
                      className={
                        model.checkpoint_exists
                          ? "font-semibold text-clinical"
                          : "font-semibold text-red-600"
                      }
                    >
                      {model.checkpoint_exists
                        ? t("modelInfo.checkpoint.ready")
                        : t("modelInfo.checkpoint.missing")}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : null}
      </div>
    </section>
  );
}
