"use client";

import { Loader2, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { explainPrediction, type PredictionResponse } from "@/lib/api";
import { useLanguage } from "@/lib/LanguageContext";
import type { Language } from "@/lib/i18n";
import MarkdownContent from "./MarkdownContent";

type PredictionExplanationProps = {
  prediction: PredictionResponse;
  language: Language;
};

export default function PredictionExplanation({
  prediction,
  language,
}: PredictionExplanationProps) {
  const { t } = useLanguage();
  const [explanation, setExplanation] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);
    setError(null);
    setExplanation(null);

    explainPrediction(prediction, language)
      .then((text) => {
        if (isMounted) {
          setExplanation(text);
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
  }, [prediction, language]);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-soft">
      <div className="mb-4 flex items-center gap-3">
        <Sparkles className="h-5 w-5 text-clinical" aria-hidden="true" />
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.12em] text-clinical">
            {t("explanation.eyebrow")}
          </p>
          <h2 className="mt-1 text-xl font-bold text-ink">
            {t("explanation.title")}
          </h2>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-2 text-sm text-slate-600">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          {t("explanation.loading")}
        </div>
      ) : null}

      {error ? (
        <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </p>
      ) : null}

      {explanation ? <MarkdownContent content={explanation} /> : null}
    </section>
  );
}
