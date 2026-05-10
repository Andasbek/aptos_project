"use client";

import { useLanguage } from "@/lib/LanguageContext";

export default function LocalizedHeader() {
  const { t } = useLanguage();
  return (
    <div>
      <p className="mb-2 text-sm font-semibold uppercase tracking-[0.12em] text-clinical">
        {t("header.eyebrow")}
      </p>
      <h1 className="max-w-4xl text-3xl font-bold tracking-normal text-ink sm:text-5xl">
        {t("header.title")}
      </h1>
      <p className="mt-4 max-w-3xl text-base leading-7 text-slate-600">
        {t("header.subtitle")}
      </p>
    </div>
  );
}
