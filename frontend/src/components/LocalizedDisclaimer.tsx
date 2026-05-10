"use client";

import { useLanguage } from "@/lib/LanguageContext";

export default function LocalizedDisclaimer() {
  const { t } = useLanguage();
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-900 shadow-soft">
      {t("disclaimer")}
    </div>
  );
}
