"use client";

import { Languages } from "lucide-react";
import { useLanguage } from "@/lib/LanguageContext";
import { LANGUAGE_LABELS, SUPPORTED_LANGUAGES, type Language } from "@/lib/i18n";

type LanguageSelectorProps = {
  variant?: "header" | "inline";
};

export default function LanguageSelector({
  variant = "header",
}: LanguageSelectorProps) {
  const { language, setLanguage, t } = useLanguage();
  const label = t("header.languagePicker");

  if (variant === "header") {
    return (
      <label className="inline-flex items-center gap-2 rounded-md border border-slate-200 bg-white px-3 py-2 shadow-soft">
        <Languages className="h-4 w-4 text-clinical" aria-hidden="true" />
        <span className="sr-only">{label}</span>
        <select
          value={language}
          onChange={(event) => setLanguage(event.target.value as Language)}
          className="bg-transparent text-sm font-semibold text-ink outline-none"
        >
          {SUPPORTED_LANGUAGES.map((lang) => (
            <option key={lang} value={lang}>
              {LANGUAGE_LABELS[lang]}
            </option>
          ))}
        </select>
      </label>
    );
  }

  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
        {label}
      </span>
      <select
        value={language}
        onChange={(event) => setLanguage(event.target.value as Language)}
        className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-ink outline-none focus:border-clinical"
      >
        {SUPPORTED_LANGUAGES.map((lang) => (
          <option key={lang} value={lang}>
            {LANGUAGE_LABELS[lang]}
          </option>
        ))}
      </select>
    </label>
  );
}
