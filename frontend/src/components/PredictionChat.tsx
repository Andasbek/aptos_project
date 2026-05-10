"use client";

import { Loader2, MessageCircle, Send } from "lucide-react";
import { useState, type FormEvent } from "react";
import { chatWithLLM, type ChatMessage, type PredictionResponse } from "@/lib/api";
import { useLanguage } from "@/lib/LanguageContext";
import type { Language } from "@/lib/i18n";
import MarkdownContent from "./MarkdownContent";

type PredictionChatProps = {
  prediction: PredictionResponse;
  language: Language;
};

export default function PredictionChat({
  prediction,
  language,
}: PredictionChatProps) {
  const { t } = useLanguage();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) {
      return;
    }

    const nextMessages: ChatMessage[] = [
      ...messages,
      { role: "user", content: trimmed },
    ];
    setMessages(nextMessages);
    setInput("");
    setError(null);
    setIsLoading(true);

    try {
      const reply = await chatWithLLM(nextMessages, prediction, language);
      setMessages([...nextMessages, { role: "assistant", content: reply }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("chat.errors.failed"));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-soft">
      <div className="mb-4 flex items-center gap-3">
        <MessageCircle className="h-5 w-5 text-clinical" aria-hidden="true" />
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.12em] text-clinical">
            {t("chat.eyebrow")}
          </p>
          <h2 className="mt-1 text-xl font-bold text-ink">{t("chat.title")}</h2>
          <p className="mt-1 text-sm text-slate-600">{t("chat.subtitle")}</p>
        </div>
      </div>

      <div className="mb-4 flex max-h-96 flex-col gap-3 overflow-y-auto">
        {messages.length === 0 ? (
          <p className="text-sm text-slate-500">{t("chat.empty")}</p>
        ) : null}
        {messages.map((message, index) =>
          message.role === "user" ? (
            <div
              key={index}
              className="self-end rounded-lg bg-clinical px-4 py-2 text-sm text-white max-w-[85%] whitespace-pre-wrap leading-6"
            >
              {message.content}
            </div>
          ) : (
            <div
              key={index}
              className="self-start rounded-lg bg-slate-100 px-4 py-2 text-slate-800 max-w-[85%]"
            >
              <MarkdownContent content={message.content} variant="chat" />
            </div>
          ),
        )}
        {isLoading ? (
          <div className="self-start flex items-center gap-2 rounded-lg bg-slate-100 px-4 py-2 text-sm text-slate-600">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            {t("chat.thinking")}
          </div>
        ) : null}
      </div>

      {error ? (
        <p className="mb-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </p>
      ) : null}

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={t("chat.placeholder")}
          disabled={isLoading}
          className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm text-ink outline-none focus:border-clinical disabled:bg-slate-50"
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-clinical px-4 text-sm font-bold text-white transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          <Send className="h-4 w-4" aria-hidden="true" />
          {t("chat.send")}
        </button>
      </form>
    </section>
  );
}
