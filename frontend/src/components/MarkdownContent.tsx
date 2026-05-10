"use client";

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

type MarkdownContentProps = {
  content: string;
  variant?: "default" | "chat";
};

const baseComponents: Components = {
  h1: ({ children }) => (
    <h3 className="mt-3 mb-2 text-base font-bold text-ink">{children}</h3>
  ),
  h2: ({ children }) => (
    <h3 className="mt-3 mb-2 text-base font-bold text-ink">{children}</h3>
  ),
  h3: ({ children }) => (
    <h4 className="mt-3 mb-1.5 text-sm font-bold text-ink">{children}</h4>
  ),
  h4: ({ children }) => (
    <h5 className="mt-2 mb-1 text-sm font-semibold text-ink">{children}</h5>
  ),
  p: ({ children }) => (
    <p className="mb-2 last:mb-0 leading-7 text-slate-700">{children}</p>
  ),
  ul: ({ children }) => (
    <ul className="mb-2 list-disc space-y-1 pl-5 text-slate-700 last:mb-0">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-2 list-decimal space-y-1 pl-5 text-slate-700 last:mb-0">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="leading-7">{children}</li>,
  strong: ({ children }) => (
    <strong className="font-semibold text-ink">{children}</strong>
  ),
  em: ({ children }) => <em className="italic">{children}</em>,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="text-clinical underline underline-offset-2 hover:text-teal-800"
    >
      {children}
    </a>
  ),
  code: ({ children }) => (
    <code className="rounded bg-slate-100 px-1 py-0.5 text-[0.85em] text-slate-800">
      {children}
    </code>
  ),
  pre: ({ children }) => (
    <pre className="mb-2 overflow-x-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-100 last:mb-0">
      {children}
    </pre>
  ),
  blockquote: ({ children }) => (
    <blockquote className="mb-2 border-l-4 border-clinical/50 bg-slate-50 px-3 py-2 italic text-slate-600 last:mb-0">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-3 border-slate-200" />,
  table: ({ children }) => (
    <div className="mb-2 overflow-x-auto last:mb-0">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-slate-200 bg-slate-50 px-2 py-1 text-left font-semibold text-ink">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-slate-200 px-2 py-1 text-slate-700">{children}</td>
  ),
};

const chatComponents: Components = {
  ...baseComponents,
  p: ({ children }) => (
    <p className="mb-1.5 last:mb-0 leading-6 text-slate-800">{children}</p>
  ),
  ul: ({ children }) => (
    <ul className="mb-1.5 list-disc space-y-0.5 pl-5 text-slate-800 last:mb-0">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-1.5 list-decimal space-y-0.5 pl-5 text-slate-800 last:mb-0">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="leading-6">{children}</li>,
};

export default function MarkdownContent({
  content,
  variant = "default",
}: MarkdownContentProps) {
  return (
    <div className="text-sm">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={variant === "chat" ? chatComponents : baseComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
