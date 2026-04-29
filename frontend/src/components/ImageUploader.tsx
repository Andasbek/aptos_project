"use client";

import { ImagePlus, Loader2, RotateCcw, Send } from "lucide-react";
import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { predictImage, type PredictionResponse } from "@/lib/api";
import PredictionResult from "./PredictionResult";

export default function ImageUploader() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setResult(null);
    setError(null);
  }

  function resetSelection() {
    setSelectedFile(null);
    setResult(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  async function handleAnalyze() {
    if (!selectedFile) {
      setError("Select a retinal fundus image first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const prediction = await predictImage(selectedFile);
      setResult(prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-soft">
        <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.12em] text-clinical">
              Upload
            </p>
            <h2 className="mt-1 text-2xl font-bold text-ink">Fundus Image</h2>
          </div>
          <button
            type="button"
            onClick={resetSelection}
            className="inline-flex h-10 items-center justify-center gap-2 rounded-md border border-slate-200 px-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
            title="Reset selected image"
          >
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
            Reset
          </button>
        </div>

        <label className="flex min-h-64 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 bg-slate-50 p-5 text-center transition hover:border-clinical hover:bg-teal-50/50">
          <input
            ref={inputRef}
            className="sr-only"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />
          {previewUrl ? (
            <div className="relative h-80 w-full overflow-hidden rounded-lg bg-black">
              <img
                src={previewUrl}
                alt="Selected retinal fundus preview"
                className="h-full w-full object-contain"
              />
            </div>
          ) : (
            <div className="flex max-w-md flex-col items-center gap-3">
              <ImagePlus className="h-12 w-12 text-clinical" aria-hidden="true" />
              <div>
                <p className="text-base font-semibold text-ink">Choose an image</p>
                <p className="mt-1 text-sm leading-6 text-slate-600">
                  PNG, JPG, or JPEG fundus image for APTOS class prediction.
                </p>
              </div>
            </div>
          )}
        </label>

        {selectedFile ? (
          <p className="mt-3 break-all text-sm text-slate-600">{selectedFile.name}</p>
        ) : null}

        {error ? (
          <p className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {error}
          </p>
        ) : null}

        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-end">
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!selectedFile || isLoading}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-clinical px-5 text-sm font-bold text-white transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            ) : (
              <Send className="h-4 w-4" aria-hidden="true" />
            )}
            Analyze Image
          </button>
        </div>
      </section>

      {result ? <PredictionResult result={result} /> : null}
    </div>
  );
}
