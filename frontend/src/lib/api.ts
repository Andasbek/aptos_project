import type { Language } from "./i18n";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type ModelOption = {
  name: string;
  checkpoint_path: string;
  checkpoint_exists: boolean;
  is_default: boolean;
};

export type ModelInfo = {
  default_model: string;
  available_models: ModelOption[];
  device: string;
  classes: Record<string, string>;
  llm_enabled: boolean;
  supported_languages: Language[];
};

export type PredictionResponse = {
  predicted_class: number;
  class_name: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_name: string;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

async function parseApiError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    // Keep the fallback below for non-JSON responses.
  }

  return `Request failed with status ${response.status}`;
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/model-info`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

  return response.json();
}

export async function predictImage(
  file: File,
  modelName: string,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model", modelName);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

  return response.json();
}

export async function explainPrediction(
  prediction: PredictionResponse,
  language: Language,
): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prediction, language }),
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

  const data: { explanation: string } = await response.json();
  return data.explanation;
}

export async function chatWithLLM(
  messages: ChatMessage[],
  prediction: PredictionResponse | null,
  language: Language,
): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, prediction, language }),
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

  const data: { reply: string } = await response.json();
  return data.reply;
}
