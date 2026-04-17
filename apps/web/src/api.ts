export type MixResponse = {
  job_id: string;
  base_name?: string | null;
  status: string;
  output_path?: string | null;
  missing_tokens?: string[];
  token_count?: number;
};

export type StitchSegment = {
  source_audio_id: string;
  start_sec: number;
  end_sec: number;
  label?: string;
};

export type HealthResponse = {
  status: string;
  asr_preferred_device: string;
  asr_resolved_device: string;
  asr_compute_type: string;
  asr_last_device_used: string;
  asr_last_compute_type: string;
};

export type AudioBaseItem = {
  base_name: string;
  audio_count: number;
  total_duration_sec: number;
  total_file_size_bytes: number;
};

export type AudioBaseImportResponse = {
  base_name: string;
  overwritten: boolean;
  cleared_audio_files: number;
  cleared_index_sources: number;
  audio_count: number;
  total_duration_sec: number;
  total_file_size_bytes: number;
  ingested_source_count: number;
  token_count: number;
  task_id?: string | null;
  task_status?: string | null;
  discarded_task_count?: number;
};

export type QueueTask = {
  task_id: string;
  base_name: string;
  status: string;
  stage?: string;
  ready_for_asr?: boolean;
  vad_total_audio_sec?: number;
  vad_processed_audio_sec?: number;
  total_files: number;
  processed_files: number;
  next_sequence_number: number;
  token_count: number;
  model_tier: string;
  created_at: string;
  updated_at: string;
  last_error?: string | null;
  overwritten?: boolean;
  cleared_audio_files?: number;
  cleared_index_sources?: number;
};

export type ImportStreamEvent =
  | { type: "status"; message: string }
  | { type: "task"; task: QueueTask }
  | { type: "overwrite"; base_name: string; cleared_audio_files: number; cleared_index_sources: number }
  | { type: "vad_start"; base_name: string; total_audio_sec: number; processed_audio_sec: number }
  | { type: "vad_progress"; base_name: string; file_name: string; total_audio_sec: number; processed_audio_sec: number }
  | { type: "vad_complete"; base_name: string; total_audio_sec: number; processed_audio_sec: number }
  | { type: "start"; base_name: string; total: number }
  | { type: "progress"; base_name: string; current: number; total: number; file_name: string; token_count: number }
  | { type: "complete"; result: AudioBaseImportResponse }
  | { type: "error"; detail: string };

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload?.detail) {
      return payload.detail;
    }
  } catch (_error) {
    // Fall back to HTTP status when body is not JSON.
  }
  return `HTTP ${response.status}`;
}

export async function requestMix(baseName: string, sentence: string, mixMode: string): Promise<MixResponse> {
  const response = await fetch("/api/v1/mix", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ base_name: baseName, sentence, mix_mode: mixMode }),
  });

  if (!response.ok) {
    throw new Error(`Mix request failed: ${await readErrorDetail(response)}`);
  }

  return (await response.json()) as MixResponse;
}

export async function requestStitch(baseName: string, segments: StitchSegment[], outputPath?: string): Promise<MixResponse> {
  const response = await fetch("/api/v1/mix/stitch", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ base_name: baseName, segments, output_path: outputPath ?? null }),
  });
  if (!response.ok) {
    throw new Error(`Stitch request failed: ${await readErrorDetail(response)}`);
  }
  return (await response.json()) as MixResponse;
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch("/api/v1/health");
  if (!response.ok) {
    throw new Error(`Health request failed: ${await readErrorDetail(response)}`);
  }
  return (await response.json()) as HealthResponse;
}

export async function listAudioBases(): Promise<AudioBaseItem[]> {
  const response = await fetch("/api/v1/audio-bases");
  if (!response.ok) {
    throw new Error(`List audio bases failed: ${await readErrorDetail(response)}`);
  }
  return (await response.json()) as AudioBaseItem[];
}

export async function getAudioBaseStats(baseName: string): Promise<AudioBaseItem> {
  const response = await fetch(`/api/v1/audio-bases/${encodeURIComponent(baseName)}/stats`);
  if (!response.ok) {
    throw new ApiError(`Get base stats failed: ${await readErrorDetail(response)}`, response.status);
  }
  return (await response.json()) as AudioBaseItem;
}

export async function importAudioBase(baseName: string, files: File[]): Promise<AudioBaseImportResponse> {
  const form = new FormData();
  form.append("base_name", baseName);
  for (const file of files) {
    form.append("files", file, file.name);
  }

  const response = await fetch("/api/v1/audio-bases/import", {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new Error(`Import base failed: ${await readErrorDetail(response)}`);
  }
  return (await response.json()) as AudioBaseImportResponse;
}

export async function importAudioBaseStream(
  baseName: string,
  files: File[],
  onEvent: (event: ImportStreamEvent) => void
): Promise<AudioBaseImportResponse> {
  const form = new FormData();
  form.append("base_name", baseName);
  for (const file of files) {
    form.append("files", file, file.name);
  }

  const response = await fetch("/api/v1/audio-bases/import/stream", {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new Error(`Import stream failed: ${await readErrorDetail(response)}`);
  }
  if (!response.body) {
    throw new Error("Import stream failed: empty response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: AudioBaseImportResponse | null = null;
  let streamError: string | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      const event = JSON.parse(trimmed) as ImportStreamEvent;
      onEvent(event);
      if (event.type === "complete") {
        finalResult = event.result;
      }
      if (event.type === "error") {
        streamError = event.detail;
      }
    }
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer.trim()) as ImportStreamEvent;
    onEvent(event);
    if (event.type === "complete") {
      finalResult = event.result;
    }
    if (event.type === "error") {
      streamError = event.detail;
    }
  }

  if (streamError) {
    throw new Error(streamError);
  }
  if (!finalResult) {
    throw new Error("Import stream ended without a completion event.");
  }
  return finalResult;
}

export async function listQueueTasks(): Promise<QueueTask[]> {
  const response = await fetch("/api/v1/tasks");
  if (!response.ok) {
    throw new Error(`List tasks failed: ${await readErrorDetail(response)}`);
  }
  return (await response.json()) as QueueTask[];
}

export async function pauseQueueTask(taskId: string): Promise<QueueTask> {
  const response = await fetch(`/api/v1/tasks/${encodeURIComponent(taskId)}/pause`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Pause task failed: ${await readErrorDetail(response)}`);
  }
  return (await response.json()) as QueueTask;
}

export async function resumeQueueTask(taskId: string): Promise<QueueTask> {
  const response = await fetch(`/api/v1/tasks/${encodeURIComponent(taskId)}/resume`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Resume task failed: ${await readErrorDetail(response)}`);
  }
  return (await response.json()) as QueueTask;
}

export async function deleteQueueTask(taskId: string): Promise<void> {
  const response = await fetch(`/api/v1/tasks/${encodeURIComponent(taskId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Delete task failed: ${await readErrorDetail(response)}`);
  }
}

export async function requestSystemExit(): Promise<void> {
  const response = await fetch("/api/v1/system/exit", {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Exit failed: ${await readErrorDetail(response)}`);
  }
}

