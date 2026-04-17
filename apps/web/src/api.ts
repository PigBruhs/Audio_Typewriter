export type MixResponse = {
  job_id: string;
  base_name?: string | null;
  status: string;
  output_path?: string | null;
  missing_tokens?: string[];
  token_count?: number;
};

export type AudioBaseItem = {
  base_name: string;
  audio_count: number;
  total_duration_sec: number;
  total_file_size_bytes: number;
};

export type AudioBaseImportResponse = {
  base_name: string;
  audio_count: number;
  total_duration_sec: number;
  total_file_size_bytes: number;
  ingested_source_count: number;
  token_count: number;
};

export async function requestMix(baseName: string, sentence: string, mixMode: string): Promise<MixResponse> {
  const response = await fetch("/api/v1/mix", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ base_name: baseName, sentence, mix_mode: mixMode }),
  });

  if (!response.ok) {
    throw new Error(`Mix request failed: ${response.status}`);
  }

  return (await response.json()) as MixResponse;
}

export async function listAudioBases(): Promise<AudioBaseItem[]> {
  const response = await fetch("/api/v1/audio-bases");
  if (!response.ok) {
    throw new Error(`List audio bases failed: ${response.status}`);
  }
  return (await response.json()) as AudioBaseItem[];
}

export async function getAudioBaseStats(baseName: string): Promise<AudioBaseItem> {
  const response = await fetch(`/api/v1/audio-bases/${encodeURIComponent(baseName)}/stats`);
  if (!response.ok) {
    throw new Error(`Get base stats failed: ${response.status}`);
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
    throw new Error(`Import base failed: ${response.status}`);
  }
  return (await response.json()) as AudioBaseImportResponse;
}

