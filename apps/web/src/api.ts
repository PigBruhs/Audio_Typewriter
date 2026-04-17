export type MixResponse = {
  job_id: string;
  status: string;
  output_path?: string | null;
};

export async function requestMix(sentence: string): Promise<MixResponse> {
  const response = await fetch("/api/v1/mix", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ sentence }),
  });

  if (!response.ok) {
    throw new Error(`Mix request failed: ${response.status}`);
  }

  return (await response.json()) as MixResponse;
}

