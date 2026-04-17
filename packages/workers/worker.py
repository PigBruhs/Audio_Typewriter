from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class JobResult:
    job_id: str
    status: str
    output_path: str | None = None


def process_mix_job(job_id: str, sentence: str) -> JobResult:
    """Worker entrypoint placeholder for background sentence mixing."""
    del sentence
    return JobResult(job_id=job_id, status="queued")

