import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import {
  ApiError,
  AudioBaseItem,
  deleteQueueTask,
  getAudioBaseStats,
  getHealth,
  HealthResponse,
  importAudioBaseStream,
  ImportStreamEvent,
  listQueueTasks,
  listAudioBases,
  pauseQueueTask,
  QueueTask,
  requestSystemExit,
  resumeQueueTask,
  requestMix,
} from "./api";

const MAX_IMPORT_LOG_LINES = 500;

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatStage(stage?: string): string {
  if (stage === "vad") {
    return "[VAD]";
  }
  if (stage === "asr") {
    return "[ASR]";
  }
  return "[UNKNOWN]";
}

function clampProgress(current: number, total: number): { value: number; max: number } {
  const safeMax = Math.max(total, 0.001);
  const safeValue = Math.min(Math.max(current, 0), safeMax);
  return { value: safeValue, max: safeMax };
}

function App(): JSX.Element {
  const [activeTab, setActiveTab] = useState<"workbench" | "tasks">("workbench");
  const [baseName, setBaseName] = useState("");
  const [baseFiles, setBaseFiles] = useState<File[]>([]);
  const [bases, setBases] = useState<AudioBaseItem[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [selectedBase, setSelectedBase] = useState("");
  const [selectedStats, setSelectedStats] = useState<AudioBaseItem | null>(null);
  const [mixMode, setMixMode] = useState("context_priority");
  const [sentence, setSentence] = useState("");
  const [result, setResult] = useState<string>("");
  const [importResult, setImportResult] = useState<string>("");
  const [importProgressCurrent, setImportProgressCurrent] = useState(0);
  const [importProgressTotal, setImportProgressTotal] = useState(0);
  const [importLogs, setImportLogs] = useState<string[]>([]);
  const [tasks, setTasks] = useState<QueueTask[]>([]);
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [exiting, setExiting] = useState(false);

  function appendImportLog(message: string): void {
    setImportLogs((prev) => {
      const next = [...prev, message];
      if (next.length <= MAX_IMPORT_LOG_LINES) {
        return next;
      }
      return next.slice(next.length - MAX_IMPORT_LOG_LINES);
    });
  }

  useEffect(() => {
    getHealth()
      .then((data) => setHealth(data))
      .catch((error: unknown) => setResult(error instanceof Error ? error.message : "Load health failed"));

    refreshBases().catch((error: unknown) => {
      setResult(error instanceof Error ? error.message : "Load bases failed");
    });

    const timer = window.setInterval(() => {
      refreshTasks().catch((_error) => {
        // Keep polling resilient; surface errors only on explicit actions.
      });
    }, 1200);

    return () => {
      window.clearInterval(timer);
    };
  }, []);

  async function refreshBases(): Promise<void> {
    const data = await listAudioBases();
    setBases(data);
    if (!selectedBase && data.length > 0) {
      const first = data[0].base_name;
      setSelectedBase(first);
      await loadBaseStats(first, true);
    }
  }

  async function loadBaseStats(base: string, allowMissingAsPending: boolean): Promise<void> {
    try {
      const stats = await getAudioBaseStats(base);
      setSelectedStats(stats);
    } catch (error) {
      if (allowMissingAsPending && error instanceof ApiError && error.status === 404) {
        setSelectedStats(null);
        appendImportLog(`[SYSTEM] Base '${base}' is queued and not materialized yet. Continue processing...`);
        return;
      }
      throw error;
    }
  }

  async function refreshTasks(): Promise<void> {
    const data = await listQueueTasks();
    setTasks(data);
  }

  async function onImportSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!baseName.trim() || baseFiles.length === 0) {
      setImportResult("Please enter base name and select a folder.");
      return;
    }

    setImporting(true);
    setImportResult("");
    setImportProgressCurrent(0);
    setImportProgressTotal(0);
    setImportLogs([]);
    await refreshTasks();
    try {
      const onStreamEvent = (streamEvent: ImportStreamEvent) => {
        if (streamEvent.type === "status") {
          appendImportLog(`[SYSTEM] ${streamEvent.message}`);
        }
        if (streamEvent.type === "task") {
          setTasks((prev) => {
            const next = prev.filter((task) => task.task_id !== streamEvent.task.task_id);
            next.unshift(streamEvent.task);
            return next;
          });
          appendImportLog(`${formatStage(streamEvent.task.stage)} Task created: ${streamEvent.task.task_id}`);
        }
        if (streamEvent.type === "vad_start") {
          setImportProgressCurrent(streamEvent.processed_audio_sec);
          setImportProgressTotal(streamEvent.total_audio_sec);
          appendImportLog(`[VAD] start: ${streamEvent.processed_audio_sec.toFixed(1)}s / ${streamEvent.total_audio_sec.toFixed(1)}s`);
        }
        if (streamEvent.type === "vad_progress") {
          setImportProgressCurrent(streamEvent.processed_audio_sec);
          setImportProgressTotal(streamEvent.total_audio_sec);
          appendImportLog(`[VAD] ${streamEvent.processed_audio_sec.toFixed(1)}s/${streamEvent.total_audio_sec.toFixed(1)}s | ${streamEvent.file_name}`);
        }
        if (streamEvent.type === "vad_complete") {
          setImportProgressCurrent(streamEvent.processed_audio_sec);
          setImportProgressTotal(streamEvent.total_audio_sec);
          appendImportLog("[VAD] completed.");
        }
        if (streamEvent.type === "overwrite") {
          appendImportLog(`[SYSTEM] Overwrite detected for base=${streamEvent.base_name}. Cleared files=${streamEvent.cleared_audio_files}, cleared indexed sources=${streamEvent.cleared_index_sources}.`);
        }
        if (streamEvent.type === "start") {
          appendImportLog(`[ASR] queued: 0/${streamEvent.total} files`);
        }
        if (streamEvent.type === "progress") {
          setImportProgressCurrent(streamEvent.current);
          setImportProgressTotal(streamEvent.total);
          appendImportLog(`[ASR] ${streamEvent.current}/${streamEvent.total} | ${streamEvent.file_name} | tokens=${streamEvent.token_count}`);
        }
        if (streamEvent.type === "complete") {
          setImportProgressCurrent(streamEvent.result.ingested_source_count);
          setImportProgressTotal(streamEvent.result.ingested_source_count);
          appendImportLog("[SYSTEM] Import request completed (ASR continues in queue).");
        }
        if (streamEvent.type === "error") {
          appendImportLog(`[ERROR] ${streamEvent.detail}`);
        }
      };

      const data = await importAudioBaseStream(baseName.trim(), baseFiles, onStreamEvent);
      const overwriteLabel = data.overwritten
        ? `overwrite=yes(cleared files=${data.cleared_audio_files}, cleared indexed sources=${data.cleared_index_sources})`
        : "overwrite=no";
      setImportResult(
        `Imported base=${data.base_name}, ${overwriteLabel}, files=${data.audio_count}, duration=${data.total_duration_sec.toFixed(1)}s, size=${formatBytes(data.total_file_size_bytes)}, tokens=${data.token_count}`
      );
      if (data.discarded_task_count && data.discarded_task_count > 0) {
        appendImportLog(`Overwrite discarded old unfinished tasks: ${data.discarded_task_count}`);
      }
      await refreshBases();
      await refreshTasks();
      setSelectedBase(data.base_name);
      await loadBaseStats(data.base_name, true);
    } catch (error) {
      setImportResult(error instanceof Error ? error.message : "Import failed");
    } finally {
      setImporting(false);
    }
  }

  async function onPauseTask(taskId: string) {
    try {
      await pauseQueueTask(taskId);
      await refreshTasks();
    } catch (error) {
      setResult(error instanceof Error ? error.message : "Pause task failed");
    }
  }

  async function onResumeTask(taskId: string) {
    try {
      await resumeQueueTask(taskId);
      await refreshTasks();
    } catch (error) {
      setResult(error instanceof Error ? error.message : "Resume task failed");
    }
  }

  async function onDeleteTask(taskId: string, baseName: string) {
    const confirmed = window.confirm(
      `Delete task and purge related data for base '${baseName}'? This removes temp, DB index, and audio_base files.`
    );
    if (!confirmed) {
      return;
    }
    try {
      await deleteQueueTask(taskId);
      await refreshTasks();
      await refreshBases();
      appendImportLog(`[SYSTEM] Deleted task ${taskId} and purged base ${baseName}.`);
    } catch (error) {
      setResult(error instanceof Error ? error.message : "Delete task failed");
    }
  }

  async function onExit() {
    setExiting(true);
    try {
      await requestSystemExit();
      alert("Backend is shutting down and queue has been flushed. Close the frontend terminal if needed.");
      window.close();
    } catch (error) {
      setResult(error instanceof Error ? error.message : "Exit failed");
    } finally {
      setExiting(false);
    }
  }

  async function onBaseChange(event: ChangeEvent<HTMLSelectElement>) {
    const nextBase = event.target.value;
    setSelectedBase(nextBase);
    if (!nextBase) {
      setSelectedStats(null);
      return;
    }
    try {
      await loadBaseStats(nextBase, true);
    } catch (error) {
      setResult(error instanceof Error ? error.message : "Get base stats failed");
    }
  }

  function onFolderSelected(event: ChangeEvent<HTMLInputElement>) {
    const files = event.target.files ? Array.from(event.target.files) : [];
    const filtered = files.filter((file) => file.name.toLowerCase().endsWith(".wav") || file.name.toLowerCase().endsWith(".mp3"));
    setBaseFiles(filtered);
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedBase) {
      setResult("Please select an audio base first.");
      return;
    }
    setLoading(true);
    setResult("");
    try {
      const data = await requestMix(selectedBase, sentence, mixMode);
      setResult(`base=${selectedBase}, job=${data.job_id}, status=${data.status}`);
    } catch (error) {
      setResult(error instanceof Error ? error.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 680, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Audio Typewriter</h1>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button type="button" onClick={() => setActiveTab("workbench")}>Workbench</button>
        <button type="button" onClick={() => setActiveTab("tasks")}>Tasks</button>
        <button type="button" onClick={onExit} disabled={exiting} style={{ marginLeft: "auto" }}>
          {exiting ? "Exiting..." : "Exit"}
        </button>
      </div>
      <p>Import a folder as audio base, then build sentence-mixed clips from that base.</p>
      {health && (
        <p>
          ASR runtime: <strong>{health.asr_resolved_device.toUpperCase()}</strong> ({health.asr_compute_type}) | preferred={health.asr_preferred_device} | last={health.asr_last_device_used}
        </p>
      )}

      {activeTab === "workbench" && (
      <>
      <section style={{ marginBottom: 24 }}>
        <h2>Import Audio Base</h2>
        <form onSubmit={onImportSubmit}>
          <input
            value={baseName}
            onChange={(event) => setBaseName(event.target.value)}
            placeholder="base name, e.g. speaker_a"
            style={{ width: "100%", marginBottom: 8 }}
          />
          <input
            type="file"
            multiple
            accept=".wav,.mp3,audio/wav,audio/mpeg"
            onChange={onFolderSelected}
            style={{ width: "100%", marginBottom: 8 }}
            {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
          />
          <div style={{ marginBottom: 8 }}>Selected files: {baseFiles.length}</div>
          <button type="submit" disabled={importing || !baseName.trim() || baseFiles.length === 0}>
            {importing ? "Importing and indexing..." : "Import Base"}
          </button>
        </form>
        {importing && importProgressTotal > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ marginBottom: 4 }}>
              Progress: {importProgressCurrent}/{importProgressTotal}
            </div>
            <progress value={importProgressCurrent} max={importProgressTotal} style={{ width: "100%" }} />
          </div>
        )}
        {importResult && <pre>{importResult}</pre>}
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2>Active Audio Base</h2>
        <select value={selectedBase} onChange={onBaseChange} style={{ width: "100%", marginBottom: 8 }}>
          <option value="">Select a base...</option>
          {bases.map((base) => (
            <option key={base.base_name} value={base.base_name}>
              {base.base_name}
            </option>
          ))}
        </select>
        {selectedStats && (
          <div>
            <div>Audio count: {selectedStats.audio_count}</div>
            <div>Total duration: {selectedStats.total_duration_sec.toFixed(1)} s</div>
            <div>Total file size: {formatBytes(selectedStats.total_file_size_bytes)}</div>
          </div>
        )}
      </section>

      <h2>Create Mix</h2>
      <form onSubmit={onSubmit}>
        <select value={mixMode} onChange={(event) => setMixMode(event.target.value)} style={{ width: "100%", marginBottom: 8 }}>
          <option value="context_priority">Context Priority (same/adjacent clip first)</option>
          <option value="all_random">All Random</option>
          <option value="nearest_gap">Nearest Gap Priority</option>
          <option value="farthest_gap">Farthest Gap Priority</option>
        </select>
        <textarea
          rows={4}
          value={sentence}
          onChange={(event) => setSentence(event.target.value)}
          style={{ width: "100%" }}
          placeholder="Type sentence here..."
        />
        <button type="submit" disabled={loading || !selectedBase || sentence.trim().length === 0}>
          {loading ? "Submitting..." : "Create Mix Job"}
        </button>
      </form>
      {result && <pre>{result}</pre>}
      </>
      )}

      {activeTab === "tasks" && (
        <section>
          <h2>Task Queue</h2>
          <p>One indexing task runs at a time.</p>
          {tasks.length === 0 && <div>No tasks yet.</div>}
          {tasks.map((task) => (
            <div key={task.task_id} style={{ border: "1px solid #ccc", padding: 8, marginBottom: 8 }}>
              <div><strong>{task.base_name}</strong> [{task.status}] {formatStage(task.stage)}</div>
              {task.stage === "vad" ? (
                <>
                  <div>
                    VAD Progress: {(task.vad_processed_audio_sec ?? 0).toFixed(1)}s/{(task.vad_total_audio_sec ?? 0).toFixed(1)}s
                  </div>
                  <progress
                    {...clampProgress(task.vad_processed_audio_sec ?? 0, task.vad_total_audio_sec ?? 0)}
                    style={{ width: "100%", marginTop: 4 }}
                  />
                </>
              ) : (
                <>
                  <div>ASR Progress: {task.processed_files}/{task.total_files}, next={task.next_sequence_number}</div>
                  <progress
                    {...clampProgress(task.processed_files, task.total_files)}
                    style={{ width: "100%", marginTop: 4 }}
                  />
                </>
              )}
              {task.last_error && <div style={{ color: "crimson" }}>Error: {task.last_error}</div>}
              {task.status === "running" && (
                <button type="button" onClick={() => onPauseTask(task.task_id)}>Pause</button>
              )}
              {(task.status === "paused" || task.status === "failed" || task.status === "queued") && (
                <button type="button" onClick={() => onResumeTask(task.task_id)}>Resume</button>
              )}
              <button type="button" onClick={() => onDeleteTask(task.task_id, task.base_name)} style={{ marginLeft: 8 }}>
                Delete
              </button>
            </div>
          ))}
        </section>
      )}

      <section style={{ marginTop: 20 }}>
        <h2>Console</h2>
        <div style={{ fontSize: 13, color: "#666", marginBottom: 6 }}>
          Runtime logs from import/VAD/ASR pipeline.
        </div>
        <pre style={{ maxHeight: 220, overflow: "auto", background: "#111", color: "#ddd", padding: 10 }}>
          {importLogs.length > 0 ? importLogs.join("\n") : "[SYSTEM] No logs yet."}
        </pre>
      </section>
    </main>
  );
}

export default App;

