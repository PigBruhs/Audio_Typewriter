import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import {
  AudioBaseItem,
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
      const stats = await getAudioBaseStats(first);
      setSelectedStats(stats);
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
    try {
      const onStreamEvent = (streamEvent: ImportStreamEvent) => {
        if (streamEvent.type === "status") {
          setImportLogs((prev) => [...prev, streamEvent.message]);
        }
        if (streamEvent.type === "vad_start") {
          setImportProgressCurrent(streamEvent.processed_audio_sec);
          setImportProgressTotal(streamEvent.total_audio_sec);
          setImportLogs((prev) => [
            ...prev,
            `VAD start: ${streamEvent.processed_audio_sec.toFixed(1)}s / ${streamEvent.total_audio_sec.toFixed(1)}s`,
          ]);
        }
        if (streamEvent.type === "vad_progress") {
          setImportProgressCurrent(streamEvent.processed_audio_sec);
          setImportProgressTotal(streamEvent.total_audio_sec);
          setImportLogs((prev) => [
            ...prev,
            `VAD ${streamEvent.processed_audio_sec.toFixed(1)}s/${streamEvent.total_audio_sec.toFixed(1)}s | ${streamEvent.file_name}`,
          ]);
        }
        if (streamEvent.type === "vad_complete") {
          setImportProgressCurrent(streamEvent.processed_audio_sec);
          setImportProgressTotal(streamEvent.total_audio_sec);
          setImportLogs((prev) => [...prev, "VAD completed."]);
        }
        if (streamEvent.type === "overwrite") {
          setImportLogs((prev) => [
            ...prev,
            `Overwrite detected for base=${streamEvent.base_name}. Cleared files=${streamEvent.cleared_audio_files}, cleared indexed sources=${streamEvent.cleared_index_sources}.`,
          ]);
        }
        if (streamEvent.type === "start") {
          setImportLogs((prev) => [...prev, `ASR queued: 0/${streamEvent.total} files`]);
        }
        if (streamEvent.type === "progress") {
          setImportProgressCurrent(streamEvent.current);
          setImportProgressTotal(streamEvent.total);
          setImportLogs((prev) => [
            ...prev,
            `ASR ${streamEvent.current}/${streamEvent.total} | ${streamEvent.file_name} | tokens=${streamEvent.token_count}`,
          ]);
        }
        if (streamEvent.type === "complete") {
          setImportProgressCurrent(streamEvent.result.ingested_source_count);
          setImportProgressTotal(streamEvent.result.ingested_source_count);
          setImportLogs((prev) => [...prev, "Import completed."]);
        }
        if (streamEvent.type === "error") {
          setImportLogs((prev) => [...prev, `Error: ${streamEvent.detail}`]);
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
        setImportLogs((prev) => [...prev, `Overwrite discarded old unfinished tasks: ${data.discarded_task_count}`]);
      }
      await refreshBases();
      await refreshTasks();
      setSelectedBase(data.base_name);
      const stats = await getAudioBaseStats(data.base_name);
      setSelectedStats(stats);
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
      const stats = await getAudioBaseStats(nextBase);
      setSelectedStats(stats);
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
        {importLogs.length > 0 && <pre style={{ maxHeight: 180, overflow: "auto" }}>{importLogs.join("\n")}</pre>}
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
          <option value="context_priority">Context Priority (same sentence first)</option>
          <option value="all_random">All Random</option>
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
          <p>One ASR indexing task runs at a time. Pause/Resume applies to ASR stage only.</p>
          {tasks.length === 0 && <div>No tasks yet.</div>}
          {tasks.map((task) => (
            <div key={task.task_id} style={{ border: "1px solid #ccc", padding: 8, marginBottom: 8 }}>
              <div><strong>{task.base_name}</strong> [{task.status}]</div>
              <div>Progress: {task.processed_files}/{task.total_files}, next={task.next_sequence_number}</div>
              <div>Tokens indexed: {task.token_count}</div>
              {task.last_error && <div style={{ color: "crimson" }}>Error: {task.last_error}</div>}
              {task.status === "running" && (
                <button type="button" onClick={() => onPauseTask(task.task_id)}>Pause</button>
              )}
              {(task.status === "paused" || task.status === "failed" || task.status === "queued") && (
                <button type="button" onClick={() => onResumeTask(task.task_id)}>Resume</button>
              )}
            </div>
          ))}
        </section>
      )}
    </main>
  );
}

export default App;

