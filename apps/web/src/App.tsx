import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  ApiError,
  AudioBaseItem,
  deleteQueueTask,
  exportLexicon,
  getAudioBaseStats,
  getHealth,
  HealthResponse,
  importAudioBaseByPathStream,
  ImportStreamEvent,
  listQueueTasks,
  listAudioBases,
  pauseQueueTask,
  QueueTask,
  MixMode,
  MixOutputMode,
  requestReAsr,
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
  if (stage === "preprocess") {
    return "[PREP]";
  }
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
  const [language, setLanguage] = useState<"zh" | "en">("zh");
  const isZh = language === "zh";
  const tt = (zh: string, en: string): string => (isZh ? zh : en);
  const mixTag = isZh ? "活字印刷" : "MIX";
  const formatTaskStatus = (status: string): string => {
    if (!isZh) {
      return status;
    }
    if (status === "running") return "运行中";
    if (status === "queued") return "排队中";
    if (status === "paused") return "已暂停";
    if (status === "completed") return "已完成";
    if (status === "failed") return "失败";
    if (status === "discarded") return "已丢弃";
    return status;
  };

  const [activeTab, setActiveTab] = useState<"workbench" | "tasks">("workbench");
  const [baseName, setBaseName] = useState("");
  const [sourceFolderPath, setSourceFolderPath] = useState("");
  const [bases, setBases] = useState<AudioBaseItem[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [selectedBase, setSelectedBase] = useState("");
  const [selectedStats, setSelectedStats] = useState<AudioBaseItem | null>(null);
  const [mixSpeed, setMixSpeed] = useState(1);
  const [mixGapMs, setMixGapMs] = useState(100);
  const [mixMode, setMixMode] = useState<MixMode>("word_phrase_sentence");
  const [mixOutputMode, setMixOutputMode] = useState<MixOutputMode>("mix");
  const [tailExtensionMs, setTailExtensionMs] = useState(20);
  const [segmentExpansionMs, setSegmentExpansionMs] = useState(250);
  const [sentence, setSentence] = useState("");
  const [result, setResult] = useState<string>("");
  const [importResult, setImportResult] = useState<string>("");
  const [importProgressCurrent, setImportProgressCurrent] = useState(0);
  const [importProgressTotal, setImportProgressTotal] = useState(0);
  const [importLogs, setImportLogs] = useState<string[]>([]);
  const [tasks, setTasks] = useState<QueueTask[]>([]);
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [reasrLoading, setReasrLoading] = useState(false);
  const [exportingLexicon, setExportingLexicon] = useState(false);
  const [exiting, setExiting] = useState(false);
  const previousTasksRef = useRef<Record<string, QueueTask>>({});
  const mixSpeedError = Number.isFinite(mixSpeed) && mixSpeed > 0 ? "" : tt("速度必须大于 0。", "Speed must be greater than 0.");
  const mixGapError = Number.isFinite(mixGapMs) && mixGapMs >= 0 ? "" : tt("间隔必须大于或等于 0。", "Gap must be 0 or greater.");
  const tailExtensionError =
    Number.isFinite(tailExtensionMs) && tailExtensionMs >= 0
      ? ""
      : tt("补尾必须大于或等于 0。", "Tail extension must be 0 or greater.");
  const segmentExpansionError =
    Number.isFinite(segmentExpansionMs) && segmentExpansionMs >= 0
      ? ""
      : tt("片段扩充时长必须大于或等于 0。", "Segment expansion must be 0 or greater.");
  const hasMixInputError =
    mixOutputMode === "segment_output"
      ? Boolean(segmentExpansionError)
      : Boolean(mixSpeedError || mixGapError || tailExtensionError);

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
      .catch((error: unknown) => setResult(error instanceof Error ? error.message : tt("加载运行状态失败", "Load health failed")));

    refreshBases().catch((error: unknown) => {
      setResult(error instanceof Error ? error.message : tt("加载音频库失败", "Load bases failed"));
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
        appendImportLog(`[SYSTEM] ${tt(`音频库 '${base}' 已进入队列，尚未落盘，继续处理中...`, `Base '${base}' is queued and not materialized yet. Continue processing...`)}`);
        return;
      }
      throw error;
    }
  }

  async function refreshTasks(): Promise<void> {
    const data = await listQueueTasks();
    const previous = previousTasksRef.current;
    const next: Record<string, QueueTask> = {};

    for (const task of data) {
      next[task.task_id] = task;
      const prev = previous[task.task_id];
      if (!prev) {
        appendImportLog(`${formatStage(task.stage)} Task detected: ${task.base_name} (${task.task_id}) status=${task.status}`);
        continue;
      }

      if (task.status !== prev.status) {
        appendImportLog(`[SYSTEM] ${task.base_name} status: ${prev.status} -> ${task.status}`);
      }

      if (task.last_error && task.last_error !== prev.last_error) {
        appendImportLog(`[ERROR] ${task.base_name}: ${task.last_error}`);
      }

      if (task.last_event && task.last_event !== prev.last_event) {
        appendImportLog(`[SYSTEM] ${task.base_name}: ${task.last_event}`);
      }

      // Single-base pipeline no longer exposes meaningful fine-grained VAD/ASR progress;
      // rely on status transitions and task events to avoid misleading bars.
    }

    previousTasksRef.current = next;
    setTasks(data);
  }

  async function onImportSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!baseName.trim() || !sourceFolderPath.trim()) {
      setImportResult(tt("请输入音频库名称和本地文件夹路径。", "Please enter base name and local folder path."));
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
        if (streamEvent.type === "preprocess_start") {
          setImportProgressCurrent(streamEvent.migrated_files);
          setImportProgressTotal(streamEvent.total_files);
          appendImportLog(`[PREP] start: ${streamEvent.migrated_files}/${streamEvent.total_files}`);
        }
        if (streamEvent.type === "preprocess_progress") {
          setImportProgressCurrent(streamEvent.migrated_files);
          setImportProgressTotal(streamEvent.total_files);
          const fileLabel = streamEvent.file_name ? ` | ${streamEvent.file_name}` : "";
          appendImportLog(`[PREP] ${streamEvent.migrated_files}/${streamEvent.total_files}${fileLabel}`);
        }
        if (streamEvent.type === "preprocess_complete") {
          setImportProgressCurrent(streamEvent.migrated_files);
          setImportProgressTotal(streamEvent.total_files);
          appendImportLog("[PREP] completed.");
        }
        if (streamEvent.type === "vad_start") {
          appendImportLog("[VAD] started.");
        }
        if (streamEvent.type === "vad_progress") {
          appendImportLog("[VAD] running...");
        }
        if (streamEvent.type === "vad_complete") {
          appendImportLog("[VAD] completed.");
        }
        if (streamEvent.type === "overwrite") {
          appendImportLog(`[SYSTEM] Overwrite detected for base=${streamEvent.base_name}. Cleared files=${streamEvent.cleared_audio_files}, cleared indexed sources=${streamEvent.cleared_index_sources}.`);
        }
        if (streamEvent.type === "start") {
          appendImportLog("[ASR] queued.");
        }
        if (streamEvent.type === "progress") {
          appendImportLog(`[ASR] processed ${streamEvent.file_name} | tokens=${streamEvent.token_count}`);
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

      const data = await importAudioBaseByPathStream(baseName.trim(), sourceFolderPath.trim(), onStreamEvent);
      const overwriteLabel = data.overwritten
        ? tt(
            `覆写=是(清理文件=${data.cleared_audio_files}, 清理索引源=${data.cleared_index_sources})`,
            `overwrite=yes(cleared files=${data.cleared_audio_files}, cleared indexed sources=${data.cleared_index_sources})`
          )
        : tt("覆写=否", "overwrite=no");
      setImportResult(
        tt(
          `已导入音频库=${data.base_name}，${overwriteLabel}，文件数=${data.audio_count}，时长=${data.total_duration_sec.toFixed(1)}s，大小=${formatBytes(data.total_file_size_bytes)}，token=${data.token_count}`,
          `Imported base=${data.base_name}, ${overwriteLabel}, files=${data.audio_count}, duration=${data.total_duration_sec.toFixed(1)}s, size=${formatBytes(data.total_file_size_bytes)}, tokens=${data.token_count}`
        )
      );
      if (data.discarded_task_count && data.discarded_task_count > 0) {
        appendImportLog(`Overwrite discarded old unfinished tasks: ${data.discarded_task_count}`);
      }
      await refreshBases();
      await refreshTasks();
      setSelectedBase(data.base_name);
      await loadBaseStats(data.base_name, true);
    } catch (error) {
      setImportResult(error instanceof Error ? error.message : tt("导入失败", "Import failed"));
    } finally {
      setImporting(false);
    }
  }

  async function onPauseTask(taskId: string) {
    try {
      await pauseQueueTask(taskId);
      await refreshTasks();
    } catch (error) {
      setResult(error instanceof Error ? error.message : tt("暂停任务失败", "Pause task failed"));
    }
  }

  async function onResumeTask(taskId: string) {
    try {
      await resumeQueueTask(taskId);
      await refreshTasks();
    } catch (error) {
      setResult(error instanceof Error ? error.message : tt("继续任务失败", "Resume task failed"));
    }
  }

  async function onDeleteTask(taskId: string, baseName: string) {
    const confirmed = window.confirm(
      tt(
        `确定删除任务并清理音频库 '${baseName}' 的相关数据吗？这会删除 temp、数据库索引和 audio_base 文件。`,
        `Delete task and purge related data for base '${baseName}'? This removes temp, DB index, and audio_base files.`
      )
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
      setResult(error instanceof Error ? error.message : tt("删除任务失败", "Delete task failed"));
    }
  }

  async function onExit() {
    setExiting(true);
    try {
      await requestSystemExit();
      alert(tt("后端正在安全退出，任务队列已保存。必要时请手动关闭前端终端。", "Backend is shutting down and queue has been flushed. Close the frontend terminal if needed."));
      window.close();
    } catch (error) {
      setResult(error instanceof Error ? error.message : tt("退出失败", "Exit failed"));
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
      setResult(error instanceof Error ? error.message : tt("读取音频库统计失败", "Get base stats failed"));
    }
  }

  async function onReAsr(): Promise<void> {
    if (!selectedBase) {
      setResult(tt("请先选择一个音频库。", "Please select an audio base first."));
      return;
    }
    setReasrLoading(true);
    try {
      const data = await requestReAsr(selectedBase);
      appendImportLog(
        `[SYSTEM] reASR queued for ${data.base_name}. purged_sources=${data.purged_sources}, purged_occurrences=${data.purged_occurrences}`
      );
      if (data.discarded_task_count > 0) {
        appendImportLog(`[SYSTEM] reASR discarded old unfinished tasks: ${data.discarded_task_count}`);
      }
      await refreshTasks();
      await loadBaseStats(selectedBase, true);
      setResult(tt(`已提交 reASR：base=${data.base_name}, task=${data.task.task_id}`, `reASR queued for base=${data.base_name}, task=${data.task.task_id}`));
    } catch (error) {
      const message = error instanceof Error ? error.message : tt("reASR 失败", "reASR failed");
      setResult(message);
      appendImportLog(`[ERROR] ${message}`);
    } finally {
      setReasrLoading(false);
    }
  }

  async function onExportLexicon(): Promise<void> {
    if (!selectedBase) {
      setResult(tt("请先选择一个音频库。", "Please select an audio base first."));
      return;
    }
    setExportingLexicon(true);
    try {
      const data = await exportLexicon(selectedBase);
      const blob = new Blob([data.content], { type: "text/plain;charset=utf-8" });
      const href = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = data.fileName;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(href);
      appendImportLog(`[SYSTEM] Lexicon exported for base=${selectedBase}: ${data.fileName}`);
      setResult(tt("词库导出成功。", "Lexicon exported successfully."));
    } catch (error) {
      const message = error instanceof Error ? error.message : tt("导出词库失败", "Export lexicon failed");
      setResult(message);
      appendImportLog(`[ERROR] ${message}`);
    } finally {
      setExportingLexicon(false);
    }
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedBase) {
      setResult(tt("请先选择一个音频库。", "Please select an audio base first."));
      return;
    }
    if (hasMixInputError) {
      const message = (
        mixOutputMode === "segment_output"
          ? [segmentExpansionError]
          : [mixSpeedError, mixGapError, tailExtensionError]
      )
        .filter(Boolean)
        .join(" ");
      setResult(message || tt("请修正活字印刷参数。", "Please fix mix input values."));
      appendImportLog(`[${mixTag}][ERROR] ${message || tt("请修正活字印刷参数。", "Please fix mix input values.")}`);
      return;
    }
    setLoading(true);
    setResult("");
    try {
      const data = await requestMix(
        selectedBase,
        sentence,
        mixSpeed,
        mixGapMs,
        mixMode,
        tailExtensionMs,
        mixOutputMode,
        segmentExpansionMs
      );
      setResult(tt(`音频库=${selectedBase}, 任务=${data.job_id}, 状态=${data.status}`, `base=${selectedBase}, job=${data.job_id}, status=${data.status}`));
      appendImportLog(`[${mixTag}] base=${selectedBase}, job=${data.job_id}, status=${data.status}`);
      if (mixOutputMode === "segment_output") {
        const fileCount = data.output_files?.length ?? 0;
        appendImportLog(`[${mixTag}] segment output ready: folder=${data.output_path ?? ""}, files=${fileCount}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : tt("未知错误", "Unknown error");
      setResult(message);
      appendImportLog(`[${mixTag}][ERROR] ${message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 680, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>{tt("活字印刷机", "Audio Typewriter")}</h1>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button type="button" onClick={() => setActiveTab("workbench")}>{tt("工作台", "Workbench")}</button>
        <button type="button" onClick={() => setActiveTab("tasks")}>{tt("任务", "Tasks")}</button>
        <button type="button" onClick={() => setLanguage((prev) => (prev === "zh" ? "en" : "zh"))}>
          {isZh ? "English" : "中文"}
        </button>
        <button type="button" onClick={onExit} disabled={exiting} style={{ marginLeft: "auto" }}>
          {exiting ? tt("退出中...", "Exiting...") : tt("退出", "Exit")}
        </button>
      </div>
      <p>{tt("导入本地文件夹作为音频库，然后基于该音频库进行句子活字印刷。", "Import a folder as audio base, then build sentence-mixed clips from that base.")}</p>
      {health && (
        <p>
          {tt("ASR 运行时：", "ASR runtime:")} <strong>{health.asr_resolved_device.toUpperCase()}</strong> ({health.asr_compute_type}) | {tt("偏好", "preferred") }={health.asr_preferred_device} | {tt("最近", "last") }={health.asr_last_device_used}
        </p>
      )}

      {activeTab === "workbench" && (
      <>
      <section style={{ marginBottom: 24 }}>
        <h2>{tt("导入音频库", "Import Audio Base")}</h2>
        <form onSubmit={onImportSubmit}>
          <input
            value={baseName}
            onChange={(event) => setBaseName(event.target.value)}
            placeholder={tt("音频库名称，例如 speaker_a", "base name, e.g. speaker_a")}
            style={{ width: "100%", marginBottom: 8 }}
          />
          <input
            value={sourceFolderPath}
            onChange={(event) => setSourceFolderPath(event.target.value)}
            placeholder={tt("本地文件夹路径，例如 E:\\Recordings\\Henry", "local folder path, e.g. E:\\Recordings\\Henry")}
            style={{ width: "100%", marginBottom: 8 }}
          />
          <div style={{ marginBottom: 8 }}>{tt("后端会从该本地路径扫描 .wav/.mp3 文件。", "Backend will scan .wav/.mp3 from this local folder path.")}</div>
          <button type="submit" disabled={importing || !baseName.trim() || !sourceFolderPath.trim()}>
            {importing ? tt("导入并索引中...", "Importing and indexing...") : tt("导入音频库", "Import Base")}
          </button>
        </form>
        {importing && importProgressTotal > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ marginBottom: 4 }}>
              {tt("进度", "Progress")}: {importProgressCurrent}/{importProgressTotal}
            </div>
            <progress value={importProgressCurrent} max={importProgressTotal} style={{ width: "100%" }} />
          </div>
        )}
        {importResult && <pre>{importResult}</pre>}
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2>{tt("当前音频库", "Active Audio Base")}</h2>
        <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
          <select value={selectedBase} onChange={onBaseChange} style={{ flex: 1 }}>
            <option value="">{tt("选择音频库...", "Select a base...")}</option>
            {bases.map((base) => (
              <option key={base.base_name} value={base.base_name}>
                {base.base_name}
              </option>
            ))}
          </select>
          <button type="button" onClick={onReAsr} disabled={!selectedBase || reasrLoading}>
            {reasrLoading ? "reASR..." : "reASR"}
          </button>
          <button type="button" onClick={onExportLexicon} disabled={!selectedBase || exportingLexicon}>
            {exportingLexicon ? tt("导出中...", "Exporting...") : tt("导出词库", "Export Lexicon")}
          </button>
        </div>
        {selectedStats && (
          <div>
            <div>{tt("音频数量", "Audio count")}: {selectedStats.audio_count}</div>
            <div>{tt("总时长", "Total duration")}: {selectedStats.total_duration_sec.toFixed(1)} s</div>
            <div>{tt("总文件大小", "Total file size")}: {formatBytes(selectedStats.total_file_size_bytes)}</div>
          </div>
        )}
      </section>

      <h2>{tt("创建活字印刷", "Create Mix")}</h2>
      <form onSubmit={onSubmit}>
        <textarea
          rows={4}
          value={sentence}
          onChange={(event) => setSentence(event.target.value)}
          style={{ width: "100%" }}
          placeholder={tt("在这里输入目标句子...", "Type sentence here...")}
        />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8, marginBottom: 8 }}>
          <label>
            {tt("输出模式", "Output Mode")}
            <select
              value={mixOutputMode}
              onChange={(event) => setMixOutputMode(event.target.value as MixOutputMode)}
              style={{ width: "100%" }}
            >
              <option value="mix">{tt("拼接输出", "Mixed Output")}</option>
              <option value="segment_output">{tt("片段输出", "Segment Output")}</option>
            </select>
          </label>
          <label>
            {tt("变速 (x)", "Speed (x)")}
            <input
              type="number"
              min={0.1}
              step={0.1}
              value={mixSpeed}
              onChange={(event) => setMixSpeed(Number(event.target.value))}
              style={{ width: "100%" }}
              disabled={mixOutputMode === "segment_output"}
            />
            {mixOutputMode !== "segment_output" && mixSpeedError && <div style={{ color: "crimson", fontSize: 12 }}>{mixSpeedError}</div>}
          </label>
          <label>
            {tt("间隔 (ms)", "Gap (ms)")}
            <input
              type="number"
              min={0}
              step={10}
              value={mixGapMs}
              onChange={(event) => setMixGapMs(Number(event.target.value))}
              style={{ width: "100%" }}
              disabled={mixOutputMode === "segment_output"}
            />
            {mixOutputMode !== "segment_output" && mixGapError && <div style={{ color: "crimson", fontSize: 12 }}>{mixGapError}</div>}
          </label>
          <label>
            {tt("拼接模式", "Mix Mode")}
            <select
              value={mixMode}
              onChange={(event) => setMixMode(event.target.value as MixMode)}
              style={{ width: "100%" }}
            >
              <option value="word">{tt("词级", "Word")}</option>
              <option value="word_phrase">{tt("词 + 短语", "Word + Phrase")}</option>
              <option value="word_phrase_sentence">{tt("词 + 短语 + 句子", "Word + Phrase + Sentence")}</option>
            </select>
          </label>
          {mixOutputMode === "segment_output" ? (
            <label>
              {tt("片段扩充时长 (ms)", "Segment Expansion (ms)")}
              <input
                type="number"
                min={0}
                step={10}
                value={segmentExpansionMs}
                onChange={(event) => setSegmentExpansionMs(Number(event.target.value))}
                style={{ width: "100%" }}
              />
              {segmentExpansionError && <div style={{ color: "crimson", fontSize: 12 }}>{segmentExpansionError}</div>}
            </label>
          ) : (
            <label>
              {tt("补尾随机上限 (ms)", "Tail Extension Max (ms)")}
              <input
                type="number"
                min={0}
                step={1}
                value={tailExtensionMs}
                onChange={(event) => setTailExtensionMs(Number(event.target.value))}
                style={{ width: "100%" }}
              />
              {tailExtensionError && <div style={{ color: "crimson", fontSize: 12 }}>{tailExtensionError}</div>}
            </label>
          )}
        </div>
        <button type="submit" disabled={loading || !selectedBase || sentence.trim().length === 0 || hasMixInputError}>
          {loading ? tt("提交中...", "Submitting...") : tt("创建活字印刷任务", "Create Mix Job")}
        </button>
      </form>
      {result && <pre>{result}</pre>}
      </>
      )}

      {activeTab === "tasks" && (
        <section>
          <h2>{tt("任务队列", "Task Queue")}</h2>
          <p>{tt("索引任务为串行执行，一次只运行一个。", "One indexing task runs at a time.")}</p>
          {tasks.length === 0 && <div>{tt("暂无任务。", "No tasks yet.")}</div>}
          {tasks.map((task) => (
            <div key={task.task_id} style={{ border: "1px solid #ccc", padding: 8, marginBottom: 8 }}>
              <div><strong>{task.base_name}</strong> [{formatTaskStatus(task.status)}] {formatStage(task.stage)}</div>
              {task.stage === "vad" ? (
                <>
                  <div>{tt("VAD 用时", "VAD Elapsed")}: {(task.vad_elapsed_sec ?? 0).toFixed(1)}s</div>
                  <div>{tt("音频总时长", "Audio Duration")}: {(task.vad_total_audio_sec ?? 0).toFixed(1)}s</div>
                </>
              ) : (
                <>
                  <div>{tt("ASR 用时", "ASR Elapsed")}: {(task.asr_elapsed_sec ?? 0).toFixed(1)}s</div>
                  <div>
                    {tt("ASR 进度", "ASR Progress")}: {(task.asr_processed_audio_sec ?? 0).toFixed(1)}s/
                    {(task.asr_total_audio_sec ?? 0).toFixed(1)}s
                  </div>
                  {(task.asr_total_audio_sec ?? 0) > 0 && (
                    <progress
                      {...clampProgress(task.asr_processed_audio_sec ?? 0, task.asr_total_audio_sec ?? 0)}
                      style={{ width: "100%", marginTop: 4 }}
                    />
                  )}
                  <div>{tt("累计 token", "Accumulated tokens")}: {task.token_count}</div>
                </>
              )}
              {task.last_error && <div style={{ color: "crimson" }}>{tt("错误", "Error")}: {task.last_error}</div>}
              {task.status === "running" && task.stage !== "asr" && (
                <button type="button" onClick={() => onPauseTask(task.task_id)}>{tt("暂停", "Pause")}</button>
              )}
              {(task.status === "paused" || task.status === "failed" || task.status === "queued") && (
                <button type="button" onClick={() => onResumeTask(task.task_id)}>{tt("继续", "Resume")}</button>
              )}
              <button type="button" onClick={() => onDeleteTask(task.task_id, task.base_name)} style={{ marginLeft: 8 }}>
                {tt("删除", "Delete")}
              </button>
            </div>
          ))}
        </section>
      )}

      <section style={{ marginTop: 20 }}>
        <h2>{tt("控制台", "Console")}</h2>
        <div style={{ fontSize: 13, color: "#666", marginBottom: 6 }}>
          {tt("显示导入 / VAD / ASR 流程运行日志。", "Runtime logs from import/VAD/ASR pipeline.")}
        </div>
        <pre style={{ maxHeight: 220, overflow: "auto", background: "#111", color: "#ddd", padding: 10 }}>
          {importLogs.length > 0 ? importLogs.join("\n") : tt("[SYSTEM] 暂无日志。", "[SYSTEM] No logs yet.")}
        </pre>
      </section>
    </main>
  );
}

export default App;

