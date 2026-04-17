import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import { AudioBaseItem, getAudioBaseStats, importAudioBase, listAudioBases, requestMix } from "./api";

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
  const [baseName, setBaseName] = useState("");
  const [baseFiles, setBaseFiles] = useState<File[]>([]);
  const [bases, setBases] = useState<AudioBaseItem[]>([]);
  const [selectedBase, setSelectedBase] = useState("");
  const [selectedStats, setSelectedStats] = useState<AudioBaseItem | null>(null);
  const [mixMode, setMixMode] = useState("context_priority");
  const [sentence, setSentence] = useState("");
  const [result, setResult] = useState<string>("");
  const [importResult, setImportResult] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);

  useEffect(() => {
    refreshBases().catch((error: unknown) => {
      setResult(error instanceof Error ? error.message : "Load bases failed");
    });
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

  async function onImportSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!baseName.trim() || baseFiles.length === 0) {
      setImportResult("Please enter base name and select a folder.");
      return;
    }

    setImporting(true);
    setImportResult("");
    try {
      const data = await importAudioBase(baseName.trim(), baseFiles);
      setImportResult(
        `Imported base=${data.base_name}, files=${data.audio_count}, duration=${data.total_duration_sec.toFixed(1)}s, size=${formatBytes(data.total_file_size_bytes)}, tokens=${data.token_count}`
      );
      await refreshBases();
      setSelectedBase(data.base_name);
      const stats = await getAudioBaseStats(data.base_name);
      setSelectedStats(stats);
    } catch (error) {
      setImportResult(error instanceof Error ? error.message : "Import failed");
    } finally {
      setImporting(false);
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
      <p>Import a folder as audio base, then build sentence-mixed clips from that base.</p>

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
    </main>
  );
}

export default App;

