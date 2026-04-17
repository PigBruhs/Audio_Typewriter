import { FormEvent, useState } from "react";
import { requestMix } from "./api";

function App(): JSX.Element {
  const [sentence, setSentence] = useState("");
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setResult("");
    try {
      const data = await requestMix(sentence);
      setResult(`job=${data.job_id}, status=${data.status}`);
    } catch (error) {
      setResult(error instanceof Error ? error.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 680, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Audio Typewriter</h1>
      <p>Enter a target sentence to build a sentence-mixed clip.</p>
      <form onSubmit={onSubmit}>
        <textarea
          rows={4}
          value={sentence}
          onChange={(event) => setSentence(event.target.value)}
          style={{ width: "100%" }}
          placeholder="Type sentence here..."
        />
        <button type="submit" disabled={loading || sentence.trim().length === 0}>
          {loading ? "Submitting..." : "Create Mix Job"}
        </button>
      </form>
      {result && <pre>{result}</pre>}
    </main>
  );
}

export default App;

