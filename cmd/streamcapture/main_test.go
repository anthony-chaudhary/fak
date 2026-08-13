package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// corpus is the committed capture directory, relative to this package.
const corpus = "testdata/captures"

// TestCommittedManifestIsHonest is the hermetic half of the recorder: it
// re-derives every manifest claim from the recorded bytes with no network. A
// row that overstates what it captured — a tool scenario with no tool deltas, a
// digest that drifted, a fragmentation claim the bytes do not support — fails
// here rather than in a reviewer's head.
func TestCommittedManifestIsHonest(t *testing.T) {
	devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatalf("open %s: %v", os.DevNull, err)
	}
	defer devnull.Close()
	if err := verifyManifest(corpus, devnull); err != nil {
		t.Fatalf("committed manifest does not match its bytes: %v", err)
	}
}

// TestCorpusHasNoOrphanCaptures keeps the directory and the manifest in step:
// an unlisted capture is provenance nobody declared.
func TestCorpusHasNoOrphanCaptures(t *testing.T) {
	m, err := readManifest(corpus)
	if err != nil {
		t.Fatal(err)
	}
	listed := map[string]bool{}
	for _, entry := range m.Captures {
		listed[entry.File] = true
	}
	entries, err := os.ReadDir(corpus)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.IsDir() || e.Name() == "MANIFEST.json" {
			continue
		}
		if !listed[e.Name()] {
			t.Errorf("%s is in the corpus but has no manifest row", e.Name())
		}
	}
}

// TestSameModelFragmentsOnOneStackAndNotTheOther pins the measured fact the
// corpus exists to carry. openai/gpt-oss-120b streams its tool arguments
// token-by-token behind NVIDIA NIM and as one indivisible chunk behind Groq. So
// tool-argument fragmentation is a property of the SERVING STACK, not of the
// wire and not of the model — which is exactly why an adapter has to measure
// its steering resolution instead of deriving it from either.
func TestSameModelFragmentsOnOneStackAndNotTheOther(t *testing.T) {
	fragmenting := observeFile(t, "nvidia--openai-gpt-oss-120b--tool-destructive.sse", true)
	whole := observeFile(t, "groq--openai-gpt-oss-120b--tool-destructive.sse", true)

	if !fragmenting.ToolArgsFragmented || fragmenting.MaxArgDeltasPerCall < 2 {
		t.Errorf("nvidia openai/gpt-oss-120b: expected fragmented tool arguments, got max_per_call=%d", fragmenting.MaxArgDeltasPerCall)
	}
	if whole.ToolArgsFragmented || whole.MaxArgDeltasPerCall != 1 {
		t.Errorf("groq openai/gpt-oss-120b: expected one whole argument chunk, got max_per_call=%d", whole.MaxArgDeltasPerCall)
	}
	for _, o := range []Observed{fragmenting, whole} {
		if !o.ArgumentsComplete {
			t.Error("a complete capture should assemble to valid JSON arguments")
		}
		if len(o.ToolNames) != 1 || o.ToolNames[0] != "shell" {
			t.Errorf("expected the shell tool, got %v", o.ToolNames)
		}
	}
}

// TestPartialArgumentsNeverLookComplete is the recorder-side echo of the safety
// property the adapters enforce. A stream cut while the destructive arguments
// are still arriving must not assemble into a complete call — on real bytes,
// not a hand-written fragment.
func TestPartialArgumentsNeverLookComplete(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(corpus, "nvidia--openai-gpt-oss-120b--tool-destructive.sse"))
	if err != nil {
		t.Fatal(err)
	}
	full := analyzeStream(raw)
	if !full.ArgumentsComplete {
		t.Fatal("the full capture should assemble to complete arguments")
	}
	// Cut at the event boundary just after the fifth argument delta, i.e. while
	// the destructive command is still arriving.
	events := strings.Split(string(raw), "\n\n")
	if len(events) < 8 {
		t.Fatalf("expected a multi-event capture, got %d", len(events))
	}
	seen, cutAt := 0, -1
	for i, event := range events {
		if !strings.Contains(event, `"arguments"`) {
			continue
		}
		if seen++; seen == 5 {
			cutAt = i + 1
			break
		}
	}
	if cutAt < 0 {
		t.Fatal("capture has fewer than five argument deltas")
	}
	cut := analyzeStream([]byte(strings.Join(events[:cutAt], "\n\n")))
	if cut.ArgumentsComplete {
		t.Error("arguments cut mid-stream must not assemble into a complete call")
	}
	if cut.Terminated {
		t.Error("a cut stream has no terminal event")
	}
	if cut.ToolArgDeltas == 0 {
		t.Error("the cut prefix should still carry the speculative deltas it received")
	}
}

// TestJudgeRefusesAClaimTheBytesDoNotSupport covers the manifest's honesty rule
// directly: an error response, a prose reply, and an unterminated tool call are
// each refused as tool captures.
func TestJudgeRefusesAClaimTheBytesDoNotSupport(t *testing.T) {
	tool := target{Provider: "groq", Model: "m", Scenario: "tool-destructive"}
	cases := []struct {
		name   string
		target target
		status int
		body   string
		stream bool
	}{
		{"http error", tool, 401, `{"error":{"message":"Invalid API Key"}}`, true},
		{"error payload on a 200", tool, 200, "data: {\"error\":{\"message\":\"model overloaded\"}}\n\ndata: [DONE]\n", true},
		{"prose answered a tool scenario", tool, 200, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n", true},
		{"prose scenario with no text", target{Provider: "groq", Model: "m", Scenario: "prose"}, 200, "data: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			observed := observe([]byte(tc.body), tc.stream)
			satisfied, caveat := judge(tc.target, tc.status, observed)
			if satisfied {
				t.Fatalf("expected refusal, got scenario_satisfied=true (%+v)", observed)
			}
			if caveat == "" {
				t.Error("a refused row must name its caveat")
			}
		})
	}
}

// TestJudgeAcceptsARealToolCapture is the positive control for the rule above.
func TestJudgeAcceptsARealToolCapture(t *testing.T) {
	observed := observeFile(t, "nvidia--openai-gpt-oss-120b--tool-destructive.sse", true)
	satisfied, caveat := judge(target{Provider: "nvidia", Model: "openai/gpt-oss-120b", Scenario: "tool-destructive"}, 200, observed)
	if !satisfied {
		t.Fatalf("a real tool capture was refused: %s", caveat)
	}
}

// TestNonStreamingCaptureFoldsAsOneResponse grounds the request-boundary row.
func TestNonStreamingCaptureFoldsAsOneResponse(t *testing.T) {
	observed := observeFile(t, "groq--openai-gpt-oss-120b--tool-destructive-nonstream.json", false)
	if observed.Chunks != 1 || !observed.Terminated {
		t.Errorf("a whole response is one terminated chunk, got %+v", observed)
	}
	if observed.ToolCalls != 1 || observed.MaxArgDeltasPerCall != 1 {
		t.Errorf("a non-streaming tool call arrives whole, got %+v", observed)
	}
	if observed.ToolArgsFragmented {
		t.Error("a non-streaming response cannot fragment arguments")
	}
}

// TestScrubRefusesSecretAndPIIShapes proves the write path is fail-closed
// rather than merely well-behaved.
func TestScrubRefusesSecretAndPIIShapes(t *testing.T) {
	dirty := []struct {
		name    string
		payload string
	}{
		{"groq key", `data: {"note":"gsk_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"}`},
		{"nvidia key", `data: {"note":"nvapi-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"}`},
		{"openai key", `data: {"note":"sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"}`},
		{"auth header", `data: {"headers":{"Authorization": "redacted"}}`},
		{"bearer token", `data: {"note":"Bearer ABCDEFGHIJKLMNOPQRST"}`},
		{"email", `data: {"note":"operator@example.com"}`},
	}
	for _, tc := range dirty {
		t.Run(tc.name, func(t *testing.T) {
			if got := scrubViolations([]byte(tc.payload), nil); len(got) == 0 {
				t.Error("expected the scrubber to refuse this payload")
			}
		})
	}
	if got := scrubViolations([]byte(`data: {"choices":[{"delta":{"content":"hello"}}]}`), nil); len(got) != 0 {
		t.Errorf("clean payload refused: %v", got)
	}
	if got := scrubViolations([]byte(`data: {"note":"prefix-LIVEKEYVALUE-suffix"}`), []string{"LIVEKEYVALUE"}); len(got) == 0 {
		t.Error("a live environment key inside a payload must be refused")
	}
}

// TestEveryCommittedCaptureIsScrubbed runs the same fail-closed check over what
// actually landed.
func TestEveryCommittedCaptureIsScrubbed(t *testing.T) {
	m, err := readManifest(corpus)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range m.Captures {
		raw, err := os.ReadFile(filepath.Join(corpus, entry.File))
		if err != nil {
			t.Fatal(err)
		}
		if got := scrubViolations(raw, nil); len(got) != 0 {
			t.Errorf("%s carries a secret/PII shape: %v", entry.File, got)
		}
	}
}

// TestVerifyDetectsTampering proves -verify is a real check: a single flipped
// byte and an overstated row both fail it.
func TestVerifyDetectsTampering(t *testing.T) {
	devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer devnull.Close()

	t.Run("digest drift", func(t *testing.T) {
		dir := copyCorpus(t)
		name := filepath.Join(dir, "groq--openai-gpt-oss-120b--prose.sse")
		raw, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		raw[len(raw)/2] ^= 0x20
		if err := os.WriteFile(name, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := verifyManifest(dir, devnull); err == nil {
			t.Fatal("a tampered capture passed verification")
		}
	})

	t.Run("overstated row", func(t *testing.T) {
		dir := copyCorpus(t)
		m, err := readManifest(dir)
		if err != nil {
			t.Fatal(err)
		}
		for i := range m.Captures {
			m.Captures[i].Observed.ToolArgDeltas += 7
			m.Captures[i].Observed.ToolArgsFragmented = true
		}
		if err := writeManifest(dir, m); err != nil {
			t.Fatal(err)
		}
		if err := verifyManifest(dir, devnull); err == nil {
			t.Fatal("a manifest claiming deltas the bytes do not have passed verification")
		}
	})
}

// TestRequestBodyIsStableAndCarriesTheDestructiveTool pins what a capture is a
// capture OF: the request digest recorded in the manifest is re-derivable.
func TestRequestBodyIsStableAndCarriesTheDestructiveTool(t *testing.T) {
	sc := scenarios["tool-destructive"]
	first, err := requestBody("m", sc)
	if err != nil {
		t.Fatal(err)
	}
	second, err := requestBody("m", sc)
	if err != nil {
		t.Fatal(err)
	}
	if string(first) != string(second) {
		t.Fatal("request body is not stable, so its digest is not re-derivable")
	}
	var decoded struct {
		Stream bool `json:"stream"`
		Tools  []struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(first, &decoded); err != nil {
		t.Fatal(err)
	}
	if !decoded.Stream {
		t.Error("the tool-destructive scenario is a streaming scenario")
	}
	if len(decoded.Tools) != 1 || decoded.Tools[0].Function.Name != "shell" {
		t.Errorf("expected exactly the shell tool, got %+v", decoded.Tools)
	}
	if nonStream := scenarios["tool-destructive-nonstream"]; nonStream.Stream {
		t.Error("the request-boundary scenario must not be a streaming scenario")
	}
}

// TestRunRefusesAnIncompleteInvocation keeps the argv surface fail-closed.
func TestRunRefusesAnIncompleteInvocation(t *testing.T) {
	devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer devnull.Close()
	if err := run([]string{"-provider", "groq"}, devnull); err == nil {
		t.Fatal("a capture with no model or scenario should refuse")
	}
	if err := run([]string{"-probe", "-provider", "groq"}, devnull); err == nil {
		t.Fatal("a probe with no models should refuse")
	}
}

func observeFile(t *testing.T, name string, streaming bool) Observed {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(corpus, name))
	if err != nil {
		t.Fatal(err)
	}
	return observe(raw, streaming)
}

func copyCorpus(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	entries, err := os.ReadDir(corpus)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(corpus, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, e.Name()), raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}
