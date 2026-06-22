package dgxbridge

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// makeTranscript builds a minimal bridge-shaped JSONL from a list of
// (event, text) pairs, so tests exercise the real transcriptStdout filter
// (process_output only) deterministically.
func makeTranscript(t *testing.T, evs [][2]string) []byte {
	t.Helper()
	var sb strings.Builder
	for _, e := range evs {
		b, err := json.Marshal(map[string]string{"event": e[0], "text": e[1]})
		if err != nil {
			t.Fatal(err)
		}
		sb.Write(b)
		sb.WriteByte('\n')
	}
	return []byte(sb.String())
}

func TestTranscriptStdout_OnlyProcessOutput(t *testing.T) {
	jsonl := makeTranscript(t, [][2]string{
		{"stdin", "echo SHOULD_NOT_APPEAR"},          // stdin echo must be excluded
		{"process_output", "real-output-line-1\n"},   // included
		{"control_command", "!dump"},                  // excluded
		{"process_output", "real-output-line-2\n"},   // included
	})
	got := transcriptStdout(jsonl)
	if strings.Contains(got, "SHOULD_NOT_APPEAR") {
		t.Fatalf("stdin echo leaked into stdout reconstruction: %q", got)
	}
	if !strings.Contains(got, "real-output-line-1") || !strings.Contains(got, "real-output-line-2") {
		t.Fatalf("process_output text missing: %q", got)
	}
}

func TestTranscriptStdout_StripsANSI(t *testing.T) {
	// A line with cursor controls + a tmux-style redraw around the payload.
	noisy := "\x1b[1;23r\x1b[1;1HHELLO_\x1b[KWORLD\x1b[1;24r\r"
	jsonl := makeTranscript(t, [][2]string{{"process_output", noisy}})
	got := transcriptStdout(jsonl)
	if !strings.Contains(got, "HELLO_WORLD") {
		t.Fatalf("ANSI not stripped to HELLO_WORLD: %q", got)
	}
	if strings.Contains(got, "\x1b") || strings.Contains(got, "\r") {
		t.Fatalf("residual control bytes: %q", got)
	}
}

func TestExtractBlock_IgnoresCommandEcho(t *testing.T) {
	nonce := "RPC42"
	// The PTY echoes the command line (which literally contains the nonce AND
	// nonce_DONE), then prints the real output, then the sentinels on their own.
	stdout := strings.Join([]string{
		"echo " + nonce + "; { uname ; } ; echo " + nonce + "_DONE", // command echo
		nonce,                  // START sentinel (output)
		"Linux dgx2 6.8.0",     // real output
		nonce + "_DONE",        // END sentinel (output)
		"root@dgx2:/#",
	}, "\n")
	block, ok := extractBlock(stdout, nonce)
	if !ok {
		t.Fatal("expected ok=true")
	}
	if strings.TrimSpace(block) != "Linux dgx2 6.8.0" {
		t.Fatalf("block should be just the output, got %q", block)
	}
}

func TestExtractBlock_NoDoneSentinel(t *testing.T) {
	if _, ok := extractBlock("partial output, command still running", "RPC1"); ok {
		t.Fatal("expected ok=false when _DONE absent")
	}
}

func TestExtractBlock_MultilineOutput(t *testing.T) {
	nonce := "RPC7"
	stdout := nonce + "\nline-a\nline-b\nline-c\n" + nonce + "_DONE"
	block, ok := extractBlock(stdout, nonce)
	if !ok {
		t.Fatal("ok=false")
	}
	for _, want := range []string{"line-a", "line-b", "line-c"} {
		if !strings.Contains(block, want) {
			t.Fatalf("missing %q in %q", want, block)
		}
	}
}

// TestExtractBlock_OutputContainsNonce reproduces the live readback bug
// (2026-06-20): SelfTest runs `echo SELFTEST_<nonce>`, so the OUTPUT line itself
// contains the nonce as a substring. The old substring-scan latched onto that
// occurrence as the START sentinel and returned an empty block -> echo_mismatch.
// Whole-line sentinel matching must keep the payload.
func TestExtractBlock_OutputContainsNonce(t *testing.T) {
	nonce := "RPC26739100"
	payload := "SELFTEST_" + nonce // the SelfTest payload embeds the nonce
	// Exactly the process_output sequence seen on the wire: bare-nonce line,
	// payload line (contains nonce), nonce_DONE line.
	stdout := nonce + "\n" + payload + "\n" + nonce + "_DONE\n"
	block, ok := extractBlock(stdout, nonce)
	if !ok {
		t.Fatal("expected ok=true")
	}
	if strings.TrimSpace(block) != payload {
		t.Fatalf("payload dropped: got %q want %q", block, payload)
	}
}

// TestExtractBlock_DoneSubstringInOutput guards the END side of the same class:
// an output line that merely CONTAINS nonce_DONE as a substring must not be
// mistaken for the sentinel line.
func TestExtractBlock_DoneSubstringInOutput(t *testing.T) {
	nonce := "RPC55"
	stdout := nonce + "\nprefix-" + nonce + "_DONE-suffix\nreal-out\n" + nonce + "_DONE\n"
	block, ok := extractBlock(stdout, nonce)
	if !ok {
		t.Fatal("expected ok=true")
	}
	if !strings.Contains(block, "real-out") {
		t.Fatalf("real output missing: %q", block)
	}
}

func TestDecodeFileBlock(t *testing.T) {
	nonce := "RPC9"
	payload := []byte("{\"hello\":\"world\",\"n\":123}\n")
	b64 := base64.StdEncoding.EncodeToString(payload)
	// Simulate PTY chunking: inject stray control-ish chars the alphabet filter drops.
	chunked := b64[:10] + " \n " + b64[10:]
	out := fileStart(nonce) + "\n" + chunked + "\n" + fileEnd(nonce)
	got, err := decodeFileBlock(out, nonce)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if string(got) != string(payload) {
		t.Fatalf("roundtrip mismatch: got %q want %q", got, payload)
	}
}

func TestDecodeFileBlock_MissingMarkers(t *testing.T) {
	if _, err := decodeFileBlock("no markers here", "RPCx"); err == nil {
		t.Fatal("expected error on missing markers")
	}
}

func TestHTMLUnescape(t *testing.T) {
	if got := htmlUnescape("a &gt; b &amp;&amp; c &lt; d"); got != "a > b && c < d" {
		t.Fatalf("got %q", got)
	}
}

// TestTranscriptStdout_RealFixture asserts the parser survives a real captured
// PTY transcript and recovers known content (the A100 GPU listing) without panicking.
func TestTranscriptStdout_RealFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/real_transcript.jsonl")
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	out := transcriptStdout(data)
	if !strings.Contains(out, "A100") {
		t.Fatalf("expected A100 in recovered stdout (len=%d)", len(out))
	}
	// Sanity: the sentinel tokens known to exist in this fixture survive parsing.
	if !strings.Contains(out, "_DONE") {
		t.Fatal("expected a _DONE sentinel in the real transcript")
	}
}

// makeThreadTranscript builds a JSONL transcript whose events carry thread_ts,
// so the thread-routing tests exercise transcriptThreadTS on realistic shapes.
// Each ev is {event, text, thread_ts}; a "" thread_ts omits the field (old-format).
func makeThreadTranscript(t *testing.T, evs [][3]string) []byte {
	t.Helper()
	var sb strings.Builder
	for _, e := range evs {
		m := map[string]string{"event": e[0], "text": e[1]}
		if e[2] != "" {
			m["thread_ts"] = e[2]
		}
		b, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		sb.Write(b)
		sb.WriteByte('\n')
	}
	return []byte(sb.String())
}

// The bridge_start event is the authoritative session identity, so it wins even
// when an earlier or later event carries a different thread_ts (e.g. a stray reply
// from another session quoted into this transcript).
func TestTranscriptThreadTS_PrefersBridgeStart(t *testing.T) {
	jsonl := makeThreadTranscript(t, [][3]string{
		{"process_output", "noise", "9999.0000"}, // earlier, non-authoritative
		{"bridge_start", "banner", "1234.5678"},   // authoritative
		{"process_output", "more", "8888.0000"},   // later, non-authoritative
	})
	if got := transcriptThreadTS(jsonl); got != "1234.5678" {
		t.Fatalf("bridge_start thread should win, got %q", got)
	}
}

// With no bridge_start, the first event that carries a thread_ts is used.
func TestTranscriptThreadTS_FallbackFirstWithThreadTS(t *testing.T) {
	jsonl := makeThreadTranscript(t, [][3]string{
		{"process_output", "no-thread-here", ""}, // skipped: no thread_ts
		{"process_output", "first-with-ts", "5555.1111"},
		{"process_output", "second", "6666.2222"},
	})
	if got := transcriptThreadTS(jsonl); got != "5555.1111" {
		t.Fatalf("expected first thread_ts as fallback, got %q", got)
	}
}

// An old-format transcript with no thread_ts anywhere returns "", which is the
// signal transcriptForThread uses to fall back to the newest candidate (no regression
// for legacy single-session bridges).
func TestTranscriptThreadTS_OldFormatNone(t *testing.T) {
	jsonl := makeThreadTranscript(t, [][3]string{
		{"bridge_start", "banner", ""},
		{"process_output", "output", ""},
	})
	if got := transcriptThreadTS(jsonl); got != "" {
		t.Fatalf("old-format transcript should yield empty thread, got %q", got)
	}
}

// SelfTest's typed reasons must map 1:1 onto the readback errors Exec returns, so
// an orchestrator can branch on the cause. Guard the wiring from the error sentinels
// to the Reason* strings (the cmd layer's readbackHint switches on exactly these).
func TestSelfTestReasons_DistinctAndWired(t *testing.T) {
	reasons := []string{ReasonNoSessionTranscript, ReasonSentinelMissing, ReasonEchoMismatch, ReasonExecError}
	seen := map[string]bool{}
	for _, r := range reasons {
		if r == "" {
			t.Fatal("a SelfTest reason is empty")
		}
		if seen[r] {
			t.Fatalf("duplicate SelfTest reason %q", r)
		}
		seen[r] = true
	}
}

func TestShellQuote(t *testing.T) {
	cases := map[string]string{
		"/tmp/x.json":   "'/tmp/x.json'",
		"a b":           "'a b'",
		"it's":          `'it'\''s'`,
	}
	for in, want := range cases {
		if got := shellQuote(in); got != want {
			t.Errorf("shellQuote(%q)=%q want %q", in, got, want)
		}
	}
}
