package guardtrace

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	// Complete() stores payloads through abi.ActiveResolver(); the blob backend's
	// init() registers the RegionBackend so the resolver is non-nil.
	_ "github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// twoTurnFixture is a small, well-formed Fixture with real per-turn usage and a
// multi-call turn, used to exercise ToBenchTrace/ToCassette/splitTurnUsage.
func twoTurnFixture() *Fixture {
	return &Fixture{
		SliceID: "demo-session",
		Turns: []Turn{
			{
				Usage: Usage{InputTokens: 100, OutputTokens: 20, CacheReadInputTokens: 10, CacheCreationInputTokens: 4},
				Calls: []Call{
					{Tool: "Read", Args: json.RawMessage(`{"file_path":"a.go"}`), Class: "allow"},
				},
			},
			{
				// Two calls sharing one turn's usage — exercises the even-split with a
				// remainder (101/2 => 51,50).
				Usage: Usage{InputTokens: 101, OutputTokens: 21, CacheReadInputTokens: 11, CacheCreationInputTokens: 5},
				Calls: []Call{
					{Tool: "Bash", Args: json.RawMessage(`{"command":"ls"}`), Class: "allow"},
					{Tool: "Write", Args: json.RawMessage(`{"file_path":"b.go"}`), Class: "deny", Reason: "POLICY_BLOCK"},
				},
			},
		},
	}
}

// ToBenchTrace flattens every turn's calls in order, and its SliceID is bound to
// the fixture's SliceID under the SessionTracePrefix — the provenance/workload
// hash acceptance criterion for #1846.
func TestToBenchTraceFlattensAndPrefixesSliceID(t *testing.T) {
	f := twoTurnFixture()
	tr := f.ToBenchTrace()

	if want := SessionTracePrefix + "demo-session"; tr.SliceID != want {
		t.Errorf("SliceID = %q, want %q", tr.SliceID, want)
	}
	if len(tr.Calls) != 3 {
		t.Fatalf("want 3 flattened calls, got %d", len(tr.Calls))
	}
	wantTools := []string{"Read", "Bash", "Write"}
	for i, c := range tr.Calls {
		if c.Tool != wantTools[i] {
			t.Errorf("call[%d].Tool = %q, want %q", i, c.Tool, wantTools[i])
		}
	}
	// The bench.Call carries the original args bytes through unmodified.
	if string(tr.Calls[0].Args) != `{"file_path":"a.go"}` {
		t.Errorf("call[0].Args = %s, want passthrough", tr.Calls[0].Args)
	}
}

// A blank fixture SliceID falls back to "captured" so ToBenchTrace never emits a
// bare "session:" prefix with nothing after it.
func TestToBenchTraceBlankSliceIDFallsBackToCaptured(t *testing.T) {
	f := &Fixture{Turns: []Turn{{Calls: []Call{{Tool: "Read", Class: "allow"}}}}}
	tr := f.ToBenchTrace()
	if want := SessionTracePrefix + "captured"; tr.SliceID != want {
		t.Errorf("SliceID = %q, want %q", tr.SliceID, want)
	}
}

// ToCassette must carry the REAL usage split onto each call's cassette entry, and
// a multi-call turn's usage is evenly distributed across the turn's calls, with
// the remainder on the first calls — the sum reconstructs the turn total exactly
// (acceptance #2 of #1846: token columns match observed usage).
func TestToCassetteCarriesRealUsagePerCall(t *testing.T) {
	f := twoTurnFixture()
	cas := f.ToCassette()
	eng := engine.NewCassetteEngine(cas)
	_ = eng // constructed to prove ToCassette's output type-checks against NewCassetteEngine

	// Turn 0: single call gets the whole turn's usage.
	k0 := engine.CallKey("Read", []byte(`{"file_path":"a.go"}`))
	// Turn 1: two calls split 101 in / 21 out / 11 cacheRead / 5 cacheCreate.
	// dist(101,2) = [51,50]; dist(21,2)=[11,10]; dist(11,2)=[6,5]; dist(5,2)=[3,2].
	k1 := engine.CallKey("Bash", []byte(`{"command":"ls"}`))
	k2 := engine.CallKey("Write", []byte(`{"file_path":"b.go"}`))

	entries := cassetteEntriesByKey(t, cas)
	if got := entries[k0]; got.InputTokens != 100 || got.OutputTokens != 20 ||
		got.CacheReadTokens != 10 || got.CacheCreationTokens != 4 {
		t.Errorf("turn-0 call usage = %+v, want the whole turn's usage", got)
	}
	e1, e2 := entries[k1], entries[k2]
	if e1.InputTokens+e2.InputTokens != 101 {
		t.Errorf("turn-1 input split sums to %d, want 101", e1.InputTokens+e2.InputTokens)
	}
	if e1.OutputTokens+e2.OutputTokens != 21 {
		t.Errorf("turn-1 output split sums to %d, want 21", e1.OutputTokens+e2.OutputTokens)
	}
	if e1.CacheReadTokens+e2.CacheReadTokens != 11 {
		t.Errorf("turn-1 cache_read split sums to %d, want 11", e1.CacheReadTokens+e2.CacheReadTokens)
	}
	if e1.CacheCreationTokens+e2.CacheCreationTokens != 5 {
		t.Errorf("turn-1 cache_creation split sums to %d, want 5", e1.CacheCreationTokens+e2.CacheCreationTokens)
	}
	// The remainder lands on the FIRST call (i=0): 51 in, 11 out, 6 cacheRead, 3 cacheCreate.
	if e1.InputTokens != 51 || e1.OutputTokens != 11 || e1.CacheReadTokens != 6 || e1.CacheCreationTokens != 3 {
		t.Errorf("first call of turn-1 usage = %+v, want {51 11 6 3}-shaped remainder-first split", e1)
	}
}

// cassetteEntriesByKey drives the Cassette through its public Complete() surface
// (via CassetteEngine) to recover each entry's Usage, since the internal map is
// unexported — this exercises exactly the path #1846 depends on (bench.RunArm's
// Meta-based token fold), not a white-box peek.
func cassetteEntriesByKey(t *testing.T, cas *engine.Cassette) map[string]engine.Usage {
	t.Helper()
	eng := engine.NewCassetteEngine(cas)
	out := map[string]engine.Usage{}
	for tool, args := range map[string]string{
		"Read":  `{"file_path":"a.go"}`,
		"Bash":  `{"command":"ls"}`,
		"Write": `{"file_path":"b.go"}`,
	} {
		key := engine.CallKey(tool, []byte(args))
		call := &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
		res, err := eng.Complete(context.Background(), call)
		if err != nil {
			t.Fatalf("Complete(%s): %v", tool, err)
		}
		in := mustAtoi(t, res.Meta["input_tokens"])
		out2 := mustAtoi(t, res.Meta["output_tokens"])
		cr := mustAtoi(t, res.Meta["cache_read_tokens"])
		cc := mustAtoi(t, res.Meta["cache_creation_tokens"])
		out[key] = engine.Usage{InputTokens: in, OutputTokens: out2, CacheReadTokens: cr, CacheCreationTokens: cc}
	}
	return out
}

// SessionEngineID is content-addressed: same bytes => same id (idempotent
// re-registration), different bytes => a different id (no collision).
func TestSessionEngineIDContentAddressed(t *testing.T) {
	a := SessionEngineID([]byte(`{"turns":[]}`))
	b := SessionEngineID([]byte(`{"turns":[]}`))
	c := SessionEngineID([]byte(`{"turns":[{}]}`))
	if a != b {
		t.Errorf("SessionEngineID not idempotent: %q != %q", a, b)
	}
	if a == c {
		t.Errorf("SessionEngineID collided for different bytes: %q", a)
	}
	if !strings.HasPrefix(a, "session:") {
		t.Errorf("SessionEngineID = %q, want a session: prefix", a)
	}
}

// LoadSessionTrace is the whole read pipeline: it must load a real fixture from
// disk, bind the trace's SliceID under the session prefix, and hand back a
// cassette + engine id consistent with the raw bytes.
func TestLoadSessionTraceEndToEnd(t *testing.T) {
	raw, err := json.Marshal(twoTurnFixture())
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "session.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	tr, cas, engineID, err := LoadSessionTrace(path)
	if err != nil {
		t.Fatalf("LoadSessionTrace: %v", err)
	}
	if !strings.HasPrefix(tr.SliceID, SessionTracePrefix) {
		t.Errorf("trace SliceID = %q, want session: prefix", tr.SliceID)
	}
	if len(tr.Calls) != 3 {
		t.Errorf("want 3 calls, got %d", len(tr.Calls))
	}
	if cas == nil {
		t.Fatal("cassette is nil")
	}
	wantID := SessionEngineID(raw)
	if engineID != wantID {
		t.Errorf("engineID = %q, want %q (content-addressed on the raw file bytes)", engineID, wantID)
	}
}

// LoadSessionTrace surfaces a missing file and malformed JSON as errors, never a
// silent empty trace.
func TestLoadSessionTraceErrors(t *testing.T) {
	if _, _, _, err := LoadSessionTrace(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Fatal("LoadSessionTrace(missing file) = nil error, want error")
	}
	bad := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(bad, []byte(`{not json`), 0o644); err != nil {
		t.Fatalf("write bad fixture: %v", err)
	}
	if _, _, _, err := LoadSessionTrace(bad); err == nil {
		t.Fatal("LoadSessionTrace(malformed json) = nil error, want error")
	}
}

func mustAtoi(t *testing.T, s string) int {
	t.Helper()
	n := 0
	neg := false
	for i, r := range s {
		if i == 0 && r == '-' {
			neg = true
			continue
		}
		if r < '0' || r > '9' {
			t.Fatalf("mustAtoi(%q): not a plain integer", s)
		}
		n = n*10 + int(r-'0')
	}
	if neg {
		n = -n
	}
	return n
}
