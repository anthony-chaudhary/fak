package gateway

import (
	_ "embed"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/guardtrace"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// guardTraceFloorJSON is a faithful copy of the SHIPPED guard floor's danger classes.
// The cmd/fak replay test asserts the shipped floor and this testdata floor agree, so it
// cannot silently drift. It lives in testdata so this internal/gateway test can fire the
// REAL floor without importing cmd/fak (which would be an import cycle).
//
//go:embed testdata/guard-trace-floor.json
var guardTraceFloorJSON []byte

// newGuardTraceServer wires a gateway whose adjudicator IS the real shipped guard floor
// (parsed from the testdata manifest), pointed at a fake upstream that replays f. Unlike
// newTestServer it registers the production floor, not the toy prefix-matcher — so a
// danger arg (rm -rf, sudo, a write into .ssh) actually FIRES the way it does for a live
// `fak guard -- claude`. provider selects the wire ("anthropic" | "openai").
func newGuardTraceServer(t *testing.T, provider string, f *guardtrace.Fixture) (*Server, *guardtrace.FakeUpstream) {
	t.Helper()
	rt, err := policy.ParseRuntime(guardTraceFloorJSON)
	if err != nil {
		t.Fatalf("parse testdata guard floor: %v", err)
	}

	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, adjudicator.New(rt.Adjudicator)) // the REAL floor, not toolAdj

	const model = "guard-trace:model"
	upstream := guardtrace.NewFakeUpstream(provider, model, f)
	t.Cleanup(upstream.Close)

	srv, err := New(Config{
		EngineID: "test",
		Model:    model,
		Provider: provider,
		BaseURL:  upstreamBaseURL(provider, upstream.URL),
		VDSO:     true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv, upstream
}

// upstreamBaseURL gives the proxy planner the base its adapter appends its endpoint to.
// The OpenAI adapter posts <base>/chat/completions, so the base must carry /v1; the
// Anthropic adapter posts <base>/v1/messages, so the base stays bare. This mirrors
// guardDefaultBaseURL / guardOpenAIV1Base in cmd/fak.
func upstreamBaseURL(provider, host string) string {
	if provider == "openai" {
		return host + "/v1"
	}
	return host
}

// runGuardTrace posts every turn of f through the gateway (one shared trace id), returns
// the per-call verdicts the gateway reported, and the trace id used.
func runGuardTrace(t *testing.T, srv *Server, provider string, f *guardtrace.Fixture) ([]guardtrace.ResponseAdjudication, string) {
	t.Helper()
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	const trace = "guard-trace-session"
	var all []guardtrace.ResponseAdjudication
	for i, turn := range f.Turns {
		raw, _, err := guardtrace.PostTurn(http.DefaultClient, ts.URL, traceHeader, trace, provider, srv.model, turn)
		if err != nil {
			t.Fatalf("turn %d post: %v", i, err)
		}
		adjs, err := guardtrace.DecodeAdjudications(raw)
		if err != nil {
			t.Fatalf("turn %d decode: %v", i, err)
		}
		if len(adjs) != len(turn.Calls) {
			t.Fatalf("turn %d: gateway reported %d adjudications, want %d (one per proposed call)", i, len(adjs), len(turn.Calls))
		}
		all = append(all, adjs...)
	}
	return all, trace
}

// TestGuardTraceFiresFloorAndRecordsJournalAnthropic is the end-to-end witness on the
// flagship `fak guard -- claude` wire: a realistic provider trace, run through the REAL
// proxy planner + the REAL shipped floor, fires the danger classes, the survivors reach
// the caller, the hash-chained decision journal records every deny and verifies, the
// exit AdjudicationSummary matches the fixture, and the token economy lands.
func TestGuardTraceFiresFloorAndRecordsJournalAnthropic(t *testing.T) {
	assertGuardTraceEndToEnd(t, "anthropic")
}

// TestGuardTraceFiresFloorAndRecordsJournalOpenAI is the same end-to-end witness on the
// OpenAI-compatible wire (codex / opencode under `fak guard --provider openai`).
func TestGuardTraceFiresFloorAndRecordsJournalOpenAI(t *testing.T) {
	assertGuardTraceEndToEnd(t, "openai")
}

func assertGuardTraceEndToEnd(t *testing.T, provider string) {
	f, err := guardtrace.LoadFixture(filepath.Join("testdata", "guard-trace-e2e.json"))
	if err != nil {
		t.Fatalf("load fixture: %v", err)
	}
	exp := f.Expect()

	srv, _ := newGuardTraceServer(t, provider, f)

	// Attach a fresh decision journal directly to the (reset) ABI, so the kernel fans
	// every verdict to it. journal.Open + abi.RegisterEmitter sidesteps the global,
	// idempotent journal.Enable, keeping this test self-contained and per-run isolated.
	jpath := filepath.Join(t.TempDir(), "guard-trace-audit.jsonl")
	j, err := journal.Open(jpath)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	defer j.Close()
	abi.RegisterEmitter(j)

	adjs, _ := runGuardTrace(t, srv, provider, f)

	// --- (a) Each call got the disposition the fixture declares (the floor FIRED). ---
	i := 0
	for _, turn := range f.Turns {
		for _, c := range turn.Calls {
			a := adjs[i]
			i++
			if a.Tool != c.Tool {
				t.Errorf("call %s: adjudication tool = %q, want %q (order skew)", c.ID, a.Tool, c.Tool)
			}
			if c.ExpectAllow() {
				if !a.Admitted {
					t.Errorf("call %s (%s): want ALLOW, got %s/%s", c.ID, c.Tool, a.Kind, a.Reason)
				}
			} else {
				if a.Admitted {
					t.Errorf("call %s (%s %s): a DANGER call was ADMITTED — the floor did not fire", c.ID, c.Tool, c.ArgPreview())
				}
				if a.Kind != "DENY" {
					t.Errorf("call %s: verdict = %q, want DENY", c.ID, a.Kind)
				}
				if c.Reason != "" && a.Reason != c.Reason {
					t.Errorf("call %s: deny reason = %q, want %q", c.ID, a.Reason, c.Reason)
				}
			}
		}
	}

	// --- (b) The decision journal recorded every deny and the hash chain verifies. ---
	if err := j.Flush(); err != nil {
		t.Fatalf("flush journal: %v", err)
	}
	n, err := journal.Verify(jpath)
	if err != nil {
		t.Fatalf("journal.Verify(%s): %v (a chain break means a tampered/garbled audit trail)", jpath, err)
	}
	// Every DENY lands as one DENY row. ALLOWs land as DECIDE rows too, so the chain has
	// AT LEAST one row per denied call; assert the denials are all present and the chain
	// is sound, the same posture `fak audit verify` checks.
	if n < exp.Denied {
		t.Fatalf("journal verified %d rows, want >= %d (one per denied danger call)", n, exp.Denied)
	}

	// --- (c) The exit AdjudicationSummary matches the fixture (the roll-up the banner prints). ---
	sum := srv.AdjudicationSummary()
	if int(sum.Denied) != exp.Denied {
		t.Errorf("summary Denied = %d, want %d", sum.Denied, exp.Denied)
	}
	if int(sum.Allowed) < exp.Allowed {
		t.Errorf("summary Allowed = %d, want >= %d (proposed-call allows)", sum.Allowed, exp.Allowed)
	}
	for reason, want := range exp.ByReason {
		if got := int(sum.ByReason[reason]); got != want {
			t.Errorf("summary ByReason[%s] = %d, want %d", reason, got, want)
		}
	}

	// --- (d) The token economy reached the summary (the trace led to real token work). ---
	// On the Anthropic wire fak preserves the provider's cache_read verbatim, so the
	// summary's cached read count equals the fixture's summed cache_read. (The OpenAI
	// proxy folds cache differently; assert the read axis only on the Anthropic wire,
	// where `fak guard -- claude` lives and the number is load-bearing.)
	if provider == "anthropic" {
		if int(sum.CachedPromptTokens) != exp.CacheReadTokens {
			t.Errorf("summary CachedPromptTokens = %d, want %d (provider cache_read forwarded verbatim)", sum.CachedPromptTokens, exp.CacheReadTokens)
		}
	}
}

// TestGuardTraceDebugStatsFireWhenWired proves the per-turn --debug-stats line — the
// glanceable "did this turn work + the token economy" an operator reads — fires once per
// turn and carries the verdict + the cache health, payload-free. This is the
// understand-what-is-happening half of the goal.
func TestGuardTraceDebugStatsFireWhenWired(t *testing.T) {
	f, err := guardtrace.LoadFixture(filepath.Join("testdata", "guard-trace-e2e.json"))
	if err != nil {
		t.Fatalf("load fixture: %v", err)
	}
	srv, _ := newGuardTraceServer(t, "anthropic", f)

	var lines []string
	srv.debugStatsf = func(format string, args ...any) {
		lines = append(lines, fmt.Sprintf(format, args...))
	}

	runGuardTrace(t, srv, "anthropic", f)

	if len(lines) != len(f.Turns) {
		t.Fatalf("debug lines = %d, want one per turn (%d):\n%s", len(lines), len(f.Turns), strings.Join(lines, "\n"))
	}
	for i, line := range lines {
		if !strings.HasPrefix(line, "fak-turn ") {
			t.Errorf("turn %d debug line not a fak-turn line: %q", i, line)
		}
		for _, want := range []string{"trace=", "cache=", "finish="} {
			if !strings.Contains(line, want) {
				t.Errorf("turn %d debug line missing %q: %s", i, want, line)
			}
		}
	}
	// No prompt content can leak onto the glanceable line.
	for i, line := range lines {
		for _, leak := range []string{"rm -rf", "authorized_keys", "sudo"} {
			if strings.Contains(line, leak) {
				t.Errorf("turn %d debug line leaked tool-arg content %q: %s", i, leak, line)
			}
		}
	}
}
