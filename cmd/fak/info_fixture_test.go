package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// demoOverlayVars is the checked-in fixture the README media is captured from: a fully-populated,
// PAYLOAD-FREE /debug/vars snapshot that gives every tab something real to show — a PROVEN cache
// with an owner split, a seat roster + serving nodes, a two-session agent tree, and a safety
// reason breakdown. Every value is a counter, gauge, generic tool/reason name, or placeholder
// identity: no prompt or result text, no local path, no real account, credential, or hostname.
// It is deliberately richer than richVisualVars() (which the render tests use) because a docs
// frame must exercise the accounts and safety tabs the live overlay populates. The headline
// numbers mirror the reference screenshot (51% reuse, +65,144 tokens saved, ~46%/54% owner
// split) so the captured frame reads like a real session.
func demoOverlayVars() guardInfoVars {
	var v guardInfoVars
	v.Gateway.UptimeSeconds = 2*3600 + 14*60
	v.Gateway.InflightRequests = 3
	v.Gateway.VDSO = true

	v.Runtime.NumGoroutine = 28
	v.Runtime.Memory.HeapAllocBytes = 47 << 20
	v.Runtime.Memory.SysBytes = 72 << 20
	v.Runtime.Memory.NumGC = 18

	v.Inference.Turns = 4
	v.Inference.OutputTokensPerSecond = 41.2
	v.Inference.MeanTTFTSeconds = 0.9
	v.Inference.InflightMaxAgeSeconds = 2.1

	// Cache: the headline economy from the reference screenshot.
	v.VCache = &struct {
		CacheReadTokens int64   `json:"cache_read_tokens"`
		SavedTokenEquiv float64 `json:"saved_token_equiv"`
		HitRate         float64 `json:"hit_rate"`
		Multiplier      float64 `json:"multiplier"`
		Status          string  `json:"status"`
	}{CacheReadTokens: 142_400, SavedTokenEquiv: 65_144, HitRate: 0.51, Multiplier: 1.60, Status: "PROVEN"}
	// Owner split: ~46% provider / ~54% fak, matching the screenshot's "split default cache 46%
	// (~65.1k tok) + fak 54% (~77.3k tok)". The sub-fields decompose each owner's net so the Cache
	// tab's live ablation section draws a full per-mechanism breakdown: provider net 65.1k = a
	// 67.8k read rebate less a 2.7k unrepaid write premium; fak 77.3k = 52.3k compaction-shed +
	// 25.0k in-kernel KV-prefix reuse; plus 3 vDSO-memo engine calls avoided.
	v.CacheAttribution = &guardInfoCacheAttribution{
		ProviderTokenEquiv:  65_100,
		FakTokenEquiv:       77_300,
		TotalTokenEquiv:     142_400,
		FakVDSOAvoidedCalls: 3,

		ProviderPromptCacheReadTokenEquiv:         67_800,
		ProviderPromptCacheWritePremiumTokenEquiv: -2_700,
		FakCompactionShedTokens:                   52_300,
		FakKVPrefixReusedTokens:                   25_000,
	}

	// Agents: a main session + one sub-agent, so the agents tab shows lineage.
	v.Sessions = []guardInfoSession{
		{TraceID: "main-trace-01", Run: "running", TokensLeft: 372_000, TurnsLeft: 6, ElapsedSeconds: 134},
		{TraceID: "sub-trace-02", Run: "running", ParentTrace: "main-trace-01", Generation: 1, TokensLeft: 118_000, TurnsLeft: 3, ElapsedSeconds: 41},
	}

	// Accounts + nodes: a small roster (active / walled / idle) and the kernel+serving nodes, so
	// the accounts tab is populated. Placeholder names + example.com logins only.
	v.Endpoints = &gateway.SessionEndpoints{
		Accounts: []gateway.SessionAccount{
			{Name: "seat-a", Email: "seat-a@example.com", Active: true, CanServe: true, LoginStatus: "ok"},
			{Name: "seat-b", Email: "seat-b@example.com", Walled: true, CanServe: false, LoginStatus: "ok"},
			{Name: "seat-c", Email: "seat-c@example.com", CanServe: true, LoginStatus: "ok"},
		},
		Nodes: []gateway.SessionNode{
			{Role: "kernel", ID: "this-host", Kind: "host", Detail: "fak guard + agent + adjudication"},
			{Role: "serving", ID: "api.provider.example", Kind: "proxy", Detail: "provider API"},
		},
	}

	// Safety: a non-trivial verdict roll-up with a reason breakdown, so the safety tab shows the
	// blocked/held/deferred story, not "nothing blocked". Generic reason codes only.
	v.Adjudication = &gateway.AdjudicationSummary{
		Total:       46,
		Allowed:     41,
		Denied:      3,
		Transformed: 1,
		Quarantined: 1,
		Deferred:    2,
		Escalated:   1,
		ByReason: map[string]uint64{
			"DEFAULT_DENY":      2,
			"SECRET_IN_ARGS":    1,
			"UNVERIFIED_RESULT": 1,
		},
	}
	return v
}

// TestGenerateOverlayFixture writes the checked-in demo fixture to visuals/info-overlay-live.json
// when FAK_UPDATE_INFO_FIXTURE=1. It is the golden-file WRITER (skipped in normal CI); the
// capture doc's regeneration command runs it. Kept in-package because demoOverlayVars uses the
// unexported guardInfoVars shape.
func TestGenerateOverlayFixture(t *testing.T) {
	if os.Getenv("FAK_UPDATE_INFO_FIXTURE") != "1" {
		t.Skip("set FAK_UPDATE_INFO_FIXTURE=1 to (re)write visuals/info-overlay-live.json")
	}
	raw, err := json.MarshalIndent(demoOverlayVars(), "", "  ")
	if err != nil {
		t.Fatalf("marshal demo fixture: %v", err)
	}
	// cmd/fak → repo root is two levels up.
	out := filepath.Join("..", "..", "visuals", "info-overlay-live.json")
	if err := os.WriteFile(out, append(raw, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", out, err)
	}
	t.Logf("wrote %s", out)
}

// TestInfoViewByName pins the --tab → view mapping (both the UI label and the internal name for
// the accounts/endpoints tab), and that an unknown word is rejected rather than silently
// resolving to overview.
func TestInfoViewByName(t *testing.T) {
	cases := []struct {
		name string
		want infoView
		ok   bool
	}{
		{"overview", viewOverview, true},
		{"", viewOverview, true}, // empty defaults to overview
		{"agents", viewAgents, true},
		{"fleet", viewFleet, true},
		{"accounts", viewEndpoints, true},
		{"endpoints", viewEndpoints, true}, // internal alias accepted
		{"cache", viewCache, true},
		{"safety", viewSafety, true},
		{"CACHE", viewCache, true}, // case-insensitive
		{"  cache  ", viewCache, true},
		{"bogus", viewOverview, false},
	}
	for _, tc := range cases {
		got, ok := infoViewByName(tc.name)
		if ok != tc.ok {
			t.Errorf("infoViewByName(%q) ok=%v, want %v", tc.name, ok, tc.ok)
			continue
		}
		if ok && got != tc.want {
			t.Errorf("infoViewByName(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// writeInfoVarsFixture serializes vars to a temp JSON file (the shape `fak info --json` emits)
// and returns its path.
func writeInfoVarsFixture(t *testing.T, v guardInfoVars) string {
	t.Helper()
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "info-vars.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

// TestRunInfoFixtureFrameRendersEachTab proves the offline path renders every tab from a recorded
// snapshot with exit 0, that the requested tab is the ACTIVE one in the tab bar, and that the
// frame is byte-identical to what the live renderer produces for the same vars — i.e. the offline
// path is a faithful capture, not a second rendering.
func TestRunInfoFixtureFrameRendersEachTab(t *testing.T) {
	v := richVisualVars()
	path := writeInfoVarsFixture(t, v)

	// The reference trend the offline path seeds: the same count of the same snapshot.
	refTrend := newGuardInfoTrend(guardInfoTrendCap)
	for i := 0; i < guardInfoTrendSeedSamples; i++ {
		refTrend.push(v)
	}

	for i := 0; i < infoViewCount(); i++ {
		view := infoView(i)
		name := infoViewName(view)
		t.Run(name, func(t *testing.T) {
			var out, errBuf bytes.Buffer
			// Fixed geometry so the frame is deterministic.
			code := runInfoFixtureFrame(&out, &errBuf, path, name, true, 120, 40)
			if code != 0 {
				t.Fatalf("runInfoFixtureFrame(%s) = %d, stderr=%q", name, code, errBuf.String())
			}
			if errBuf.Len() != 0 {
				t.Errorf("unexpected stderr: %q", errBuf.String())
			}
			got := strings.TrimRight(out.String(), "\n")

			want := renderGuardInfoInteractiveBlock(infoViewState{active: view}, v, refTrend, 120, 40)
			if got != want {
				t.Errorf("offline frame for tab %s differs from the live renderer output.\n--- got ---\n%s\n--- want ---\n%s", name, got, want)
			}

			// The requested tab must be the highlighted/active one: the tab bar (row 1) marks the
			// active view with its «guillemet» chrome (buildInfoTabBar). Assert the tab word is present
			// on the first row, a cheap "we rendered the right view" check that does not couple to the
			// exact highlight glyphs.
			firstRow := got
			if nl := strings.IndexByte(got, '\n'); nl >= 0 {
				firstRow = got[:nl]
			}
			if !strings.Contains(firstRow, name) {
				t.Errorf("tab bar row %q does not mention the requested tab %q", firstRow, name)
			}
		})
	}
}

// TestRunInfoFixtureFrameErrors covers the three rejection paths: unknown tab, missing file,
// malformed JSON, and --frame=false (unsupported today) — each a distinct exit and a house error
// on stderr, nothing on stdout.
func TestRunInfoFixtureFrameErrors(t *testing.T) {
	good := writeInfoVarsFixture(t, richVisualVars())

	t.Run("unknown tab", func(t *testing.T) {
		var out, errBuf bytes.Buffer
		if code := runInfoFixtureFrame(&out, &errBuf, good, "bogus", true, 0, 0); code != 2 {
			t.Fatalf("exit = %d, want 2", code)
		}
		if out.Len() != 0 {
			t.Errorf("stdout should be empty, got %q", out.String())
		}
		if !strings.Contains(errBuf.String(), "unknown --tab") {
			t.Errorf("stderr = %q, want it to name the unknown tab", errBuf.String())
		}
	})

	t.Run("missing file", func(t *testing.T) {
		var out, errBuf bytes.Buffer
		missing := filepath.Join(t.TempDir(), "nope.json")
		if code := runInfoFixtureFrame(&out, &errBuf, missing, "cache", true, 0, 0); code != 1 {
			t.Fatalf("exit = %d, want 1", code)
		}
		if !strings.Contains(errBuf.String(), "reading fixture") {
			t.Errorf("stderr = %q, want a read error", errBuf.String())
		}
	})

	t.Run("bad json", func(t *testing.T) {
		bad := filepath.Join(t.TempDir(), "bad.json")
		if err := os.WriteFile(bad, []byte("{not json"), 0o644); err != nil {
			t.Fatal(err)
		}
		var out, errBuf bytes.Buffer
		if code := runInfoFixtureFrame(&out, &errBuf, bad, "cache", true, 0, 0); code != 1 {
			t.Fatalf("exit = %d, want 1", code)
		}
		if !strings.Contains(errBuf.String(), "parsing fixture") {
			t.Errorf("stderr = %q, want a parse error", errBuf.String())
		}
	})

	t.Run("frame=false unsupported", func(t *testing.T) {
		var out, errBuf bytes.Buffer
		if code := runInfoFixtureFrame(&out, &errBuf, good, "cache", false, 0, 0); code != 2 {
			t.Fatalf("exit = %d, want 2", code)
		}
		if !strings.Contains(errBuf.String(), "only --frame") {
			t.Errorf("stderr = %q, want the frame-only note", errBuf.String())
		}
	})
}

// TestRunInfoFixtureFrameToleratesExtraFields proves a raw gateway /debug/vars snapshot — which
// carries many blocks the overlay does not render — decodes fine (the same tolerance the live
// fetch relies on), so an operator can capture directly from a gateway, not only via `fak info
// --json`.
func TestRunInfoFixtureFrameToleratesExtraFields(t *testing.T) {
	raw, err := json.Marshal(richVisualVars())
	if err != nil {
		t.Fatal(err)
	}
	// Splice an unknown top-level block in.
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	m["some_unrendered_block"] = json.RawMessage(`{"a":1,"b":[2,3]}`)
	spliced, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "raw-vars.json")
	if err := os.WriteFile(path, spliced, 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errBuf bytes.Buffer
	if code := runInfoFixtureFrame(&out, &errBuf, path, "cache", true, 120, 40); code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, errBuf.String())
	}
	if out.Len() == 0 {
		t.Error("expected a rendered frame")
	}
}
