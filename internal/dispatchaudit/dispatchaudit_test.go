package dispatchaudit

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func mustTime(t *testing.T, s string) time.Time {
	t.Helper()
	tt, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		t.Fatalf("parse time %q: %v", s, err)
	}
	return tt.UTC()
}

func TestClassifyShipped(t *testing.T) {
	w := Worker{Log: "resolve-100.log", Issue: "100", Lane: "tools", HeaderBackend: BackendClaude, SidecarBackend: BackendClaude, CommitSHA: "b68ead49"}
	c := Classify(w, DefaultThresholds())
	if c.Outcome != OutcomeShipped {
		t.Fatalf("want SHIPPED, got %s (%s)", c.Outcome, c.Reason)
	}
	if c.Backend != BackendClaude {
		t.Fatalf("want claude backend, got %s", c.Backend)
	}
}

func TestClassifyQuotaWalledWithMissingSidecar(t *testing.T) {
	// The issue's worked example: opencode hard-walled (15 errors / 17 min) with a
	// MISSING .backend sidecar — the header still resolves the backend.
	w := Worker{
		Log:            "resolve-1346.log",
		Issue:          "1346",
		Lane:           "docs",
		HeaderBackend:  BackendOpencode,
		SidecarMissing: true,
		CapHit:         true,
		ErrorLines:     15,
		FirstError:     mustTime(t, "2026-06-30T00:01:22Z"),
		LastError:      mustTime(t, "2026-06-30T00:33:26Z"),
	}
	c := Classify(w, DefaultThresholds())
	if c.Outcome != OutcomeQuotaWalled {
		t.Fatalf("want QUOTA_WALLED, got %s (%s)", c.Outcome, c.Reason)
	}
	if c.Backend != BackendOpencode {
		t.Fatalf("missing sidecar must fall back to header backend; got %s", c.Backend)
	}
	if !c.Misattributed {
		t.Fatal("missing sidecar with a declared header backend must flag misattributed")
	}
	if c.WallMinutes < 30 {
		t.Fatalf("want ~32 wasted minutes, got %v", c.WallMinutes)
	}
}

func TestClassifyWastedSpawnShortCap(t *testing.T) {
	w := Worker{Log: "resolve-9.log", Lane: "model", HeaderBackend: BackendOpencode, SidecarBackend: BackendOpencode, CapHit: true, ErrorLines: 1}
	c := Classify(w, DefaultThresholds())
	if c.Outcome != OutcomeWastedSpawn {
		t.Fatalf("a brief cap hit with no storm is WASTED_SPAWN, got %s", c.Outcome)
	}
}

func TestClassifyRetryStorm(t *testing.T) {
	w := Worker{
		Log: "resolve-7.log", Lane: "ci", HeaderBackend: BackendClaude, SidecarBackend: BackendClaude,
		ErrorLines: 8,
		FirstError: mustTime(t, "2026-06-30T00:00:00Z"),
		LastError:  mustTime(t, "2026-06-30T00:12:00Z"),
	}
	c := Classify(w, DefaultThresholds())
	if c.Outcome != OutcomeRetryStorm {
		t.Fatalf("want RETRY_STORM, got %s (%s)", c.Outcome, c.Reason)
	}
}

func TestClassifyNoOpBanner(t *testing.T) {
	w := Worker{Log: "resolve-3.log", Lane: "docs", HeaderBackend: BackendClaude, SidecarBackend: BackendClaude, BannerOnly: true}
	c := Classify(w, DefaultThresholds())
	if c.Outcome != OutcomeNoOp {
		t.Fatalf("banner-only is NO_OP, got %s", c.Outcome)
	}
}

func TestClassifyNoOpProgressTick(t *testing.T) {
	w := Worker{Log: "resolve-4.log", Lane: "docs", HeaderBackend: BackendClaude, SidecarBackend: BackendClaude, ProgressTicks: 3, ProgressMoved: false}
	c := Classify(w, DefaultThresholds())
	if c.Outcome != OutcomeNoOp {
		t.Fatalf("a progress tick that never moved is NO_OP, got %s", c.Outcome)
	}
}

func TestFingerprintStability(t *testing.T) {
	a := Classify(Worker{Log: "resolve-1346.log", Lane: "docs", HeaderBackend: BackendOpencode, SidecarMissing: true, CapHit: true, ErrorLines: 15, FirstError: mustTime(t, "2026-06-30T00:01:22Z"), LastError: mustTime(t, "2026-06-30T00:33:26Z")}, DefaultThresholds())
	// Same outcome+backend+lane, DIFFERENT log/timestamp -> SAME fingerprint.
	b := Classify(Worker{Log: "resolve-9999.log", Lane: "docs", HeaderBackend: BackendOpencode, SidecarMissing: true, CapHit: true, ErrorLines: 20, FirstError: mustTime(t, "2026-07-01T00:00:00Z"), LastError: mustTime(t, "2026-07-01T01:00:00Z")}, DefaultThresholds())
	if Fingerprint(a) != Fingerprint(b) {
		t.Fatalf("fingerprint must be stable across logs for the same code-site: %s != %s", Fingerprint(a), Fingerprint(b))
	}
	// A different lane -> different fingerprint.
	c := Classify(Worker{Log: "resolve-5.log", Lane: "ci", HeaderBackend: BackendOpencode, SidecarMissing: true, CapHit: true, ErrorLines: 15, FirstError: mustTime(t, "2026-06-30T00:01:22Z"), LastError: mustTime(t, "2026-06-30T00:33:26Z")}, DefaultThresholds())
	if Fingerprint(a) == Fingerprint(c) {
		t.Fatal("a different lane must yield a different fingerprint")
	}
}

func TestFoldRollupAndDedup(t *testing.T) {
	th := DefaultThresholds()
	workers := []Worker{
		{Log: "resolve-100.log", Issue: "100", Lane: "tools", SidecarBackend: BackendClaude, HeaderBackend: BackendClaude, CommitSHA: "abc1234"},
		// two walled opencode/docs workers -> ONE finding (deduped fingerprint).
		{Log: "resolve-1346.log", Issue: "1346", Lane: "docs", HeaderBackend: BackendOpencode, SidecarMissing: true, CapHit: true, ErrorLines: 15, FirstError: mustTime(t, "2026-06-30T00:01:22Z"), LastError: mustTime(t, "2026-06-30T00:33:26Z")},
		{Log: "resolve-1350.log", Issue: "1350", Lane: "docs", HeaderBackend: BackendOpencode, SidecarMissing: true, CapHit: true, ErrorLines: 18, FirstError: mustTime(t, "2026-06-30T01:00:00Z"), LastError: mustTime(t, "2026-06-30T01:30:00Z")},
	}
	rep := Fold(workers, th)

	if len(rep.Classifications) != 3 {
		t.Fatalf("want 3 classifications, got %d", len(rep.Classifications))
	}
	// One finding only — the two docs/opencode walls share a fingerprint.
	if len(rep.Findings) != 1 {
		t.Fatalf("want 1 deduped finding, got %d", len(rep.Findings))
	}
	if rep.Findings[0].Outcome != OutcomeQuotaWalled {
		t.Fatalf("finding outcome = %s", rep.Findings[0].Outcome)
	}

	var claude, opencode *BackendRollup
	for i := range rep.Rollups {
		switch rep.Rollups[i].Backend {
		case BackendClaude:
			claude = &rep.Rollups[i]
		case BackendOpencode:
			opencode = &rep.Rollups[i]
		}
	}
	if claude == nil || claude.Shipped != 1 {
		t.Fatalf("claude rollup wrong: %+v", claude)
	}
	if opencode == nil || opencode.QuotaWalled != 2 {
		t.Fatalf("opencode rollup must count BOTH walls: %+v", opencode)
	}
	if opencode.WastedMinutes < 60 {
		t.Fatalf("opencode wasted-minutes must sum both walls (~62), got %v", opencode.WastedMinutes)
	}
	if opencode.Misattributed != 2 {
		t.Fatalf("both missing-sidecar workers must count as misattributed, got %d", opencode.Misattributed)
	}
}

func TestNewFindingsDedup(t *testing.T) {
	f := Finding{Fingerprint: "deadbeef", Title: "dispatch audit: QUOTA_WALLED on opencode (lane docs)"}
	// already-marked fingerprint -> dropped.
	got := NewFindings([]Finding{f}, map[string]bool{"deadbeef": true}, nil)
	if len(got) != 0 {
		t.Fatalf("marked fingerprint must be filtered, got %d", len(got))
	}
	// title already open -> dropped.
	got = NewFindings([]Finding{f}, nil, map[string]bool{f.Title: true})
	if len(got) != 0 {
		t.Fatalf("existing open title must be filtered, got %d", len(got))
	}
	// genuinely new -> kept.
	got = NewFindings([]Finding{f}, nil, nil)
	if len(got) != 1 {
		t.Fatalf("a new finding must survive, got %d", len(got))
	}
}

// TestScanDirFixture proves the I/O shell parses a real on-disk .dispatch-runs/
// fixture: a real ship, a quota-walled opencode log with a MISSING .backend
// sidecar, and a banner-only no-op log.
func TestScanDirFixture(t *testing.T) {
	dir := t.TempDir()

	// 1) a real ship (claude, with a matching .backend sidecar)
	writeFile(t, dir, "resolve-100-20260628-105439.log",
		"# fak-spawn 20260628-105439 issue=100 lane=tools backend=claude argv0=claude\n"+
			"working...\n"+
			"✅ Commit created: `b68ead49` - implements the thing (closes #100)\n")
	writeFile(t, dir, "resolve-100-20260628-105439.backend", "claude")

	// 2) a quota-walled opencode worker with NO .backend sidecar
	walled := "# fak-spawn 20260629-235906 issue=1346 lane=docs backend=opencode argv0=opencode.CMD\n"
	for _, ts := range []string{
		"2026-06-30T00:01:22.783Z", "2026-06-30T00:03:32.310Z", "2026-06-30T00:07:49.424Z",
		"2026-06-30T00:16:22.188Z", "2026-06-30T00:33:26.827Z",
	} {
		walled += "timestamp=" + ts + " level=ERROR run=b5da99f7 message=\"stream error\" error.error=\"AI_APICallError: Weekly/Monthly Limit Exhausted. Your limit will reset at 2026-07-04 00:56:38\"\n"
	}
	writeFile(t, dir, "resolve-1346-20260629-235906.log", walled)

	// 3) a banner-only no-op log (#1275)
	writeFile(t, dir, "resolve-3-20260629-000000.log",
		"# fak-spawn 20260629-000000 issue=3 lane=docs backend=claude argv0=claude\n"+
			"fak guard: ANTHROPIC_API_KEY is set but fak defaults to your subscription\n"+
			"fak guard — kernel-adjudicated: claude -p\n"+
			"  gateway    : http://127.0.0.1:51903\n")
	writeFile(t, dir, "resolve-3-20260629-000000.backend", "claude")

	// progress ledger: a no-op tick for issue 3 (never moved).
	writeFile(t, dir, "progress.jsonl",
		"{\"witnessed_numbers\":[3],\"resolved_toward_target\":4}\n"+
			"{\"witnessed_numbers\":[3],\"resolved_toward_target\":4}\n")

	workers, err := ScanDir(dir)
	if err != nil {
		t.Fatalf("ScanDir: %v", err)
	}
	if len(workers) != 3 {
		t.Fatalf("want 3 workers parsed, got %d", len(workers))
	}

	rep := Fold(workers, DefaultThresholds())
	got := map[string]Outcome{}
	for _, c := range rep.Classifications {
		got[c.Issue] = c.Outcome
	}
	if got["100"] != OutcomeShipped {
		t.Errorf("issue 100 should be SHIPPED, got %s", got["100"])
	}
	if got["1346"] != OutcomeQuotaWalled {
		t.Errorf("issue 1346 should be QUOTA_WALLED, got %s", got["1346"])
	}
	if got["3"] != OutcomeNoOp {
		t.Errorf("issue 3 should be NO_OP, got %s", got["3"])
	}

	// The walled worker had no sidecar -> backend resolved from header, flagged.
	for _, c := range rep.Classifications {
		if c.Issue == "1346" {
			if c.Backend != BackendOpencode {
				t.Errorf("walled worker backend should fall back to header opencode, got %s", c.Backend)
			}
			if !c.Misattributed {
				t.Errorf("walled worker with missing sidecar should be flagged misattributed")
			}
		}
	}
}

func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

func TestMarkAndAlreadyFiled(t *testing.T) {
	dir := t.TempDir()
	fp := "abc123def456"
	if AlreadyFiled(dir, fp) {
		t.Fatal("fingerprint should not be filed in a fresh dir")
	}
	if err := MarkFiled(dir, fp); err != nil {
		t.Fatalf("MarkFiled: %v", err)
	}
	if !AlreadyFiled(dir, fp) {
		t.Fatal("fingerprint should be filed after MarkFiled")
	}
}
