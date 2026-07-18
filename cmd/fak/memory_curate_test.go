package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/memq"
)

// fixtureCurateStore writes a two-note markdown store whose notes are UNTYPED
// (frontmatter without metadata.type → session durability), so both are
// eviction candidates — never protected by the durable floor.
func fixtureCurateStore(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"MEMORY.md": "# Memory index\n\n" +
			"- [Alpha](alpha.md) — first fact\n" +
			"- [Beta](beta.md) — second fact\n",
		"alpha.md": "---\nname: alpha\ndescription: first fact\n---\n\nAlpha body fact.\n",
		"beta.md":  "---\nname: beta\ndescription: second fact\n---\n\nBeta body fact.\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// #5110 done-condition at the CLI: `fak memory curate` renders the byte budget,
// the evicted set, and the running regret rate (#3908 DoD 4). A 1-byte cap over
// two session notes evicts both; naming one as later-needed witnesses regret.
func TestMemoryCurate_rendersBudgetEvictedAndRegret(t *testing.T) {
	dir := fixtureCurateStore(t)
	var out, errb strings.Builder
	code := runMemoryCurate(&out, &errb, []string{"--store", dir, "--budget", "1", "--needed", "alpha.md"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "under a 1-byte cap; 2 evicted") {
		t.Errorf("report must render the byte budget and eviction count; got:\n%s", s)
	}
	if !strings.Contains(s, "evict alpha.md") || !strings.Contains(s, "evict beta.md") {
		t.Errorf("report must name the evicted set; got:\n%s", s)
	}
	if !strings.Contains(s, "regret: 1/1") || !strings.Contains(s, "rate 1.000") {
		t.Errorf("report must render the running regret rate; got:\n%s", s)
	}
	if !strings.Contains(s, "proposal only") {
		t.Errorf("without --apply the eviction must stay a fail-closed proposal; got:\n%s", s)
	}
}

// The --json path emits the typed CurateReport + RegretReport envelope, and the
// eviction effect stays un-applied without --apply (fail-closed).
func TestMemoryCurate_jsonEnvelope(t *testing.T) {
	dir := fixtureCurateStore(t)
	var out, errb strings.Builder
	if code := runMemoryCurate(&out, &errb, []string{"--store", dir, "--budget", "1", "--json"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var env curateEnvelope
	if err := json.Unmarshal([]byte(out.String()), &env); err != nil {
		t.Fatalf("envelope must parse: %v\n%s", err, out.String())
	}
	if env.Report.Reason != memq.CurateReason {
		t.Errorf("report reason = %q, want %q", env.Report.Reason, memq.CurateReason)
	}
	if len(env.Report.Evicted) != 2 {
		t.Errorf("evicted = %d cells, want 2:\n%s", len(env.Report.Evicted), out.String())
	}
	if env.Regret.Reason != memq.RegretReason {
		t.Errorf("regret reason = %q, want %q", env.Regret.Reason, memq.RegretReason)
	}
	if env.Effect == nil {
		t.Fatalf("an eviction must carry the effect record:\n%s", out.String())
	}
	if env.Effect.Applied {
		t.Errorf("without --apply (and against the read-only notes store) the effect must never apply:\n%s", out.String())
	}
}

// A missing/non-positive --budget is refused up front: curate without a hard
// cap is not a pass.
func TestMemoryCurate_requiresBudget(t *testing.T) {
	var out, errb strings.Builder
	if code := runMemoryCurate(&out, &errb, []string{"--store", t.TempDir()}); code != 2 {
		t.Fatalf("exit=%d, want 2; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "--budget") {
		t.Errorf("refusal must name the missing flag; got: %s", errb.String())
	}
}
