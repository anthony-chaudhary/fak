package memvaluescore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
)

// staleByValue marks exactly the claims whose value contains the marker as
// stale; everything else verifies fresh — the same injectable-verifier move
// the memq notes backend tests use.
func staleByValue(marker string) recall.ArtifactVerifier {
	return func(_ context.Context, claims []recall.ArtifactClaim) []recall.ArtifactFinding {
		out := make([]recall.ArtifactFinding, 0, len(claims))
		for _, c := range claims {
			st := recall.ArtifactFresh
			detail := ""
			if strings.Contains(c.Value, marker) {
				st, detail = recall.ArtifactStale, "gone from the checkout"
			}
			out = append(out, recall.ArtifactFinding{Claim: c, Status: st, Detail: detail})
		}
		return out
	}
}

func allUnverifiable(_ context.Context, claims []recall.ArtifactClaim) []recall.ArtifactFinding {
	out := make([]recall.ArtifactFinding, 0, len(claims))
	for _, c := range claims {
		out = append(out, recall.ArtifactFinding{Claim: c, Status: recall.ArtifactUnverifiable})
	}
	return out
}

const goodNote = `---
name: good-note
description: A healthy note whose claims verify.
metadata:
  type: project
---

The scorecard lives at internal/memvaluescore/score.go and links [[other-note]].
`

const otherNote = `---
name: other-note
description: The wikilink target.
metadata:
  type: reference
---

Nothing checkable here, prose only.
`

func writeStore(t *testing.T, index string, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte(index), 0o644); err != nil {
		t.Fatal(err)
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

const cleanIndex = "- [Good](good-note.md) — hook\n- [Other](other-note.md) — hook\n"

func cleanFiles() map[string]string {
	return map[string]string{"good-note.md": goodNote, "other-note.md": otherNote}
}

func TestCleanStore_ZeroDebtZeroPressure_FrontierFailsLow(t *testing.T) {
	store := writeStore(t, cleanIndex, cleanFiles())
	p := BuildWith(context.Background(), store, filepath.Join(store, "no-ledger.jsonl"), staleByValue("@@none@@"))
	if p.Corpus[DebtKey] != 0 {
		t.Fatalf("memory_debt = %v, want 0", p.Corpus[DebtKey])
	}
	if p.Corpus["memory_rot_pressure"] != 0 {
		t.Fatalf("pressure = %v, want 0", p.Corpus["memory_rot_pressure"])
	}
	if p.Corpus["memory_value_frontier"] != 0 {
		t.Fatalf("frontier = %v, want 0 with no ledger (fails low)", p.Corpus["memory_value_frontier"])
	}
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("ok=%v verdict=%q, want OK on a clean store", p.OK, p.Verdict)
	}
}

func TestDanglingIndexRow_IsHardDebt(t *testing.T) {
	store := writeStore(t, "- [Gone](gone.md) — hook\n", nil)
	a := AuditStore(context.Background(), store, staleByValue("@@none@@"))
	if len(a.Debt) != 1 || a.Debt[0].Kind != "dangling_index_row" {
		t.Fatalf("Debt = %+v, want one dangling_index_row", a.Debt)
	}
	p := BuildWith(context.Background(), store, filepath.Join(store, "no.jsonl"), staleByValue("@@none@@"))
	if p.Corpus[DebtKey] != 1 || p.OK {
		t.Fatalf("debt=%v ok=%v, want gate red on structural debt", p.Corpus[DebtKey], p.OK)
	}
}

func TestOrphanAndFrontmatterAreDebt_WikilinkIsSoft(t *testing.T) {
	misnamed := strings.Replace(goodNote, "name: good-note", "name: wrong-slug", 1)
	files := map[string]string{
		"misnamed.md": misnamed,
		"orphan.md":   "no frontmatter at all, and a [[missing-target]] link\n",
	}
	store := writeStore(t, "- [Misnamed](misnamed.md) — hook\n", files)
	a := AuditStore(context.Background(), store, staleByValue("@@none@@"))

	kinds := map[string]int{}
	for _, it := range a.Debt {
		kinds[it.Kind]++
	}
	if kinds["orphan_fact_file"] != 1 || kinds["frontmatter_violation"] == 0 {
		t.Fatalf("Debt kinds = %v, want orphan + frontmatter violations", kinds)
	}
	if kinds["broken_wikilink"] != 0 {
		t.Fatal("a forward-reference wikilink must never be HARD debt")
	}
	// Two unresolved links: [[missing-target]] (orphan) and [[other-note]] (misnamed).
	if len(a.Soft) != 2 {
		t.Fatalf("Soft = %+v, want the two forward references", a.Soft)
	}
	_, byTerm := Pressure(a)
	if byTerm["broken_wikilink"] != 2*SevStructural {
		t.Fatalf("wikilink pressure = %d, want %d", byTerm["broken_wikilink"], 2*SevStructural)
	}
}

func TestStaleClaims_ArePressureNotDebt(t *testing.T) {
	store := writeStore(t, cleanIndex, cleanFiles())
	p := BuildWith(context.Background(), store, filepath.Join(store, "no.jsonl"),
		staleByValue("internal/memvaluescore/score.go"))
	if p.Corpus[DebtKey] != 0 {
		t.Fatalf("memory_debt = %v — stale claims must never gate as HARD debt", p.Corpus[DebtKey])
	}
	if p.Corpus["stale_claims"] != 1 {
		t.Fatalf("stale_claims = %v, want 1", p.Corpus["stale_claims"])
	}
	if p.Corpus["memory_rot_pressure"] != SevStaleClaim {
		t.Fatalf("pressure = %v, want %d", p.Corpus["memory_rot_pressure"], SevStaleClaim)
	}
	if !p.OK {
		t.Fatal("the gate must stay green under external-drift staleness")
	}
}

func TestUnverifiableClaims_NeverCountStale(t *testing.T) {
	store := writeStore(t, cleanIndex, cleanFiles())
	p := BuildWith(context.Background(), store, filepath.Join(store, "no.jsonl"), allUnverifiable)
	if p.Corpus["stale_claims"] != 0 || p.Corpus["memory_rot_pressure"] != 0 {
		t.Fatalf("stale=%v pressure=%v, want zero for unverifiable-only",
			p.Corpus["stale_claims"], p.Corpus["memory_rot_pressure"])
	}
	if p.Corpus["unverifiable_claims"] == 0 {
		t.Fatal("unverifiable claims must be reported for visibility")
	}
}

func TestFoldLedger_SumsAndReportsSkipped(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "memory-value.jsonl")
	rows := []string{
		`{"schema":"fak-memory-value-ledger/1","fresh":3,"withheld_stale":1}`,
		`{"schema":"fak-memory-value-ledger/1","fresh":2,"withheld_stale":0,"lessons":1}`,
		`{"schema":"some-other-ledger/9","fresh":100}`,
		`not json at all`,
	}
	if err := os.WriteFile(ledger, []byte(strings.Join(rows, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	f := FoldLedger(ledger)
	if f.Rows != 2 || f.SkippedRows != 2 {
		t.Fatalf("rows=%d skipped=%d, want 2/2", f.Rows, f.SkippedRows)
	}
	total, byTerm := Frontier(f.Events)
	want := 2*5 + 8*1 + 4*1
	if total != want {
		t.Fatalf("frontier = %d, want %d", total, want)
	}
	sum := 0
	for _, v := range byTerm {
		sum += v
	}
	if sum != total {
		t.Fatalf("by_term sums to %d, want %d", sum, total)
	}
}

func TestFrontier_MonotoneUnboundedFailsLow(t *testing.T) {
	base := map[string]int{"fresh_rendered": 10, "stale_withheld": 2}
	total, _ := Frontier(base)
	for term, weight := range FrontierUnits {
		bumped := map[string]int{}
		for k, v := range base {
			bumped[k] = v
		}
		bumped[term]++
		bigger, _ := Frontier(bumped)
		if bigger != total+weight {
			t.Fatalf("bump %s: %d, want %d", term, bigger, total+weight)
		}
	}
	huge, _ := Frontier(map[string]int{"fresh_rendered": 10000, "stale_withheld": 10000, "lesson_distilled": 10000})
	if huge <= 100 {
		t.Fatal("no 0-100 clamp: the frontier is unbounded")
	}
	zero, _ := Frontier(map[string]int{})
	if zero != 0 {
		t.Fatalf("empty events = %d, want 0 (fails low)", zero)
	}
}

func TestMissingStore_IsEmptyCorpusNotError(t *testing.T) {
	dir := t.TempDir()
	p := BuildWith(context.Background(), filepath.Join(dir, "nope"),
		filepath.Join(dir, "nope.jsonl"), staleByValue("@@none@@"))
	if !p.OK || p.Corpus[DebtKey] != 0 {
		t.Fatalf("ok=%v debt=%v, want a missing store to fold clean", p.OK, p.Corpus[DebtKey])
	}
}

// The live-tree floor (the conflation-card precedent): the COMMITTED memory
// mirror must hold zero structural debt. Debt deliberately excludes anything
// external drift can move, so this pin cannot red a peer who never touched
// memory. Claim checks use the real recall verifier — the same seam
// `fak memory recall` gates page-ins with.
func TestLiveTree_CommittedMirrorHoldsZeroDebt(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	a := BuildWith(context.Background(), filepath.Join(root, DefaultStoreRel),
		filepath.Join(root, DefaultLedgerRel), nil)
	if a.Corpus[DebtKey] != 0 {
		t.Fatalf("committed mirror memory_debt = %v, want 0; defects: %v",
			a.Corpus[DebtKey], a.KPIs)
	}
	b := BuildWith(context.Background(), filepath.Join(root, DefaultStoreRel),
		filepath.Join(root, DefaultLedgerRel), nil)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("two folds over the same tree must be identical (deterministic)")
	}
	if _, err := json.Marshal(a); err != nil {
		t.Fatalf("payload must marshal end to end: %v", err)
	}
}
