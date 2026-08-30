package gateway

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// contractFixture is one page set carrying all three survival classes, with element bytes of
// distinct lengths so the byte count can only come from the pages the plan actually named.
func contractFixture() ([]ctxplan.Page, [][]byte) {
	pages := []ctxplan.Page{
		{ID: "p1", Kind: ctxplan.KindSystemInvariant}, // PINNED
		{ID: "p2", Kind: ctxplan.KindToolSchema},      // REPLAYABLE
		{ID: "p3", Kind: ctxplan.KindTranscriptProse}, // EVICTABLE
	}
	els := [][]byte{[]byte("aaa"), []byte("bbbb"), []byte("ccccc")}
	return pages, els
}

// TestCompactionContractFromProjectsThePlan pins the fold the whole record rests on: the split is
// the plan's OWN kept/evicted counts per class, the byte count sums only evicted elements, and a
// digest is issued for a REPLAYABLE eviction and withheld from an EVICTABLE one — because a handle
// to bytes that cannot be paged back is a false promise of recovery.
func TestCompactionContractFromProjectsThePlan(t *testing.T) {
	pages, els := contractFixture()
	plan := ctxplan.EvictionPlan{Keep: []string{"p1"}, Evict: []string{"p2", "p3"}}

	c := compactionContractFrom(pages, els, plan)
	if c == nil {
		t.Fatal("contract is nil for a plan that evicted two pages")
	}
	if c.ContractVersion != compactionContractVersion {
		t.Errorf("contract_version = %d, want %d", c.ContractVersion, compactionContractVersion)
	}
	if c.Instruction != compactionContractInstruction {
		t.Errorf("instruction = %q, want %q", c.Instruction, compactionContractInstruction)
	}
	// Evicted bytes are p2 (4) + p3 (5); the kept p1 (3) must not be counted as a loss.
	if c.EvictedByteCount != 9 {
		t.Errorf("evicted_byte_count = %d, want 9 (p2+p3 only, never the kept p1)", c.EvictedByteCount)
	}
	// Resistance order, one row per class the page set actually held.
	want := []PreservedClassSplit{
		{Class: "PINNED", Kept: 1, Evicted: 0},
		{Class: "REPLAYABLE", Kept: 0, Evicted: 1},
		{Class: "EVICTABLE", Kept: 0, Evicted: 1},
	}
	if len(c.PreservedClasses) != len(want) {
		t.Fatalf("preserved_classes = %+v, want %+v", c.PreservedClasses, want)
	}
	for i, w := range want {
		if c.PreservedClasses[i] != w {
			t.Errorf("preserved_classes[%d] = %+v, want %+v (resistance order)", i, c.PreservedClasses[i], w)
		}
	}
	// Exactly one digest, and it is p2's — the EVICTABLE p3 earns none.
	sum := sha256.Sum256([]byte("bbbb"))
	if len(c.ReplayablePageDigests) != 1 || c.ReplayablePageDigests[0] != hex.EncodeToString(sum[:]) {
		t.Errorf("replayable digests = %v, want exactly p2's digest (an EVICTABLE drop is unrecoverable and must carry none)", c.ReplayablePageDigests)
	}
}

// TestCompactionContractFromOmitsAbsentClasses pins the "no vacuous row" rule: a class no page
// carried must be absent, not reported as an all-zero split, which would read as a loss that
// never happened.
func TestCompactionContractFromOmitsAbsentClasses(t *testing.T) {
	pages := []ctxplan.Page{
		{ID: "a", Kind: ctxplan.KindTranscriptProse},
		{ID: "b", Kind: ctxplan.KindTranscriptProse},
	}
	els := [][]byte{[]byte("x"), []byte("yy")}
	c := compactionContractFrom(pages, els, ctxplan.EvictionPlan{Keep: []string{"a"}, Evict: []string{"b"}})
	if c == nil {
		t.Fatal("contract is nil for a plan that evicted a page")
	}
	if len(c.PreservedClasses) != 1 || c.PreservedClasses[0].Class != "EVICTABLE" {
		t.Errorf("preserved_classes = %+v, want only the EVICTABLE row the pages actually carried", c.PreservedClasses)
	}
	if len(c.ReplayablePageDigests) != 0 {
		t.Errorf("digests = %v, want none (no REPLAYABLE page was evicted)", c.ReplayablePageDigests)
	}
}

// TestCompactionContractFromNilWhenNothingEvicted pins the emission floor: a boundary that
// dropped no page is not announced at all, so neither reader learns to ignore a routinely-empty
// field.
func TestCompactionContractFromNilWhenNothingEvicted(t *testing.T) {
	pages, els := contractFixture()
	if c := compactionContractFrom(pages, els, ctxplan.EvictionPlan{Keep: []string{"p1", "p2", "p3"}}); c != nil {
		t.Errorf("contract = %+v for a plan that evicted nothing, want nil", c)
	}
}

// TestCompactionContractDigestsAreCapped pins the bound that keeps the record from inflating the
// very response it exists to shrink: a shed far larger than the cap still yields at most
// maxReplayablePageDigests handles.
func TestCompactionContractDigestsAreCapped(t *testing.T) {
	n := maxReplayablePageDigests + 10
	var pages []ctxplan.Page
	var els [][]byte
	var evict []string
	for i := 0; i < n; i++ {
		id := "r" + string(rune('a'+i%26)) + string(rune('a'+i/26))
		pages = append(pages, ctxplan.Page{ID: id, Kind: ctxplan.KindToolSchema})
		els = append(els, []byte(id))
		evict = append(evict, id)
	}
	c := compactionContractFrom(pages, els, ctxplan.EvictionPlan{Evict: evict})
	if c == nil {
		t.Fatal("contract is nil for a large eviction")
	}
	if len(c.ReplayablePageDigests) != maxReplayablePageDigests {
		t.Errorf("digests = %d, want the cap %d", len(c.ReplayablePageDigests), maxReplayablePageDigests)
	}
	// The cap bounds the handle list, never the accounting: every evicted page's bytes count.
	if c.PreservedClasses[0].Evicted != n {
		t.Errorf("evicted count = %d, want %d — the digest cap must not shrink the split", c.PreservedClasses[0].Evicted, n)
	}
}

// TestCompactionContractTakeIsOnce pins the TAKE-ONCE latch the file's header calls out against a
// seen-set: the reporting turn CONSUMES the record, so a later turn on the same trace cannot
// re-announce a boundary it never crossed — while a genuinely new boundary re-arms it.
func TestCompactionContractTakeIsOnce(t *testing.T) {
	s := &Server{}
	first := &CompactionContract{ContractVersion: compactionContractVersion, EvictedByteCount: 11}

	if got := s.takeCompactionContract("t1"); got != nil {
		t.Errorf("take before any boundary = %+v, want nil", got)
	}
	s.noteCompactionContract("t1", first)
	got := s.takeCompactionContract("t1")
	if got == nil || got.EvictedByteCount != 11 {
		t.Fatalf("first take = %+v, want the noted contract", got)
	}
	if again := s.takeCompactionContract("t1"); again != nil {
		t.Errorf("second take = %+v, want nil — the record must be consumed, not latched open", again)
	}
	// A new boundary on the same trace re-arms: take-once is per boundary, not per session.
	s.noteCompactionContract("t1", &CompactionContract{ContractVersion: compactionContractVersion, EvictedByteCount: 22})
	if got := s.takeCompactionContract("t1"); got == nil || got.EvictedByteCount != 22 {
		t.Errorf("take after a second boundary = %+v, want the newer contract", got)
	}
	// A trace that never compacted is unaffected by another trace's boundary.
	s.noteCompactionContract("t2", first)
	if got := s.takeCompactionContract("t3"); got != nil {
		t.Errorf("take on an unrelated trace = %+v, want nil", got)
	}
}

func TestCompactionContractProjectsBoundedRetentionEffects(t *testing.T) {
	pages := []ctxplan.Page{
		{ID: "keep", Kind: ctxplan.KindTranscriptProse, Retention: []ctxplan.RetentionAnnotation{{Intent: ctxplan.RetentionKeep, Source: "deterministic:needle", ReasonCode: "private_reason"}}},
		{ID: "drop", Kind: ctxplan.KindTranscriptProse, Retention: []ctxplan.RetentionAnnotation{{Intent: ctxplan.RetentionDrop, Source: "agent:ranker", ReasonCode: "private_reason"}}},
	}
	plan := ctxplan.EvictionPlan{Keep: []string{"keep"}, Evict: []string{"drop"}}
	contract := compactionContractFrom(pages, [][]byte{[]byte("secret keep content"), []byte("secret drop content")}, plan)
	if contract == nil {
		t.Fatal("contract = nil, want a projected eviction")
	}
	want := []RetentionEffect{
		{PageID: "keep", Intent: "keep", Source: "deterministic:needle", Outcome: "kept"},
		{PageID: "drop", Intent: "drop", Source: "agent:ranker", Outcome: "evicted"},
	}
	if !reflect.DeepEqual(contract.RetentionEffects, want) {
		t.Fatalf("RetentionEffects = %+v, want %+v", contract.RetentionEffects, want)
	}
	wire, err := json.Marshal(contract)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"secret keep content", "secret drop content", "private_reason"} {
		if bytes.Contains(wire, []byte(forbidden)) {
			t.Fatalf("contract leaked %q: %s", forbidden, wire)
		}
	}
}

func TestCompactionContractCapsRetentionEffects(t *testing.T) {
	pages := make([]ctxplan.Page, maxRetentionEffects+5)
	els := make([][]byte, len(pages))
	plan := ctxplan.EvictionPlan{}
	for i := range pages {
		id := fmt.Sprintf("p%d", i)
		pages[i] = ctxplan.Page{ID: id, Kind: ctxplan.KindTranscriptProse, Retention: []ctxplan.RetentionAnnotation{{Intent: ctxplan.RetentionDrop, Source: "agent:ranker"}}}
		els[i] = []byte(id)
		plan.Evict = append(plan.Evict, id)
	}
	contract := compactionContractFrom(pages, els, plan)
	if got := len(contract.RetentionEffects); got != maxRetentionEffects {
		t.Fatalf("retention effects = %d, want cap %d", got, maxRetentionEffects)
	}
}
