package agent

// syspromptmmu_seam_test.go — the #1322 acceptance witness for the owned loop's first
// non-test importer of internal/syspromptmmu. It proves: the system block is built via
// syspromptmmu.BuildSystemValue with fak-concepts pinned FIRST, an overlay item is authored
// through the witness-gated GateEdit/ApplyEdit, AuditRealizedPrefix confirms the realized
// resident prefix stayed cache-stable, and the resident head is reused VERBATIM as the
// overlay grows (the "without re-serializing the prompt head" win). It also pins the
// fail-closed gate: no witness ⇒ no authored overlay.

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

// spmmuPassWitness is the injected success predicate ApplyEdit gates an authored overlay
// item on — the external witness bus (the agent never grades its own edit).
func spmmuPassWitness(syspromptmmu.BaseEdit) bool { return true }

// spmmuDecodeBlocks decodes an Anthropic system[] JSON value into its text blocks.
func spmmuDecodeBlocks(t *testing.T, value []byte) []struct {
	Type string `json:"type"`
	Text string `json:"text"`
} {
	t.Helper()
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(value, &blocks); err != nil {
		t.Fatalf("decode system value: %v", err)
	}
	return blocks
}

// TestBuildOwnedSystemBlockPinsSpineFirstAndStaysCacheStable is the #1322 acceptance
// witness: the owned loop authors/queries its own system block — spine pinned first, overlay
// authored through ApplyEdit, realized prefix proven cache-stable.
func TestBuildOwnedSystemBlockPinsSpineFirstAndStaysCacheStable(t *testing.T) {
	items := [][]byte{
		[]byte("capability card: search_flights(origin, destination, date) — queried on demand"),
		[]byte("capability card: book_flight(flight_id) — queried on demand"),
	}
	b := BuildOwnedSystemBlock(items, spmmuPassWitness)

	// Every authored item was admitted past the witness gate.
	if b.Overlays != len(items) {
		t.Fatalf("admitted overlays = %d, want %d; refused=%v", b.Overlays, len(items), b.Refused)
	}

	// Cache-stable: the realized resident prefix equals the planned spine (Rung-6 audit).
	if !b.CacheStable() {
		t.Fatalf("system block not cache-stable: audit status = %q (want %q)", b.Audit.Status, syspromptmmu.AuditOK)
	}
	if b.Audit.GotDigest != b.Audit.ExpectDigest {
		t.Fatalf("realized spine digest %q != planned %q", b.Audit.GotDigest, b.Audit.ExpectDigest)
	}

	// fak-concepts pinned FIRST: the first realized block is fak's spine identity, byte-for
	// byte the leading BaseContextPlan segment; the block count is resident + overlay.
	plan := syspromptmmu.BaseContextPlan()
	blocks := spmmuDecodeBlocks(t, b.Value)
	if len(blocks) != len(plan)+len(items) {
		t.Fatalf("realized %d blocks, want %d resident + %d overlay", len(blocks), len(plan), len(items))
	}
	if blocks[0].Text != string(plan[0].Content) {
		t.Fatalf("first block is not the pinned spine:\n got: %q\nwant: %q", blocks[0].Text, string(plan[0].Content))
	}

	// The cache-stability WIN: the resident head is NOT re-serialized as the overlay grows.
	// head0 (no overlay) is reused VERBATIM as the head of the with-overlay value — the
	// overlay rides strictly after the breakpoint, so the cached prefix bytes are untouched.
	head0 := OwnedResidentHead()
	if len(head0) == 0 {
		t.Fatal("OwnedResidentHead is empty")
	}
	residentBytes := head0[:len(head0)-1] // drop the trailing ']' — the head with the array still open
	if !bytes.HasPrefix(b.Value, residentBytes) {
		t.Fatal("resident head was re-serialized: the with-overlay value does not carry the no-overlay head verbatim")
	}
}

// TestBuildOwnedSystemBlockFailsClosedWithoutWitness pins the fail-closed gate: a nil
// witness admits no overlay (the agent never grades its own edit), so the block carries the
// bare spine — still cache-stable, because the spine is untouched.
func TestBuildOwnedSystemBlockFailsClosedWithoutWitness(t *testing.T) {
	items := [][]byte{[]byte("unwitnessed card — must be refused")}
	b := BuildOwnedSystemBlock(items, nil)

	if b.Overlays != 0 {
		t.Fatalf("a nil witness must admit no overlay, got %d", b.Overlays)
	}
	if len(b.Refused) != len(items) {
		t.Fatalf("refused %d items, want %d (every item gated)", len(b.Refused), len(items))
	}
	if got := b.Refused[0].Reason; got != syspromptmmu.EditRefusedNoWitness {
		t.Fatalf("refusal reason = %q, want %q", got, syspromptmmu.EditRefusedNoWitness)
	}
	if !b.CacheStable() {
		t.Fatalf("bare-spine block must be cache-stable; audit status = %q", b.Audit.Status)
	}
	// The refused-all block equals the bare resident head exactly (no overlay appended).
	if !bytes.Equal(b.Value, OwnedResidentHead()) {
		t.Fatal("refused-all block must equal the bare resident head")
	}
}
