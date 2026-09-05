package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// compaction_contract_note_test.go — the #2422 ACCEPTANCE witness.
//
// compaction_contract_test.go tests the projection in isolation. This tests the WIRING: a
// real compaction boundary (the #2421 survival gate, on the passthrough where it is the only
// place it runs) must reach a completed turn on BOTH channels — the in-band `[fak]` note the
// model reads and the `fak.compaction` extension an orchestrator reads — and must do so
// exactly once per boundary rather than once per turn.
//
// The bug this pins is not hypothetical: the contract was taken from the trace and rendered
// into the note while the constructed turn dropped the record on the floor, so
// `fak.compaction` was silently never emitted and the two channels disagreed about every
// boundary the gateway crossed. A note-only test passes on that tree.

// armContractBoundary puts srv in the passthrough shape the survival gate requires and
// returns a restore func. The gate only runs on the anthropic→anthropic passthrough
// (compactAnthropicRawWithReason bails on every other wire), so a boundary cannot be produced
// any other way; the reporting turn is then completed with the server's own local planner,
// which is the half a test may not dial upstream for.
func armContractBoundary(srv *Server) func() {
	prev := srv.planner
	srv.planner = &agent.HTTPPlanner{Provider: agent.ProviderAnthropic}
	srv.compactHistoryBudget = 1200
	srv.compactAnchorHead = true
	b := 1200
	a := 1
	prevVC := srv.VersionedConfig()
	_, _, _ = srv.PatchScalarConfig(ScalarConfigPatch{
		CompactHistoryBudget: &b,
		CompactAnchorHead:    &a,
	})
	return func() {
		srv.planner = prev
		_, _ = srv.SetScalarConfig(prevVC.Config)
	}
}

// crossCompactionBoundary drives ONE real compaction on trace and returns the body it
// compacted. It fails the test rather than returning quietly when the fixture did not
// actually shed: a boundary that never happened would make every assertion below vacuous.
func crossCompactionBoundary(t *testing.T, srv *Server, trace string, nMsgs int) []byte {
	t.Helper()
	defer armContractBoundary(srv)()
	raw := pinSurvivalWireBody(t, nMsgs, 10, 3)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	fired, reason := srv.compactAnthropicRawWithReason(req, 1000, trace)
	if !fired {
		t.Fatalf("fixture sanity: the %d-message body did not compact (reason %q) — there is no boundary to report", nMsgs, reason)
	}
	return raw
}

// completeContractTurn completes one turn on trace with the server's local planner. This is
// the seam under test: the turn builder is where the boundary's pending contract is consumed,
// rendered into the leading text block, and attached to the response extension.
func completeContractTurn(t *testing.T, srv *Server, trace string, raw []byte) *anthropicTurn {
	t.Helper()
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode turn body: %v", err)
	}
	turn, err := srv.completeAnthropicTurn(context.Background(), req, trace, servedSessionTurn{}, "", "", "", "")
	if err != nil {
		t.Fatalf("completeAnthropicTurn: %v", err)
	}
	return turn
}

// boundaryNoteOf returns the turn's leading in-band compaction note, or "" when the turn
// carries none. The note is PREPENDED, so it can only be block 0 — one that landed anywhere
// else would be read after the per-call verdicts it has to precede.
func boundaryNoteOf(turn *anthropicTurn) string {
	if len(turn.Blocks) == 0 {
		return ""
	}
	if b := turn.Blocks[0]; b.Type == "text" && strings.HasPrefix(b.Text, "[fak] context compaction boundary") {
		return b.Text
	}
	return ""
}

// contractClassCensus counts the PRE-compaction body's pages per survival class, read straight
// from the classifier. It is the INDEPENDENT check the contract is measured against: the
// record's per-class denominators come from the eviction plan, so a class the body never held
// — or a denominator larger than the body's own page count for that class — means the record
// is describing something other than the boundary it was emitted for.
func contractClassCensus(t *testing.T, raw []byte) map[string]int {
	t.Helper()
	pages, _, ok := anthropicSurvivalPages(raw)
	if !ok {
		t.Fatal("fixture sanity: the body has no classifiable eviction domain")
	}
	census := map[string]int{}
	for _, p := range pages {
		census[p.Class().String()]++
	}
	return census
}

func TestCompactionContractNote(t *testing.T) {
	srv := newTestServer(t)
	const trace = "gw-compaction-contract"

	// --- Boundary 1: both channels carry the SAME record --------------------
	raw := crossCompactionBoundary(t, srv, trace, 160)
	turn := completeContractTurn(t, srv, trace, raw)

	if turn.Compaction == nil {
		t.Fatalf("a turn that crossed a compaction boundary carries no contract — the in-band note may " +
			"still render, but `fak.compaction` is then silently never emitted and an orchestrator that " +
			"reads no prose cannot see the boundary at all")
	}
	if ext := turn.fakExt(); ext == nil || ext.Compaction != turn.Compaction {
		t.Fatalf("fak extension = %+v, want it to carry the SAME record as the turn — taking a take-once "+
			"contract twice (once per channel) would serve one and starve the other", ext)
	}

	// The contract as an orchestrator actually receives it: serialized on the response, exactly
	// as every anthropic response writer builds it.
	body, err := json.Marshal(anthropicMessageResponse{
		ID: turn.ID, Type: "message", Role: "assistant", Model: turn.Model,
		Content: turn.Blocks, StopReason: turn.Stop, Usage: turn.Usage, Fak: turn.fakExt(),
	})
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	var wire struct {
		Fak *struct {
			Compaction *CompactionContract `json:"compaction"`
		} `json:"fak"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if wire.Fak == nil || wire.Fak.Compaction == nil {
		t.Fatalf("the serialized response carries no fak.compaction:\n%s", body)
	}
	got := wire.Fak.Compaction
	if got.ContractVersion != compactionContractVersion {
		t.Errorf("contract_version = %d, want %d", got.ContractVersion, compactionContractVersion)
	}
	if got.Instruction != compactionContractInstruction {
		t.Errorf("instruction = %q, want the closed continuation token %q", got.Instruction, compactionContractInstruction)
	}
	if len(got.PreservedClasses) == 0 {
		t.Fatalf("preserved_classes is empty — the record says nothing about what survived: %+v", got)
	}

	// preserved_classes must describe THIS body's survival sets.
	census := contractClassCensus(t, raw)
	evictedSomething := false
	for _, split := range got.PreservedClasses {
		held, ok := census[split.Class]
		if !ok {
			t.Errorf("preserved_classes names class %q, of which the pre-compaction body held no page: census %v", split.Class, census)
			continue
		}
		if split.Kept+split.Evicted > held {
			t.Errorf("class %q reports %d kept + %d evicted = %d pages, but the body held only %d of that class",
				split.Class, split.Kept, split.Evicted, split.Kept+split.Evicted, held)
		}
		if split.Evicted > 0 {
			evictedSomething = true
		}
		// The survival contract this record projects (#2421): a PINNED page is never evicted, so
		// a contract reporting one would be announcing a loss the gate forbids.
		if split.Class == ctxplan.ClassPinned.String() && split.Evicted != 0 {
			t.Errorf("PINNED reports %d evicted pages; the survival gate must never evict one", split.Evicted)
		}
	}
	if !evictedSomething {
		t.Errorf("no class reports an eviction, yet a contract was emitted — a boundary that shed nothing must emit nothing: %+v", got)
	}
	if got.EvictedByteCount <= 0 || got.EvictedByteCount >= len(raw) {
		t.Errorf("evicted_byte_count = %d, want a positive count below the whole body (%d)", got.EvictedByteCount, len(raw))
	}
	// The digest list is capped, so it may be SHORTER than the replayable eviction count — but
	// never longer, which would promise recovery of pages the plan never evicted.
	if n := replayableEvictedCount(got); len(got.ReplayablePageDigests) > n {
		t.Errorf("%d replayable digests for %d replayable evictions — a digest is a promise those bytes are recoverable",
			len(got.ReplayablePageDigests), n)
	}

	// The model-facing half of the same record.
	note := boundaryNoteOf(turn)
	if note == "" {
		t.Fatalf("the turn carries no leading in-band boundary note; blocks = %+v", turn.Blocks)
	}
	if note != compactionContractNote(turn.Compaction) {
		t.Errorf("the in-band note is not a rendering of the turn's own contract:\n note: %s\n want: %s",
			note, compactionContractNote(turn.Compaction))
	}
	if !strings.Contains(note, compactionContractInstruction) {
		t.Errorf("the note omits the continuation instruction — reporting only the loss is exactly the input that makes a model wrap up early: %s", note)
	}

	// --- Once per BOUNDARY, not once per turn -------------------------------
	// The next turn on the same trace crossed no boundary, so it must announce none on either
	// channel. A latched record would tell the model its history was shed again when it was not.
	quiet := completeContractTurn(t, srv, trace, raw)
	if quiet.Compaction != nil {
		t.Errorf("a turn that crossed no boundary carries a contract: %+v", quiet.Compaction)
	}
	if n := boundaryNoteOf(quiet); n != "" {
		t.Errorf("a turn that crossed no boundary emitted a boundary note: %s", n)
	}
	if ext := quiet.fakExt(); ext != nil && ext.Compaction != nil {
		t.Errorf("a turn that crossed no boundary carries fak.compaction: %+v", ext.Compaction)
	}

	// --- The 3-compaction fixture the done condition names ------------------
	// Three boundaries, each followed by a reporting turn AND a quiet turn: exactly three
	// notes, and each describes its OWN boundary rather than repeating the first.
	notes := 0
	distinct := map[string]bool{}
	for round := 0; round < 3; round++ {
		shed := crossCompactionBoundary(t, srv, trace, 160+40*round)
		reporting := completeContractTurn(t, srv, trace, shed)
		if n := boundaryNoteOf(reporting); n != "" {
			notes++
			distinct[n] = true
		} else {
			t.Errorf("boundary %d produced no note", round)
		}
		if reporting.Compaction == nil {
			t.Errorf("boundary %d produced no fak.compaction record", round)
		}
		if n := boundaryNoteOf(completeContractTurn(t, srv, trace, shed)); n != "" {
			t.Errorf("boundary %d was announced a second time, on a turn that crossed no boundary: %s", round, n)
		}
	}
	if notes != 3 {
		t.Fatalf("the 3-compaction fixture emitted %d notes, want exactly one per boundary", notes)
	}
	if len(distinct) != 3 {
		t.Errorf("the 3 boundaries emitted %d DISTINCT notes; each must describe its own shed, not repeat the first: %v", len(distinct), distinct)
	}
}
