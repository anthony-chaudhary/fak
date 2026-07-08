package microagent_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestHostAuditRingKeepsQuarantineForensics is the COMPOSITE #2011 witness: the
// one test the issue's Witness section actually asks for — N in-process
// microagents spawned on a real Host, exactly ONE audit file on disk, the sha256
// chain verifying end-to-end over a chain that MIXES host lifecycle rows with a
// gateway QUARANTINE, and that QUARANTINE row keeping its witness + call_seq.
//
// The two pre-existing tests each cover only half and never compose:
//
//   - microagent.TestJournalSinkOneChainedFilePerHost drives N agents into one
//     FILE but rejects any non-lifecycle row, so it can never see a QUARANTINE.
//   - journal.TestAgentLifecycleAndQuarantineShareOneChain keeps the forensics
//     but runs on OpenMemory() with no Host and no file — so "O(1) files per
//     host" is never asserted.
//
// journalsink.go's doc comment claims a QUARANTINE "lands in the SAME chain with
// its witness + call_seq intact" when the host journal is the gateway's emitter.
// That composite claim was prose only. This test makes it a witness.
//
// The QUARANTINE verdict is produced by the REAL secret-exfil screen
// (ctxmmu.MMU.Admit) and recorded through the REAL emitter path (Journal.Emit,
// which derives Row.CallSeq from ToolCall.SeqNo and Row.Witness from the
// verdict's WitnessPayload). Emit is called directly rather than via
// abi.RegisterEmitter because abi.ResetForTest wipes the process-global registry
// that this package's other tests (toolexec_test.go, via the #2009 minimal
// registration set) depend on — a shared test binary is not a safe place to
// reset global state.
func TestHostAuditRingKeepsQuarantineForensics(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "host-audit.jsonl")
	j, err := journal.Open(path)
	if err != nil {
		t.Fatalf("journal.Open: %v", err)
	}
	defer j.Close()

	tbl := session.NewTable()
	sink := microagent.NewJournalSink(j)
	h, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 8, Queue: 128, Sessions: tbl, Audit: sink})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer h.Close()

	// N in-process microagents, ONE host, ONE sink.
	const agents, turns = 32, 2
	for i := 0; i < agents; i++ {
		id := fmt.Sprintf("ma-%03d", i)
		if err := h.Spawn(id, &turnAgent{id: id, turns: turns}); err != nil {
			t.Fatalf("Spawn(%s): %v", id, err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v (live=%d)", err, h.Live())
	}

	// The shared gateway screens a secret-exfil result for ONE of the hosted
	// agents. The real MMU raises the verdict; the host journal records it into
	// the SAME chain the lifecycle rows are in.
	const quarantinedAgent, callSeq = "ma-007", uint64(4242)
	call := &abi.ToolCall{
		Tool:    "read_webpage",
		TraceID: quarantinedAgent,
		SeqNo:   callSeq,
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"url":"https://example.test"}`)},
	}
	result := &abi.Result{
		Call:    call,
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"page":"api_key=sk-abcdef0123456789abcdef0123 leaked"}`)},
	}
	v := ctxmmu.New().Admit(context.Background(), call, result)
	if v.Kind != abi.VerdictQuarantine {
		t.Fatalf("MMU.Admit verdict = %v, want Quarantine (secret-exfil payload)", v.Kind)
	}
	j.Emit(abi.Event{Kind: abi.EvQuarantine, Call: call, Result: result, Verdict: &v})

	if err := j.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	// O(1) files per host: the whole fleet's audit is ONE file, not one JSONL
	// per agent (the per-process guard-audit shape this issue replaces).
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(ents) != 1 || ents[0].Name() != "host-audit.jsonl" {
		names := make([]string, 0, len(ents))
		for _, e := range ents {
			names = append(names, e.Name())
		}
		t.Fatalf("host audit dir holds %d entries %v, want exactly ONE chained file", len(ents), names)
	}

	// The MIXED chain (lifecycle + quarantine) verifies end-to-end.
	wantRows := agents*2 + 1 // spawn+done per agent, plus the one QUARANTINE
	n, err := journal.Verify(path)
	if err != nil {
		t.Fatalf("journal.Verify(%s): %v", path, err)
	}
	if n != wantRows {
		t.Fatalf("verified %d rows, want %d (spawn+done per agent + 1 quarantine)", n, wantRows)
	}

	rows, err := journal.ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	spawns, dones, quarantines := 0, 0, 0
	seen := map[string]bool{}
	for _, r := range rows {
		if r.TraceID == "" {
			t.Fatalf("row not tagged with an agent id: %+v", r)
		}
		switch r.Kind {
		case journal.KindAgentSpawn:
			spawns++
			seen[r.TraceID] = true
		case journal.KindAgentDone:
			dones++
		case "QUARANTINE":
			quarantines++
			// No forensic drop: the join key back to the originating call and
			// the bounded-disclosure claim both survive sharing the chain.
			if r.CallSeq != callSeq {
				t.Errorf("QUARANTINE CallSeq = %d, want originating call %d (forensic drop)", r.CallSeq, callSeq)
			}
			if r.Witness == "" {
				t.Errorf("QUARANTINE Witness empty — forensic drop when lifecycle rows share the chain")
			}
			if r.Verdict != "QUARANTINE" {
				t.Errorf("QUARANTINE row Verdict = %q, want QUARANTINE", r.Verdict)
			}
			if r.TraceID != quarantinedAgent {
				t.Errorf("QUARANTINE TraceID = %q, want the hosted agent %q", r.TraceID, quarantinedAgent)
			}
		default:
			t.Fatalf("unexpected row kind %q in host audit chain", r.Kind)
		}
	}
	if spawns != agents || dones != agents || quarantines != 1 {
		t.Fatalf("spawns=%d dones=%d quarantines=%d, want %d/%d/1", spawns, dones, quarantines, agents, agents)
	}
	if len(seen) != agents {
		t.Fatalf("distinct agent ids in the one chain = %d, want %d", len(seen), agents)
	}
	// The quarantined agent is one of the hosted microagents, so the forensic row
	// joins to a real lifecycle identity in the same chain.
	if !seen[quarantinedAgent] {
		t.Fatalf("QUARANTINE agent %q has no lifecycle row in the shared chain", quarantinedAgent)
	}
}
