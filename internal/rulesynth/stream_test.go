package rulesynth

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// streamEvent builds an adjudication event carrying a command-bearing call, the shape
// the kernel emits.
func streamEvent(kind abi.EventKind, vk abi.VerdictKind, tool, cmd string) abi.Event {
	b, _ := json.Marshal(map[string]string{"command": cmd})
	return abi.Event{
		Kind:    kind,
		Call:    &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: b}},
		Verdict: &abi.Verdict{Kind: vk},
	}
}

// TestEmitCapturesOnlyAdmittedNearMiss pins the Emit filter: a near-miss is mined only
// from an ADMITTED (Allow) call on the verdict-resolved event (EvDecide). A deny is
// already caught, an unguarded command has nothing to guard, and a non-EvDecide event
// for the same admitted call must not double-record it.
func TestEmitCapturesOnlyAdmittedNearMiss(t *testing.T) {
	corpus := NewNearMissCorpus()
	h := NewHarvester(corpus, guarded)

	nearMiss := `ruby -e 'File.write("internal/adjudicator/x.go","")'`

	// 1. Admitted near-miss on the verdict-resolved event -> captured.
	h.Emit(streamEvent(abi.EvDecide, abi.VerdictAllow, "Bash", nearMiss))
	// 2. A DENY of the same command is already caught -> not a near-miss.
	h.Emit(streamEvent(abi.EvDecide, abi.VerdictDeny, "Bash", nearMiss))
	// 3. An admitted command naming no guarded tree -> nothing to mine.
	h.Emit(streamEvent(abi.EvDecide, abi.VerdictAllow, "Bash", `ruby -e 'File.write("/tmp/x","")'`))
	// 4. The SAME admitted near-miss on a non-decide event (EvDispatch fires for every
	//    allowed call too) must not double-record — Emit keys on EvDecide only.
	h.Emit(streamEvent(abi.EvDispatch, abi.VerdictAllow, "Bash", nearMiss))

	rows := corpus.Rows()
	if len(rows) != 1 {
		t.Fatalf("corpus = %+v; want exactly 1 (only the admitted EvDecide near-miss)", rows)
	}
	if rows[0].GuardedGlob != "internal/adjudicator/" {
		t.Fatalf("mined glob = %q, want internal/adjudicator/", rows[0].GuardedGlob)
	}
}

// --- a real-kernel integration: prove the harvester mines the ACTUAL log ----------

type inlineRes struct{}

func (inlineRes) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) { return r.Inline, nil }
func (inlineRes) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	return abi.Ref{Kind: abi.RefInline, Inline: append([]byte(nil), b...), Len: int64(len(b))}, nil
}

type inlineBackend struct{}

func (inlineBackend) Resolver() abi.Resolver { return inlineRes{} }
func (inlineBackend) Caps() []abi.Capability { return nil }

type countEngine struct{ n int64 }

func (e *countEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	e.n++
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}, nil
}
func (e *countEngine) Caps() []abi.Capability { return nil }

func bash(cmd string) *abi.ToolCall {
	b, _ := json.Marshal(map[string]string{"command": cmd})
	return &abi.ToolCall{Tool: "Bash", Args: abi.Ref{Kind: abi.RefInline, Inline: b}}
}

// TestHarvesterMinesLiveStream drives a REAL kernel over a realistic coding-agent floor
// (Bash allowed, the harness trees guarded) with the harvester attached as an Emitter,
// and proves the rung-1 loop end to end: an unrecognized shell write that slips the
// floor is mined from the live LOG, an already-caught write and a benign write are not,
// and the mined corpus feeds Propose into the next structural rule.
func TestHarvesterMinesLiveStream(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})

	corpus := NewNearMissCorpus()
	abi.RegisterEmitter(NewHarvester(corpus, guarded))

	// The realistic floor: a coding agent's Bash is ALLOWED; the harness trees are
	// guarded. So a recognized shell write is denied, an unrecognized one slips.
	abi.RegisterAdjudicator(100, adjudicator.New(adjudicator.Policy{
		Allow:           map[string]bool{"Bash": true},
		SelfModifyGlobs: guarded,
	}))
	abi.RegisterEngine("e", &countEngine{})

	k := kernel.New("e")
	ctx := context.Background()

	// near-miss: ruby -e writes into a guarded tree by a verb not in shellWriteVerbs -> admitted.
	k.Syscall(ctx, bash(`ruby -e 'File.write("internal/adjudicator/decide.go","x")'`))
	// already caught: sed -i is a recognized write verb -> denied, never a near-miss.
	k.Syscall(ctx, bash(`sed -i s/a/b/ internal/adjudicator/decide.go`))
	// benign: a write to an unguarded path -> admitted but names no guarded tree.
	k.Syscall(ctx, bash(`tee /tmp/out.txt`))

	rows := corpus.Rows()
	if len(rows) != 1 {
		t.Fatalf("want exactly 1 mined near-miss (the ruby -e write), got %d: %+v", len(rows), rows)
	}
	if rows[0].GuardedGlob != "internal/adjudicator/" {
		t.Fatalf("near-miss glob = %q, want internal/adjudicator/", rows[0].GuardedGlob)
	}

	// The mined LOG must feed the proposer: Propose yields the ruby -e candidate, and
	// because it guards a harness tree it is flagged SelfModify (require-witness).
	cands := Propose(rows)
	if len(cands) != 1 || cands[0].Verb != "ruby -e" {
		t.Fatalf("Propose(live corpus) = %+v, want one ruby -e candidate", cands)
	}
	if !cands[0].SelfModify {
		t.Fatalf("a candidate guarding internal/adjudicator must be flagged SelfModify")
	}
}
