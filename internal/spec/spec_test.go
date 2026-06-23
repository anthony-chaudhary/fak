package spec

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// ---------------------------------------------------------------------------
// Test fixtures: small CPU synthetic models (no GPU, no weight download), the same
// shape cmd/polymodelbench uses. VocabSize 256 means every byte is a valid token, so
// any drafted id is valid for the target.
// ---------------------------------------------------------------------------

func cfg(hidden, layers, nHeads, nKV, headDim, inter int) model.Config {
	return model.Config{
		HiddenSize:        hidden,
		NumLayers:         layers,
		NumHeads:          nHeads,
		NumKVHeads:        nKV,
		HeadDim:           headDim,
		IntermediateSize:  inter,
		VocabSize:         256,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1, // never early-stop; decode a fixed length
	}
}

func bytesToIDs(b []byte) []int {
	ids := make([]int, len(b))
	for i, c := range b {
		ids[i] = int(c)
	}
	return ids
}

// greedyDecode is plain autoregressive greedy decoding — the lossless reference.
func greedyDecode(m *model.Model, prompt []int, n int) []int {
	s := m.NewSession()
	logits := s.Prefill(prompt)
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		t := argmax(logits)
		out = append(out, t)
		logits = s.Step(t)
	}
	return out
}

// modelDrafter is a real co-resident draft model: a (typically cheaper) model whose
// argmax proposes the draft, threading its own KV so it continues from the committed
// context. It rolls back its OWN speculative span in Commit with the same bit-exact
// Evict — a wrong draft costs only that rollback, never target correctness.
type modelDrafter struct {
	s        *model.Session
	logits   []float32
	specFrom int
}

func newModelDrafter(m *model.Model, prompt []int) *modelDrafter {
	s := m.NewSession()
	return &modelDrafter{s: s, logits: s.Prefill(prompt)}
}

func (d *modelDrafter) Draft(k int) []int {
	d.specFrom = d.s.Cache.Len()
	drafts := make([]int, 0, k)
	l := d.logits
	for j := 0; j < k; j++ {
		t := argmax(l)
		drafts = append(drafts, t)
		l = d.s.Step(t)
	}
	return drafts
}

func (d *modelDrafter) Commit(committed []int) {
	// Drop the speculative draft span, then re-thread the truly committed tokens so the
	// next Draft continues from the real context.
	if grew := d.s.Cache.Len() - d.specFrom; grew > 0 {
		d.s.Cache.Evict(d.specFrom, grew)
	}
	for _, t := range committed {
		d.logits = d.s.Step(t)
	}
}

// advDrafter is an ADVERSARIAL proposer (a deterministic counter independent of the
// target) that forces rejections nearly every round, so the bit-exact squash path runs
// hard. It is not a model — only the target has a KV cache to roll back.
type advDrafter struct{ round int }

func (d *advDrafter) Draft(k int) []int {
	drafts := make([]int, 0, k)
	for j := 0; j < k; j++ {
		drafts = append(drafts, ((d.round*13+j*7+1)%256+256)%256)
	}
	return drafts
}

func (d *advDrafter) Commit(_ []int) { d.round++ }

func assertEqualTokens(t *testing.T, label string, got, want []int) {
	t.Helper()
	if len(got) < len(want) {
		t.Fatalf("%s: short decode got %d want %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: LOSSLESS VIOLATED at token %d: speculative=%d greedy=%d "+
				"(the KV rollback of a rejected draft was not bit-exact)", label, i, got[i], want[i])
		}
	}
}

// assertContinuationsMatch proves a rolled-back session is bit-exact to a never-drafted
// one BEHAVIORALLY: it greedily decodes `steps` tokens from each (the internal K/V
// slices are unexported, so behavior is the observable). A non-bit-exact Evict leaves a
// survivor's K rotated at the wrong position, which diverges the continuation within a
// few steps. Both sessions must be at the same cached position on entry.
func assertContinuationsMatch(t *testing.T, label string, a, b *model.Session, seed, steps int) {
	t.Helper()
	if a.Cache.Len() != b.Cache.Len() {
		t.Fatalf("%s: cache length mismatch %d vs %d before continuation", label, a.Cache.Len(), b.Cache.Len())
	}
	la, lb := a.Step(seed), b.Step(seed)
	for i := 0; i < steps; i++ {
		ta, tb := argmax(la), argmax(lb)
		if ta != tb {
			t.Fatalf("%s: continuation diverged at step %d: %d vs %d (Evict was not bit-exact)", label, i, ta, tb)
		}
		la, lb = a.Step(ta), b.Step(tb)
	}
}

// ---------------------------------------------------------------------------
// The headline witness: greedy speculative decode THROUGH the ProvisionalSink seam is
// token-identical to plain greedy — the native form of the cmd/polymodelbench check.
// ---------------------------------------------------------------------------

func TestSpeculativeGreedyLossless(t *testing.T) {
	t.Setenv(polymodel.FlagEnv, "on")
	abi.ResetForTest()
	defer abi.ResetForTest()

	sink := Install()
	if sink == nil {
		t.Fatal("Install returned nil with the lane enabled")
	}

	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	prompt := bytesToIDs([]byte("speculative decoding is lossless when verified greedily"))
	const N, K = 24, 4
	want := greedyDecode(target, prompt, N)

	// (a) Real co-resident draft model (cheaper, different weights). Whatever the
	//     acceptance, the output must be token-identical to greedy.
	draft := model.NewSynthetic(cfg(32, 2, 2, 1, 16, 64))
	gotA, draftedA, acceptedA, rolledA := SpeculativeGreedy(
		context.Background(), sink, target.NewSession(), prompt, N, K,
		newModelDrafter(draft, prompt))
	assertEqualTokens(t, "real-draft-model", gotA, want)
	if sink.OpenCount() != 0 {
		t.Fatalf("real-draft-model: %d speculations left unresolved (a leak)", sink.OpenCount())
	}
	t.Logf("real draft model: proposed %d, accepted %d, rolled back %d", draftedA, acceptedA, rolledA)

	// (b) Adversarial proposer: forces rejections, so the bit-exact Evict path runs
	//     hard. Output STILL token-identical, and rollbacks must actually happen — else
	//     the witness is vacuous (it never exercised the squash).
	gotB, draftedB, acceptedB, rolledB := SpeculativeGreedy(
		context.Background(), sink, target.NewSession(), prompt, N, K, &advDrafter{})
	assertEqualTokens(t, "adversarial-draft", gotB, want)
	if rolledB == 0 {
		t.Fatal("VACUOUS WITNESS: adversarial draft caused 0 rollbacks — the squash path was never exercised")
	}
	if sink.OpenCount() != 0 {
		t.Fatalf("adversarial-draft: %d speculations left unresolved (a leak)", sink.OpenCount())
	}
	t.Logf("adversarial draft: proposed %d, accepted %d, rolled back %d bit-exact spans", draftedB, acceptedB, rolledB)
}

// ---------------------------------------------------------------------------
// The reserved OpsSpec ops resolve provisional state through the frozen kernel op
// table: OpSpecSquash → the registered Sink's Rollback → bit-exact KVCache.Evict.
// ---------------------------------------------------------------------------

func TestReservedOpsSpecResolveThroughKernelTable(t *testing.T) {
	t.Setenv(polymodel.FlagEnv, "on")
	abi.ResetForTest()
	defer abi.ResetForTest()

	sink := Install()
	if sink == nil {
		t.Fatal("Install returned nil with the lane enabled")
	}

	// Both reserved ops are registered and sit inside the OpsSpec range.
	for _, code := range []abi.OpCode{OpSpecCommit, OpSpecSquash} {
		if uint32(code) < abi.OpsSpec.Lo || uint32(code) >= abi.OpsSpec.Hi {
			t.Errorf("op %d is outside the reserved OpsSpec range [%d,%d)", code, abi.OpsSpec.Lo, abi.OpsSpec.Hi)
		}
		if _, ok := abi.LookupOp(code); !ok {
			t.Errorf("reserved op %d was not registered by Install", code)
		}
	}

	// Build a real session, append a span, register it as provisional, then drive the
	// squash op through the op table and confirm the span was evicted and the cache is
	// bit-exact to a never-drafted one.
	m := model.NewSynthetic(cfg(48, 3, 3, 1, 16, 96))
	prompt := bytesToIDs([]byte("the cache is the lever"))

	s := m.NewSession()
	baseLogits := s.Prefill(prompt)
	base := s.Cache.Len()
	last := argmax(baseLogits)
	for i := 0; i < 3; i++ { // append 3 speculative positions
		s.Step(last)
	}
	if s.Cache.Len() != base+3 {
		t.Fatalf("setup: cache len %d, want %d", s.Cache.Len(), base+3)
	}
	const txn = abi.TxnID(7)
	sink.Open(txn, EpochReject, s.Cache, base, 3)

	squash, ok := abi.LookupOp(OpSpecSquash)
	if !ok {
		t.Fatal("OpSpecSquash not registered")
	}
	res, v := squash.Invoke(context.Background(), nil, &abi.ToolCall{
		Op:   OpSpecSquash,
		Txn:  txn,
		Spec: abi.SpeculationContext{Speculative: true, Epoch: EpochReject},
	})
	if v.Kind != abi.VerdictAllow {
		t.Errorf("squash verdict = %v, want Allow", v.Kind)
	}
	if res.Outcome != abi.OutcomeSquashed {
		t.Errorf("squash Outcome = %v, want OutcomeSquashed", res.Outcome)
	}
	if s.Cache.Len() != base {
		t.Fatalf("after squash: cache len %d, want %d (the provisional span was not evicted)", s.Cache.Len(), base)
	}
	if sink.OpenCount() != 0 {
		t.Fatalf("after squash: %d speculations still open", sink.OpenCount())
	}

	ref := m.NewSession()
	ref.Prefill(prompt)
	assertContinuationsMatch(t, "squash-bit-exact", s, ref, 42, 6)

	// OpSpecCommit on an unknown txn is a clean no-op committed.
	commit, _ := abi.LookupOp(OpSpecCommit)
	cres, _ := commit.Invoke(context.Background(), nil, &abi.ToolCall{Op: OpSpecCommit, Txn: 999})
	if cres.Outcome != abi.OutcomeCommitted {
		t.Errorf("commit Outcome = %v, want OutcomeCommitted", cres.Outcome)
	}
}

// ---------------------------------------------------------------------------
// Gating: Install is a no-op that touches no global registry while the lane is off.
// ---------------------------------------------------------------------------

func TestInstallNoopWhenDisabled(t *testing.T) {
	t.Setenv(polymodel.FlagEnv, "off")
	abi.ResetForTest()
	defer abi.ResetForTest()

	if s := Install(); s != nil {
		t.Fatal("Install must return nil when the poly-model lane is off")
	}
	if got := len(abi.ProvisionalSinks()); got != 0 {
		t.Fatalf("Install registered %d sink(s) while the lane is off", got)
	}
	for _, code := range []abi.OpCode{OpSpecCommit, OpSpecSquash} {
		if _, ok := abi.LookupOp(code); ok {
			t.Fatalf("Install registered op %d while the lane is off", code)
		}
	}
}

// ---------------------------------------------------------------------------
// Direct ProvisionalSink contract: Promote drops the bookkeeping (KV survives);
// Rollback evicts bit-exactly; both are idempotent.
// ---------------------------------------------------------------------------

func TestSinkPromoteAndRollback(t *testing.T) {
	m := model.NewSynthetic(cfg(32, 2, 2, 1, 16, 64))
	prompt := bytesToIDs([]byte("promote keeps, rollback retracts"))
	sink := NewSink()
	ctx := context.Background()

	// Promote keeps the appended positions.
	{
		s := m.NewSession()
		l := s.Prefill(prompt)
		base := s.Cache.Len()
		last := argmax(l)
		s.Step(last)
		s.Step(last)
		sink.Open(1, EpochAccept, s.Cache, base, 2)
		if err := sink.Promote(ctx, 1, EpochAccept); err != nil {
			t.Fatalf("Promote: %v", err)
		}
		if s.Cache.Len() != base+2 {
			t.Fatalf("Promote evicted KV: len %d, want %d", s.Cache.Len(), base+2)
		}
		if err := sink.Promote(ctx, 1, EpochAccept); err != nil { // idempotent
			t.Fatalf("Promote (2nd): %v", err)
		}
		if sink.OpenCount() != 0 {
			t.Fatalf("Promote left %d open", sink.OpenCount())
		}
	}

	// Rollback evicts the span bit-exactly and is idempotent.
	{
		s := m.NewSession()
		l := s.Prefill(prompt)
		base := s.Cache.Len()
		last := argmax(l)
		s.Step(last)
		s.Step(last)
		sink.Open(2, EpochReject, s.Cache, base, 2)
		if err := sink.Rollback(ctx, 2, EpochReject); err != nil {
			t.Fatalf("Rollback: %v", err)
		}
		if s.Cache.Len() != base {
			t.Fatalf("Rollback did not evict: len %d, want %d", s.Cache.Len(), base)
		}
		if err := sink.Rollback(ctx, 2, EpochReject); err != nil { // idempotent, no double-evict
			t.Fatalf("Rollback (2nd): %v", err)
		}
		if s.Cache.Len() != base {
			t.Fatalf("second Rollback mutated the cache: len %d, want %d", s.Cache.Len(), base)
		}
		ref := m.NewSession()
		ref.Prefill(prompt)
		assertContinuationsMatch(t, "rollback-bit-exact", s, ref, 42, 6)
	}
}
