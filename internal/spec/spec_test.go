package spec

import (
	"context"
	"errors"
	"sync"
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

// scriptedPartialDrafter forces a PARTIAL acceptance every round (0 < Accepted < kk): its
// first proposed token is the target's true next token (the precomputed greedy sequence
// `want`, so it is accepted) and the rest are deliberately wrong (so they are rejected).
// This is the offset-sensitive case the full-accept / full-reject subtests miss: the
// squash evicts the tail span starting at from+Accepted with Accepted>0.
type scriptedPartialDrafter struct {
	want []int
	pos  int
}

func (d *scriptedPartialDrafter) Draft(k int) []int {
	drafts := make([]int, 0, k)
	for j := 0; j < k; j++ {
		idx := d.pos + j
		w := 0
		if idx < len(d.want) {
			w = d.want[idx]
		}
		if j == 0 {
			drafts = append(drafts, w) // the target's true next token → accepted
		} else {
			drafts = append(drafts, (w+1)%256) // guaranteed != the greedy token → rejected
		}
	}
	return drafts
}

func (d *scriptedPartialDrafter) Commit(committed []int) { d.pos += len(committed) }

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
// one by comparing the FULL LOGIT VECTORS (exact float32 equality), not just the argmax:
// the internal K/V slices are unexported, but a single corrupted surviving-KV byte
// changes the attention dot-products and therefore the logits, even when the argmax is
// unchanged. (Argmax alone is too weak — the synthetic model is a fixed-point attractor,
// so a zeroed survivor K still produces the same argmax; exact logits catch it.) Both
// sessions must be at the same cached position on entry. The structural max|Δ|=0 proof of
// Evict's survivor re-RoPE lives in internal/model/evict_test.go; this is the seam-level
// behavioral check that the squash routed through the ProvisionalSink is lossless.
func assertContinuationsMatch(t *testing.T, label string, a, b *model.Session, seed, steps int) {
	t.Helper()
	if a.Cache.Len() != b.Cache.Len() {
		t.Fatalf("%s: cache length mismatch %d vs %d before continuation", label, a.Cache.Len(), b.Cache.Len())
	}
	la, lb := a.Step(seed), b.Step(seed)
	for i := 0; i <= steps; i++ {
		if len(la) != len(lb) {
			t.Fatalf("%s: logit width mismatch at step %d: %d vs %d", label, i, len(la), len(lb))
		}
		for j := range la {
			if la[j] != lb[j] {
				t.Fatalf("%s: logits diverge at step %d, index %d: %v vs %v "+
					"(the squash was not bit-exact — a surviving KV byte differs)", label, i, j, la[j], lb[j])
			}
		}
		if i == steps {
			break
		}
		t := argmax(la)
		la, lb = a.Step(t), b.Step(t)
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

	// (c) Scripted PARTIAL acceptance: drafts[0] is the target's true next token
	//     (accepted), the rest deliberately wrong (rejected) → 0 < Accepted < kk EVERY
	//     round. This is the offset-sensitive squash (Evict at from+Accepted, Accepted>0)
	//     that full-accept (a) and full-reject (b) never exercise. Output must STILL be
	//     token-identical to greedy, with both an accept and a rollback every round.
	gotC, draftedC, acceptedC, rolledC := SpeculativeGreedy(
		context.Background(), sink, target.NewSession(), prompt, N, K,
		&scriptedPartialDrafter{want: want})
	assertEqualTokens(t, "partial-split", gotC, want)
	if acceptedC == 0 || acceptedC >= draftedC {
		t.Fatalf("partial-split: expected a partial accept (0 < accepted=%d < drafted=%d)", acceptedC, draftedC)
	}
	if rolledC == 0 {
		t.Fatal("partial-split: expected rollbacks (a rejected suffix every round)")
	}
	if sink.OpenCount() != 0 {
		t.Fatalf("partial-split: %d speculations left unresolved (a leak)", sink.OpenCount())
	}
	t.Logf("partial-split: proposed %d, accepted %d, rolled back %d (offset Evict at from+Accepted)", draftedC, acceptedC, rolledC)
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

// ---------------------------------------------------------------------------
// MULTIPLE-SINK TRANSMISSION — one kernel op (OpSpecCommit / OpSpecSquash) fans its
// resolution across EVERY registered ProvisionalSink for one (Txn, Spec.Epoch). That
// fan-out (op.Invoke ranging abi.ProvisionalSinks()) is the cache-lifecycle lever: a
// single squash retracts the provisional state of every cache that registered a sink,
// so a misspeculated turn leaves no stale span in ANY cache — without each cache
// exposing its own private rollback verb the caller must remember to fan by hand.
//
// Every other witness in this file registers exactly ONE sink (the real KV spec.Sink),
// so the >1-registrant fan loop and its error path have never run. These two witnesses
// register a SECOND sink alongside the KV sink and pin the contract a future real
// second cache (a radixkv prefix node, a ctxplan.Index Retract) will inherit:
//   - the fan reaches every registrant for the same (Txn, Spec.Epoch);
//   - a single cross-sink DRAIN invariant: after an epoch resolves, NO registered
//     sink holds an open provisional span;
//   - the fan is BEST-EFFORT, not two-phase: a mid-fan retract failure neither
//     short-circuits the remaining sinks nor compensates the ones already resolved,
//     and the op still reports the success Outcome with only Status flipped.
// ---------------------------------------------------------------------------

// countingSink is a minimal second abi.ProvisionalSink — a stand-in for any future
// retractable cache tier that would join the same per-epoch fan-out. It records, per
// (txn, epoch), whether the kernel Promoted or Rolled it back, and is idempotent on an
// unknown / already-resolved key (the no-lock fan relies on every sink being
// idempotent). A non-nil rollbackErr models a sink whose retract FAILS: it returns the
// error and leaves the provisional span open (a torn retract), the case the headline
// KV sink — whose Rollback always succeeds — can never exercise.
type countingSink struct {
	mu          sync.Mutex
	open        map[key]bool
	promoted    int
	rolledBack  int
	rollbackErr error
}

var _ abi.ProvisionalSink = (*countingSink)(nil)

func newCountingSink() *countingSink { return &countingSink{open: map[key]bool{}} }

// record marks a provisional span open under (txn, epoch) — the second sink's analogue
// of spec.Sink.Open (the caller produced a retractable effect the kernel will resolve).
func (c *countingSink) record(txn abi.TxnID, epoch uint64) {
	c.mu.Lock()
	c.open[key{txn, epoch}] = true
	c.mu.Unlock()
}

func (c *countingSink) Promote(_ context.Context, txn abi.TxnID, epoch uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.open[key{txn, epoch}] {
		delete(c.open, key{txn, epoch})
		c.promoted++
	}
	return nil
}

func (c *countingSink) Rollback(_ context.Context, txn abi.TxnID, epoch uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.rollbackErr != nil {
		return c.rollbackErr // failed retract: the provisional span stays open (torn)
	}
	if c.open[key{txn, epoch}] {
		delete(c.open, key{txn, epoch})
		c.rolledBack++
	}
	return nil
}

func (c *countingSink) openCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.open)
}

// TestMultiSinkTransmissionFansToEveryRegistrant proves the reserved OpsSpec ops fan
// across BOTH a real KV sink and a second registered sink for one (Txn, Spec.Epoch):
// squash retracts every registrant (a cross-sink drain), commit promotes every
// registrant. This is the >1-registrant path the single-sink witnesses never reach.
func TestMultiSinkTransmissionFansToEveryRegistrant(t *testing.T) {
	t.Setenv(polymodel.FlagEnv, "on")
	abi.ResetForTest()
	defer abi.ResetForTest()

	kv := Install() // the real KV spec.Sink (first registrant) + the reserved OpsSpec ops
	if kv == nil {
		t.Fatal("Install returned nil with the lane enabled")
	}
	other := newCountingSink()
	abi.RegisterProvisionalSink(other) // a SECOND retractable-cache tier joins the fan
	if got := len(abi.ProvisionalSinks()); got != 2 {
		t.Fatalf("multi-sink premise: want 2 registered sinks, got %d", got)
	}

	m := model.NewSynthetic(cfg(48, 3, 3, 1, 16, 96))
	prompt := bytesToIDs([]byte("one op resolves every cache"))

	// Open a real provisional KV span on the KV sink AND a provisional record on the
	// second sink, both under the SAME (txn, epoch). Returns the session so the caller
	// can check the KV span survived (commit) or was evicted (squash).
	openBoth := func(txn abi.TxnID, epoch uint64) *model.Session {
		s := m.NewSession()
		l := s.Prefill(prompt)
		base := s.Cache.Len()
		last := argmax(l)
		for i := 0; i < 3; i++ {
			s.Step(last)
		}
		kv.Open(txn, epoch, s.Cache, base, 3)
		other.record(txn, epoch)
		return s
	}

	// (a) SQUASH fans Rollback to BOTH sinks: the KV span is evicted AND the second
	//     sink is rolled back — neither leaves an open span (the cross-sink drain).
	{
		const txn = abi.TxnID(11)
		s := openBoth(txn, EpochReject)
		base := s.Cache.Len() - 3
		squash, ok := abi.LookupOp(OpSpecSquash)
		if !ok {
			t.Fatal("OpSpecSquash not registered")
		}
		res, v := squash.Invoke(context.Background(), nil, &abi.ToolCall{
			Op:   OpSpecSquash,
			Txn:  txn,
			Spec: abi.SpeculationContext{Speculative: true, Epoch: EpochReject},
		})
		if v.Kind != abi.VerdictAllow || res.Outcome != abi.OutcomeSquashed || res.Status != abi.StatusOK {
			t.Fatalf("squash: verdict=%v outcome=%v status=%v (want Allow/Squashed/OK)", v.Kind, res.Outcome, res.Status)
		}
		if s.Cache.Len() != base {
			t.Fatalf("squash: KV span not evicted: len %d, want %d", s.Cache.Len(), base)
		}
		if kv.OpenCount() != 0 || other.openCount() != 0 {
			t.Fatalf("squash: cross-sink drain violated: kv open=%d, other open=%d (want 0,0)", kv.OpenCount(), other.openCount())
		}
		if other.rolledBack != 1 {
			t.Fatalf("squash: second sink rolledBack=%d, want 1 (the fan never reached it)", other.rolledBack)
		}
	}

	// (b) COMMIT fans Promote to BOTH sinks: the KV span stays durable AND the second
	//     sink is promoted; neither leaves an open span.
	{
		const txn = abi.TxnID(12)
		s := openBoth(txn, EpochAccept)
		full := s.Cache.Len()
		commit, ok := abi.LookupOp(OpSpecCommit)
		if !ok {
			t.Fatal("OpSpecCommit not registered")
		}
		res, _ := commit.Invoke(context.Background(), nil, &abi.ToolCall{
			Op:   OpSpecCommit,
			Txn:  txn,
			Spec: abi.SpeculationContext{Speculative: true, Epoch: EpochAccept},
		})
		if res.Outcome != abi.OutcomeCommitted || res.Status != abi.StatusOK {
			t.Fatalf("commit: outcome=%v status=%v (want Committed/OK)", res.Outcome, res.Status)
		}
		if s.Cache.Len() != full {
			t.Fatalf("commit: KV span must stay durable: len %d, want %d", s.Cache.Len(), full)
		}
		if kv.OpenCount() != 0 || other.openCount() != 0 {
			t.Fatalf("commit: open spans after promote: kv=%d, other=%d (want 0,0)", kv.OpenCount(), other.openCount())
		}
		if other.promoted != 1 {
			t.Fatalf("commit: second sink promoted=%d, want 1", other.promoted)
		}
	}
}

// TestMultiSinkFanOutIsBestEffortNotAtomic pins the HONEST semantics of the fan-out: it
// is best-effort, not two-phase. With the fan order [kv, bad, good], the bad sink's
// Rollback errors AFTER kv has already evicted its span and BEFORE good is reached. The
// op (1) does NOT short-circuit — good still resolves; (2) does NOT compensate — kv is
// not re-added and bad is left torn; (3) still returns the success Outcome (Squashed)
// with only Status flipped to StatusError. A future caller wanting all-or-nothing must
// change this contract (a 2PC / compensation upgrade), and this witness is what such a
// change would have to rewrite — the non-atomicity is documented, not silent.
func TestMultiSinkFanOutIsBestEffortNotAtomic(t *testing.T) {
	t.Setenv(polymodel.FlagEnv, "on")
	abi.ResetForTest()
	defer abi.ResetForTest()

	kv := Install()
	if kv == nil {
		t.Fatal("Install returned nil with the lane enabled")
	}
	// Registration order is fan order: [kv, bad, good].
	bad := newCountingSink()
	bad.rollbackErr = errors.New("retract failed")
	good := newCountingSink()
	abi.RegisterProvisionalSink(bad)
	abi.RegisterProvisionalSink(good)

	m := model.NewSynthetic(cfg(32, 2, 2, 1, 16, 64))
	prompt := bytesToIDs([]byte("best effort, not two-phase"))
	s := m.NewSession()
	l := s.Prefill(prompt)
	base := s.Cache.Len()
	last := argmax(l)
	for i := 0; i < 2; i++ {
		s.Step(last)
	}
	const txn = abi.TxnID(21)
	kv.Open(txn, EpochReject, s.Cache, base, 2)
	bad.record(txn, EpochReject)
	good.record(txn, EpochReject)

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
		t.Errorf("verdict=%v, want Allow", v.Kind)
	}
	if res.Status != abi.StatusError {
		t.Errorf("Status=%v, want StatusError (the bad sink's error must surface)", res.Status)
	}
	if res.Outcome != abi.OutcomeSquashed {
		t.Errorf("Outcome=%v, want OutcomeSquashed (the fan reports the success outcome despite a torn retract)", res.Outcome)
	}
	// kv ran before the failure: it resolved and its KV was evicted bit-exactly.
	if kv.OpenCount() != 0 || s.Cache.Len() != base {
		t.Errorf("kv sink not resolved: open=%d cacheLen=%d (want 0, %d)", kv.OpenCount(), s.Cache.Len(), base)
	}
	// good sits AFTER the failing sink and STILL resolves — the fan does not short-circuit.
	if good.openCount() != 0 || good.rolledBack != 1 {
		t.Errorf("good sink (after the failure) did not resolve: open=%d rolledBack=%d (the fan short-circuited)", good.openCount(), good.rolledBack)
	}
	// bad stays torn: its failed retract is neither retried nor compensated by the op.
	if bad.openCount() != 1 {
		t.Errorf("bad sink openCount=%d, want 1 (a failed retract is left torn, not compensated)", bad.openCount())
	}
}
