package model

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

// expert_checkpoint_q3k_decline_test.go — the Q3_K SETTLEMENT witness for fak#13122.
//
// The DeepSeek-V4.1 Q2_K artifact is published with a Q3_K down-projection form, so the question
// this file settles is whether the checkpoint tier should ADMIT Q3_K alongside Q2_K. The answer is
// no, and it is forced by the representative downstream, not by taste: staging().mk must return a
// compute.Tensor, and no Q3_K host-tensor constructor exists (the enum, blockGeometry(), dtype()
// and weight() have no Q3_K member, and grep finds no compute.Q3_K / compute.NewQ3K). The quant
// registry confirms the same asymmetry from the other side — Q3_K is registered as a NON-HAL
// descriptor with a zero ComputeDtype, while Q2_K is HAL-supported with a real constructor.
//
// Admitting Q3_K would therefore be a silent half-path: the tier would fault Q3_K bytes that
// nothing downstream can stage. Worse, because dtype() and weight() fall through to a default
// (compute.Q4_K / q4kTensor), a Q3_K entry indexed under an unset enum would be read at the wrong
// block geometry and rebuilt as the wrong representation — a silent corruption rather than a
// refusal. The tier must decline instead, which is exactly what an unknown ExpertCheckpointQuant
// does: blockGeometry() returns ok=false and AddShardData refuses it by name.

// TestExpertCheckpointQ3KIsNotStaged pins the settlement: the tier carries no Q3_K member, and an
// unknown/out-of-range quant is declined rather than silently mapped onto a Q4_K fall-through.
func TestExpertCheckpointQ3KIsNotStaged(t *testing.T) {
	// An unset/unknown quant must be declined by the geometry, not coerced to a real one. This is
	// the property that keeps a Q3_K description from being read at the Q4_K block geometry the
	// default arm would impose.
	if _, _, ok := ExpertCheckpointQuant(-1).blockGeometry(); ok {
		t.Fatal("blockGeometry accepted an unknown quant; an unstageable representation must be declined")
	}
	if _, _, ok := ExpertCheckpointQuant(999).blockGeometry(); ok {
		t.Fatal("blockGeometry accepted an out-of-range quant; the tier must not fabricate geometry")
	}

	// A name is diagnosed, never invented. If a future edit added a Q3_K member without a compute
	// tensor, this is the assertion that would still distinguish the enum from a real representation:
	// an out-of-range value reports itself numerically rather than borrowing another kind's name.
	if got := ExpertCheckpointQuant(999).String(); got == "Q3_K" {
		t.Fatalf("out-of-range quant printed %q; a diagnostic must not masquerade as a real representation", got)
	} else if !strings.Contains(got, "ExpertCheckpointQuant(") {
		t.Fatalf("out-of-range quant printed %q, want the ExpertCheckpointQuant(n) diagnostic form", got)
	}

	// No value of the enum may name itself Q3_K. This is the property the settlement rests on: a
	// Q3_K member would make staging().mk reachable with a representation no compute tensor exists
	// for. Sweeping the full enum range (not just the four known members) means a FUTURE Q3_K arm
	// added at the next iota is caught here rather than at decode, so this test fails when the
	// settlement is violated rather than merely restating today's constant list.
	for q := ExpertCheckpointQuant(0); q <= ExpertCheckpointQuant(64); q++ {
		if q.String() == "Q3_K" {
			t.Fatalf("ExpertCheckpointQuant(%d) is named Q3_K; Q3_K has no compute tensor kind and must not be admitted", q)
		}
	}
}

// TestExpertCheckpointQ2KAndQ3KAreDistinctRegistryOutcomes pins the asymmetry that justifies the
// settlement, read through the model package's own registry. Q2_K is HAL-stageable so the tier can
// serve it; Q3_K is not, so the tier correctly declines it. This is the registry-side proof that
// the tier's refusal tracks the representative downstream rather than a guess.
func TestExpertCheckpointQ2KAndQ3KAreDistinctRegistryOutcomes(t *testing.T) {
	// The settlement is only sound if the two outcomes actually differ: the tier admits Q2_K
	// because the resident store stages it and declines Q3_K because it does not. Assert the
	// specific direction (Q2_K true, Q3_K false) rather than mere inequality, so a registry that
	// flipped Q2_K to non-HAL would fail here instead of accidentally satisfying an "!=".
	if !SupportsHALKQuant(kindQ2K) {
		t.Fatal("Q2_K is not HAL-supported; the tier's Q2_K admission would be the same half-path Q3_K avoids")
	}
	if SupportsHALKQuant(kindQ3K) {
		t.Fatal("Q3_K reports HAL-supported; the tier must then admit it, contradicting this settlement")
	}

	// The difference the settlement rests on is concrete: Q2_K has a host-tensor constructor a
	// faulted stride can rebuild into, Q3_K has none. This is the representative capability the
	// issue names — staging().mk must return a compute.Tensor — observed directly rather than
	// inferred from the HAL boolean above. A zero Dtype is the registry's own statement that no
	// compute tensor kind exists to build.
	q2Desc, ok := LookupQuantDescriptor(kindQ2K)
	if !ok {
		t.Fatal("kindQ2K is not registered; the Q2_K staging arm has no descriptor to build from")
	}
	q3Desc, ok := LookupQuantDescriptor(kindQ3K)
	if !ok {
		t.Fatal("kindQ3K is not registered; expected a non-HAL descriptor")
	}
	if q2Desc.Dtype() == 0 {
		t.Fatal("Q2_K descriptor dtype = 0; the tier could not name a compute dtype to stage")
	}
	if q3Desc.Dtype() != 0 {
		t.Fatalf("Q3_K descriptor dtype = %s, want 0 (no compute tensor kind exists)", q3Desc.Dtype())
	}
}

// TestExpertCheckpointUnknownQuantShardIsRefused pins the load-time refusal: a fused tensor that
// declares an unstageable representation (the Q3_K slot has no enum member, so an unknown value is
// the only way to express one) aborts AddShardData with ErrGGUFExpertMetadata rather than being
// half-indexed. This is the "no silent half-path" rule at the shard boundary.
func TestExpertCheckpointUnknownQuantShardIsRefused(t *testing.T) {
	const H = 256
	blob := make([]byte, H*H)
	tier := NewExpertCheckpointTier(0)
	err := tier.AddShardData(bytes.NewReader(blob), int64(len(blob)), nil, []FusedExpertTensor{{
		Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj",
		Quant: ExpertCheckpointQuant(999), Offset: 0, Experts: 1, Rows: H, Cols: H,
	}})
	if err == nil {
		t.Fatal("AddShardData admitted an unstageable representation; the tier must refuse the shard")
	}
	if !errors.Is(err, ErrGGUFExpertMetadata) {
		t.Fatalf("refusal = %v, want it to wrap ErrGGUFExpertMetadata", err)
	}
	if !strings.Contains(err.Error(), "unstageable representation") {
		t.Fatalf("refusal = %v, want it to name the unstageable representation", err)
	}
	// Nothing may have been indexed: a refused shard leaves the tier exactly as it was found.
	if got := tier.Stats().Tensors; got != 0 {
		t.Fatalf("tier indexed %d tensors after a refused shard, want 0", got)
	}
}
