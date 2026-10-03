package model

import (
	"errors"
	"testing"
)

// fakeWholeSequenceSession is a non-Qwen WholeSequenceSession double. It proves
// the backend-neutral seam can be driven without any Qwen35/Metal type, and that
// the operation validator is expressed purely against the neutral tokens.
type fakeWholeSequenceSession struct {
	path      string
	executed  WholeSequenceEvidenceState
	selector  WholeSequenceSelectorState
	enable    func() error
	finalize  func() (bool, error)
	receipt   func() WholeSequenceReceipt
	handoff   func() WholeSequenceHandoff
	enableN   int
	finalizeN int
}

func (f *fakeWholeSequenceSession) EnableWholeSequence() error {
	f.enableN++
	if f.enable != nil {
		return f.enable()
	}
	return nil
}

func (f *fakeWholeSequenceSession) FinalizeWholeSequence() (bool, error) {
	f.finalizeN++
	if f.finalize != nil {
		return f.finalize()
	}
	return true, nil
}

func (f *fakeWholeSequenceSession) WholeSequenceReceipt() WholeSequenceReceipt {
	if f.receipt != nil {
		return f.receipt()
	}
	return WholeSequenceReceipt{}
}

func (f *fakeWholeSequenceSession) WholeSequenceHandoffReceipt() WholeSequenceHandoff {
	if f.handoff != nil {
		return f.handoff()
	}
	return WholeSequenceHandoff{Mode: "AUTO"}
}

func (f *fakeWholeSequenceSession) WholeSequencePath() string { return f.path }

func (f *fakeWholeSequenceSession) WholeSequenceExecutedEvidence() WholeSequenceEvidenceState {
	return f.executed
}

func (f *fakeWholeSequenceSession) WholeSequenceSelectorOnState() WholeSequenceSelectorState {
	return f.selector
}

// TestWholeSequenceSeam drives the backend-neutral whole-sequence witness with a
// non-Qwen test double and asserts the same operation-validation outcomes the
// concrete Metal receipt produced. It pins the seam contract: enable/finalize,
// before/after receipts, handoff counters, and the path + evidence-state tokens
// are all reachable without naming Qwen35 or Metal.
func TestWholeSequenceSeam(t *testing.T) {
	seam := &fakeWholeSequenceSession{
		path:     "native/whole-sequence-v1",
		executed: WholeSequenceEvidenceExecuted,
		selector: WholeSequenceSelectorOn,
	}
	if err := seam.EnableWholeSequence(); err != nil {
		t.Fatalf("EnableWholeSequence: %v", err)
	}
	if executed, err := seam.FinalizeWholeSequence(); err != nil || !executed {
		t.Fatalf("FinalizeWholeSequence executed=%v err=%v", executed, err)
	}
	if seam.enableN != 1 || seam.finalizeN != 1 {
		t.Fatalf("lifecycle calls enable=%d finalize=%d, want 1/1", seam.enableN, seam.finalizeN)
	}
	if seam.WholeSequencePath() != "" && seam.WholeSequencePath() != seam.path {
		t.Fatalf("path token mismatch")
	}
	if seam.WholeSequenceExecutedEvidence() != WholeSequenceEvidenceExecuted {
		t.Fatalf("evidence token mismatch")
	}
	if seam.WholeSequenceSelectorOnState() != WholeSequenceSelectorOn {
		t.Fatalf("selector token mismatch")
	}

	wholeTokenAfter := WholeSequenceReceipt{
		Path: seam.WholeSequencePath(), Available: true,
		SelectorState: seam.WholeSequenceSelectorOnState(), EvidenceState: seam.WholeSequenceExecutedEvidence(),
		Tokens: 1, CommandBuffers: 1, TerminalWaits: 1, TerminalReadbacks: 1,
		Committed: true, CompletedWait: true,
	}
	// The whole-token route requires a fresh receipt and exactly one accepted
	// block call. This is the same outcome the concrete Metal witness asserts.
	wholeToken := WholeSequenceOperation{
		Before: WholeSequenceReceipt{Path: seam.WholeSequencePath(), Available: true, SelectorState: seam.WholeSequenceSelectorOnState(), EvidenceState: seam.WholeSequenceExecutedEvidence(), Tokens: 32, CommandBuffers: 1},
		After:  wholeTokenAfter,
		CountsBefore: WholeSequenceHandoff{
			Mode: "AUTO",
		},
		CountsAfter: WholeSequenceHandoff{
			Mode: "AUTO", BlockAcceptedCalls: 1,
		},
		CacheBefore: 32, CacheAfter: 33,
	}
	if err := ValidateWholeSequenceOperation("whole-token", seam.WholeSequencePath(), seam.WholeSequenceExecutedEvidence(), wholeToken); err != nil {
		t.Fatalf("valid whole-token operation rejected: %v", err)
	}

	// A per-layer fallback that advanced the block counter, or one whose after
	// receipt is not a fresh executed receipt, must be rejected.
	fallback := wholeToken
	fallback.After = fallback.Before
	if err := ValidateWholeSequenceOperation("whole-token", seam.WholeSequencePath(), seam.WholeSequenceExecutedEvidence(), fallback); err == nil {
		t.Fatal("accepted a non-fresh whole-token receipt (fallback is non-qualifying)")
	}
	stale := wholeToken
	stale.After.EvidenceState = WholeSequenceEvidenceUnavailable
	if err := ValidateWholeSequenceOperation("whole-token", seam.WholeSequencePath(), seam.WholeSequenceExecutedEvidence(), stale); err == nil {
		t.Fatal("accepted an unavailable evidence-state token")
	}
	wrongPath := wholeToken
	wrongPath.After.Path = "some/other-backend-v1"
	if err := ValidateWholeSequenceOperation("whole-token", seam.WholeSequencePath(), seam.WholeSequenceExecutedEvidence(), wrongPath); err == nil {
		t.Fatal("accepted a receipt from a different capability path")
	}
	wrongCount := wholeToken
	wrongCount.CountsAfter.ResidentAcceptedCalls = 1
	if err := ValidateWholeSequenceOperation("whole-token", seam.WholeSequencePath(), seam.WholeSequenceExecutedEvidence(), wrongCount); err == nil {
		t.Fatal("accepted a resident-accept advance on the whole-token route")
	}

	// The source-control per-layer route requires an unchanged receipt and exactly
	// one block call, with no mixer/resident advance.
	perLayer := WholeSequenceOperation{
		Before:       wholeToken.Before,
		After:        wholeToken.Before,
		CountsBefore: WholeSequenceHandoff{Mode: "AUTO"},
		CountsAfter:  WholeSequenceHandoff{Mode: "AUTO", BlockAcceptedCalls: 1},
		CacheBefore:  32, CacheAfter: 33,
	}
	if err := ValidateWholeSequenceOperation("per-layer", seam.WholeSequencePath(), seam.WholeSequenceExecutedEvidence(), perLayer); err != nil {
		t.Fatalf("valid per-layer operation rejected: %v", err)
	}
	twoBlocks := perLayer
	twoBlocks.CountsAfter.BlockAcceptedCalls = 2
	if err := ValidateWholeSequenceOperation("per-layer", seam.WholeSequencePath(), seam.WholeSequenceExecutedEvidence(), twoBlocks); err == nil {
		t.Fatal("accepted two block calls for one per-layer Step")
	}
	if err := ValidateWholeSequenceOperation("bogus", seam.WholeSequencePath(), seam.WholeSequenceExecutedEvidence(), perLayer); err == nil {
		t.Fatal("accepted an unknown route")
	}
	staleCache := wholeToken
	staleCache.CacheAfter = 35
	if err := ValidateWholeSequenceOperation("whole-token", seam.WholeSequencePath(), seam.WholeSequenceExecutedEvidence(), staleCache); err == nil {
		t.Fatal("accepted a Step that did not advance exactly one cache position")
	}

	// Readback of a serialized backend-general artifact must pass the capability
	// token the report recorded at build time (the concrete runtime token, not the
	// neutral contract constant). The validator must never trust a path the receipt
	// declares for itself, and must refuse an empty required path.
	concretePath := wholeToken
	concretePath.After.Path = Qwen35MetalGDNSequenceForwardPath
	concretePath.Before.Path = Qwen35MetalGDNSequenceForwardPath
	if err := ValidateWholeSequenceOperation("whole-token", Qwen35MetalGDNSequenceForwardPath, WholeSequenceEvidenceExecuted, concretePath); err != nil {
		t.Fatalf("readback rejected the recorded capability path %q: %v", concretePath.After.Path, err)
	}
	noPath := concretePath
	noPath.After.Path = ""
	if err := ValidateWholeSequenceOperation("whole-token", "", WholeSequenceEvidenceExecuted, noPath); err == nil {
		t.Fatal("accepted a whole-token operation with no required capability path")
	}
	fabricated := concretePath
	fabricated.After.Path = "fabricated/other-backend-v1"
	if err := ValidateWholeSequenceOperation("whole-token", Qwen35MetalGDNSequenceForwardPath, WholeSequenceEvidenceExecuted, fabricated); err == nil {
		t.Fatal("trusted a receipt's self-declared path over the required token")
	}
}

// TestWholeSequenceReceiptAdapterIsNeutral pins that the concrete Qwen35 Metal
// receipt adapts into the neutral shape field-for-field (minus the Metal-only
// state-identity binding) and that the reverse handoff conversion is exact.
func TestWholeSequenceReceiptAdapterIsNeutral(t *testing.T) {
	concrete := Qwen35MetalForwardSequenceReceipt{
		Path: Qwen35MetalGDNSequenceForwardPath, Available: true,
		SelectorState: Qwen35MetalSequenceSelectorOn, EvidenceState: Qwen35MetalSequenceEvidenceExecuted,
		Tokens: 1, CommandBuffers: 1, Encoders: 2, TerminalWaits: 1, TerminalReadbacks: 1,
		HostUploadBytes: 8, HostReadbackBytes: 16, Committed: true, CompletedWait: true,
	}
	neutral := WholeSequenceReceiptFromQwen35(concrete)
	if neutral.Path != concrete.Path || neutral.Available != concrete.Available ||
		neutral.SelectorState != WholeSequenceSelectorState(concrete.SelectorState) ||
		neutral.EvidenceState != WholeSequenceEvidenceState(concrete.EvidenceState) ||
		neutral.Tokens != concrete.Tokens || neutral.CommandBuffers != concrete.CommandBuffers ||
		neutral.HostUploadBytes != concrete.HostUploadBytes || neutral.HostReadbackBytes != concrete.HostReadbackBytes ||
		!neutral.Committed || !neutral.CompletedWait {
		t.Fatalf("adapter drifted: %+v", neutral)
	}
	handoff := WholeSequenceHandoffFromQwen35(Qwen35DecodeHandoffReceipt{
		Mode: Qwen35DecodeHandoffAuto, BlockAcceptedCalls: 2, MixerAcceptedCalls: 1, ResidentGDNAcceptedCalls: 3,
	})
	if handoff.Mode != "AUTO" || handoff.BlockAcceptedCalls != 2 || handoff.MixerAcceptedCalls != 1 || handoff.ResidentAcceptedCalls != 3 {
		t.Fatalf("handoff adapter drifted: %+v", handoff)
	}
}

// neutralWholeTokenOperation builds one valid neutral whole-token operation for
// the given capability path. It names only the neutral tokens and receipt shape:
// no Qwen35/Metal type and no session is constructed, so it is the receipt half
// of the contract, decoupled from whichever model owns the execution.
func neutralWholeTokenOperation(path string) WholeSequenceOperation {
	return WholeSequenceOperation{
		Before: WholeSequenceReceipt{
			Path: path, Available: true,
			SelectorState: WholeSequenceSelectorOn, EvidenceState: WholeSequenceEvidenceExecuted,
			Tokens: 32, CommandBuffers: 1,
		},
		After: WholeSequenceReceipt{
			Path: path, Available: true,
			SelectorState: WholeSequenceSelectorOn, EvidenceState: WholeSequenceEvidenceExecuted,
			Tokens: 1, CommandBuffers: 1, TerminalWaits: 1, TerminalReadbacks: 1,
			Committed: true, CompletedWait: true,
		},
		CountsBefore: WholeSequenceHandoff{Mode: "AUTO"},
		CountsAfter:  WholeSequenceHandoff{Mode: "AUTO", BlockAcceptedCalls: 1},
		CacheBefore:  32, CacheAfter: 33,
	}
}

// TestV41WholeSequenceOwnerRefusedByCapabilityNotReceipt drives the two distinct
// predicates the whole-token witness gates on: a real deepseek_v41_text session
// advertises NO whole-sequence capability (admission) and reports a typed V4.1
// refusal (diagnosis), even when handed a well-formed neutral operation receipt.
// Ownership admission is deliberately separate from receipt validation: the
// validator only checks the receipt shape against a path the caller must already
// own, so a plausible unrelated (Qwen/Metal) receipt cannot promote an
// unsupported session, and a nil/unknown-session control is refused too.
func TestV41WholeSequenceOwnerRefusedByCapabilityNotReceipt(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	m := &Model{Cfg: cfg}
	if !m.Cfg.IsDeepSeekV41() {
		t.Fatal("probe config is not recognized as V4.1")
	}
	s := &Session{M: m}

	path, admissible := s.WholeSequenceCapability()
	if admissible || path != "" {
		t.Fatalf("V4.1 session advertised whole-sequence capability path=%q admissible=%t, want none", path, admissible)
	}
	err := s.WholeSequenceUnsupportedReason()
	if !errors.Is(err, ErrV41WholeSequenceUnsupported) {
		t.Fatalf("V4.1 refusal=%v want ErrV41WholeSequenceUnsupported", err)
	}
	var typed *UnsupportedV41WholeSequenceError
	if !errors.As(err, &typed) || typed.Path != WholeSequencePath {
		t.Fatalf("V4.1 refusal=%v typed=%#v, want a named refusal on %q", err, typed, WholeSequencePath)
	}

	// A plausible neutral receipt of the unrelated Qwen/Metal capability is not an
	// ownership credential. Validated against the borrowed capability token it
	// still passes, which proves the session's refusal above came from admission,
	// not from receipt shape — the validator does not authenticate a model owner.
	borrowed := neutralWholeTokenOperation(Qwen35MetalGDNSequenceForwardPath)
	if err := ValidateWholeSequenceOperation("whole-token", Qwen35MetalGDNSequenceForwardPath, WholeSequenceEvidenceExecuted, borrowed); err != nil {
		t.Fatalf("valid borrowed-capability receipt rejected: %v", err)
	}
	// Under the V4.1 session's own (empty) capability path the same operation is
	// refused: an unsupported model cannot be promoted by a well-formed receipt.
	if err := ValidateWholeSequenceOperation("whole-token", path, WholeSequenceEvidenceExecuted, borrowed); err == nil {
		t.Fatal("accepted a whole-token operation under the V4.1 session's empty capability path")
	}

	// Nil and non-V4.1 controls: no V4.1 identity means no named refusal, so the
	// refusal is gated on IsDeepSeekV41 and cannot leak onto another family.
	if err := (*Session)(nil).WholeSequenceUnsupportedReason(); err != nil {
		t.Fatalf("nil session refusal=%v want nil", err)
	}
	llama := &Session{M: &Model{Cfg: Config{ModelType: "llama", HiddenSize: 4, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 4}}}
	if p, ok := llama.WholeSequenceCapability(); ok || p != "" {
		t.Fatalf("non-V4.1 session advertised capability path=%q ok=%t, want none", p, ok)
	}
	if err := llama.WholeSequenceUnsupportedReason(); err != nil {
		t.Fatalf("non-V4.1 refusal=%v want nil (refusal must be IsDeepSeekV41-gated)", err)
	}
}

// TestWholeSequenceEvidenceRefusesMalformedNeutralReceipts mutates a valid
// neutral whole-token operation one field at a time and requires
// ValidateWholeSequenceOperation to refuse each, while the unmutated control
// still passes. It exercises the real validation contract with no session and no
// model named, so receipt validation and ownership admission stay independent.
func TestWholeSequenceEvidenceRefusesMalformedNeutralReceipts(t *testing.T) {
	path := Qwen35MetalGDNSequenceForwardPath
	valid := neutralWholeTokenOperation(path)
	if err := ValidateWholeSequenceOperation("whole-token", path, WholeSequenceEvidenceExecuted, valid); err != nil {
		t.Fatalf("valid control rejected: %v", err)
	}
	mutations := map[string]func(WholeSequenceOperation) WholeSequenceOperation{
		"stale receipt has no fresh token": func(op WholeSequenceOperation) WholeSequenceOperation {
			op.After = op.Before
			return op
		},
		"evidence token unavailable": func(op WholeSequenceOperation) WholeSequenceOperation {
			op.After.EvidenceState = WholeSequenceEvidenceUnavailable
			return op
		},
		"declared path differs from required token": func(op WholeSequenceOperation) WholeSequenceOperation {
			op.After.Path = "some/other-backend-v1"
			return op
		},
		"step did not advance the cache": func(op WholeSequenceOperation) WholeSequenceOperation {
			op.CacheAfter = op.CacheBefore
			return op
		},
		"resident accept advanced on whole-token": func(op WholeSequenceOperation) WholeSequenceOperation {
			op.CountsAfter.ResidentAcceptedCalls = 1
			return op
		},
		"block accept not advanced": func(op WholeSequenceOperation) WholeSequenceOperation {
			op.CountsAfter.BlockAcceptedCalls = op.CountsBefore.BlockAcceptedCalls
			return op
		},
		"no capability path": func(op WholeSequenceOperation) WholeSequenceOperation {
			op.Before.Path, op.After.Path = "", ""
			return op
		},
		"sink exceeds terminal waits": func(op WholeSequenceOperation) WholeSequenceOperation {
			op.After.TerminalReadbacks = 2
			return op
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			if err := ValidateWholeSequenceOperation("whole-token", path, WholeSequenceEvidenceExecuted, mutate(valid)); err == nil {
				t.Fatal("accepted malformed whole-token evidence")
			}
		})
	}
	if err := ValidateWholeSequenceOperation("bogus-route", path, WholeSequenceEvidenceExecuted, valid); err == nil {
		t.Fatal("accepted an unknown whole-sequence route")
	}
}
