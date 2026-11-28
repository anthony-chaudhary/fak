package modelengine

// Adapted from deepseek-ai/DeepEP (MIT; Copyright (c) 2025 DeepSeek; see
// THIRD_PARTY_NOTICES.md), csrc/hybrid_ep/backend/hybrid_ep_backend.cuh and
// deep_ep/hybrid_ep_buffer.py at pinned revision
// 4e890fd19a3a52f23cc7bd38a0d9b0b1276f70d6 (HybridEP PR #766).
//
// Clean-room host Go contract and in-memory oracle only; no RDMA hardware or
// device kernels are invoked.
//
// When capacity-overflowing non-blocking MoE dispatch clamps a dropped
// token-expert route so its destination map entry has no source, the fused
// unpermute-combine pass must zero each fabricated dummy entry (payload and,
// under backward combine, probability and local expert id) and execute a
// memory fence before signaling barrier arrival so fully dropped tokens
// contribute exact zero instead of stale-memory NaNs. Dispatch and combine
// must also propagate the non-blocking flag symmetrically.

import (
	"errors"
	"fmt"
	"math"
)

var (
	// ErrMoECombineNonBlockingMismatch reports dispatch and combine invoked with
	// asymmetric non-blocking flags. Callers may use errors.Is.
	ErrMoECombineNonBlockingMismatch = errors.New("modelengine: moe combine: dispatch/combine non-blocking flag mismatch")
	// ErrMoECombineInvalidConfig reports invalid shape, capacity, or non-finite
	// input during drop-aware dispatch or combine. Callers may use errors.Is.
	ErrMoECombineInvalidConfig = errors.New("modelengine: moe combine: invalid dispatch/combine configuration")
)

// MoEDropNoSource marks a clamped destination entry with no source token slot.
const MoEDropNoSource = -1

// MoEDispatchDropConfig configures capacity-bounded MoE dispatch with drop
// accounting and symmetric non-blocking flag propagation.
type MoEDispatchDropConfig struct {
	NumTokens      int
	TopK           int
	NumExperts     int
	HiddenDim      int
	ExpertCapacity int
	NonBlocking    bool
}

// MoEDestinationEntry records one (token, topk-slot) route after capacity
// clamping. When Dropped is true, SourceSlot is MoEDropNoSource (-1) and no
// expert output is read for that slot.
type MoEDestinationEntry struct {
	TokenIdx      int
	TopKSlot      int
	LocalExpertID int
	SourceSlot    int
	Prob          float32
	Dropped       bool
}

// MoEDispatchDropState carries the post-dispatch routing state and propagated
// non-blocking flag consumed by MoEUnpermuteCombine.
type MoEDispatchDropState struct {
	Config             MoEDispatchDropConfig
	NonBlocking        bool
	TokenDropTriggered bool
	DroppedSlots       int
	FullyDroppedTokens int
	// Entries is indexed as [tokenIdx * TopK + slot].
	Entries []MoEDestinationEntry
	// ExpertPayloads holds [numAccepted][HiddenDim] simulated expert outputs
	// indexed by Entry.SourceSlot.
	ExpertPayloads [][]float32
}

// MoECombinePhase records ordered synchronization checkpoints during fused
// unpermute-combine so callers can assert fence-before-barrier ordering.
type MoECombinePhase string

const (
	MoECombinePhaseZeroDummy     MoECombinePhase = "zero_dummy"
	MoECombinePhaseFence         MoECombinePhase = "fence"
	MoECombinePhaseBarrierArrive MoECombinePhase = "barrier_arrive"
	MoECombinePhaseAccumulate    MoECombinePhase = "accumulate"
)

// MoECombineBarrierObserver receives ordered barrier/fence phases during
// combine execution. Optional; nil skips tracing.
type MoECombineBarrierObserver func(phase MoECombinePhase)

// MoECombineBackwardEntry captures the combine backward metadata for a token's
// dummy or routed entry.
type MoECombineBackwardEntry struct {
	LocalExpertID int
	Prob          float32
	GradPayload   []float32
}

// MoECombineDropConfig configures the fused unpermute-combine pass.
type MoECombineDropConfig struct {
	NonBlocking bool
	Backward    bool
	// StaleBuffer, when non-nil, simulates pre-existing destination/dummy
	// staging memory ([NumTokens][HiddenDim]) that may contain stale NaNs or
	// garbage and must be explicitly zeroed before barrier arrival for dummy
	// entries.
	StaleBuffer [][]float32
	// StaleBackward, when non-nil and Backward is true, simulates pre-existing
	// backward dummy metadata ([NumTokens]) that must be zeroed before barrier
	// arrival for fully dropped tokens.
	StaleBackward []MoECombineBackwardEntry
	// UpstreamGrad supplies [NumTokens][HiddenDim] upstream gradient when
	// Backward is true; if nil under Backward, unit ones are used.
	UpstreamGrad [][]float32
	// Observer records synchronization ordering across dummy zero, fence, and
	// barrier arrival.
	Observer MoECombineBarrierObserver
}

// MoECombineDropOutput holds the combined forward payloads and (when Backward
// is enabled) zero-safe backward metadata per token.
type MoECombineDropOutput struct {
	NonBlocking        bool
	TokenDropTriggered bool
	Combined           [][]float32
	BackwardDummy      []MoECombineBackwardEntry
	BackwardSlots      [][]MoECombineBackwardEntry
	Phases             []MoECombinePhase
}

// MoEDispatchWithDrop runs capacity-bounded expert routing and clamps dropped
// slots so their destination map entries carry MoEDropNoSource (-1).
func MoEDispatchWithDrop(cfg MoEDispatchDropConfig, tokens [][]float32, topkExperts [][]int, topkProbs [][]float32, expertOutFn func(tokenIdx, expertID int, payload []float32) []float32) (*MoEDispatchDropState, error) {
	if cfg.NumTokens < 0 || cfg.TopK <= 0 || cfg.NumExperts <= 0 || cfg.HiddenDim <= 0 || cfg.ExpertCapacity < 0 {
		return nil, fmt.Errorf("%w: invalid dimensions %+v", ErrMoECombineInvalidConfig, cfg)
	}
	if len(tokens) != cfg.NumTokens || len(topkExperts) != cfg.NumTokens || len(topkProbs) != cfg.NumTokens {
		return nil, fmt.Errorf("%w: token/routing slice length mismatch", ErrMoECombineInvalidConfig)
	}

	expertCounts := make([]int, cfg.NumExperts)
	entries := make([]MoEDestinationEntry, cfg.NumTokens*cfg.TopK)
	var expertPayloads [][]float32
	droppedSlots := 0
	fullyDropped := 0

	for t := 0; t < cfg.NumTokens; t++ {
		if len(tokens[t]) != cfg.HiddenDim {
			return nil, fmt.Errorf("%w: token %d hidden dim %d != %d", ErrMoECombineInvalidConfig, t, len(tokens[t]), cfg.HiddenDim)
		}
		if len(topkExperts[t]) != cfg.TopK || len(topkProbs[t]) != cfg.TopK {
			return nil, fmt.Errorf("%w: token %d topk length mismatch", ErrMoECombineInvalidConfig, t)
		}
		keptForToken := 0
		for k := 0; k < cfg.TopK; k++ {
			exp := topkExperts[t][k]
			prob := topkProbs[t][k]
			if exp < 0 || exp >= cfg.NumExperts {
				return nil, fmt.Errorf("%w: token %d slot %d expert %d out of [0,%d)", ErrMoECombineInvalidConfig, t, k, exp, cfg.NumExperts)
			}
			if math.IsNaN(float64(prob)) || math.IsInf(float64(prob), 0) || prob < 0 {
				return nil, fmt.Errorf("%w: token %d slot %d invalid prob %v", ErrMoECombineInvalidConfig, t, k, prob)
			}
			idx := t*cfg.TopK + k
			if expertCounts[exp] >= cfg.ExpertCapacity {
				// Capacity overflow: clamp destination map entry so it has no source.
				entries[idx] = MoEDestinationEntry{
					TokenIdx:      t,
					TopKSlot:      k,
					LocalExpertID: MoEDropNoSource,
					SourceSlot:    MoEDropNoSource,
					Prob:          prob,
					Dropped:       true,
				}
				droppedSlots++
				continue
			}
			expertCounts[exp]++
			slotIdx := len(expertPayloads)
			var out []float32
			if expertOutFn != nil {
				out = append([]float32(nil), expertOutFn(t, exp, tokens[t])...)
			} else {
				out = append([]float32(nil), tokens[t]...)
			}
			if len(out) != cfg.HiddenDim {
				return nil, fmt.Errorf("%w: expert output dim %d != %d", ErrMoECombineInvalidConfig, len(out), cfg.HiddenDim)
			}
			expertPayloads = append(expertPayloads, out)
			entries[idx] = MoEDestinationEntry{
				TokenIdx:      t,
				TopKSlot:      k,
				LocalExpertID: exp,
				SourceSlot:    slotIdx,
				Prob:          prob,
				Dropped:       false,
			}
			keptForToken++
		}
		if keptForToken == 0 {
			fullyDropped++
		}
	}

	return &MoEDispatchDropState{
		Config:             cfg,
		NonBlocking:        cfg.NonBlocking,
		TokenDropTriggered: droppedSlots > 0,
		DroppedSlots:       droppedSlots,
		FullyDroppedTokens: fullyDropped,
		Entries:            entries,
		ExpertPayloads:     expertPayloads,
	}, nil
}

// MoEUnpermuteCombine executes the fused unpermute-combine pass over a
// dispatched state. It enforces symmetric non-blocking flag propagation,
// zero-fills dummy entries (payload and, under backward combine, prob and
// local-expert-id) with a fence strictly before barrier arrival for tokens with
// no surviving source, and accumulates prob-scaled payloads for surviving slots.
func MoEUnpermuteCombine(state *MoEDispatchDropState, cfg MoECombineDropConfig) (*MoECombineDropOutput, error) {
	if state == nil {
		return nil, fmt.Errorf("%w: nil dispatch state", ErrMoECombineInvalidConfig)
	}
	if cfg.NonBlocking != state.NonBlocking {
		return nil, fmt.Errorf("%w: dispatch non_blocking=%t combine non_blocking=%t",
			ErrMoECombineNonBlockingMismatch, state.NonBlocking, cfg.NonBlocking)
	}

	numTokens := state.Config.NumTokens
	topK := state.Config.TopK
	hidden := state.Config.HiddenDim

	if cfg.StaleBuffer != nil && len(cfg.StaleBuffer) != numTokens {
		return nil, fmt.Errorf("%w: stale buffer token count %d != %d", ErrMoECombineInvalidConfig, len(cfg.StaleBuffer), numTokens)
	}
	if cfg.StaleBackward != nil && len(cfg.StaleBackward) != numTokens {
		return nil, fmt.Errorf("%w: stale backward count %d != %d", ErrMoECombineInvalidConfig, len(cfg.StaleBackward), numTokens)
	}
	if cfg.Backward && cfg.UpstreamGrad != nil && len(cfg.UpstreamGrad) != numTokens {
		return nil, fmt.Errorf("%w: upstream grad token count %d != %d", ErrMoECombineInvalidConfig, len(cfg.UpstreamGrad), numTokens)
	}

	// Start from stale buffer if supplied so tests can verify dummy zero-fill
	// overwrites stale NaNs/garbage before barrier arrival.
	combined := make([][]float32, numTokens)
	for t := 0; t < numTokens; t++ {
		if cfg.StaleBuffer != nil {
			if len(cfg.StaleBuffer[t]) != hidden {
				return nil, fmt.Errorf("%w: stale buffer token %d hidden %d != %d", ErrMoECombineInvalidConfig, t, len(cfg.StaleBuffer[t]), hidden)
			}
			combined[t] = append([]float32(nil), cfg.StaleBuffer[t]...)
		} else {
			combined[t] = make([]float32, hidden)
		}
	}

	var backwardDummy []MoECombineBackwardEntry
	var backwardSlots [][]MoECombineBackwardEntry
	if cfg.Backward {
		backwardDummy = make([]MoECombineBackwardEntry, numTokens)
		backwardSlots = make([][]MoECombineBackwardEntry, numTokens)
		for t := 0; t < numTokens; t++ {
			if cfg.StaleBackward != nil {
				sb := cfg.StaleBackward[t]
				backwardDummy[t] = MoECombineBackwardEntry{
					LocalExpertID: sb.LocalExpertID,
					Prob:          sb.Prob,
					GradPayload:   append([]float32(nil), sb.GradPayload...),
				}
				if len(backwardDummy[t].GradPayload) != hidden {
					backwardDummy[t].GradPayload = make([]float32, hidden)
					copy(backwardDummy[t].GradPayload, sb.GradPayload)
				}
			} else {
				backwardDummy[t] = MoECombineBackwardEntry{GradPayload: make([]float32, hidden)}
			}
			backwardSlots[t] = make([]MoECombineBackwardEntry, topK)
			for k := 0; k < topK; k++ {
				backwardSlots[t][k].GradPayload = make([]float32, hidden)
			}
		}
	}

	var phases []MoECombinePhase
	emitPhase := func(p MoECombinePhase) {
		phases = append(phases, p)
		if cfg.Observer != nil {
			cfg.Observer(p)
		}
	}

	// Phase 1: For every token, check if all sources were dropped. If any dummy
	// entries must be fabricated, zero-fill their payload (and backward prob +
	// local-expert-id), issue a fence, then arrive at the barrier BEFORE
	// unpermute-combine reads/accumulates.
	anyDummy := false
	fullyDroppedToken := make([]bool, numTokens)
	for t := 0; t < numTokens; t++ {
		hasSource := false
		for k := 0; k < topK; k++ {
			e := state.Entries[t*topK+k]
			if !e.Dropped && e.SourceSlot >= 0 {
				hasSource = true
				break
			}
		}
		if !hasSource {
			fullyDroppedToken[t] = true
			anyDummy = true
			// Zero dummy payload explicitly (no stale-memory NaNs).
			for d := 0; d < hidden; d++ {
				combined[t][d] = 0
			}
			if cfg.Backward {
				backwardDummy[t].Prob = 0
				backwardDummy[t].LocalExpertID = 0
				for d := 0; d < hidden; d++ {
					backwardDummy[t].GradPayload[d] = 0
				}
			}
		}
	}
	if anyDummy {
		emitPhase(MoECombinePhaseZeroDummy)
	}
	emitPhase(MoECombinePhaseFence)
	emitPhase(MoECombinePhaseBarrierArrive)

	// Phase 2: Accumulate scaled payloads for surviving slots (and zero-init
	// partial tokens' output before accumulating so stale buffer values do not
	// pollute partial tokens either).
	emitPhase(MoECombinePhaseAccumulate)
	for t := 0; t < numTokens; t++ {
		if fullyDroppedToken[t] {
			continue
		}
		for d := 0; d < hidden; d++ {
			combined[t][d] = 0
		}
		var upGrad []float32
		if cfg.Backward {
			if cfg.UpstreamGrad != nil {
				if len(cfg.UpstreamGrad[t]) != hidden {
					return nil, fmt.Errorf("%w: upstream grad token %d hidden %d != %d", ErrMoECombineInvalidConfig, t, len(cfg.UpstreamGrad[t]), hidden)
				}
				upGrad = cfg.UpstreamGrad[t]
			}
		}
		for k := 0; k < topK; k++ {
			e := state.Entries[t*topK+k]
			if e.Dropped || e.SourceSlot < 0 {
				if cfg.Backward {
					backwardSlots[t][k].LocalExpertID = 0
					backwardSlots[t][k].Prob = 0
					for d := 0; d < hidden; d++ {
						backwardSlots[t][k].GradPayload[d] = 0
					}
				}
				continue
			}
			src := state.ExpertPayloads[e.SourceSlot]
			for d := 0; d < hidden; d++ {
				combined[t][d] += e.Prob * src[d]
			}
			if cfg.Backward {
				backwardSlots[t][k].LocalExpertID = e.LocalExpertID
				backwardSlots[t][k].Prob = e.Prob
				for d := 0; d < hidden; d++ {
					g := float32(1.0)
					if upGrad != nil {
						g = upGrad[d]
					}
					backwardSlots[t][k].GradPayload[d] = e.Prob * g
				}
			}
		}
	}

	return &MoECombineDropOutput{
		NonBlocking:        cfg.NonBlocking,
		TokenDropTriggered: state.TokenDropTriggered,
		Combined:           combined,
		BackwardDummy:      backwardDummy,
		BackwardSlots:      backwardSlots,
		Phases:             phases,
	}, nil
}
