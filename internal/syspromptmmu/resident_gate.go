package syspromptmmu

import "github.com/anthony-chaudhary/fak/internal/cachemeta"

// StableBlockInputs is the complete input surface a RESIDENT prompt-block gate may
// inspect. Its fields are deliberately private and conversation-stable: the current
// user message, turn text, request body, clock, nonce, and every other per-request
// value are structurally absent. A new field changes the cache-stability contract and
// is pinned by TestStableBlockInputsFieldSet.
//
// Strings keep the snapshot immutable after construction: no caller-owned slice/map
// can mutate under a gate between turns.
type StableBlockInputs struct {
	policyVersion     string
	toolsetDigest     string
	deploymentProfile string
}

// NewStableBlockInputs snapshots the declared conversation-stable facts a resident
// gate may read. Callers must create this from session/deployment configuration, never
// from the current user message. The returned value contains no mutable references.
func NewStableBlockInputs(policyVersion, toolsetDigest, deploymentProfile string) StableBlockInputs {
	return StableBlockInputs{
		policyVersion:     policyVersion,
		toolsetDigest:     toolsetDigest,
		deploymentProfile: deploymentProfile,
	}
}

// ResidentBlockID is the stable identity whose admission ResidentBlocks latches.
// Content may be deliberately versioned later, but presence is tracked by this id.
type ResidentBlockID string

// blockCondition is the only predicate shape admitted by the resident registry. The
// volatile turn/request operands are not in its type, so a gate cannot accidentally
// branch on the current message.
type blockCondition func(StableBlockInputs) bool

// residentBlockSpec is one declared resident prompt-block builder. Every fak-authored
// resident block lives in the baseContext registry in syspromptmmu.go; tests pin the
// ids and reject a missing gate.
type residentBlockSpec struct {
	id      ResidentBlockID
	tier    Tier
	content string
	gate    blockCondition
}

func alwaysResident(StableBlockInputs) bool { return true }

// ResidentBlocks owns the append-only presence state for ONE conversation.
// Reuse the same builder for every turn in that conversation. Once a gate admits an
// id, Next keeps it present even if a later input would make the predicate false.
//
// Admission order is retained, not registry order re-sorted: a block that turns on
// later is appended after the already-resident prefix, so every prior snapshot is a
// segment-for-segment prefix of the next one. The builder is intentionally not safe
// for concurrent use; a conversation turn loop is its single owner.
type ResidentBlocks struct {
	specs   []residentBlockSpec
	present map[ResidentBlockID]residentBlockSpec
	order   []ResidentBlockID
}

// NewResidentBlocks returns a conversation-scoped builder over the complete
// fak-authored resident-block registry.
func NewResidentBlocks() *ResidentBlocks {
	return newResidentBlocks(baseContext)
}

func newResidentBlocks(specs []residentBlockSpec) *ResidentBlocks {
	return &ResidentBlocks{
		specs:   append([]residentBlockSpec(nil), specs...),
		present: make(map[ResidentBlockID]residentBlockSpec, len(specs)),
		order:   make([]ResidentBlockID, 0, len(specs)),
	}
}

// Next evaluates only blocks not already present, latches newly admitted ids, and
// returns the complete resident plan in admission order. A nil gate fails closed
// (absent); the registry coverage test makes such a declaration a red build.
func (b *ResidentBlocks) Next(inputs StableBlockInputs) []Segment {
	if b == nil {
		return nil
	}
	for _, spec := range b.specs {
		if _, ok := b.present[spec.id]; ok {
			continue
		}
		if spec.gate == nil || !spec.gate(inputs) {
			continue
		}
		b.present[spec.id] = spec
		b.order = append(b.order, spec.id)
	}

	out := make([]Segment, 0, len(b.order))
	for _, id := range b.order {
		spec := b.present[id]
		content := []byte(spec.content)
		out = append(out, Segment{
			Tier: spec.tier,
			PromptSegment: cachemeta.PromptSegment{
				Kind:    cachemeta.SegStable,
				Tokens:  estTokens(content),
				Content: content,
				Witness: WitnessFor(content),
			},
		})
	}
	return out
}
