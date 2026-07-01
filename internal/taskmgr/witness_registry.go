package taskmgr

import (
	"fmt"
	"sort"
	"strings"
)

const (
	// PathRefKind is the EvidenceRef.Kind PathWitness reads back.
	PathRefKind = "path"
	// kindWitnessSource is the WitnessRecord.Source for the by-kind registry.
	kindWitnessSource = "kind-registry"
)

// WithOriginWitnessByKind installs a by-kind origin witness registry. It is the
// ergonomic version of WithOriginWitness for callers that already know the
// EvidenceRef.Kind but should not have to select PathWitness/ShapeWitness by hand.
func WithOriginWitnessByKind(witnesses map[string]Witness) Option {
	return WithOriginWitness(KindWitness{Witnesses: cloneWitnessMap(witnesses)})
}

// WithDefaultOriginWitnesses installs the built-in origin witnesses for the ref
// kinds taskmgr knows how to read locally: path existence and output shape.
func WithDefaultOriginWitnesses() Option {
	return WithOriginWitnessByKind(DefaultOriginWitnesses())
}

// DefaultOriginWitnesses returns the built-in kind registry. The returned map is a
// fresh copy so callers can add or replace witnesses without mutating package state.
func DefaultOriginWitnesses() map[string]Witness {
	return map[string]Witness{
		PathRefKind:   PathWitness{},
		OutputRefKind: ShapeWitness{},
	}
}

// KindWitness dispatches each EvidenceRef.Kind to the witness registered for that
// kind and folds the per-kind verdicts into one task/step witness record.
type KindWitness struct {
	Witnesses map[string]Witness
}

// WitnessClaim verifies every ref whose kind is registered. Any refused kind
// refuses the whole claim. Unknown or unavailable kinds make the whole claim
// unavailable rather than silently passing as verified.
func (w KindWitness) WitnessClaim(c Claim) WitnessRecord {
	byKind := map[string][]EvidenceRef{}
	for _, ref := range c.Refs {
		byKind[ref.Kind] = append(byKind[ref.Kind], ref)
	}
	kinds := make([]string, 0, len(byKind))
	for kind := range byKind {
		kinds = append(kinds, kind)
	}
	sort.Strings(kinds)

	out := WitnessRecord{Source: kindWitnessSource}
	var verified []string
	var unavailable []string
	var refused []string
	for _, kind := range kinds {
		refs := byKind[kind]
		witness := w.Witnesses[kind]
		if witness == nil {
			unavailable = append(unavailable, fmt.Sprintf("%s:no registered witness", kind))
			out.EvidenceRefs = append(out.EvidenceRefs, cloneEvidenceRefs(refs)...)
			continue
		}
		claim := c
		claim.Refs = cloneEvidenceRefs(refs)
		rec := witness.WitnessClaim(claim)
		if len(rec.EvidenceRefs) > 0 {
			out.EvidenceRefs = append(out.EvidenceRefs, rec.EvidenceRefs...)
		} else {
			out.EvidenceRefs = append(out.EvidenceRefs, cloneEvidenceRefs(refs)...)
		}
		detail := kind
		if rec.Detail != "" {
			detail += ":" + rec.Detail
		}
		switch rec.VerifiedState {
		case VerifiedRefused:
			refused = append(refused, detail)
		case VerifiedDone:
			verified = append(verified, kind)
		default:
			unavailable = append(unavailable, detail)
		}
	}

	switch {
	case len(refused) > 0:
		out.VerifiedState = VerifiedRefused
		out.Verdict = "evidence kind refused"
		out.Detail = strings.Join(refused, "; ")
	case len(unavailable) > 0:
		out.VerifiedState = VerifiedUnavailable
		out.Verdict = "evidence kind unavailable"
		out.Detail = strings.Join(unavailable, "; ")
	case len(verified) > 0:
		out.VerifiedState = VerifiedDone
		out.Detail = "verified evidence kinds: " + strings.Join(verified, ",")
	default:
		out.VerifiedState = VerifiedUnavailable
		out.Detail = "no evidence refs to read back"
	}
	return out
}

func cloneWitnessMap(in map[string]Witness) map[string]Witness {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]Witness, len(in))
	for kind, witness := range in {
		out[kind] = witness
	}
	return out
}
