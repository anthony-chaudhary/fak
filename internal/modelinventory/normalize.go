package modelinventory

import (
	"path"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Normalize produces an inventory only when every observation is structurally
// sound and currently witnessed. Any diagnostic empties the candidate set.
func Normalize(in Observations, asOf time.Time) (Inventory, Diagnostics) {
	if asOf.IsZero() {
		return Inventory{Schema: Schema}, Diagnostics{newDiagnostic(
			CodeAsOfRequired, "", "as_of", "", "normalization time is required",
			"pass the explicit UTC time at which evidence is being evaluated",
		)}
	}
	asOf = asOf.UTC().Truncate(time.Second)
	out := Inventory{Schema: Schema, AsOf: asOf.Format(time.RFC3339)}
	for _, raw := range in.Providers {
		out.Candidates = append(out.Candidates, normalizeProvider(raw))
	}
	for _, raw := range in.Locals {
		out.Candidates = append(out.Candidates, normalizeLocal(raw))
	}
	out = canonicalizeInventory(out)
	if diagnostics := out.ValidateAt(asOf); len(diagnostics) != 0 {
		out.Candidates = nil
		return out, diagnostics
	}
	return out, nil
}

func normalizeProvider(raw ProviderObservation) Candidate {
	return normalizeEntry(Candidate{
		ID: strings.TrimSpace(raw.ID),
		Identity: Identity{
			Source:    SourceProvider,
			Provider:  strings.ToLower(strings.TrimSpace(raw.Provider)),
			Artifact:  strings.Trim(strings.TrimSpace(raw.Repository), "/"),
			Revision:  strings.ToLower(strings.TrimSpace(raw.Revision)),
			Digest:    strings.ToLower(strings.TrimSpace(raw.Digest)),
			Format:    strings.ToLower(strings.TrimSpace(raw.Format)),
			Witnesses: raw.IdentityEvidence,
		},
		Evidence: raw.Evidence,
	})
}

func normalizeLocal(raw LocalObservation) Candidate {
	artifact := strings.ReplaceAll(strings.TrimSpace(raw.Artifact), "\\", "/")
	artifact = strings.TrimPrefix(path.Clean(artifact), "./")
	return normalizeEntry(Candidate{
		ID: strings.TrimSpace(raw.ID),
		Identity: Identity{
			Source:    SourceLocal,
			Artifact:  artifact,
			Digest:    strings.ToLower(strings.TrimSpace(raw.Digest)),
			Format:    strings.ToLower(strings.TrimSpace(raw.Format)),
			Witnesses: raw.IdentityEvidence,
		},
		Evidence: raw.Evidence,
	})
}

func canonicalizeInventory(in Inventory) Inventory {
	out := Inventory{
		Schema: strings.TrimSpace(in.Schema),
		AsOf:   canonicalTime(in.AsOf),
	}
	out.Candidates = make([]Candidate, len(in.Candidates))
	for i, candidate := range in.Candidates {
		out.Candidates[i] = normalizeEntry(candidate)
	}
	sort.SliceStable(out.Candidates, func(i, j int) bool {
		a, b := out.Candidates[i], out.Candidates[j]
		if a.ID != b.ID {
			return a.ID < b.ID
		}
		if a.Identity.Source != b.Identity.Source {
			return a.Identity.Source < b.Identity.Source
		}
		return identityKey(a.Identity) < identityKey(b.Identity)
	})
	return out
}

func normalizeEntry(in Candidate) Candidate {
	out := Candidate{
		ID: strings.TrimSpace(in.ID),
		Identity: Identity{
			Source:    SourceKind(strings.ToLower(strings.TrimSpace(string(in.Identity.Source)))),
			Provider:  strings.ToLower(strings.TrimSpace(in.Identity.Provider)),
			Artifact:  strings.TrimSpace(in.Identity.Artifact),
			Revision:  strings.ToLower(strings.TrimSpace(in.Identity.Revision)),
			Digest:    strings.ToLower(strings.TrimSpace(in.Identity.Digest)),
			Format:    strings.ToLower(strings.TrimSpace(in.Identity.Format)),
			Witnesses: canonicalEvidence(in.Identity.Witnesses),
		},
	}
	out.Evidence.Availability = canonicalFact(in.Evidence.Availability)
	if out.Evidence.Availability.Name == "" {
		out.Evidence.Availability.Name = "available"
	}
	out.Evidence.Serving = canonicalFacts(in.Evidence.Serving)
	out.Evidence.Platform = canonicalFacts(in.Evidence.Platform)
	out.Evidence.Policy = canonicalFacts(in.Evidence.Policy)
	out.Evidence.Capabilities = canonicalFacts(in.Evidence.Capabilities)
	return out
}

func canonicalFacts(in []Fact) []Fact {
	out := make([]Fact, len(in))
	for i, fact := range in {
		out[i] = canonicalFact(fact)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return valueKey(out[i].Value) < valueKey(out[j].Value)
	})
	merged := make([]Fact, 0, len(out))
	for _, fact := range out {
		last := len(merged) - 1
		if last >= 0 && merged[last].Name == fact.Name && valueKey(merged[last].Value) == valueKey(fact.Value) {
			merged[last].Witnesses = canonicalEvidence(append(merged[last].Witnesses, fact.Witnesses...))
			continue
		}
		merged = append(merged, fact)
	}
	return merged
}

func canonicalFact(in Fact) Fact {
	return Fact{
		Name:      strings.ToLower(strings.TrimSpace(in.Name)),
		Value:     cloneValue(in.Value),
		Witnesses: canonicalEvidence(in.Witnesses),
	}
}

func canonicalEvidence(in []Witness) []Witness {
	out := make([]Witness, len(in))
	for i, witness := range in {
		out[i] = Witness{
			Kind:       WitnessKind(strings.ToLower(strings.TrimSpace(string(witness.Kind)))),
			Source:     strings.TrimSpace(witness.Source),
			ObservedAt: canonicalTime(witness.ObservedAt),
			ExpiresAt:  canonicalTime(witness.ExpiresAt),
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		if a.Source != b.Source {
			return a.Source < b.Source
		}
		if a.ObservedAt != b.ObservedAt {
			return a.ObservedAt < b.ObservedAt
		}
		return a.ExpiresAt < b.ExpiresAt
	})
	return compactEvidence(out)
}

func compactEvidence(in []Witness) []Witness {
	out := in[:0]
	for _, witness := range in {
		if len(out) == 0 || out[len(out)-1] != witness {
			out = append(out, witness)
		}
	}
	return out
}

func cloneValue(in Value) Value {
	var out Value
	if in.Bool != nil {
		v := *in.Bool
		out.Bool = &v
	}
	if in.Integer != nil {
		v := *in.Integer
		out.Integer = &v
	}
	if in.Text != nil {
		v := strings.TrimSpace(*in.Text)
		out.Text = &v
	}
	return out
}

func valueKey(v Value) string {
	switch {
	case v.Bool != nil && v.Integer == nil && v.Text == nil:
		if *v.Bool {
			return "b:1"
		}
		return "b:0"
	case v.Integer != nil && v.Bool == nil && v.Text == nil:
		return "i:" + strconv.FormatInt(*v.Integer, 10)
	case v.Text != nil && v.Bool == nil && v.Integer == nil:
		return "s:" + *v.Text
	default:
		return "invalid"
	}
}

func identityKey(id Identity) string {
	return strings.Join([]string{string(id.Source), id.Provider, id.Artifact, id.Revision, id.Digest, id.Format}, "\x00")
}
