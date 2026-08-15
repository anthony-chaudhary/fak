package quantmeta

// adjudicate.go answers exactly one question about a descriptor: can fak
// describe this artifact without guessing? It never answers whether the
// quantization is good, fast or preferable -- #6221's guardrail is that fak
// selects no universal quantization winner, so nothing here compares methods,
// formats or runtimes to one another.
//
// The four outcomes are deliberately distinct, because collapsing them is how a
// tool ends up silently pretending:
//
//   - SUPPORTED  -- fully declared and self-consistent; fak can describe it.
//   - DELEGATE   -- describable, but its meaning is owned by the producing
//     runtime (a learned codebook), so fak routes instead of claiming.
//   - ABSTAIN    -- fak cannot read it: an unknown schema version, an unknown
//     format, or nothing declared at all. Not an error, just not ours to
//     interpret.
//   - REFUSE     -- fak CAN read it and it contradicts itself. This is the only
//     outcome that says the descriptor is wrong.
//
// The abstain/refuse split is the load-bearing one. An unknown format abstains
// because the artifact may be perfectly valid and merely newer than this build;
// treating "new" as "broken" would make fak a brake on the ecosystem. A
// self-contradictory descriptor refuses because no future version makes
// "per-group with no group size" mean something.

// Outcome is the typed adjudication result. There is no fifth, implicit
// "silently assume" outcome -- that absence is the point.
type Outcome string

// Reason is a public, stable reason code. Every non-supported outcome carries at
// least one, so a caller never has to infer why from prose.
type Reason string

// ClaimClass names what kind of user-facing claim a descriptor licenses. #6222
// requires these to stay separate so an artifact's existence is never reported
// as a measured performance result.
type ClaimClass string

const (
	OutcomeSupported Outcome = "supported"
	OutcomeDelegate  Outcome = "delegate"
	OutcomeAbstain   Outcome = "abstain"
	OutcomeRefuse    Outcome = "refuse"

	// ReasonFullyDeclared accompanies OutcomeSupported.
	ReasonFullyDeclared Reason = "fully-declared"
	// ReasonSchemaUnknown: the descriptor states a schema version this build
	// does not implement. It is not read as if it were SchemaVersion.
	ReasonSchemaUnknown Reason = "schema-unknown"
	// ReasonFormatUnknown: a declared format is outside this build's vocabulary.
	ReasonFormatUnknown Reason = "format-unknown"
	// ReasonNothingDeclared: no weight, activation or KV quantization is
	// described. Undescribed is not the same as unquantized.
	ReasonNothingDeclared Reason = "nothing-declared"
	// ReasonRuntimeOwned: describable, but the producing runtime owns what it
	// means.
	ReasonRuntimeOwned Reason = "runtime-owned"
	// ReasonGroupSizeMissing: grouping was declared without a group size.
	ReasonGroupSizeMissing Reason = "group-size-missing"
	// ReasonZeroPointConflict: an explicitly symmetric quantization carries a
	// present zero point.
	ReasonZeroPointConflict Reason = "zero-point-conflict"
	// ReasonSparsityPatternInvalid: a structured N:M pattern whose N and M
	// cannot describe a real pattern.
	ReasonSparsityPatternInvalid Reason = "sparsity-pattern-invalid"

	// ClaimNone: the descriptor licenses no claim at all.
	ClaimNone ClaimClass = ""
	// ClaimArtifact: a statement about the artifact's own structure.
	ClaimArtifact ClaimClass = "artifact"
	// ClaimRecipe: a statement about how the artifact was produced.
	ClaimRecipe ClaimClass = "recipe"
	// ClaimRuntimeDelegated: a statement that belongs to the runtime, not fak.
	ClaimRuntimeDelegated ClaimClass = "runtime-delegated"
	// ClaimMeasuredEnvelope: a statement bound to a real device and runtime.
	ClaimMeasuredEnvelope ClaimClass = "measured-envelope"
)

// Known reports whether r is a registered public reason code. Callers surfacing
// a reason to a user can check this rather than trusting an arbitrary string.
func (r Reason) Known() bool {
	switch r {
	case ReasonFullyDeclared, ReasonSchemaUnknown, ReasonFormatUnknown,
		ReasonNothingDeclared, ReasonRuntimeOwned, ReasonGroupSizeMissing,
		ReasonZeroPointConflict, ReasonSparsityPatternInvalid:
		return true
	}
	return false
}

// Result is the typed adjudication of one descriptor.
type Result struct {
	Outcome Outcome    `json:"outcome"`
	Reasons []Reason   `json:"reasons"`
	Claim   ClaimClass `json:"claim,omitempty"`
}

// Adjudicate classifies a descriptor. It is a pure function of the descriptor:
// no method identity, producer name or format family changes the outcome for a
// descriptor of the same shape, which is the no-universal-winner guardrail
// expressed as code rather than as a promise.
func Adjudicate(d Descriptor) Result {
	if d.Schema != SchemaVersion {
		return result(OutcomeAbstain, ReasonSchemaUnknown, ClaimNone)
	}
	if d.Weight == nil && d.Activation == nil && d.KV == nil {
		return result(OutcomeAbstain, ReasonNothingDeclared, ClaimNone)
	}

	specs := []*TensorSpec{d.Weight, d.Activation}
	if d.KV != nil {
		specs = append(specs, d.KV.Key, d.KV.Value)
	}
	for _, s := range specs {
		if s == nil {
			continue
		}
		if !s.Format.Known() {
			return result(OutcomeAbstain, ReasonFormatUnknown, ClaimNone)
		}
		if s.Granularity == GranularityPerGroup && s.GroupSize == 0 &&
			(s.Scale == nil || s.Scale.GroupSize == 0) {
			return result(OutcomeRefuse, ReasonGroupSizeMissing, ClaimNone)
		}
		if s.Symmetric != nil && *s.Symmetric && s.ZeroPoint != nil && s.ZeroPoint.Present {
			return result(OutcomeRefuse, ReasonZeroPointConflict, ClaimNone)
		}
	}

	if sp := d.Sparsity; sp != nil && (sp.Kind == "structured" || sp.Pattern == "n:m") {
		if sp.N <= 0 || sp.M <= 0 || sp.N >= sp.M {
			return result(OutcomeRefuse, ReasonSparsityPatternInvalid, ClaimNone)
		}
	}

	// A learned codebook is describable, but only its producer knows what the
	// entries mean, so fak routes rather than claiming support.
	if d.Weight != nil && (d.Weight.Format == FormatCodebook || d.Weight.Codebook != nil) {
		return result(OutcomeDelegate, ReasonRuntimeOwned, ClaimRuntimeDelegated)
	}

	return result(OutcomeSupported, ReasonFullyDeclared, claimClass(d))
}

// claimClass reports the strongest claim the descriptor licenses. The ordering
// is about EVIDENCE, not quality: a measured envelope outranks a named recipe
// because it is bound to a real device, not because it is better quantization.
func claimClass(d Descriptor) ClaimClass {
	if e := d.Envelope; e != nil && e.DeviceID != "" && e.RuntimeID != "" && e.MeasuredOn != "" {
		return ClaimMeasuredEnvelope
	}
	if d.Provenance.MethodID != "" {
		return ClaimRecipe
	}
	return ClaimArtifact
}

func result(o Outcome, r Reason, c ClaimClass) Result {
	return Result{Outcome: o, Reasons: []Reason{r}, Claim: c}
}
