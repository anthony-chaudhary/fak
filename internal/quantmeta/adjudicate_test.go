package quantmeta

import (
	"os"
	"path/filepath"
	"testing"
)

// adjudicate_test.go is the #6222 witness for the "explicit typed result"
// half of the acceptance gate: unknown and unsupported input must produce a
// named outcome and reason code, never a silent fallback -- and the adjudicator
// must stay method-neutral, which is the parent epic's (#6221) standing
// guardrail against selecting a universal quantization winner.

func parseFixture(t *testing.T, name string) Descriptor {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name+".input.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	d, err := Parse(b)
	if err != nil {
		t.Fatalf("Parse(%s): %v", name, err)
	}
	return d
}

func hasReason(r Result, want Reason) bool {
	for _, got := range r.Reasons {
		if got == want {
			return true
		}
	}
	return false
}

// TestUnknownSchemaAbstains: a descriptor from a future schema version is not
// guessed at and not silently downgraded to v1 -- it abstains with a reason.
func TestUnknownSchemaAbstains(t *testing.T) {
	got := Adjudicate(parseFixture(t, "future_schema"))
	if got.Outcome != OutcomeAbstain {
		t.Errorf("Outcome = %q, want %q", got.Outcome, OutcomeAbstain)
	}
	if !hasReason(got, ReasonSchemaUnknown) {
		t.Errorf("Reasons = %v, want to include %q", got.Reasons, ReasonSchemaUnknown)
	}
	// An abstain must never license a claim about the artifact.
	if got.Claim != ClaimNone {
		t.Errorf("Claim = %q, want %q on abstain", got.Claim, ClaimNone)
	}
}

// TestFullyDeclaredDescriptorsAreSupported: the ordinary path stays green for
// unrelated industry families, so the refusal paths below mean something.
func TestFullyDeclaredDescriptorsAreSupported(t *testing.T) {
	for _, name := range []string{"gguf_q4k", "gptq_int4", "fp8_weight_activation", "bitnet_ternary", "kv_int4_sparse"} {
		t.Run(name, func(t *testing.T) {
			got := Adjudicate(parseFixture(t, name))
			if got.Outcome != OutcomeSupported {
				t.Errorf("Outcome = %q (reasons %v), want %q", got.Outcome, got.Reasons, OutcomeSupported)
			}
		})
	}
}

// TestCodebookDelegates: a learned codebook is describable but its realization
// is owned by the producing runtime, so fak routes rather than claiming.
func TestCodebookDelegates(t *testing.T) {
	got := Adjudicate(parseFixture(t, "codebook_delegate"))
	if got.Outcome != OutcomeDelegate {
		t.Errorf("Outcome = %q (reasons %v), want %q", got.Outcome, got.Reasons, OutcomeDelegate)
	}
	if !hasReason(got, ReasonRuntimeOwned) {
		t.Errorf("Reasons = %v, want to include %q", got.Reasons, ReasonRuntimeOwned)
	}
	if got.Claim != ClaimRuntimeDelegated {
		t.Errorf("Claim = %q, want %q", got.Claim, ClaimRuntimeDelegated)
	}
}

// TestIncoherentDescriptorsRefuse: a descriptor that contradicts itself is a
// REFUSE (we can read it and it is wrong), which is a different typed result
// from an ABSTAIN (we cannot read it at all).
func TestIncoherentDescriptorsRefuse(t *testing.T) {
	base := parseFixture(t, "gptq_int4")

	t.Run("per-group without a group size", func(t *testing.T) {
		d := base
		w := *base.Weight
		w.GroupSize = 0
		scale := *w.Scale
		scale.GroupSize = 0
		w.Scale = &scale
		d.Weight = &w
		got := Adjudicate(d)
		if got.Outcome != OutcomeRefuse {
			t.Errorf("Outcome = %q (reasons %v), want %q", got.Outcome, got.Reasons, OutcomeRefuse)
		}
		if !hasReason(got, ReasonGroupSizeMissing) {
			t.Errorf("Reasons = %v, want to include %q", got.Reasons, ReasonGroupSizeMissing)
		}
	})

	t.Run("zero point on a symmetric quantization", func(t *testing.T) {
		d := base
		w := *base.Weight
		yes := true
		w.Symmetric = &yes
		w.ZeroPoint = &ZeroPointSpec{Present: true, Format: FormatF16, Granularity: GranularityPerGroup}
		d.Weight = &w
		got := Adjudicate(d)
		if got.Outcome != OutcomeRefuse {
			t.Errorf("Outcome = %q (reasons %v), want %q", got.Outcome, got.Reasons, OutcomeRefuse)
		}
		if !hasReason(got, ReasonZeroPointConflict) {
			t.Errorf("Reasons = %v, want to include %q", got.Reasons, ReasonZeroPointConflict)
		}
	})

	t.Run("structured sparsity without n or m", func(t *testing.T) {
		d := parseFixture(t, "kv_int4_sparse")
		s := *d.Sparsity
		s.N = 0
		d.Sparsity = &s
		got := Adjudicate(d)
		if got.Outcome != OutcomeRefuse {
			t.Errorf("Outcome = %q (reasons %v), want %q", got.Outcome, got.Reasons, OutcomeRefuse)
		}
		if !hasReason(got, ReasonSparsityPatternInvalid) {
			t.Errorf("Reasons = %v, want to include %q", got.Reasons, ReasonSparsityPatternInvalid)
		}
	})
}

// TestUnknownFormatAbstains: an unrecognized format is not refused (it may be
// perfectly valid and simply newer than fak) and it is not guessed at -- it
// abstains, which is the honest answer for "I cannot describe this".
func TestUnknownFormatAbstains(t *testing.T) {
	d := parseFixture(t, "gptq_int4")
	w := *d.Weight
	w.Format = Format("some-2027-format")
	d.Weight = &w
	got := Adjudicate(d)
	if got.Outcome != OutcomeAbstain {
		t.Errorf("Outcome = %q (reasons %v), want %q", got.Outcome, got.Reasons, OutcomeAbstain)
	}
	if !hasReason(got, ReasonFormatUnknown) {
		t.Errorf("Reasons = %v, want to include %q", got.Reasons, ReasonFormatUnknown)
	}
}

// TestNothingDeclaredAbstains: an empty descriptor is not "unquantized", it is
// undescribed. Treating it as supported would be exactly the silent fallback
// #6222 forbids.
func TestNothingDeclaredAbstains(t *testing.T) {
	got := Adjudicate(Descriptor{Schema: SchemaVersion})
	if got.Outcome != OutcomeAbstain {
		t.Errorf("Outcome = %q, want %q", got.Outcome, OutcomeAbstain)
	}
	if !hasReason(got, ReasonNothingDeclared) {
		t.Errorf("Reasons = %v, want to include %q", got.Reasons, ReasonNothingDeclared)
	}
}

// TestEveryOutcomeHasAReason is the no-silent-result guard: whatever the input,
// a non-supported outcome always carries at least one public reason code.
func TestEveryOutcomeHasAReason(t *testing.T) {
	inputs := []Descriptor{
		{},
		{Schema: SchemaVersion},
		parseFixture(t, "future_schema"),
		parseFixture(t, "codebook_delegate"),
		parseFixture(t, "gguf_q4k"),
	}
	for i, d := range inputs {
		got := Adjudicate(d)
		if got.Outcome == "" {
			t.Errorf("input %d: Outcome is empty, want an explicit typed outcome", i)
		}
		if got.Outcome != OutcomeSupported && len(got.Reasons) == 0 {
			t.Errorf("input %d: outcome %q carries no reason code", i, got.Outcome)
		}
		for _, r := range got.Reasons {
			if !r.Known() {
				t.Errorf("input %d: reason %q is not a registered public reason code", i, r)
			}
		}
	}
}

// TestNoUniversalWinner is the parent epic's guardrail as an executable check.
// Holding the descriptor's SHAPE fixed and varying only the method identity, the
// adjudicated result must not move. If any method were privileged -- ranked,
// preferred, or specially blessed -- this test would catch it.
func TestNoUniversalWinner(t *testing.T) {
	methods := []string{
		"gptq", "awq", "hqq", "rtn", "bitnet-b1.58", "llm-compressor",
		"torchao", "bitsandbytes", "marlin", "mlx", "some-unheard-of-method",
	}
	base := parseFixture(t, "gptq_int4")
	want := Adjudicate(base)
	for _, m := range methods {
		d := base
		p := base.Provenance
		p.MethodID = m
		d.Provenance = p
		got := Adjudicate(d)
		if got.Outcome != want.Outcome {
			t.Errorf("method %q: Outcome = %q, want %q -- the adjudicator prefers a method", m, got.Outcome, want.Outcome)
		}
		if got.Claim != want.Claim {
			t.Errorf("method %q: Claim = %q, want %q -- the adjudicator ranks a method", m, got.Claim, want.Claim)
		}
		if len(got.Reasons) != len(want.Reasons) {
			t.Errorf("method %q: Reasons = %v, want %v", m, got.Reasons, want.Reasons)
		}
	}

	// The same neutrality across FORMAT families of equal descriptive
	// completeness: a 4-bit integer artifact and an fp8 artifact both describe
	// themselves fully, so both are supported. Neither is "the winner".
	for _, name := range []string{"gguf_q4k", "gptq_int4", "fp8_weight_activation", "bitnet_ternary"} {
		if got := Adjudicate(parseFixture(t, name)); got.Outcome != OutcomeSupported {
			t.Errorf("%s: Outcome = %q, want %q -- a fully-declared family was disadvantaged", name, got.Outcome, OutcomeSupported)
		}
	}
}

// TestClaimClassSeparation: #6222 requires user-facing claims to distinguish
// artifact, recipe, runtime delegation and measured hardware envelope. The
// load-bearing half is the fence -- a measured-envelope claim is unreachable
// without a real device and runtime recorded on the descriptor.
func TestClaimClassSeparation(t *testing.T) {
	t.Run("measured envelope needs a device and a runtime", func(t *testing.T) {
		got := Adjudicate(parseFixture(t, "fp8_weight_activation"))
		if got.Claim != ClaimMeasuredEnvelope {
			t.Errorf("Claim = %q, want %q for a descriptor carrying a real envelope", got.Claim, ClaimMeasuredEnvelope)
		}
	})

	t.Run("no envelope never yields a measured claim", func(t *testing.T) {
		d := parseFixture(t, "fp8_weight_activation")
		d.Envelope = nil
		if got := Adjudicate(d); got.Claim == ClaimMeasuredEnvelope {
			t.Error("Claim = measured-envelope with no envelope recorded -- a fabricated hardware claim")
		}
	})

	t.Run("a half-filled envelope never yields a measured claim", func(t *testing.T) {
		d := parseFixture(t, "fp8_weight_activation")
		for _, partial := range []MeasuredEnvelope{
			{RuntimeID: "vllm", MeasuredOn: "2026-08-04"},
			{DeviceID: "NVIDIA H100 80GB HBM3", MeasuredOn: "2026-08-04"},
			{DeviceID: "NVIDIA H100 80GB HBM3", RuntimeID: "vllm"},
		} {
			e := partial
			d.Envelope = &e
			if got := Adjudicate(d); got.Claim == ClaimMeasuredEnvelope {
				t.Errorf("envelope %+v yielded a measured-envelope claim; want it withheld", partial)
			}
		}
	})

	t.Run("a declared method is a recipe claim, a bare artifact is not", func(t *testing.T) {
		d := parseFixture(t, "gptq_int4")
		if got := Adjudicate(d); got.Claim != ClaimRecipe {
			t.Errorf("Claim = %q, want %q for a descriptor naming its method", got.Claim, ClaimRecipe)
		}
		p := d.Provenance
		p.MethodID = ""
		d.Provenance = p
		if got := Adjudicate(d); got.Claim != ClaimArtifact {
			t.Errorf("Claim = %q, want %q for a descriptor with no method recorded", got.Claim, ClaimArtifact)
		}
	})
}
