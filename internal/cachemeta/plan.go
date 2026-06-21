package cachemeta

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// PlanTemplate is the field-only shape for an advisory, reusable plan-template
// cache entry (Agentic Plan Caching). A template is keyed by task class, parameter
// digest, plan-schema digest, tool-manifest digest, and policy version. §2.6's gap:
// no PlanePlanTemplate / intent-key cache existed yet, and the first version must be
// read-only/advisory and abstain aggressively.
//
// A cached plan is NEVER an execution permit (refusal rule 5): a HIT yields a
// candidate that must re-enter plancfi/adjudication before any tool effect. That is
// encoded mechanically via AdmissionDefer + the must_reenter_plancfi Meta flag.
type PlanTemplate struct {
	TaskClass          string
	ParamsDigest       string
	PlanSchemaDigest   string
	ToolManifestDigest string
	PolicyVersion      string
	Steps              []string // candidate tool graph (advisory; not an execution permit)
	Producer           string
	TTLMillis          int64
	Scope              abi.ShareScope
}

// PlanCacheRequest is what the current task asks of the plan-template cache.
type PlanCacheRequest struct {
	TaskClass          string
	ParamsDigest       string
	PlanSchemaDigest   string
	ToolManifestDigest string
	PolicyVersion      string
	StateWitness       string // "" => current state/witness missing => REVALIDATE, not HIT
}

// FromPlanTemplate lowers an advisory plan template into a cachemeta entry on the
// plan_template plane. The entry is admitted as Defer (a candidate awaiting
// re-adjudication), never Allow-as-execution-permit.
func FromPlanTemplate(p PlanTemplate, opts ...Option) Entry {
	producer := p.Producer
	if producer == "" {
		producer = "plan-cache"
	}
	// ShareScope's zero value is ScopeAgent (private) — the correct fail-closed
	// default for an advisory candidate, so no explicit default is needed.
	e := Entry{
		ID: EntryID{
			Digest:    DigestPlanTemplate(p),
			MediaType: MediaPlanTemplate,
			Length:    int64(len(p.Steps)),
			Unit:      UnitBytes,
		},
		Plane: PlanePlanTemplate,
		Derivation: Derivation{
			Producer:   producer,
			SourceRefs: nil,
		},
		Validity: Validity{
			PolicyVersion: p.PolicyVersion,
			TTLMillis:     p.TTLMillis,
		},
		Security: Security{
			Taint:            abi.TaintTrusted,
			Scope:            p.Scope,
			AdmissionVerdict: AdmissionDefer,
			AdmittedBy:       producer,
			Reason:           "advisory_reenter_plancfi",
		},
		Residency: Residency{Tier: TierRecompute, Owner: producer},
		Coherence: Coherence{InvalidationMode: InvalidationPolicy},
		Labels: map[string]string{
			"task_class":           p.TaskClass,
			"params_digest":        p.ParamsDigest,
			"plan_schema_digest":   p.PlanSchemaDigest,
			"tool_manifest_digest": p.ToolManifestDigest,
		},
	}
	apply(&e, opts)
	return e
}

// PlanTemplateLookup is the abstaining verdict for plan-template reuse (Step 5):
//
//   - exact task class + params + schema + tool manifest + policy, WITH a present
//     state witness -> HIT (an advisory candidate; must re-enter plancfi).
//   - exact binding but no state witness -> REVALIDATE (stale state).
//   - any binding axis mismatch -> MISS.
//
// A HIT is never permission to execute: the returned entry carries AdmissionDefer
// and the verdict Meta marks must_reenter_plancfi=true.
func PlanTemplateLookup(req PlanCacheRequest, tpl PlanTemplate) LookupVerdict {
	e := FromPlanTemplate(tpl)
	if req.TaskClass != tpl.TaskClass ||
		req.ParamsDigest != tpl.ParamsDigest ||
		req.PlanSchemaDigest != tpl.PlanSchemaDigest ||
		req.ToolManifestDigest != tpl.ToolManifestDigest ||
		req.PolicyVersion != tpl.PolicyVersion {
		return Miss(ReasonPolicyMismatch)
	}
	if req.StateWitness == "" {
		return Revalidate(e, ReasonStale)
	}
	v := Hit(e)
	if v.Meta == nil {
		v.Meta = map[string]string{}
	}
	v.Meta["must_reenter_plancfi"] = "true"
	return v
}

// DigestPlanTemplate is the deterministic identity over a plan template's binding
// axes. Length counts steps (advisory); the digest is the reuse key.
func DigestPlanTemplate(p PlanTemplate) string {
	h := sha256.New()
	writeNullField := func(s string) {
		_, _ = h.Write([]byte(s))
		_, _ = h.Write([]byte{0})
	}
	writeNullField(p.TaskClass)
	writeNullField(p.ParamsDigest)
	writeNullField(p.PlanSchemaDigest)
	writeNullField(p.ToolManifestDigest)
	writeNullField(p.PolicyVersion)
	for _, s := range p.Steps {
		writeNullField(s)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// IntentKey is the field-only shape for a semantic/intent-cache entry. Intent
// canonicalization fails when keys optimize for generic similarity instead of
// precision; this record carries a measured precision and a threshold so the lookup
// can abstain (Structured Intent Canonicalization).
type IntentKey struct {
	IntentDigest       string
	PrecisionThreshold float64 // cluster precision required to act on a hit
	Precision          float64 // measured precision of this particular match
	Producer           string
	PolicyVersion      string
	Scope              abi.ShareScope
	TTLMillis          int64
}

// IntentCacheRequest is what the current call asks of the intent cache.
type IntentCacheRequest struct {
	IntentDigest string
}

// FromIntentKey lowers a semantic-intent entry onto the semantic_intent plane. Like
// a plan template, it is advisory (Defer) and may never directly execute effects
// (refusal rule 4: a semantic cache may suggest; it may not bypass adjudication).
func FromIntentKey(k IntentKey, opts ...Option) Entry {
	producer := k.Producer
	if producer == "" {
		producer = "intent-cache"
	}
	// ShareScope zero value == ScopeAgent (private) is the correct default.
	digest := k.IntentDigest
	if digest == "" {
		digest = DigestIntentKey(k)
	}
	e := Entry{
		ID: EntryID{
			Digest:    digest,
			MediaType: MediaIntentKey,
			Length:    0,
			Unit:      UnitBytes,
		},
		Plane: PlaneSemanticIntent,
		Derivation: Derivation{
			Producer: producer,
		},
		Validity: Validity{
			PolicyVersion: k.PolicyVersion,
			TTLMillis:     k.TTLMillis,
		},
		Security: Security{
			Taint:            abi.TaintTrusted,
			Scope:            k.Scope,
			AdmissionVerdict: AdmissionDefer,
			AdmittedBy:       producer,
			Reason:           "advisory_no_direct_effect",
		},
		Residency: Residency{Tier: TierRecompute, Owner: producer},
		Coherence: Coherence{InvalidationMode: InvalidationPolicy},
		Metrics: Metrics{
			QualityDeltaProbe: k.Precision,
			Coverage:          k.PrecisionThreshold,
		},
		Labels: map[string]string{
			"precision_threshold": strconv.FormatFloat(k.PrecisionThreshold, 'g', -1, 64),
			"precision":           strconv.FormatFloat(k.Precision, 'g', -1, 64),
		},
	}
	apply(&e, opts)
	return e
}

// IntentKeyLookup is the abstaining verdict for semantic-intent reuse. It HITs only
// on an exact intent digest AND precision at/above threshold; otherwise it abstains
// (MISS with approximate_fault for a sub-threshold near match, or absent for a
// digest mismatch). Abstention is the safe default — a false positive executes the
// wrong cached answer.
func IntentKeyLookup(req IntentCacheRequest, k IntentKey) LookupVerdict {
	e := FromIntentKey(k)
	if req.IntentDigest != k.IntentDigest {
		if k.Precision > 0 && k.Precision < k.PrecisionThreshold {
			return Miss(ReasonApproxFault)
		}
		return Miss(ReasonAbsent)
	}
	if k.Precision < k.PrecisionThreshold {
		return Miss(ReasonApproxFault)
	}
	v := Hit(e)
	if v.Meta == nil {
		v.Meta = map[string]string{}
	}
	v.Meta["advisory_only"] = "true"
	return v
}

// DigestIntentKey is the deterministic identity for an intent entry.
func DigestIntentKey(k IntentKey) string {
	h := sha256.New()
	_, _ = h.Write([]byte(k.IntentDigest))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(strconv.FormatFloat(k.PrecisionThreshold, 'g', -1, 64)))
	return hex.EncodeToString(h.Sum(nil))
}
