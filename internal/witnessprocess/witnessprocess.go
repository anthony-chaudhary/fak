// Package witnessprocess defines the witness-first contract shared by issue and worker packets.
package witnessprocess

import (
	"errors"
	"fmt"
	"strings"
)

type Context string

const (
	Logic             Context = "logic-correctness"
	Visual            Context = "tui-visual"
	Security          Context = "security-policy"
	Reliability       Context = "reliability-operations"
	Cost              Context = "cost-token"
	NativePerformance Context = "native-performance"
)

type Policy string

const (
	Warn    Policy = "warn"
	Enforce Policy = "enforce"
)

type Exception string

const (
	NoException         Exception = ""
	ReadOnlyTriage      Exception = "read-only-triage"
	TrivialChange       Exception = "trivial-change"
	UrgentResponse      Exception = "urgent-safety-outage"
	ExternalUnavailable Exception = "external-system-unavailable"
)

type NegativeResult struct {
	Lever     string `json:"lever"`
	Artifact  string `json:"artifact"`
	Falsifier string `json:"falsifier"`
}

type Block struct {
	Context           Context          `json:"context"`
	Envelope          string           `json:"envelope,omitempty"`
	BaselineArtifact  string           `json:"baseline_artifact,omitempty"`
	Lever             string           `json:"lever,omitempty"`
	CandidateArtifact string           `json:"candidate_artifact,omitempty"`
	PromotionGate     string           `json:"promotion_gate,omitempty"`
	DurableWitness    string           `json:"durable_witness,omitempty"`
	Policy            Policy           `json:"policy,omitempty"`
	Exception         Exception        `json:"exception,omitempty"`
	ExceptionReason   string           `json:"exception_reason,omitempty"`
	NegativeResults   []NegativeResult `json:"negative_results,omitempty"`
}

type Adapter struct {
	Context            Context `json:"context"`
	EnvelopePrompt     string  `json:"envelope_prompt"`
	BaselineClass      string  `json:"baseline_artifact_class"`
	CandidateClass     string  `json:"candidate_artifact_class"`
	PromotionPrimitive string  `json:"promotion_primitive"`
	DurablePrimitive   string  `json:"durable_witness_primitive"`
}

var adapters = map[Context]Adapter{
	Logic:             {Logic, "inputs, expected behavior, and deterministic environment", "failing behavior repro", "passing behavior repro", "focused Go test", "regression test"},
	Visual:            {Visual, "terminal dimensions, renderer, and interaction state", "captured failing render", "captured corrected render", "render-witness assertion", "render-witness test and, for live UI, before/after screenshot"},
	Security:          {Security, "policy, capability set, tool call, and adversary input", "denied/allowed preflight or red-team receipt", "candidate preflight or red-team receipt", "policy/attestation gate", "policy fixture or red-team regression"},
	Reliability:       {Reliability, "load, duration, dependencies, and operating envelope", "failure/incident or soak artifact", "candidate soak/operation artifact", "end-to-end reliability SLO including recovery cost", "deterministic fault test or bounded soak witness"},
	Cost:              {Cost, "provider/model, cache state, workload, and accounting boundary", "net-true usage receipt", "candidate net-true usage receipt", "quality-constrained end-to-end cost/token gate", "usage fixture or benchmark regression"},
	NativePerformance: {NativePerformance, "native engine, model, hardware, workload, quality, and accounting envelope", "native baseline benchmark receipt", "native candidate benchmark receipt", "matched-envelope quality-constrained net performance gate", "native benchmark/receipt regression"},
}

func AdapterFor(context Context) (Adapter, bool) { a, ok := adapters[context]; return a, ok }

func Contexts() []Context {
	return []Context{Logic, Visual, Security, Reliability, Cost, NativePerformance}
}

func (b Block) Validate() (warnings []string, err error) {
	if b.Exception != NoException {
		if !validException(b.Exception) {
			return nil, fmt.Errorf("unknown witness-first exception %q", b.Exception)
		}
		if strings.TrimSpace(b.ExceptionReason) == "" {
			return nil, errors.New("witness-first exception requires a reason")
		}
		return nil, nil
	}
	if _, ok := AdapterFor(b.Context); !ok {
		return nil, fmt.Errorf("unknown witness-first context %q", b.Context)
	}
	var missing []string
	for name, value := range map[string]string{
		"context/envelope": b.Envelope, "baseline artifact": b.BaselineArtifact,
		"one declared lever": b.Lever, "candidate artifact": b.CandidateArtifact,
		"promotion gate": b.PromotionGate, "durable witness": b.DurableWitness,
	} {
		if strings.TrimSpace(value) == "" {
			missing = append(missing, name)
		}
	}
	// Stable field order for packets and fixtures.
	ordered := []string{"context/envelope", "baseline artifact", "one declared lever", "candidate artifact", "promotion gate", "durable witness"}
	missing = missing[:0]
	values := map[string]string{"context/envelope": b.Envelope, "baseline artifact": b.BaselineArtifact, "one declared lever": b.Lever, "candidate artifact": b.CandidateArtifact, "promotion gate": b.PromotionGate, "durable witness": b.DurableWitness}
	for _, name := range ordered {
		if strings.TrimSpace(values[name]) == "" {
			missing = append(missing, name)
		}
	}
	if len(missing) == 0 {
		return nil, nil
	}
	message := "witness-first block missing " + strings.Join(missing, ", ")
	if b.Policy == Enforce {
		return nil, errors.New(message)
	}
	return []string{message}, nil
}

func validException(e Exception) bool {
	return e == ReadOnlyTriage || e == TrivialChange || e == UrgentResponse || e == ExternalUnavailable
}

func (b Block) RenderMarkdown() (string, error) {
	warnings, err := b.Validate()
	if err != nil {
		return "", err
	}
	if b.Exception != NoException {
		return fmt.Sprintf("### Witness-first contract\n\n- **Exception:** `%s` — %s\n", b.Exception, strings.TrimSpace(b.ExceptionReason)), nil
	}
	a, _ := AdapterFor(b.Context)
	var out strings.Builder
	fmt.Fprintf(&out, "### Witness-first contract — %s\n\n", b.Context)
	fmt.Fprintf(&out, "- **Context / envelope** (%s): %s\n", a.EnvelopePrompt, b.Envelope)
	fmt.Fprintf(&out, "- **Baseline artifact** (%s): %s\n", a.BaselineClass, b.BaselineArtifact)
	fmt.Fprintf(&out, "- **One declared lever:** %s\n", b.Lever)
	fmt.Fprintf(&out, "- **Candidate artifact** (%s): %s\n", a.CandidateClass, b.CandidateArtifact)
	fmt.Fprintf(&out, "- **Promotion gate** (%s): %s\n", a.PromotionPrimitive, b.PromotionGate)
	fmt.Fprintf(&out, "- **Durable witness** (%s): %s\n", a.DurablePrimitive, b.DurableWitness)
	for _, n := range b.NegativeResults {
		fmt.Fprintf(&out, "- **Rejected lever:** %s — artifact: %s; falsifier: %s\n", n.Lever, n.Artifact, n.Falsifier)
	}
	for _, warning := range warnings {
		fmt.Fprintf(&out, "- **Warning:** %s\n", warning)
	}
	return out.String(), nil
}
