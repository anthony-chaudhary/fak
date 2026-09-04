package composition

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"sort"
)

// Schema identifies the composition snapshot version format.
const Schema = "fak.composition-snapshot/1"

// Intent describes scheduling goals, latency classes, and policy bounds.
type Intent struct {
	WorkID, Quality, LatencyClass, CostClass, PolicyID, Continuity string
	Tools, Computer                                                bool
}

// Model identifies the engine, checkpoint revision, and declared capabilities.
type Model struct {
	ID, Revision, Provenance, Engine string
	Capabilities                     []string
}

// Execution specifies backend and quantization parameters alongside pipeline phases.
type Execution struct {
	Backend, Quantization string
	Phases                []string
}

// ResourceClaim records memory, device locality, and lifetime constraints for allocated state.
type ResourceClaim struct {
	Kind, Owner, Lifetime, Locality, Compatibility string
	Bytes                                          int64
}

// Edge represents state transfer dependencies between pipeline execution phases.
type Edge struct {
	From, To, Kind string
	Bytes          int64
}

// Snapshot captures the complete immutable pre-allocation execution graph.
type Snapshot struct {
	Schema    string
	Intent    Intent
	Model     Model
	Execution Execution
	Claims    []ResourceClaim
	Edges     []Edge
	Forbidden [][]string
	Digest    string
}

// Reason represents a typed validation failure class.
type Reason string

const (
	// ReasonMissingCapability indicates a required resource or capability was omitted.
	ReasonMissingCapability Reason = "missing_capability"
	// ReasonForbiddenCombination indicates incompatible runtime features were requested simultaneously.
	ReasonForbiddenCombination Reason = "forbidden_combination"
	// ReasonEngineAmbiguous indicates a model engine other than fak-native was specified.
	ReasonEngineAmbiguous Reason = "engine_ambiguous"
	// ReasonUnauthorizedScope indicates missing policy authorization identity.
	ReasonUnauthorizedScope Reason = "unauthorized_scope"
)

// ValidationError records structured diagnostic reasons for graph rejection.
type ValidationError struct {
	Reason Reason
	Detail string
}

// Error formats the validation failure reason and details.
func (e *ValidationError) Error() string { return string(e.Reason) + ": " + e.Detail }

// Receipt records verified execution parameters and graph digests after validation.
type Receipt struct {
	Schema, WorkID, GraphDigest, ModelID, Backend, Engine, Outcome string
	StateKinds, Phases                                             []string
}

// Handle wraps a successfully resolved and validated snapshot.
type Handle struct{ snapshot *Snapshot }

// Snapshot returns the underlying validated composition snapshot.
func (h Handle) Snapshot() *Snapshot { return h.snapshot }

// Resolve verifies backend compatibility and declared capabilities to produce an immutable execution handle.
func Resolve(s Snapshot) (Handle, Receipt, error) {
	s.Schema = Schema
	if s.Model.Engine != "fak-native" {
		return Handle{}, Receipt{}, &ValidationError{ReasonEngineAmbiguous, s.Model.Engine}
	}
	present := map[string]bool{}
	for _, x := range s.Model.Capabilities {
		present[x] = true
	}
	present[s.Execution.Backend] = true
	present[s.Execution.Quantization] = true
	for _, c := range s.Claims {
		if c.Kind == "" || c.Owner == "" || c.Compatibility == "" {
			return Handle{}, Receipt{}, &ValidationError{ReasonMissingCapability, "resource claim"}
		}
		present[c.Kind] = true
	}
	for _, combo := range s.Forbidden {
		all := true
		for _, x := range combo {
			all = all && present[x]
		}
		if all {
			return Handle{}, Receipt{}, &ValidationError{ReasonForbiddenCombination, "declared feature interaction"}
		}
	}
	if s.Intent.PolicyID == "" {
		return Handle{}, Receipt{}, &ValidationError{ReasonUnauthorizedScope, "policy identity required"}
	}
	digest, err := digest(s)
	if err != nil {
		return Handle{}, Receipt{}, err
	}
	s.Digest = digest
	kinds := make([]string, 0, len(s.Claims))
	for _, c := range s.Claims {
		kinds = append(kinds, c.Kind)
	}
	sort.Strings(kinds)
	r := Receipt{Schema: "fak.composition-receipt/1", WorkID: s.Intent.WorkID, GraphDigest: digest, ModelID: s.Model.ID, Backend: s.Execution.Backend, Engine: s.Model.Engine, Outcome: "validated", StateKinds: kinds, Phases: append([]string(nil), s.Execution.Phases...)}
	return Handle{snapshot: &s}, r, nil
}

func digest(s Snapshot) (string, error) {
	s.Digest = ""
	b, e := json.Marshal(s)
	if e != nil {
		return "", e
	}
	x := sha256.Sum256(b)
	return hex.EncodeToString(x[:]), nil
}

// IsReason reports whether err contains a validation error matching the specified reason class.
func IsReason(err error, r Reason) bool {
	var v *ValidationError
	return errors.As(err, &v) && v.Reason == r
}
