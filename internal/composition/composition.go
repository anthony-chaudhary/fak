// Package composition resolves an immutable pre-allocation execution graph.
package composition

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"sort"
)

const Schema = "fak.composition-snapshot/1"

type Intent struct {
	WorkID, Quality, LatencyClass, CostClass, PolicyID, Continuity string
	Tools, Computer                                                bool
}
type Model struct {
	ID, Revision, Provenance, Engine string
	Capabilities                     []string
}
type Execution struct {
	Backend, Quantization string
	Phases                []string
}
type ResourceClaim struct {
	Kind, Owner, Lifetime, Locality, Compatibility string
	Bytes                                          int64
}
type Edge struct {
	From, To, Kind string
	Bytes          int64
}
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
type Reason string

const (
	ReasonMissingCapability    Reason = "missing_capability"
	ReasonForbiddenCombination Reason = "forbidden_combination"
	ReasonEngineAmbiguous      Reason = "engine_ambiguous"
	ReasonUnauthorizedScope    Reason = "unauthorized_scope"
)

type ValidationError struct {
	Reason Reason
	Detail string
}

func (e *ValidationError) Error() string { return string(e.Reason) + ": " + e.Detail }

type Receipt struct {
	Schema, WorkID, GraphDigest, ModelID, Backend, Engine, Outcome string
	StateKinds, Phases                                             []string
}
type Handle struct{ snapshot *Snapshot }

func (h Handle) Snapshot() *Snapshot { return h.snapshot }
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
func IsReason(err error, r Reason) bool {
	var v *ValidationError
	return errors.As(err, &v) && v.Reason == r
}
