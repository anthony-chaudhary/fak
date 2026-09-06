// Package sessionreplay freezes a turn's regime-conditioned harness decision
// into a deterministically replayable regression fixture.
package sessionreplay

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

// SchemaV1 is the fixture schema format identifier.
const SchemaV1 = "fak.sessionreplay.v1"

// DecisionInputs captures the tool call and raw JSON arguments adjudicated for a turn.
type DecisionInputs struct {
	Tool string          `json:"tool"`
	Args json.RawMessage `json:"args,omitempty"`
}

// Verdict represents the wire-format projection of an adjudication decision.
type Verdict struct {
	Kind   string `json:"kind"`
	Reason string `json:"reason,omitempty"`
}

// Equal reports whether two projected verdicts are identical in kind and reason.
func (v Verdict) Equal(o Verdict) bool { return v.Kind == o.Kind && v.Reason == o.Reason }

// String renders the verdict as KIND or KIND/REASON for legible test failures.
func (v Verdict) String() string {
	if v.Reason == "" {
		return v.Kind
	}
	return v.Kind + "/" + v.Reason
}

// Fixture represents a captured turn and its expected adjudication verdict under an active regime.
type Fixture struct {
	Schema       string         `json:"schema"`
	Turn         DecisionInputs `json:"turn"`
	ActiveRegime string         `json:"active_regime"`
	Expect       Verdict        `json:"expect"`
	Note         string         `json:"note,omitempty"`
	Issue        string         `json:"issue,omitempty"`
}

// Capture constructs a Fixture from captured decision inputs, regime name, and expected verdict.
func Capture(turn DecisionInputs, activeRegime string, expect Verdict) Fixture {
	return Fixture{
		Schema:       SchemaV1,
		Turn:         turn,
		ActiveRegime: activeRegime,
		Expect:       expect,
	}
}

// Marshal encodes the fixture as formatted JSON.
func (f Fixture) Marshal() ([]byte, error) {
	return json.MarshalIndent(f, "", "  ")
}

// ParseFixture decodes fixture bytes with strict schema and unknown-field validation.
func ParseFixture(b []byte) (Fixture, error) {
	var f Fixture
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		return Fixture{}, fmt.Errorf("sessionreplay: invalid fixture: %w", err)
	}
	if f.Schema != SchemaV1 {
		return Fixture{}, fmt.Errorf("sessionreplay: unsupported schema %q (want %q)", f.Schema, SchemaV1)
	}
	return f, nil
}

// LoadFixture reads and parses a fixture from a file path.
func LoadFixture(path string) (Fixture, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Fixture{}, fmt.Errorf("sessionreplay: %w", err)
	}
	return ParseFixture(b)
}
