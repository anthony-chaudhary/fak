// Package sessionreplay freezes one turn's regime-conditioned harness decision
// into a checked-in, deterministically replayable regression fixture (#4425,
// part of the managed-turn epic #4107).
//
// The problem it closes: a mode/regime bug — "the plan regime let a write
// through", "autonomy=low still auto-approved" — is reproducible only if you can
// pin the AMBIENT STATE that shaped the decision, not just the transcript. fak
// records the turn and adjudicates every tool call against a NAMED capability
// floor (a "regime"), but nothing folds `(captured decision inputs, active
// regime)` into a golden a regression test re-runs and asserts. So a
// regime-conditioned defect gets fixed by inspection and can silently regress:
// the next refactor of the regime floor has no failing witness. This leaf is the
// missing test-time artifact — freeze the offending `(turn, active_regime)` into
// a fixture, replay the DECISION path deterministically (no live model call, no
// world mutation), and assert the frozen verdict. It is the eval sibling of the
// epic's R3 idempotent runtime replay: R3 makes runtime replay never
// double-apply an effect; this makes captured replay assert a fixed verdict.
//
// What a "regime" is here. fak's harness-native program (#2405/#2409/#2759)
// mints NAMED regimes; a regime is a named capability FLOOR, and the concrete,
// reviewed realization of that floor is the internal/policy preset ladder. So a
// fixture's active_regime resolves to a real policy floor and the captured tool
// call is re-adjudicated through the REAL internal/adjudicator — Replay re-runs
// the genuine harness/policy decision, not a toy stand-in (see replay.go).
//
// Scope honesty: the issue's capture design assumes #2405 already stamps the
// active regime name onto every gateway WireVerdict, so a capturer reads the
// stamp rather than re-deriving it. That stamp is not present on WireVerdict in
// the current tree, so Capture takes the active regime as an explicit argument
// (the value that stamp WOULD carry). Wiring Capture to read a live WireVerdict
// stamp is a follow-on once the stamp lands; it does not change this fixture
// format or Replay.
package sessionreplay

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

// SchemaV1 is the fixture format tag. A fixture whose Schema is not exactly this
// is refused by Replay rather than replayed under a guessed shape.
const SchemaV1 = "fak.sessionreplay.v1"

// DecisionInputs is the captured per-turn decision slice: the tool call the
// harness adjudicated. It is deliberately the small, effect-bearing slice the
// decision is a pure function of — the tool name and its arguments — never the
// model's token stream. Args is kept as raw JSON so it round-trips byte-stable
// and is handed to the adjudicator verbatim as inline argument bytes.
type DecisionInputs struct {
	Tool string          `json:"tool"`
	Args json.RawMessage `json:"args,omitempty"`
}

// Verdict is the stable, named projection of the regime-conditioned decision —
// the assertable half of the fixture. It mirrors the gateway's WireVerdict
// kind/reason vocabulary (the closed, stable names) so a frozen fixture asserts
// against names, never against an integer that could silently renumber.
type Verdict struct {
	Kind   string `json:"kind"`             // ALLOW|DENY|TRANSFORM|QUARANTINE|REQUIRE_WITNESS|DEFER|INDETERMINATE|KIND_<n>
	Reason string `json:"reason,omitempty"` // closed refusal vocabulary, e.g. DEFAULT_DENY|POLICY_BLOCK
}

// Equal reports whether two projected verdicts are identical (kind and reason).
func (v Verdict) Equal(o Verdict) bool { return v.Kind == o.Kind && v.Reason == o.Reason }

// String renders the verdict as KIND or KIND/REASON for legible test failures.
func (v Verdict) String() string {
	if v.Reason == "" {
		return v.Kind
	}
	return v.Kind + "/" + v.Reason
}

// Fixture is one `fak.sessionreplay.v1` record: a captured turn's decision
// inputs, the ACTIVE REGIME that shaped the decision, and the FROZEN verdict the
// regime-conditioned decision produced. The regime is load-bearing: a
// regime-blind fixture cannot catch a mode-conditioned bug, which is the whole
// point — Replay re-derives the verdict from `(Turn, ActiveRegime)` and a
// regression test asserts it equals Expect.
type Fixture struct {
	Schema string `json:"schema"`

	// Turn is the captured decision slice re-adjudicated on replay.
	Turn DecisionInputs `json:"turn"`

	// ActiveRegime is the named regime in force for this turn (e.g. "plan",
	// "autonomous"). It resolves to a real policy floor at replay time.
	ActiveRegime string `json:"active_regime"`

	// Expect is the frozen regime-conditioned verdict this fixture guards. A
	// regression test replays Turn under ActiveRegime and asserts the result
	// equals Expect; a paired negative test flips ActiveRegime alone and asserts
	// the verdict changes.
	Expect Verdict `json:"expect"`

	// Provenance (advisory, never re-adjudicated): what bug this froze and where
	// it came from. Kept out of the decision so the replay stays a pure function
	// of (Turn, ActiveRegime).
	Note  string `json:"note,omitempty"`
	Issue string `json:"issue,omitempty"`
}

// Capture tags a per-turn decision slice with its active regime and the frozen
// verdict, producing a `fak.sessionreplay.v1` fixture. This is the capture point
// the issue frames against the gateway's regime stamp: once #2405's active-regime
// stamp rides the WireVerdict, a capturer passes that stamped value as
// activeRegime; today the caller supplies it explicitly. Capture performs no I/O
// and mutates nothing.
func Capture(turn DecisionInputs, activeRegime string, expect Verdict) Fixture {
	return Fixture{
		Schema:       SchemaV1,
		Turn:         turn,
		ActiveRegime: activeRegime,
		Expect:       expect,
	}
}

// Marshal renders the fixture as indented JSON (the on-disk golden form). Pure;
// no filesystem access.
func (f Fixture) Marshal() ([]byte, error) {
	return json.MarshalIndent(f, "", "  ")
}

// ParseFixture decodes fixture bytes, rejecting unknown fields so a misspelled
// key is a hard error rather than a silently dropped assertion, and validating
// the schema tag so a foreign shape never replays under a guessed interpretation.
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

// LoadFixture reads and parses a fixture file from disk.
func LoadFixture(path string) (Fixture, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Fixture{}, fmt.Errorf("sessionreplay: %w", err)
	}
	return ParseFixture(b)
}
