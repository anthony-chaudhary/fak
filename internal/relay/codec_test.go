package relay

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// C2 (issue #1871) done condition: Parse/Marshal are deterministic and the round-trip is
// byte-identical. These tests are that witness (run: `go test ./internal/relay -run
// Deterministic`). C2 adds no persistence path and no content validation — only the pure,
// no-I/O, no-clock codec and its equality proof.

// sampleBaton builds a fully-populated baton (the schema doc's example, mirroring
// TestBatonConstruction) so the codec is exercised over every field, including the
// embedded ObjectivePin and the slice fields the schema pins to `[]`.
func sampleBaton() Baton {
	return Baton{
		Schema:      Schema,
		RelayID:     "RLY-20260701-0001",
		Leg:         7,
		ParentTrace: "trace-relay-leg-7",
		Objective:   ctxplan.NewObjectivePin("pin-relay-schema", "Ship the relay baton codec and close #1871.", 3),
		DoneWhen:    "A pushed commit resolves issue #1871 and dos commit-audit passes.",
		ProgressCursor: ProgressCursor{
			StartSHA:   "0123456789abcdef0123456789abcdef01234567",
			LedgerRef:  ".dos/runs/relay-demo.jsonl#L12",
			HeldRegion: []string{"internal/relay/**"},
		},
		NextAction:    "Run the C2 witnesses and close #1871.",
		OpenQuestions: []string{"issue:#1872 decides the artifact-pointer resolver interface"},
		Artifacts: []Artifact{
			{Kind: string(ArtifactIssue), Ref: "#1871"},
			{Kind: string(ArtifactFile), Ref: "internal/relay/codec.go"},
		},
		DoNotRederive: []string{"memory:relay-codec-freeform-draft"},
		Tombstone: Tombstone{
			Reason: "RELAY_ROTATED",
			AtSHA:  "0123456789abcdef0123456789abcdef01234567",
			Note:   "codec ready; successor verifies refs",
		},
	}
}

// TestBatonCodecDeterministic is the C2 witness: it asserts both halves of the done
// condition in one place. (1) Marshal is deterministic — the same value encodes to the
// exact same bytes across independent calls, and two independently-constructed equal
// batons encode identically (no map iteration or clock leaks into the wire form). (2)
// The round-trip is byte-identical — Marshal(Parse(Marshal(b))) equals Marshal(b) to the
// byte. (3) Parse is deterministic — decoding the same bytes twice yields DeepEqual
// values.
func TestBatonCodecDeterministic(t *testing.T) {
	b := sampleBaton()

	first, err := Marshal(b)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	second, err := Marshal(b)
	if err != nil {
		t.Fatalf("Marshal (repeat): %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Errorf("Marshal is not deterministic:\n first=%s\nsecond=%s", first, second)
	}
	if other, err := Marshal(sampleBaton()); err != nil {
		t.Fatalf("Marshal (independent build): %v", err)
	} else if !bytes.Equal(first, other) {
		t.Errorf("two equal batons encode differently:\n a=%s\n b=%s", first, other)
	}

	// Byte-identical round-trip: decode the canonical bytes and re-encode them.
	parsed, err := Parse(first)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	reencoded, err := Marshal(parsed)
	if err != nil {
		t.Fatalf("Marshal (round-trip): %v", err)
	}
	if !bytes.Equal(first, reencoded) {
		t.Errorf("round-trip is not byte-identical:\n in =%s\n out=%s", first, reencoded)
	}

	// Parse determinism: the same bytes decode to DeepEqual values every time.
	again, err := Parse(first)
	if err != nil {
		t.Fatalf("Parse (repeat): %v", err)
	}
	if !reflect.DeepEqual(parsed, again) {
		t.Errorf("Parse is not deterministic: %+v vs %+v", parsed, again)
	}
}

// TestBatonRoundTripValueDeterministic pins the value-level round-trip: Parse(Marshal(b))
// reconstructs the canonical baton exactly (DeepEqual against project(b)), so a reader
// inspecting the parsed baton sees the same value the writer emitted. Named to run under
// the `-run Deterministic` witness alongside the byte-identity check.
func TestBatonRoundTripValueDeterministic(t *testing.T) {
	b := sampleBaton()
	data, err := Marshal(b)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	got, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if want := project(b); !reflect.DeepEqual(want, got) {
		t.Errorf("round-trip value mismatch:\n want=%+v\n got =%+v", want, got)
	}
	if got.Schema != Schema {
		t.Errorf("parsed schema = %q, want %q", got.Schema, Schema)
	}
	if !got.Objective.Verify() {
		t.Errorf("parsed objective digest must still verify: %+v", got.Objective)
	}
}

// TestBatonProjectEmptySlicesDeterministic asserts the one wire ambiguity the schema
// calls out is closed: a baton built with nil slices and one built with empty slices
// Marshal to the SAME bytes, and those bytes carry `[]` (never `null`) for the four
// list fields. This is the "project" half of the rung — without it the round-trip would
// not be deterministic across the nil/empty distinction.
func TestBatonProjectEmptySlicesDeterministic(t *testing.T) {
	nilSlices := Baton{Schema: Schema, RelayID: "RLY-empty"}
	emptySlices := Baton{
		Schema:         Schema,
		RelayID:        "RLY-empty",
		OpenQuestions:  []string{},
		Artifacts:      []Artifact{},
		DoNotRederive:  []string{},
		ProgressCursor: ProgressCursor{HeldRegion: []string{}},
	}

	a, err := Marshal(nilSlices)
	if err != nil {
		t.Fatalf("Marshal(nil slices): %v", err)
	}
	b, err := Marshal(emptySlices)
	if err != nil {
		t.Fatalf("Marshal(empty slices): %v", err)
	}
	if !bytes.Equal(a, b) {
		t.Errorf("nil and empty slices must encode identically:\n nil  =%s\n empty=%s", a, b)
	}
	for _, field := range []string{`"open_questions":[]`, `"artifacts":[]`, `"do_not_rederive":[]`, `"held_region":[]`} {
		if !strings.Contains(string(a), field) {
			t.Errorf("canonical baton must contain %s (never null); got %s", field, a)
		}
	}
	if bytes.Contains(a, []byte("null")) {
		t.Errorf("canonical baton must not contain a null list field: %s", a)
	}
}

// TestParseRejectsNonObject confirms Parse fails closed on input that is not a JSON
// object — a bare number, string, or array is corrupt baton input, not an empty baton.
func TestParseRejectsNonObject(t *testing.T) {
	for _, bad := range []string{`123`, `"a string"`, `[1,2,3]`, `not json`} {
		if _, err := Parse([]byte(bad)); err == nil {
			t.Errorf("Parse(%q) = nil error, want a decode error", bad)
		}
	}
	// A well-formed empty object decodes to the zero-identity baton without error.
	if got, err := Parse([]byte(`{}`)); err != nil {
		t.Errorf("Parse({}) errored: %v", err)
	} else if !got.IsZero() {
		t.Errorf("Parse({}) must be the zero baton, got %+v", got)
	}
}

// TestMarshalMatchesEncodingJSON is a guardrail: the canonical bytes are exactly what
// encoding/json produces over the projected value, so no hidden re-ordering or escaping
// is introduced by the codec. If a future edit swaps in a custom encoder, this pins the
// expectation that the wire form stays the stdlib stable-ordering form.
func TestMarshalMatchesEncodingJSON(t *testing.T) {
	b := sampleBaton()
	got, err := Marshal(b)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	want, err := json.Marshal(project(b))
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("Marshal diverged from encoding/json:\n got =%s\n want=%s", got, want)
	}
}
