package relay

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// B3 (issue #1867): the shadow would-be-baton emitter. The projection must carry the
// objective pin VERBATIM and only durable pointers (no bytes, no recap, structurally no
// `claimed` field anywhere in the type tree), and the sidecar it writes must round-trip
// through the C2 codec unchanged (run: `go test ./internal/relay -run ShadowBaton`).

// shadowBatonInput builds the shared test input around a verified objective pin.
func shadowBatonInput(t *testing.T) (ShadowBatonInput, ctxplan.ObjectivePin) {
	t.Helper()
	pin := ctxplan.NewObjectivePin("pin-1867", "Carry the objective across the shadow projection unchanged.", 4)
	if !pin.Verify() {
		t.Fatalf("precondition: a freshly minted pin must verify against its own pin_id+text: %+v", pin)
	}
	in := ShadowBatonInput{
		RelayID:     "RLY-1867",
		Leg:         3,
		ParentTrace: "trace-b3",
		Objective:   pin,
		DoneWhen:    "done_when: issue #1867 closed",
		NextAction:  "emit shadow baton",
		StartSHA:    "abc123",
		HeldRegion:  []string{"internal/relay/**"},
		Artifacts: []Artifact{
			{Kind: string(ArtifactCommit), Ref: "abc123"},
			{Kind: string(ArtifactIssue), Ref: "#1867"},
			{Kind: string(ArtifactFile), Ref: "internal/relay/shadowbaton.go"},
		},
	}
	return in, pin
}

// TestShadowBatonCarriesObjectivePinAndOnlyDurablePointers asserts the projected baton
// carries the objective pin verbatim, anchors progress on the re-verifiable cursor,
// carries only closed-vocabulary durable pointers, and has no `claimed` field anywhere
// in its type tree.
func TestShadowBatonCarriesObjectivePinAndOnlyDurablePointers(t *testing.T) {
	in, pin := shadowBatonInput(t)
	b := ProjectShadowBaton(in)

	if b.IsZero() {
		t.Fatal("a projected shadow baton must not be the zero baton")
	}
	if b.Schema != Schema {
		t.Errorf("Schema = %q, want the package constant %q", b.Schema, Schema)
	}
	if b.RelayID != in.RelayID {
		t.Errorf("RelayID = %q, want %q", b.RelayID, in.RelayID)
	}
	if b.Leg != 3 {
		t.Errorf("Leg = %d, want 3", b.Leg)
	}

	// The objective pin is carried verbatim — the done condition.
	if b.Objective != pin {
		t.Errorf("objective pin not carried verbatim:\n want=%+v\n got =%+v", pin, b.Objective)
	}
	if !b.Objective.Verify() {
		t.Errorf("carried objective must still Verify(): %+v", b.Objective)
	}

	if b.ProgressCursor.StartSHA != "abc123" {
		t.Errorf("ProgressCursor.StartSHA = %q, want %q", b.ProgressCursor.StartSHA, "abc123")
	}
	if len(b.ProgressCursor.HeldRegion) != 1 || b.ProgressCursor.HeldRegion[0] != "internal/relay/**" {
		t.Errorf("ProgressCursor.HeldRegion = %v, want [internal/relay/**]", b.ProgressCursor.HeldRegion)
	}

	// Only durable pointers: every artifact carries a non-empty ref and a kind drawn
	// from the closed ArtifactKind vocabulary.
	kinds := map[string]bool{
		string(ArtifactCommit): true,
		string(ArtifactIssue):  true,
		string(ArtifactMemory): true,
		string(ArtifactLedger): true,
		string(ArtifactFile):   true,
		string(ArtifactImage):  true,
	}
	for i, a := range b.Artifacts {
		if a.Ref == "" {
			t.Errorf("artifact %d has an empty ref: %+v", i, a)
		}
		if !kinds[a.Kind] {
			t.Errorf("artifact %d kind %q is outside the closed ArtifactKind vocabulary", i, a.Kind)
		}
	}

	if b.Tombstone.Reason != "RELAY_ARMED" {
		t.Errorf("Tombstone.Reason = %q, want RELAY_ARMED (the default)", b.Tombstone.Reason)
	}
	if b.Tombstone.AtSHA != "abc123" {
		t.Errorf("Tombstone.AtSHA = %q, want %q (defaulted to StartSHA)", b.Tombstone.AtSHA, "abc123")
	}

	// The pointer-only / no-`claimed` invariant, reflectively: no exported field
	// anywhere in the Baton type tree may be named "claimed" (case-insensitive).
	assertNoClaimedField(t, reflect.TypeOf(b), "Baton")
}

// assertNoClaimedField walks exported struct fields recursively (through structs,
// slices, arrays, pointers, and maps) and fails if any field name lowercases to
// "claimed".
func assertNoClaimedField(t *testing.T, typ reflect.Type, path string) {
	t.Helper()
	switch typ.Kind() {
	case reflect.Ptr, reflect.Slice, reflect.Array:
		assertNoClaimedField(t, typ.Elem(), path)
	case reflect.Map:
		assertNoClaimedField(t, typ.Key(), path)
		assertNoClaimedField(t, typ.Elem(), path)
	case reflect.Struct:
		for i := 0; i < typ.NumField(); i++ {
			f := typ.Field(i)
			if !f.IsExported() {
				continue
			}
			if strings.ToLower(f.Name) == "claimed" {
				t.Errorf("field %s.%s violates the no-`claimed` invariant", path, f.Name)
			}
			assertNoClaimedField(t, f.Type, path+"."+f.Name)
		}
	}
}

// TestShadowBatonSidecarRoundTrips asserts the emitter writes one sidecar under dir
// whose bytes Parse back (C2 codec) to a baton carrying the same relay id and a
// still-verifying objective pin with the same digest.
func TestShadowBatonSidecarRoundTrips(t *testing.T) {
	in, pin := shadowBatonInput(t)
	dir := t.TempDir()

	path, err := EmitShadowBaton(dir, in)
	if err != nil {
		t.Fatalf("EmitShadowBaton: %v", err)
	}
	if filepath.Dir(path) != dir {
		t.Fatalf("sidecar path %q is not under dir %q", path, dir)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%q): %v", path, err)
	}
	got, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if got.Objective.Digest != pin.Digest {
		t.Errorf("objective digest drifted across the sidecar round-trip:\n before=%q\n after =%q", pin.Digest, got.Objective.Digest)
	}
	if !got.Objective.Verify() {
		t.Errorf("round-tripped objective must still Verify(): %+v", got.Objective)
	}
	if got.RelayID != in.RelayID {
		t.Errorf("RelayID = %q, want %q", got.RelayID, in.RelayID)
	}
}
