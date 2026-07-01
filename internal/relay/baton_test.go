package relay

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// C1 (issue #1870) done condition: the Baton type compiles with all fak.relay.baton.v1
// schema fields and a passing construction test, and no claimed-progress field exists.
// These tests are that witness. Parse/validate is a later rung (C2 #1871); here we only
// assert the SHAPE — construction, the zero-value predicate, and the structural
// no-`claimed` invariant.

// TestBatonConstruction builds a fully-populated baton mirroring the schema doc's
// example (RELAY-BATON-SCHEMA-2026-07-01.md) and checks every field round-trips through
// the struct. It also confirms the Objective is a real ctxplan.ObjectivePin whose digest
// Verify()s — the "carried verbatim, digest-checkable" property the reader contract relies
// on — so the objective half of the baton cannot silently lie about its own text.
func TestBatonConstruction(t *testing.T) {
	obj := ctxplan.NewObjectivePin("pin-relay-schema", "Ship the relay baton schema and close #1863.", 3)
	b := Baton{
		Schema:      Schema,
		RelayID:     "RLY-20260701-0001",
		Leg:         7,
		ParentTrace: "trace-relay-leg-7",
		Objective:   obj,
		DoneWhen:    "A pushed commit resolves issue #1863 and dos commit-audit passes.",
		ProgressCursor: ProgressCursor{
			StartSHA:   "0123456789abcdef0123456789abcdef01234567",
			LedgerRef:  ".dos/runs/relay-demo.jsonl#L12",
			HeldRegion: []string{"INDEX.md", "docs/notes/**"},
		},
		NextAction:    "Run the schema doc witnesses and close #1863.",
		OpenQuestions: []string{"issue:#1870 decides the Go package for Baton"},
		Artifacts: []Artifact{
			{Kind: string(ArtifactIssue), Ref: "#1863"},
			{Kind: string(ArtifactFile), Ref: "docs/notes/RELAY-BATON-SCHEMA-2026-07-01.md"},
		},
		DoNotRederive: []string{"memory:relay-schema-freeform-draft"},
		Tombstone: Tombstone{
			Reason: "RELAY_ROTATED",
			AtSHA:  "0123456789abcdef0123456789abcdef01234567",
			Note:   "schema handoff ready; successor must verify refs",
		},
	}

	if b.Schema != "fak.relay.baton.v1" {
		t.Errorf("schema tag = %q, want fak.relay.baton.v1", b.Schema)
	}
	if b.RelayID == "" || b.ParentTrace == "" {
		t.Error("relay_id and parent_trace must be set on a constructed baton")
	}
	if b.Leg != 7 {
		t.Errorf("leg = %d, want 7", b.Leg)
	}
	if !b.Objective.Verify() {
		t.Errorf("objective digest must verify against its own pin_id+text; pin=%+v", b.Objective)
	}
	if b.ProgressCursor.StartSHA == "" {
		t.Error("progress_cursor.start_sha is the ground-truth anchor and must be set")
	}
	if len(b.ProgressCursor.HeldRegion) != 2 {
		t.Errorf("held_region round-trip = %v, want 2 globs", b.ProgressCursor.HeldRegion)
	}
	if len(b.Artifacts) != 2 || b.Artifacts[0].Kind != "issue" || b.Artifacts[1].Kind != "file" {
		t.Errorf("artifacts round-trip = %+v", b.Artifacts)
	}
	if b.Tombstone.Reason != "RELAY_ROTATED" || b.Tombstone.AtSHA == "" {
		t.Errorf("tombstone round-trip = %+v", b.Tombstone)
	}
	if b.IsZero() {
		t.Error("a fully-populated baton must not report IsZero")
	}
}

// TestBatonIsZero pins the zero-value predicate: an unset baton is zero, and a baton is
// no longer zero as soon as it carries either identity field (Schema or RelayID). This
// matches IsZero's contract — a partially-built baton must not slip through as "absent."
func TestBatonIsZero(t *testing.T) {
	if !(Baton{}).IsZero() {
		t.Error("the zero Baton must report IsZero")
	}
	if (Baton{Schema: Schema}).IsZero() {
		t.Error("a baton carrying the schema tag is not the zero value")
	}
	if (Baton{RelayID: "RLY-x"}).IsZero() {
		t.Error("a baton carrying a relay_id is not the zero value")
	}
	// A non-identity field alone (e.g. a next_action) still leaves the baton zero by the
	// documented predicate — IsZero keys off identity, not arbitrary content.
	if !(Baton{NextAction: "do the thing"}).IsZero() {
		t.Error("a baton with only a next_action but no identity is still the zero value")
	}
}

// TestBatonHasNoClaimedField is the load-bearing structural invariant of the schema:
// "No `claimed` field, by construction." Progress must be a re-verifiable cursor, never
// an asserted number. Walking the type tree reflectively means a future edit that adds a
// `claimed` field anywhere — a Go field name or a json tag, at any nesting depth — fails
// this test instead of silently reopening the self-report door.
func TestBatonHasNoClaimedField(t *testing.T) {
	seen := map[reflect.Type]bool{}
	var walk func(t reflect.Type, path string)
	walk = func(rt reflect.Type, path string) {
		for rt.Kind() == reflect.Ptr || rt.Kind() == reflect.Slice || rt.Kind() == reflect.Array {
			rt = rt.Elem()
		}
		if rt.Kind() != reflect.Struct || seen[rt] {
			return
		}
		seen[rt] = true
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			jsonTag := strings.Split(f.Tag.Get("json"), ",")[0]
			if strings.EqualFold(f.Name, "claimed") || strings.EqualFold(jsonTag, "claimed") {
				t.Errorf("forbidden `claimed` field at %s.%s (json:%q) — progress must be a re-verifiable cursor", path, f.Name, jsonTag)
			}
			walk(f.Type, path+"."+f.Name)
		}
	}
	walk(reflect.TypeOf(Baton{}), "Baton")
}
