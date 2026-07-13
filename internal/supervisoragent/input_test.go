package supervisoragent

import (
	"reflect"
	"sort"
	"strings"
	"testing"
)

// walkFields flattens a type into a stable, sorted set of "path:kind" leaf
// descriptors, recursing structs by field name and slices/arrays via a "[]"
// segment. A named scalar (WorkerState, time.Duration) reduces to its underlying
// Kind, so the pin tracks shape, not spelling.
func walkFields(t reflect.Type, path string, out *[]string) {
	switch t.Kind() {
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			child := f.Name
			if path != "" {
				child = path + "." + f.Name
			}
			walkFields(f.Type, child, out)
		}
	case reflect.Slice, reflect.Array:
		walkFields(t.Elem(), path+"[]", out)
	default:
		*out = append(*out, path+":"+t.Kind().String())
	}
}

// TestSupervisorInputClosedFieldSet is the golden pin (leaf #4478's Witness): the
// projection's leaf field set is exactly this closed list. Any added field — most
// dangerously a "just in case" transcript/last-message field — fails this test
// until the closed set is DELIBERATELY updated, forcing a reviewer to see the
// payload creeping in. That structural closedness is what keeps the supervisor a
// router (safe), not a recognizer (forbidden).
func TestSupervisorInputClosedFieldSet(t *testing.T) {
	want := []string{
		"Escalations.Present:bool",
		"Escalations.Value[].Class:string",
		"Escalations.Value[].ID:string",
		"Escalations.Value[].Issue:string",
		"Escalations.Value[].ReasonCode:string",
		"Escalations.Value[].RunID:string",
		"Escalations.Value[].Severity:string",
		"Leases.Present:bool",
		"Leases.Value[].Kind:string",
		"Leases.Value[].Lane:string",
		"Leases.Value[].Tree[]:string",
		"Liveness.Present:bool",
		"Liveness.Value.Class:string",
		"Liveness.Value.Commits:int",
		"Liveness.Value.Region[].Kind:string",
		"Liveness.Value.Region[].Lane:string",
		"Liveness.Value.RunID:string",
		"Liveness.Value.SilentFor:int64",
		"Workers.Present:bool",
		"Workers.Value[].Issue:string",
		"Workers.Value[].Lane:string",
		"Workers.Value[].RunID:string",
		"Workers.Value[].State:string",
	}
	sort.Strings(want)

	var got []string
	walkFields(reflect.TypeOf(SupervisorInput{}), "", &got)
	sort.Strings(got)

	if !reflect.DeepEqual(got, want) {
		t.Errorf("SupervisorInput field set drifted from the closed contract.\n got: %v\nwant: %v", got, want)
	}
}

// TestSupervisorInputNoPayloadFieldName is the intent guard paired with the pin:
// no field name anywhere in the projection may name a transcript / message / body /
// free-text / log payload. This catches a payload field even if the closed set
// above were updated carelessly — the two together make adding prose a two-place,
// obviously-wrong change.
func TestSupervisorInputNoPayloadFieldName(t *testing.T) {
	forbidden := []string{
		"transcript", "message", "msg", "body", "payload", "prose",
		"freetext", "rawlog", "log", "content", "chat", "narrative",
		"summary", "comment", "text",
	}
	var names []string
	collectFieldNames(reflect.TypeOf(SupervisorInput{}), &names)
	for _, n := range names {
		low := strings.ToLower(n)
		for _, bad := range forbidden {
			if strings.Contains(low, bad) {
				t.Errorf("projection field %q contains forbidden payload token %q — a transcript/payload field turns the supervisor into a recognizer", n, bad)
			}
		}
	}
}

// collectFieldNames gathers every struct field name reachable from t.
func collectFieldNames(t reflect.Type, out *[]string) {
	switch t.Kind() {
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			*out = append(*out, f.Name)
			collectFieldNames(f.Type, out)
		}
	case reflect.Slice, reflect.Array:
		collectFieldNames(t.Elem(), out)
	}
}

// TestWithheldWitnessMarkedAbsent proves the absent-marker behavior: a surface
// that could not be read is reported absent (and drives AnyAbsent), while a
// present-but-empty surface is NOT absent — the "green absence is not a green
// witness" distinction.
func TestWithheldWitnessMarkedAbsent(t *testing.T) {
	in := SupervisorInput{
		Liveness:    Seen(Liveness{RunID: "RID-1", Class: "moving", Commits: 3}),
		Workers:     Seen([]WorkerVerdict{}), // present, but empty: no workers running
		Escalations: Absent[[]Escalation](),  // the escalation queue could not be read
		Leases:      Absent[[]Lease](),       // the lease store could not be read
	}

	got := in.AbsentWitnesses()
	want := []string{"escalations", "leases"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("AbsentWitnesses() = %v, want %v", got, want)
	}
	if !in.AnyAbsent() {
		t.Error("AnyAbsent() = false; a withheld witness must gate the decision layer to escalate")
	}

	// A present-but-empty surface must NOT read as absent.
	if in.Workers.Present != true {
		t.Error("a present, empty Workers witness must stay present, not collapse to absent")
	}

	// Fully-witnessed input has nothing absent.
	full := SupervisorInput{
		Liveness:    Seen(Liveness{}),
		Workers:     Seen([]WorkerVerdict{}),
		Escalations: Seen([]Escalation{}),
		Leases:      Seen([]Lease{}),
	}
	if full.AnyAbsent() {
		t.Errorf("fully-witnessed input reports absent surfaces: %v", full.AbsentWitnesses())
	}
}
