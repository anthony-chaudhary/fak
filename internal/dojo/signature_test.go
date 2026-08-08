package dojo

import "testing"

func TestSignatureExcludesDurationAndNodeOrder(t *testing.T) {
	first := Signature("complete", []NodeOutcome{
		{ID: "fetch", Status: "ok", DurationMS: 12},
		{ID: "grade", Status: "ok", DurationMS: 90},
	}, false)
	second := Signature("complete", []NodeOutcome{
		{ID: "grade", Status: "ok", DurationMS: 9_000},
		{ID: "fetch", Status: "ok", DurationMS: 1_200},
	}, false)

	if first != second {
		t.Fatalf("duration and input order must not make a structural outcome novel:\nfirst  %s\nsecond %s", first, second)
	}
}

func TestSignatureChangesWithNodeStatus(t *testing.T) {
	ok := Signature("complete", []NodeOutcome{
		{ID: "fetch", Status: "ok", DurationMS: 12},
		{ID: "grade", Status: "ok", DurationMS: 90},
	}, false)
	failed := Signature("complete", []NodeOutcome{
		{ID: "fetch", Status: "ok", DurationMS: 12},
		{ID: "grade", Status: "failed", DurationMS: 90},
	}, true)

	if ok == failed {
		t.Fatalf("changed structural outcome must have a different signature: %s", ok)
	}
}
