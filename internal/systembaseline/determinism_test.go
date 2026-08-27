package systembaseline

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

func TestDeterminism(t *testing.T) {
	policy := DefaultPolicy()
	policy.IncludeTopConsumers = true
	build := func() Report {
		return Build(
			quietFixture(100e6),
			fixture(100e6, 1),
			10,
			time.Second,
			policy,
			0,
			false,
		)
	}

	first := build()
	second := build()
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("identical inputs produced different reports:\nfirst:  %+v\nsecond: %+v", first, second)
	}
	firstJSON, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	secondJSON, err := json.Marshal(second)
	if err != nil {
		t.Fatal(err)
	}
	if string(firstJSON) != string(secondJSON) {
		t.Fatalf("identical reports encoded differently:\nfirst:  %s\nsecond: %s", firstJSON, secondJSON)
	}
	if err := first.Validate(); err != nil {
		t.Fatalf("deterministic report is invalid: %v", err)
	}
}
