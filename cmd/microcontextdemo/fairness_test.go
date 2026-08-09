package main

import "testing"

func TestFairnessFixture(t *testing.T) {
	r, err := runFairnessFixture()
	if err != nil {
		t.Fatal(err)
	}
	if r.PerTenant["interactive"] != 594 || r.PerTenant["bulk"] != 200 {
		t.Fatalf("report=%+v", r)
	}
}
