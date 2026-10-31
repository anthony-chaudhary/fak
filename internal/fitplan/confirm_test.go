package fitplan

// fak-test:runtime fast est=0s

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// TestFitConfirmSchemaGolden pins the JSON shape of the post-load two-phase fit
// confirmation: fixed inputs must produce a byte-stable record — per-class
// planned vs allocated rows in canonical order, both scopes, and the
// unaccounted residual total - free_after - self. A drift here breaks the
// operator's startup log line and the /debug/vars consumers at once.
func TestFitConfirmSchemaGolden(t *testing.T) {
	const (
		gib = int64(1) << 30
		mib = int64(1) << 20
	)
	rec := BuildFitConfirm("synth-q4k.gguf", "test-device",
		ConfirmInput{
			Scope:     "device",
			Total:     32 * gib,
			FreeAfter: 5*gib + 3*mib,
			Known:     true,
			FreeKnown: true,
			Planned: map[ConfirmClass]int64{
				ConfirmWeights:        24 * gib,
				ConfirmKV:             2 * gib,
				ConfirmComputeStaging: 1 * gib,
				ConfirmRing:           512 * mib,
				ConfirmTransient:      128 * mib,
			},
		},
		ConfirmInput{
			Scope:     "host",
			Total:     64 * gib,
			FreeAfter: 60 * gib,
			Known:     true,
			FreeKnown: true,
			Planned: map[ConfirmClass]int64{
				ConfirmWeights: 4 * gib,
			},
		},
	)
	if err := rec.Validate(); err != nil {
		t.Fatalf("golden record fails Validate: %v", err)
	}
	raw, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["schema"] != FitConfirmSchema {
		t.Fatalf("schema = %v, want %q", got["schema"], FitConfirmSchema)
	}
	if got["model"] != "synth-q4k.gguf" || got["backend"] != "test-device" {
		t.Fatalf("identity = model=%v backend=%v", got["model"], got["backend"])
	}
	scopes, ok := got["scopes"].([]any)
	if !ok || len(scopes) != 2 {
		t.Fatalf("scopes = %#v, want device+host", got["scopes"])
	}
	dev := scopes[0].(map[string]any)
	host := scopes[1].(map[string]any)
	if dev["scope"] != "device" || host["scope"] != "host" {
		t.Fatalf("scope names = %v, %v", dev["scope"], host["scope"])
	}
	// Device: self = 24g + 2g + 1g + 512m + 128m; unaccounted = 32g - (5g+3m) - self.
	selfDev := int64(24*gib + 2*gib + 1*gib + 512*mib + 128*mib)
	wantUnaccountedDev := 32*gib - (5*gib + 3*mib) - selfDev
	if dev["unaccounted_bytes"].(float64) != float64(wantUnaccountedDev) {
		t.Fatalf("device unaccounted = %v, want %d", dev["unaccounted_bytes"], wantUnaccountedDev)
	}
	wantOrder := []string{"weights", "kv", "compute/staging", "ring", "transient"}
	classes := dev["classes"].([]any)
	if len(classes) != len(wantOrder) {
		t.Fatalf("device classes = %d rows, want %d in canonical order", len(classes), len(wantOrder))
	}
	for i, c := range classes {
		row := c.(map[string]any)
		if row["class"] != wantOrder[i] {
			t.Fatalf("class %d = %v, want %q (canonical order)", i, row["class"], wantOrder[i])
		}
		if row["planned_bytes"].(float64) != row["allocated_bytes"].(float64) {
			t.Fatalf("class %s planned != allocated without a post-load measurement", row["class"])
		}
	}
	// Host: one class, unaccounted = 64g - 60g - 4g = 0.
	if host["unaccounted_bytes"].(float64) != 0 {
		t.Fatalf("host unaccounted = %v, want 0", host["unaccounted_bytes"])
	}
	if hcls := host["classes"].([]any); len(hcls) != 1 || hcls[0].(map[string]any)["class"] != "weights" {
		t.Fatalf("host classes = %#v, want weights only", host["classes"])
	}
	// Byte-stability: marshaling twice yields identical bytes.
	raw2, _ := json.Marshal(rec)
	if string(raw) != string(raw2) {
		t.Fatal("record is not byte-stable across marshals")
	}
}

// TestFitConfirmUnknownScopeReportsUnprobed asserts an unprobeable scope keeps
// its row with no invented numbers and still validates.
func TestFitConfirmUnknownScopeReportsUnprobed(t *testing.T) {
	rec := BuildFitConfirm("m.gguf", "",
		ConfirmInput{Scope: "device", Total: 1 << 30, FreeAfter: 0, Known: false, FreeKnown: false},
		ConfirmInput{Scope: "host", Total: 8 << 30, FreeAfter: 4 << 30, Known: true, FreeKnown: true, Planned: map[ConfirmClass]int64{ConfirmWeights: 2 << 30}},
	)
	if err := rec.Validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if rec.Scopes[0].TotalBytes != 0 || rec.Scopes[0].FreeKnown {
		t.Fatalf("unknown scope row = %+v, want zero total and FreeKnown=false", rec.Scopes[0])
	}
	if !strings.Contains(rec.Text(), "unprobed") {
		t.Fatalf("log line %q does not mark the unknown scope unprobed", rec.Text())
	}
	if rec.Scopes[1].UnaccountedBytes != (8<<30)-(4<<30)-(2<<30) {
		t.Fatalf("host unaccounted = %d", rec.Scopes[1].UnaccountedBytes)
	}
}

// TestFitConfirmValidateRejectsBadRecords pins the closed contract every
// consumer may rely on: errors.Is matches ErrFitConfirmInvalid.
func TestFitConfirmValidateRejectsBadRecords(t *testing.T) {
	cases := []struct {
		name string
		rec  FitConfirmRecord
	}{
		{"wrong schema", FitConfirmRecord{Schema: "fak.fit_confirm.v999", Scopes: []ConfirmScopeRow{{Scope: "device"}}}},
		{"no scopes", FitConfirmRecord{Schema: FitConfirmSchema}},
		{"bad scope name", FitConfirmRecord{Schema: FitConfirmSchema, Scopes: []ConfirmScopeRow{{Scope: "gpu"}}}},
		{"negative class bytes", FitConfirmRecord{Schema: FitConfirmSchema, Scopes: []ConfirmScopeRow{{Scope: "host", Classes: []ConfirmClassRow{{Class: ConfirmWeights, PlannedBytes: -1}}}}}},
	}
	for _, tc := range cases {
		err := tc.rec.Validate()
		if err == nil {
			t.Fatalf("%s: Validate passed, want failure", tc.name)
		}
		if !errors.Is(err, ErrFitConfirmInvalid) {
			t.Fatalf("%s: errors.Is(err, ErrFitConfirmInvalid) = false (%v)", tc.name, err)
		}
	}
}
