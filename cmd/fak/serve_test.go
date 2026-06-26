package main

import (
	"flag"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestCompactHistoryBudgetDefaultsToDefaultConst pins the default-on sprawl trigger: the
// --compact-history-budget flag (defined identically in runServe and runGuard) now defaults
// to gateway.DefaultCompactHistoryBudget — a non-zero default so a sprawling conversation is
// compacted with NO operator configuration — while an explicit =0 preserves the byte-for-byte
// OFF opt-out. This mirrors the live flag registration in serve.go:98 / guard.go:84; if either
// is changed away from the const, this test fails.
func TestCompactHistoryBudgetDefaultsToDefaultConst(t *testing.T) {
	if gateway.DefaultCompactHistoryBudget <= 0 {
		t.Fatalf("DefaultCompactHistoryBudget must be a non-zero default-on value, got %d", gateway.DefaultCompactHistoryBudget)
	}

	// Default with no flag passed → the const (default-on).
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	budget := fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "")
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty: %v", err)
	}
	if *budget != gateway.DefaultCompactHistoryBudget {
		t.Fatalf("default = %d, want %d (the default-on trigger)", *budget, gateway.DefaultCompactHistoryBudget)
	}

	// Explicit =0 → OFF opt-out (the only way to get the old byte-for-byte path now).
	fs2 := flag.NewFlagSet("serve", flag.ContinueOnError)
	budget2 := fs2.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "")
	if err := fs2.Parse([]string{"--compact-history-budget=0"}); err != nil {
		t.Fatalf("parse =0: %v", err)
	}
	if *budget2 != 0 {
		t.Fatalf("explicit =0 must override to OFF, got %d", *budget2)
	}
}

func TestRepeatedStringFlagAccumulatesTrimmedValues(t *testing.T) {
	var f repeatedStringFlag
	if err := f.Set(" http://127.0.0.1:8001/v1 "); err != nil {
		t.Fatalf("Set first: %v", err)
	}
	if err := f.Set("http://127.0.0.1:8002/v1"); err != nil {
		t.Fatalf("Set second: %v", err)
	}
	want := []string{"http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"}
	if got := f.Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Values() = %v, want %v", got, want)
	}
	if got := f.String(); got != "http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1" {
		t.Fatalf("String() = %q", got)
	}
	got := f.Values()
	got[0] = "mutated"
	if again := f.Values(); !reflect.DeepEqual(again, want) {
		t.Fatalf("Values() returned internal storage: %v", again)
	}
}

func TestRepeatedStringFlagRejectsEmptyValue(t *testing.T) {
	var f repeatedStringFlag
	if err := f.Set(" \t "); err == nil {
		t.Fatal("Set blank value succeeded, want error")
	}
}
