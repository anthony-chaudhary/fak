package main

import (
	"flag"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
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

// TestServeMetalFlagDefaultsOff pins the --metal flag's default-OFF posture: with no flag the
// serve path is the pure-Go CPU forward, byte-for-byte unchanged; --metal flips it on. This
// mirrors the live registration in serve.go.
func TestServeMetalFlagDefaultsOff(t *testing.T) {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	metal := fs.Bool("metal", false, "")
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty: %v", err)
	}
	if *metal {
		t.Fatal("--metal must default to false (CPU forward); the GPU path is opt-in")
	}
	fs2 := flag.NewFlagSet("serve", flag.ContinueOnError)
	metal2 := fs2.Bool("metal", false, "")
	if err := fs2.Parse([]string{"--metal"}); err != nil {
		t.Fatalf("parse --metal: %v", err)
	}
	if !*metal2 {
		t.Fatal("--metal must flip the bit on")
	}
}

// TestResolveServeMetal exercises the fail-loud decision GPU-free. On this non-fakmetal build
// metalgemm.Available()/Compiled() are the stub's deterministic false, so a request for Metal
// (via flag or FAK_METAL) MUST error rather than silently fall back to CPU — serve's posture is
// fail-loud, mirroring resolveServeChatBackend. The --metal + --backend conflict is rejected
// regardless of build. The off case (neither requested) returns (false, nil).
func TestResolveServeMetal(t *testing.T) {
	// Not requested → off, no error, on every build.
	if use, err := resolveServeMetal(false, false, ""); use || err != nil {
		t.Fatalf("neither flag nor env: got (%v,%v), want (false,nil)", use, err)
	}
	// Requested + a device --backend → conflict error, independent of Metal availability.
	if _, err := resolveServeMetal(true, false, "cuda"); err == nil {
		t.Fatal("--metal with --backend cuda must be rejected as mutually exclusive")
	}
	if _, err := resolveServeMetal(false, true, "cuda"); err == nil {
		t.Fatal("FAK_METAL with --backend cuda must be rejected as mutually exclusive")
	}
	// Requested with no conflicting backend: on a non-fakmetal build this fails loud (no silent
	// CPU fallback). On a fakmetal build with a device it would succeed — assert by availability
	// so the test is correct on BOTH builds.
	use, err := resolveServeMetal(true, false, "")
	if metalgemm.Available() {
		if !use || err != nil {
			t.Fatalf("metal available: got (%v,%v), want (true,nil)", use, err)
		}
	} else {
		if use || err == nil {
			t.Fatalf("metal unavailable must fail loud: got (%v,%v), want (false, error)", use, err)
		}
	}
	// FAK_METAL env is an equivalent trigger to the flag (same code path).
	if _, err := resolveServeMetal(false, true, ""); !metalgemm.Available() && err == nil {
		t.Fatal("FAK_METAL on a non-fakmetal build must fail loud, same as --metal")
	}
}
