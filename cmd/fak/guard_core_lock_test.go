package main

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// TestStripCoreLockAllFlagPresent: the flag is detected and removed from argv.
func TestStripCoreLockAllFlagPresent(t *testing.T) {
	core, rest := stripCoreLockAllFlag([]string{"--quiet", "--core-lock-all", "--policy", "p.json"})
	if !core {
		t.Fatal("--core-lock-all must be detected")
	}
	if want := []string{"--quiet", "--policy", "p.json"}; !reflect.DeepEqual(rest, want) {
		t.Fatalf("flag not stripped cleanly: got %v want %v", rest, want)
	}
}

// TestStripCoreLockAllFlagAbsent: without the flag, core is false and argv is
// returned unchanged.
func TestStripCoreLockAllFlagAbsent(t *testing.T) {
	core, rest := stripCoreLockAllFlag([]string{"--quiet"})
	if core {
		t.Fatal("must not report core-lock when flag absent")
	}
	if want := []string{"--quiet"}; !reflect.DeepEqual(rest, want) {
		t.Fatalf("argv changed unexpectedly: %v", rest)
	}
}

// TestCoreLockAllRefusesWiden: under core-lock-all, an added Allow (widening) is
// refused.
func TestCoreLockAllRefusesWiden(t *testing.T) {
	cur := adjudicator.Policy{Allow: map[string]bool{"read_file": true}}
	proposed := adjudicator.Policy{Allow: map[string]bool{"read_file": true, "write_file": true}}
	admit, reason := coreLockAllReloadVerdict(true, cur, proposed)
	if admit {
		t.Fatalf("core-lock-all must refuse a widening reload (%s)", reason)
	}
}

// TestCoreLockAllAdmitsTighten: under core-lock-all, an added Deny (tightening)
// is admitted.
func TestCoreLockAllAdmitsTighten(t *testing.T) {
	cur := adjudicator.Policy{}
	proposed := adjudicator.Policy{Deny: map[string]abi.ReasonCode{"dangerous_tool": abi.ReasonPolicyBlock}}
	admit, reason := coreLockAllReloadVerdict(true, cur, proposed)
	if !admit {
		t.Fatalf("core-lock-all must admit a tighten-only reload (%s)", reason)
	}
}

// TestCoreLockAllInactiveAdmitsWiden: when the mode is off, even a widening
// reload is admitted here (normal gating applies elsewhere).
func TestCoreLockAllInactiveAdmitsWiden(t *testing.T) {
	cur := adjudicator.Policy{Allow: map[string]bool{"read_file": true}}
	proposed := adjudicator.Policy{Allow: map[string]bool{"read_file": true, "write_file": true}}
	if admit, _ := coreLockAllReloadVerdict(false, cur, proposed); !admit {
		t.Fatal("inactive core-lock-all must not refuse anything")
	}
}
