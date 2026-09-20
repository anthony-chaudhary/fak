package main

import (
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// The flag is the STRICT door: a value that is not a boolean refuses the launch, and it does so BEFORE
// anything is mutated. A refusal that had already written the env var would leave the process carrying
// a setting the operator never successfully asked for.
func TestServeCrossLayerGatePrefetchRefusesABadValueWithoutMutatingTheEnv(t *testing.T) {
	for _, bad := range []string{"maybe", "2", "onoff", "tru"} {
		t.Run(bad, func(t *testing.T) {
			t.Setenv(agent.CrossLayerGatePrefetchEnv, "sentinel")
			err := applyServeCrossLayerGatePrefetch(bad)
			if err == nil {
				t.Fatalf("--%s %q was admitted; a value that is not a boolean must not fall back to a setting the operator did not choose", serveCrossLayerGatePrefetchFlag, bad)
			}
			if !strings.Contains(err.Error(), serveCrossLayerGatePrefetchFlag) {
				t.Errorf("refusal %q does not name the flag the operator typed", err)
			}
			if got := os.Getenv(agent.CrossLayerGatePrefetchEnv); got != "sentinel" {
				t.Errorf("a REFUSED value mutated %s to %q — validation must precede the write", agent.CrossLayerGatePrefetchEnv, got)
			}
		})
	}
}

// An un-passed flag must be byte-for-byte the previous path, INCLUDING on a host whose profile already
// exports the env var. Clobbering an ambient export with "the flag's default" would silently disable a
// setting the operator configured elsewhere.
func TestServeCrossLayerGatePrefetchUnpassedLeavesTheAmbientEnvAlone(t *testing.T) {
	for _, unpassed := range []string{"", "   "} {
		t.Setenv(agent.CrossLayerGatePrefetchEnv, "true")
		if err := applyServeCrossLayerGatePrefetch(unpassed); err != nil {
			t.Fatalf("an un-passed flag (%q) refused: %v", unpassed, err)
		}
		if got := os.Getenv(agent.CrossLayerGatePrefetchEnv); got != "true" {
			t.Fatalf("un-passed flag (%q) changed %s to %q, want the ambient \"true\" untouched", unpassed, agent.CrossLayerGatePrefetchEnv, got)
		}
	}
}

// A passed flag wins over the ambient environment. The load-bearing case is the explicit "false": the
// operator at the terminal is more specific than the host profile, so typing =false on a box that
// exports FAK_CROSS_LAYER_GATE_PREFETCH=1 must actually serve with the knob off.
func TestServeCrossLayerGatePrefetchPassedValueWinsOverTheAmbientEnv(t *testing.T) {
	for _, tc := range []struct {
		name    string
		ambient string
		flag    string
		want    string
	}{
		{"false overrides an ambient true", "true", "false", "false"},
		{"true overrides an ambient false", "false", "true", "true"},
		{"on normalizes to true", "", "on", "true"},
		{"off normalizes to false", "", "off", "false"},
		{"case and padding are normalized", "", "  TRUE ", "true"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(agent.CrossLayerGatePrefetchEnv, tc.ambient)
			if err := applyServeCrossLayerGatePrefetch(tc.flag); err != nil {
				t.Fatalf("--%s %q refused: %v", serveCrossLayerGatePrefetchFlag, tc.flag, err)
			}
			if got := os.Getenv(agent.CrossLayerGatePrefetchEnv); got != tc.want {
				t.Fatalf("--%s %q over ambient %q left %s=%q, want %q", serveCrossLayerGatePrefetchFlag, tc.flag, tc.ambient, agent.CrossLayerGatePrefetchEnv, got, tc.want)
			}
		})
	}
}

// Whatever the flag writes must parse back to the SAME setting the planner's env seam will read, or
// the flag and the planner disagree about what was asked for.
func TestServeCrossLayerGatePrefetchWritesAValueThePlannerParsesIdentically(t *testing.T) {
	for _, in := range []string{"true", "TRUE", " off ", "1", "0", "on", "yes", "no"} {
		wantOn, wantSet, err := agent.ParseCrossLayerGatePrefetch(in)
		if err != nil {
			t.Fatalf("fixture %q is not a valid value: %v", in, err)
		}
		t.Setenv(agent.CrossLayerGatePrefetchEnv, "")
		if err := applyServeCrossLayerGatePrefetch(in); err != nil {
			t.Fatalf("--%s %q refused: %v", serveCrossLayerGatePrefetchFlag, in, err)
		}
		gotOn, gotSet, err := agent.ParseCrossLayerGatePrefetch(os.Getenv(agent.CrossLayerGatePrefetchEnv))
		if err != nil {
			t.Fatalf("--%s %q wrote %q, which the planner then REFUSES: %v", serveCrossLayerGatePrefetchFlag, in, os.Getenv(agent.CrossLayerGatePrefetchEnv), err)
		}
		if gotOn != wantOn || gotSet != wantSet {
			t.Fatalf("--%s %q parsed to (on=%v set=%v) but reaches the planner as (on=%v set=%v)", serveCrossLayerGatePrefetchFlag, in, wantOn, wantSet, gotOn, gotSet)
		}
	}
}

// The whole point of #1401's operator half: the knob is discoverable from `fak serve --help`, not only
// from a source file. A registration that drifts out of the flag set is the regression.
func TestServeCrossLayerGatePrefetchIsRegisteredAndDocumented(t *testing.T) {
	fs, sf := newServeFlagSet()
	f := fs.Lookup(serveCrossLayerGatePrefetchFlag)
	if f == nil {
		t.Fatalf("`fak serve --help` does not list --%s — the knob stays reachable only by env var", serveCrossLayerGatePrefetchFlag)
	}
	if sf.crossLayerGatePrefetch == nil {
		t.Fatal("serveFlags.crossLayerGatePrefetch is not bound to the registered flag")
	}
	if f.DefValue != "false" {
		t.Errorf("--%s defaults to %q; it must default to false so an unset flag is byte-for-byte the previous path", serveCrossLayerGatePrefetchFlag, f.DefValue)
	}
	for _, want := range []string{"per-session", "DEFAULT OFF", "HINTS", "ring", agent.CrossLayerGatePrefetchEnv} {
		if !strings.Contains(f.Usage, want) {
			t.Errorf("--%s usage does not mention %q, so an operator cannot tell what passing it does", serveCrossLayerGatePrefetchFlag, want)
		}
	}
	// The flag must actually parse into the bound field.
	if err := fs.Parse([]string{"--" + serveCrossLayerGatePrefetchFlag, "true"}); err != nil {
		t.Fatalf("parse --%s true: %v", serveCrossLayerGatePrefetchFlag, err)
	}
	if !*sf.crossLayerGatePrefetch {
		t.Fatalf("--%s true bound false", serveCrossLayerGatePrefetchFlag)
	}
	if !sf.isExplicitFlag(serveCrossLayerGatePrefetchFlag) {
		t.Fatalf("--%s was passed but isExplicitFlag reports it unset; an explicit =false would be lost", serveCrossLayerGatePrefetchFlag)
	}
}
