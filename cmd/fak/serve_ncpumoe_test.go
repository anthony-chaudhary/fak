package main

import (
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// The flag is the STRICT door: a grade that is not auto/off/a count refuses the launch, and it does
// so BEFORE anything is mutated. A refusal that had already written the env var would leave the
// process carrying a placement the operator never successfully asked for.
func TestServeNCPUMoERefusesABadGradeWithoutMutatingTheEnv(t *testing.T) {
	for _, bad := range []string{"atuo", "-1", "abc", "auto3", "1.5"} {
		t.Run(bad, func(t *testing.T) {
			t.Setenv(agent.ExpertSpillEnv, "sentinel")
			err := applyServeNCPUMoE(bad)
			if err == nil {
				t.Fatalf("--%s %q was admitted; a misspelled grade must not fall back to a placement the operator did not choose", serveNCPUMoEFlag, bad)
			}
			if !strings.Contains(err.Error(), serveNCPUMoEFlag) {
				t.Errorf("refusal %q does not name the flag the operator typed", err)
			}
			if got := os.Getenv(agent.ExpertSpillEnv); got != "sentinel" {
				t.Errorf("a REFUSED grade mutated %s to %q — validation must precede the write", agent.ExpertSpillEnv, got)
			}
		})
	}
}

// An un-passed flag must be byte-for-byte the previous path, INCLUDING on a host whose profile
// already exports the env var. Clobbering an ambient export with "the flag's default" would silently
// disable a spill the operator configured elsewhere.
func TestServeNCPUMoEUnpassedLeavesTheAmbientEnvAlone(t *testing.T) {
	for _, unpassed := range []string{"", "   "} {
		t.Setenv(agent.ExpertSpillEnv, "auto")
		if err := applyServeNCPUMoE(unpassed); err != nil {
			t.Fatalf("an un-passed flag (%q) refused: %v", unpassed, err)
		}
		if got := os.Getenv(agent.ExpertSpillEnv); got != "auto" {
			t.Fatalf("un-passed flag (%q) changed %s to %q, want the ambient \"auto\" untouched", unpassed, agent.ExpertSpillEnv, got)
		}
	}
}

// A passed flag wins over the ambient environment. The load-bearing case is the explicit "off":
// the operator at the terminal is more specific than the host profile, so typing off on a box that
// exports FAK_N_CPU_MOE=auto must actually serve the ungraded placement.
func TestServeNCPUMoEPassedGradeWinsOverTheAmbientEnv(t *testing.T) {
	for _, tc := range []struct {
		name    string
		ambient string
		flag    string
		want    string
	}{
		{"off overrides an ambient auto", "auto", "off", "off"},
		{"auto overrides an ambient off", "off", "auto", "auto"},
		{"a count overrides an ambient auto", "auto", "12", "12"},
		{"case and padding are normalized", "", "  AUTO ", "auto"},
		{"zero is a real grade, not an absent one", "auto", "0", "0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(agent.ExpertSpillEnv, tc.ambient)
			if err := applyServeNCPUMoE(tc.flag); err != nil {
				t.Fatalf("--%s %q refused: %v", serveNCPUMoEFlag, tc.flag, err)
			}
			if got := os.Getenv(agent.ExpertSpillEnv); got != tc.want {
				t.Fatalf("--%s %q over ambient %q left %s=%q, want %q", serveNCPUMoEFlag, tc.flag, tc.ambient, agent.ExpertSpillEnv, got, tc.want)
			}
		})
	}
}

// Whatever the flag writes must parse back to the SAME grade the operator's spelling parsed to,
// or the flag and the planner disagree about what was asked for.
func TestServeNCPUMoEWritesAGradeThePlannerParsesIdentically(t *testing.T) {
	for _, in := range []string{"auto", "AUTO", " off ", "0", "7", "none"} {
		wantN, wantSet, err := agent.ParseExpertSpillGrade(in)
		if err != nil {
			t.Fatalf("fixture %q is not a valid grade: %v", in, err)
		}
		t.Setenv(agent.ExpertSpillEnv, "")
		if err := applyServeNCPUMoE(in); err != nil {
			t.Fatalf("--%s %q refused: %v", serveNCPUMoEFlag, in, err)
		}
		gotN, gotSet, err := agent.ParseExpertSpillGrade(os.Getenv(agent.ExpertSpillEnv))
		if err != nil {
			t.Fatalf("--%s %q wrote %q, which the planner then REFUSES: %v", serveNCPUMoEFlag, in, os.Getenv(agent.ExpertSpillEnv), err)
		}
		if gotN != wantN || gotSet != wantSet {
			t.Fatalf("--%s %q parsed to (n=%d set=%v) but reaches the planner as (n=%d set=%v)", serveNCPUMoEFlag, in, wantN, wantSet, gotN, gotSet)
		}
	}
}

// The whole point of #5628's operator half: the knob is discoverable from `fak serve --help`, not
// only from a source file. A registration that drifts out of the flag set is the regression.
func TestServeNCPUMoEIsRegisteredAndDocumented(t *testing.T) {
	fs, sf := newServeFlagSet()
	f := fs.Lookup(serveNCPUMoEFlag)
	if f == nil {
		t.Fatalf("`fak serve --help` does not list --%s — the grade stays reachable only by env var", serveNCPUMoEFlag)
	}
	if sf.nCPUMoE == nil {
		t.Fatal("serveFlags.nCPUMoE is not bound to the registered flag")
	}
	if f.DefValue != "" {
		t.Errorf("--%s defaults to %q; it must default to empty so an un-passed flag cannot clobber the ambient %s", serveNCPUMoEFlag, f.DefValue, agent.ExpertSpillEnv)
	}
	for _, want := range []string{"auto", "off", agent.ExpertSpillEnv} {
		if !strings.Contains(f.Usage, want) {
			t.Errorf("--%s usage does not mention %q, so an operator cannot tell what to pass", serveNCPUMoEFlag, want)
		}
	}
	// The flag must actually parse into the bound field.
	if err := fs.Parse([]string{"--" + serveNCPUMoEFlag, "auto"}); err != nil {
		t.Fatalf("parse --%s auto: %v", serveNCPUMoEFlag, err)
	}
	if *sf.nCPUMoE != "auto" {
		t.Fatalf("--%s auto bound %q", serveNCPUMoEFlag, *sf.nCPUMoE)
	}
}
