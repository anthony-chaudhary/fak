package hooks

import (
	"strings"
	"testing"
)

// gate_githygiene_test.go — the witness for the GIT_HYGIENE_BYPASS advisory gate (#5588).
//
// The issue's done condition is a two-sided claim: the gate WARNS on a synthetic violating commit
// and stays SILENT on a clean one. A one-sided test is the failure mode worth naming — a gate that
// fires on everything passes "it warns" and is worthless, because the fleet learns to ignore it.
// So every warn case below is paired with the near-miss that must stay quiet: a diagnostic that
// merely NAMES index.lock, the owner packages that implement the reclamation, and the two
// suppressions (an attestation, or code already routed through the daily tick).
//
// Fixtures are built with the in-package diffOf helper (hooks_test.go), the same style the other
// gate tests use. Note that the violating fixtures deliberately avoid the strings "gitdaily" and
// "git-daily": those are the gate's own whole-diff routed-token suppression, so spelling one in a
// fixture would silence the gate under test and turn every warn assertion vacuous.

// hygieneLockPath spells `.git/index.lock` without putting the literal in a fixture line that the
// suppression scan could confuse for prose. Kept as a helper so the needle lives in one place.
func hygieneLockPath() string { return ".git/index.lock" }

// TestGateGitHygiene_WarnsOnAdHocLockReclamation is the headline: a commit that adds its own
// removal of a git transaction lock, outside the packages that own that decision, gets exactly one
// advisory finding naming the evidence-gated route.
func TestGateGitHygiene_WarnsOnAdHocLockReclamation(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"cmd/fak/unwedge.go": {
			"package main",
			"func unwedge(root string) error {",
			"\treturn os.Remove(filepath.Join(root, \"" + hygieneLockPath() + "\"))",
			"}",
		},
	})
	f, err := gateGitHygieneBypass(d)
	if err != nil {
		t.Fatalf("the gate must be fail-open (pure string work over the staged diff), got err=%v", err)
	}
	if len(f) != 1 {
		t.Fatalf("expected exactly one GIT_HYGIENE_BYPASS finding, got %d: %+v", len(f), f)
	}
	if f[0].Gate != "GIT_HYGIENE_BYPASS" {
		t.Errorf("gate name = %q, want GIT_HYGIENE_BYPASS", f[0].Gate)
	}
	if f[0].File != "cmd/fak/unwedge.go" {
		t.Errorf("finding File = %q, want the offending staged path", f[0].File)
	}
	if !f[0].Advisory {
		t.Errorf("the finding must be Advisory — a block-mode default is this issue's explicit out-of-scope line")
	}
	// The nudge is only useful if it names where the sanctioned route lives.
	if !hasFindingFor(f, "GIT_HYGIENE_BYPASS", "internal/gitdaily") {
		t.Errorf("advisory should name the evidence-gated route; got %+v", f)
	}
	if !hasFindingFor(f, "GIT_HYGIENE_BYPASS", "index.lock") {
		t.Errorf("advisory should name the lock it saw reclaimed; got %+v", f)
	}
}

// TestGateGitHygiene_WarnsOnObjectMaintenance covers the second family. Unlike the lock case this
// needs no removal verb: running `git gc` from new code IS the bypass, because gitgate's tiers
// exist to decide whether that is safe right now.
func TestGateGitHygiene_WarnsOnObjectMaintenance(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"tools/nightly_cleanup.sh": {
			"#!/usr/bin/env bash",
			"git gc --aggressive",
		},
	})
	f, err := gateGitHygieneBypass(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 1 {
		t.Fatalf("expected one finding for ad-hoc object maintenance, got %d: %+v", len(f), f)
	}
	if !strings.Contains(f[0].Detail, "object-database maintenance") {
		t.Errorf("finding should classify the bypass as object maintenance; got %q", f[0].Detail)
	}
}

// TestGateGitHygiene_SilentOnCleanCommit is the other half of the done condition, and the one that
// keeps the gate worth reading. Three near-misses that must all stay quiet:
//
//	an ordinary commit that touches git at all;
//	a DIAGNOSTIC that names index.lock without removing it (refusal text, error strings);
//	a serializer releasing its OWN non-git .lock file.
func TestGateGitHygiene_SilentOnCleanCommit(t *testing.T) {
	for _, tc := range []struct {
		name  string
		lines []string
	}{
		{"ordinary code", []string{"package main", "func main() { fmt.Println(\"hello\") }"}},
		{"names the lock without removing it", []string{
			"package main",
			"// A held " + hygieneLockPath() + " means a peer is mid-commit; wait for it.",
			"return fmt.Errorf(\"index.lock is held by pid %d\", pid)",
		}},
		{"releases its own non-git lock", []string{
			"package main",
			"defer os.Remove(statePath + \".lock\")",
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := diffOf("/r", map[string][]string{"cmd/fak/whatever.go": tc.lines})
			f, err := gateGitHygieneBypass(d)
			if err != nil {
				t.Fatal(err)
			}
			if len(f) != 0 {
				t.Fatalf("a clean commit must produce no finding, got %+v", f)
			}
		})
	}
}

// TestGateGitHygiene_SilentInsideOwnerPackages pins the scope rule. The packages that IMPLEMENT
// lock reclamation must be able to write the very code the gate flags elsewhere — otherwise the
// gate reds the spine it is defending on every commit that touches it.
func TestGateGitHygiene_SilentInsideOwnerPackages(t *testing.T) {
	violating := []string{"package x", "os.Remove(filepath.Join(root, \"" + hygieneLockPath() + "\"))"}
	for _, owner := range []string{
		"internal/gitdaily/indexlocks.go",
		"internal/gitgate/tiers.go",
		"internal/treedoctor/residue.go",
		"internal/leaseref/reap.go",
		"internal/commitlane/frozen.go",
		"internal/flock/flock.go",
		"internal/hooks/gate_githygiene.go",
	} {
		t.Run(owner, func(t *testing.T) {
			d := diffOf("/r", map[string][]string{owner: violating})
			f, err := gateGitHygieneBypass(d)
			if err != nil {
				t.Fatal(err)
			}
			if len(f) != 0 {
				t.Fatalf("%s owns this decision and must not be nudged, got %+v", owner, f)
			}
		})
	}
}

// TestGateGitHygiene_NonSourceFileStaysQuiet: a doc or fixture that SPELLS `git gc` describes the
// behaviour rather than performing it. Nudging on prose trains the reader to ignore the gate.
func TestGateGitHygiene_NonSourceFileStaysQuiet(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"docs/notes/incident.md": {"we ran `git gc --prune=now` by hand and lost objects"},
	})
	f, err := gateGitHygieneBypass(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 0 {
		t.Fatalf("prose that merely names object maintenance must stay quiet, got %+v", f)
	}
}

// TestGateGitHygiene_Suppressions covers both escape hatches an author has INSIDE the diff (the
// env-var escape is the runner's, tested via the registry below): an explicit attestation, and
// code that already routes through the sanctioned tick.
func TestGateGitHygiene_Suppressions(t *testing.T) {
	violating := "os.Remove(filepath.Join(root, \"" + hygieneLockPath() + "\"))"
	for _, tc := range []struct {
		name, note string
	}{
		{"attestation", "// git-hygiene: this clone is exclusive to the test harness; no peer can hold it."},
		{"already routed", "// prefer `fak git-daily` when the clone is shared"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := diffOf("/r", map[string][]string{
				"cmd/fak/unwedge.go": {"package main", tc.note, violating},
			})
			f, err := gateGitHygieneBypass(d)
			if err != nil {
				t.Fatal(err)
			}
			if len(f) != 0 {
				t.Fatalf("%s must silence the advisory, got %+v", tc.name, f)
			}
		})
	}
}

// TestGateGitHygiene_OneFindingPerFileSortedByPath pins the output shape: a file that bypasses on
// three lines is ONE nudge, and multiple files come out in a deterministic order so the hook's
// output does not churn between runs.
func TestGateGitHygiene_OneFindingPerFileSortedByPath(t *testing.T) {
	line := "os.Remove(filepath.Join(root, \"" + hygieneLockPath() + "\"))"
	d := diffOf("/r", map[string][]string{
		"cmd/fak/zeta.go":  {"package main", line, line},
		"cmd/fak/alpha.go": {"package main", line},
	})
	f, err := gateGitHygieneBypass(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 2 {
		t.Fatalf("expected one finding per offending file, got %d: %+v", len(f), f)
	}
	if f[0].File != "cmd/fak/alpha.go" || f[1].File != "cmd/fak/zeta.go" {
		t.Errorf("findings must be sorted by path, got %q then %q", f[0].File, f[1].File)
	}
}

// TestGateGitHygiene_FailsOpenOnEmptyDiff is the issue's first confusion risk stated as a test: a
// broken gate must not wedge the trunk. This one cannot break — it reads no evidence outside the
// diff already in memory — so the assertion is that it returns cleanly (and still records its
// denominator) even when handed a commit that staged nothing it cares about.
func TestGateGitHygiene_FailsOpenOnEmptyDiff(t *testing.T) {
	d := emptyStagedDiff(t.TempDir())
	f, err := gateGitHygieneBypass(d)
	if err != nil {
		t.Fatalf("the gate must never error, got %v", err)
	}
	if len(f) != 0 {
		t.Fatalf("an empty diff must produce no finding, got %+v", f)
	}
	if _, _, ok := d.Candidates("GIT_HYGIENE_BYPASS"); !ok {
		t.Errorf("the gate must record a candidate denominator even at zero (#5602)")
	}
}

// TestGateGitHygiene_RegisteredAdvisoryInPreCommitGates is the acceptance gate the issue names, and
// the PARITY half of this file: the contract gate_githygiene.go documents in prose (warn by
// default, FLEET_GIT_HYGIENE_GUARD to block, ALLOW_GIT_HYGIENE_BYPASS to skip once) must be the
// contract the runner actually reads off the registry. Prose and wiring drift silently; this does
// not.
func TestGateGitHygiene_RegisteredAdvisoryInPreCommitGates(t *testing.T) {
	var got *Gate
	for i, g := range PreCommitGates() {
		if g.Name == "GIT_HYGIENE_BYPASS" {
			got = &PreCommitGates()[i]
			break
		}
	}
	if got == nil {
		t.Fatalf("GIT_HYGIENE_BYPASS is not registered in PreCommitGates()")
	}
	if got.DefaultMode != "warn" {
		t.Errorf("DefaultMode = %q, want \"warn\" — advisory-first is the issue's out-of-scope line", got.DefaultMode)
	}
	if got.ModeEnv != "FLEET_GIT_HYGIENE_GUARD" {
		t.Errorf("ModeEnv = %q, want FLEET_GIT_HYGIENE_GUARD (the token the gate's doc promises)", got.ModeEnv)
	}
	if got.EscapeEnv != "ALLOW_GIT_HYGIENE_BYPASS" {
		t.Errorf("EscapeEnv = %q, want ALLOW_GIT_HYGIENE_BYPASS (the token the gate's doc promises)", got.EscapeEnv)
	}
	if got.Check == nil {
		t.Fatalf("the registered gate has no Check")
	}
	// The registered Check is the scopeGates wrapper, not the bare function, so exercise it the
	// way the runner does: a violating diff must still warn THROUGH the wrapper.
	d := diffOf(t.TempDir(), map[string][]string{
		"cmd/fak/unwedge.go": {"package main", "os.Remove(filepath.Join(root, \"" + hygieneLockPath() + "\"))"},
	})
	f, err := got.Check(d)
	if err != nil {
		t.Fatalf("registered gate errored: %v", err)
	}
	if len(f) != 1 || f[0].Gate != "GIT_HYGIENE_BYPASS" {
		t.Fatalf("the registered gate must warn on a violating commit, got %+v", f)
	}
}
