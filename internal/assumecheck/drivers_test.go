package assumecheck

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
)

// scriptedRunner scripts a Runner/CommandRunner from a map keyed by the joined
// argv, so driver tests drive deterministic evidence with no repo and no
// subprocess. A missing key fails the test — a driver may only run the probes
// its contract declares.
func scriptedRunner(t *testing.T, script map[string]struct {
	out  string
	code int
	err  error
}) Runner {
	t.Helper()
	return func(_ context.Context, _ string, args ...string) (string, int, error) {
		key := strings.Join(args, " ")
		stub, ok := script[key]
		if !ok {
			t.Fatalf("driver ran an unscripted probe: git %s", key)
		}
		return stub.out, stub.code, stub.err
	}
}

type gitStub = struct {
	out  string
	code int
	err  error
}

// checkAgainst runs the pure kernel over the driver's evidence for an
// assumption declaring the driver's kind — proving each driver's evidence
// passes Check's kind rule and lands on the expected closed outcome.
func checkAgainst(kind WitnessKind, ev Evidence) Outcome {
	return Check(Assumption{ID: "probe-under-test", Level: LevelSubsystem, WitnessKind: kind}, ev).Outcome
}

func TestGitAncestryDriverTable(t *testing.T) {
	const ref = "abc123"
	const full = "abc123def456abc123def456abc123def456abc1"
	catFile := "cat-file -e " + ref + "^{commit}"
	revParse := "rev-parse --verify " + ref + "^{commit}"
	mergeBase := "merge-base --is-ancestor " + full + " HEAD"
	revScan := "log --format=%H%x00%B%x00 " + full + "..HEAD"

	cases := []struct {
		name          string
		script        map[string]gitStub
		wantWitnessed bool
		wantHolds     bool
		wantOutcome   Outcome
		wantDetail    string
	}{
		{
			name: "reachable and not reverted holds",
			script: map[string]gitStub{
				catFile:   {code: 0},
				revParse:  {out: full + "\n", code: 0},
				mergeBase: {code: 0},
				revScan:   {out: "d00d\x00fix: unrelated\x00", code: 0},
			},
			wantWitnessed: true, wantHolds: true, wantOutcome: OutcomeHolds,
			wantDetail: "not reverted",
		},
		{
			name: "resolves but not reachable is refuted",
			script: map[string]gitStub{
				catFile:   {code: 0},
				revParse:  {out: full + "\n", code: 0},
				mergeBase: {code: 1},
			},
			wantWitnessed: true, wantOutcome: OutcomeViolated,
			wantDetail: "not reachable from HEAD",
		},
		{
			name: "reachable but reverted is refuted",
			script: map[string]gitStub{
				catFile:   {code: 0},
				revParse:  {out: full + "\n", code: 0},
				mergeBase: {code: 0},
				revScan:   {out: "feedface00feedface00feedface00feedface00\x00Revert \"x\"\n\nThis reverts commit " + full + ".\x00", code: 0},
			},
			wantWitnessed: true, wantOutcome: OutcomeViolated,
			wantDetail: "reverted by feedface00fe",
		},
		{
			name:          "nonexistent ref is a witnessed refute",
			script:        map[string]gitStub{catFile: {code: 1}},
			wantWitnessed: true, wantOutcome: OutcomeViolated,
			wantDetail: "does not resolve to a commit",
		},
		{
			name:        "cat-file hard failure cannot witness",
			script:      map[string]gitStub{catFile: {code: 128}},
			wantOutcome: OutcomeUnverifiable,
		},
		{
			name:        "git missing cannot witness",
			script:      map[string]gitStub{catFile: {code: -1, err: errors.New("git not found")}},
			wantOutcome: OutcomeUnverifiable,
		},
		{
			name: "merge-base hard failure cannot witness",
			script: map[string]gitStub{
				catFile:   {code: 0},
				revParse:  {out: full + "\n", code: 0},
				mergeBase: {code: 128},
			},
			wantOutcome: OutcomeUnverifiable,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := NewGitAncestryDriverWithRunner(scriptedRunner(t, tc.script), "")
			ev := d.Gather(context.Background(), Target{Ref: ref})
			if ev.Kind != WitnessGitAncestry {
				t.Fatalf("driver stamped kind %s, want %s", ev.Kind, WitnessGitAncestry)
			}
			if ev.Witnessed != tc.wantWitnessed || ev.Holds != tc.wantHolds {
				t.Fatalf("evidence witnessed=%t holds=%t (detail=%q), want witnessed=%t holds=%t",
					ev.Witnessed, ev.Holds, ev.Detail, tc.wantWitnessed, tc.wantHolds)
			}
			if tc.wantDetail != "" && !strings.Contains(ev.Detail, tc.wantDetail) {
				t.Fatalf("detail %q does not carry %q", ev.Detail, tc.wantDetail)
			}
			if got := checkAgainst(WitnessGitAncestry, ev); got != tc.wantOutcome {
				t.Fatalf("kernel outcome = %s, want %s (evidence %+v)", got, tc.wantOutcome, ev)
			}
		})
	}
}

func TestGitAncestryDriverEmptyRefCannotWitness(t *testing.T) {
	d := NewGitAncestryDriverWithRunner(func(context.Context, string, ...string) (string, int, error) {
		t.Fatal("an empty ref must not reach git")
		return "", 0, nil
	}, "")
	ev := d.Gather(context.Background(), Target{})
	if ev.Witnessed || checkAgainst(WitnessGitAncestry, ev) != OutcomeUnverifiable {
		t.Fatalf("empty ref produced a decision: %+v", ev)
	}
}

func TestWorktreeGrepDriverTable(t *testing.T) {
	const token = "elideStaleReads"
	grepKey := "grep -F -- " + token + " -- ."
	cases := []struct {
		name          string
		stub          gitStub
		wantWitnessed bool
		wantHolds     bool
		wantOutcome   Outcome
	}{
		{"present holds", gitStub{code: 0}, true, true, OutcomeHolds},
		{"absent is refuted", gitStub{code: 1}, true, false, OutcomeViolated},
		{"grep hard failure cannot witness", gitStub{code: 2}, false, false, OutcomeUnverifiable},
		{"git missing cannot witness", gitStub{code: -1, err: errors.New("git not found")}, false, false, OutcomeUnverifiable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := NewWorktreeGrepDriverWithRunner(scriptedRunner(t, map[string]gitStub{grepKey: tc.stub}), "")
			ev := d.Gather(context.Background(), Target{Pattern: token})
			if ev.Kind != WitnessWorktreeGrep {
				t.Fatalf("driver stamped kind %s, want %s", ev.Kind, WitnessWorktreeGrep)
			}
			if ev.Witnessed != tc.wantWitnessed || ev.Holds != tc.wantHolds {
				t.Fatalf("evidence witnessed=%t holds=%t (detail=%q), want witnessed=%t holds=%t",
					ev.Witnessed, ev.Holds, ev.Detail, tc.wantWitnessed, tc.wantHolds)
			}
			if got := checkAgainst(WitnessWorktreeGrep, ev); got != tc.wantOutcome {
				t.Fatalf("kernel outcome = %s, want %s", got, tc.wantOutcome)
			}
		})
	}
	t.Run("empty pattern cannot witness", func(t *testing.T) {
		d := NewWorktreeGrepDriverWithRunner(func(context.Context, string, ...string) (string, int, error) {
			t.Fatal("an empty pattern must not reach git")
			return "", 0, nil
		}, "")
		if ev := d.Gather(context.Background(), Target{}); ev.Witnessed {
			t.Fatalf("empty pattern produced a decision: %+v", ev)
		}
	})
}

func TestCommandProbeDriverArgvTable(t *testing.T) {
	cases := []struct {
		name          string
		code          int
		err           error
		wantWitnessed bool
		wantHolds     bool
		wantOutcome   Outcome
	}{
		{"exit 0 holds", 0, nil, true, true, OutcomeHolds},
		{"exit 1 is refuted", 1, nil, true, false, OutcomeViolated},
		{"other exit cannot witness", 3, nil, false, false, OutcomeUnverifiable},
		{"start failure cannot witness", -1, errors.New("spawn failed"), false, false, OutcomeUnverifiable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var gotArgv []string
			d := NewCommandProbeDriverWithRunner(func(_ context.Context, _ string, argv ...string) (string, int, error) {
				gotArgv = argv
				return "probe output", tc.code, tc.err
			})
			ev := d.Gather(context.Background(), Target{Argv: []string{"some-probe", "--flag"}})
			if ev.Kind != WitnessCommandProbe {
				t.Fatalf("driver stamped kind %s, want %s", ev.Kind, WitnessCommandProbe)
			}
			if len(gotArgv) != 2 || gotArgv[0] != "some-probe" {
				t.Fatalf("driver ran argv %v, want the target's", gotArgv)
			}
			if ev.Witnessed != tc.wantWitnessed || ev.Holds != tc.wantHolds {
				t.Fatalf("evidence witnessed=%t holds=%t (detail=%q), want witnessed=%t holds=%t",
					ev.Witnessed, ev.Holds, ev.Detail, tc.wantWitnessed, tc.wantHolds)
			}
			if got := checkAgainst(WitnessCommandProbe, ev); got != tc.wantOutcome {
				t.Fatalf("kernel outcome = %s, want %s", got, tc.wantOutcome)
			}
		})
	}
}

func TestCommandProbeDriverInProcessProbe(t *testing.T) {
	cases := []struct {
		name        string
		code        int
		err         error
		wantOutcome Outcome
	}{
		{"probe tri-state 0 holds", 0, nil, OutcomeHolds},
		{"probe tri-state 1 is refuted", 1, nil, OutcomeViolated},
		{"probe error cannot witness", -1, errors.New("signal unreadable"), OutcomeUnverifiable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := NewCommandProbeDriverWithRunner(func(context.Context, string, ...string) (string, int, error) {
				t.Fatal("an in-process probe must not spawn")
				return "", 0, nil
			})
			ev := d.Gather(context.Background(), Target{Probe: func(context.Context) (string, int, error) {
				return "in-process signal", tc.code, tc.err
			}})
			if got := checkAgainst(WitnessCommandProbe, ev); got != tc.wantOutcome {
				t.Fatalf("kernel outcome = %s, want %s (evidence %+v)", got, tc.wantOutcome, ev)
			}
			if tc.err == nil && !strings.Contains(ev.Detail, "in-process signal") {
				t.Fatalf("detail %q drops the probe's trace", ev.Detail)
			}
		})
	}
	t.Run("empty target cannot witness", func(t *testing.T) {
		d := NewCommandProbeDriverWithRunner(func(context.Context, string, ...string) (string, int, error) {
			t.Fatal("an empty target must not spawn")
			return "", 0, nil
		})
		if ev := d.Gather(context.Background(), Target{}); ev.Witnessed {
			t.Fatalf("empty target produced a decision: %+v", ev)
		}
	})
}

func TestConfigFlagDriverTable(t *testing.T) {
	cases := []struct {
		name          string
		statErr       error
		wantWitnessed bool
		wantHolds     bool
		wantOutcome   Outcome
	}{
		{"path exists holds", nil, true, true, OutcomeHolds},
		{"path missing is refuted", os.ErrNotExist, true, false, OutcomeViolated},
		{"unreadable path cannot witness", errors.New("permission denied"), false, false, OutcomeUnverifiable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath string
			d := NewConfigFlagDriverWithStat(func(p string) (os.FileInfo, error) {
				gotPath = p
				return nil, tc.statErr
			})
			ev := d.Gather(context.Background(), Target{Path: "/seats/alpha"})
			if ev.Kind != WitnessConfigFlag {
				t.Fatalf("driver stamped kind %s, want %s", ev.Kind, WitnessConfigFlag)
			}
			if gotPath != "/seats/alpha" {
				t.Fatalf("driver statted %q, want the target path", gotPath)
			}
			if ev.Witnessed != tc.wantWitnessed || ev.Holds != tc.wantHolds {
				t.Fatalf("evidence witnessed=%t holds=%t (detail=%q), want witnessed=%t holds=%t",
					ev.Witnessed, ev.Holds, ev.Detail, tc.wantWitnessed, tc.wantHolds)
			}
			if got := checkAgainst(WitnessConfigFlag, ev); got != tc.wantOutcome {
				t.Fatalf("kernel outcome = %s, want %s", got, tc.wantOutcome)
			}
		})
	}
	t.Run("relative path anchors under Target.Dir", func(t *testing.T) {
		var gotPath string
		d := NewConfigFlagDriverWithStat(func(p string) (os.FileInfo, error) {
			gotPath = p
			return nil, nil
		})
		d.Gather(context.Background(), Target{Path: "seat-dir", Dir: "anchor"})
		if !strings.Contains(gotPath, "anchor") || !strings.Contains(gotPath, "seat-dir") {
			t.Fatalf("relative path statted as %q, want it joined under the target dir", gotPath)
		}
	})
	t.Run("empty path cannot witness", func(t *testing.T) {
		d := NewConfigFlagDriverWithStat(func(string) (os.FileInfo, error) {
			t.Fatal("an empty path must not be statted")
			return nil, nil
		})
		if ev := d.Gather(context.Background(), Target{}); ev.Witnessed {
			t.Fatalf("empty path produced a decision: %+v", ev)
		}
	})
}

// TestDriverRegistryResolvesAllFourKinds proves the init()-registered defaults
// cover exactly the four probe kinds this layer ships, each resolving to a
// driver that stamps its own kind, and that resolution fails closed for kinds
// with no driver (ledger-read is a bespoke shell gatherer; session-report can
// never positively confirm and gets no driver).
func TestDriverRegistryResolvesAllFourKinds(t *testing.T) {
	for _, kind := range []WitnessKind{WitnessGitAncestry, WitnessWorktreeGrep, WitnessCommandProbe, WitnessConfigFlag} {
		d, ok := ResolveDriver(kind)
		if !ok {
			t.Fatalf("no driver registered for kind %s", kind)
		}
		if d.Kind() != kind {
			t.Fatalf("driver resolved for %s stamps kind %s", kind, d.Kind())
		}
	}
	for _, kind := range []WitnessKind{WitnessLedgerRead, WitnessSessionReport} {
		if _, ok := ResolveDriver(kind); ok {
			t.Fatalf("kind %s resolved a driver; it must fail closed", kind)
		}
	}
	if got := len(Drivers()); got != 4 {
		t.Fatalf("Drivers() = %d entries, want the four registered kinds", got)
	}
}

// TestDriversStableOrder proves the coverage menu is deterministically ordered
// by kind, so lists and docs render the same on every call.
func TestDriversStableOrder(t *testing.T) {
	first := Drivers()
	for i := 1; i < len(first); i++ {
		if first[i-1].Kind() >= first[i].Kind() {
			t.Fatalf("Drivers() unordered at %d: %s >= %s", i, first[i-1].Kind(), first[i].Kind())
		}
	}
}

// badKindDriver declares a kind outside the closed vocabulary — registration
// must refuse it loudly.
type badKindDriver struct{}

func (badKindDriver) Kind() WitnessKind                       { return WitnessKind("bogus") }
func (badKindDriver) Gather(context.Context, Target) Evidence { return Evidence{} }

func TestRegisterDriverFailsLoud(t *testing.T) {
	mustPanic := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s did not panic", name)
			}
		}()
		fn()
	}
	mustPanic("nil driver", func() { RegisterDriver(nil) })
	mustPanic("kind outside the closed vocabulary", func() { RegisterDriver(badKindDriver{}) })
	mustPanic("duplicate kind", func() { RegisterDriver(NewConfigFlagDriver()) })
}
