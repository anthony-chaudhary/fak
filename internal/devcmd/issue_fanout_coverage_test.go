package devcmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuefanout"
)

// Coverage for the witness-gathering shell behind `fak-dev issue fanout
// --coverage` (#5313). The pure fold lives in internal/issuefanout and is
// already tested; what was untested is the part that decides WHAT to ask git and
// gh, which is exactly where a silent regression turns the honesty meter into a
// confident wrong number. So these tests pin the argv of every witness read, not
// just the report that comes back.
//
// The shell's fanoutCoverageDeps seam injects both effectful runners, so nothing
// here shells out to a real git or gh.

// fanoutCovWindow is the --since selector every test threads through. It carries
// a space on purpose: a git revision selector is one argv element, and a shell
// that split it would still "work" against a fake that only prefix-matched.
const fanoutCovWindow = "45 days ago"

// fanoutCovGit is a fake git runner. It records every argv it is handed and
// answers each of the three witness reads from a canned table, so a test can
// assert both what was asked and what the fold made of the answer.
type fanoutCovGit struct {
	calls   [][]string
	added   string // stdout for the --since read (paths added inside the window)
	before  string // stdout for the --until read (paths added before the window)
	tracked string // stdout for ls-files
	failOn  string // "since" | "until" | "ls-files": which surface reports failure
}

// fanoutCovSurface names which of the three witness reads an argv is, or "" when
// the shell asked something no witness accounts for.
func fanoutCovSurface(args []string) string {
	if len(args) > 0 && args[0] == "ls-files" {
		return "ls-files"
	}
	for _, a := range args {
		switch {
		case strings.HasPrefix(a, "--since="):
			return "since"
		case strings.HasPrefix(a, "--until="):
			return "until"
		}
	}
	return ""
}

func (g *fanoutCovGit) run(args []string) (string, string, bool) {
	g.calls = append(g.calls, append([]string(nil), args...))
	surface := fanoutCovSurface(args)
	if surface == "" {
		return "", "unexpected git argv: " + strings.Join(args, " "), false
	}
	if g.failOn == surface {
		return "", "fatal: " + surface + " witness is wedged", false
	}
	switch surface {
	case "since":
		return g.added, "", true
	case "until":
		return g.before, "", true
	default:
		return g.tracked, "", true
	}
}

// fanoutCovGH is a fake gh runner returning a tracker export whose bodies carry
// the given marker keys, and recording the argv so the scan bound and repo can
// be pinned too.
func fanoutCovGH(calls *[][]string, markers ...string) issueCreateRunner {
	return func(args []string) (string, string, bool) {
		if calls != nil {
			*calls = append(*calls, append([]string(nil), args...))
		}
		return fmt.Sprintf(`[{"number": 7, "body": %q}]`, strings.Join(markers, " ")), "", true
	}
}

// fanoutCovTracked is the tracked file list every test scores against: alpha
// carries both spine artifacts (a leaf test and a runnable verb), beta carries
// only a test, gamma is a leaf that predates the window.
const fanoutCovTracked = "internal/alpha/alpha.go\n" +
	"internal/alpha/alpha_test.go\n" +
	"cmd/fak/alpha.go\n" +
	"internal/beta/beta.go\n" +
	"internal/beta/beta_test.go\n" +
	"internal/gamma/gamma.go\n"

// fanoutCovOKGit is a git fake whose window yields exactly one new leaf (alpha)
// with a full spine, so the report can come back OK.
func fanoutCovOKGit() *fanoutCovGit {
	return &fanoutCovGit{
		added:   "internal/alpha/alpha.go\ninternal/alpha/alpha_test.go\ninternal/gamma/gamma.go\n",
		before:  "internal/gamma/gamma.go\n",
		tracked: fanoutCovTracked,
	}
}

func TestFanoutCoverageGathersThreeGitWitnesses(t *testing.T) {
	git := &fanoutCovGit{
		// alpha and beta are added inside the window; gamma is added inside it
		// too but already existed before it, so it is NOT a new leaf.
		added:   "internal/alpha/alpha.go\ninternal/beta/beta.go\ninternal/gamma/more.go\n\n",
		before:  "internal/gamma/gamma.go\n",
		tracked: fanoutCovTracked,
	}
	var ghCalls [][]string
	gh := fanoutCovGH(&ghCalls, "fanout-alpha-a", "fanout-alpha-b", "fanout-alpha-c", "fanout-delta-x")

	rep, err := gatherFanoutCoverage(fanoutCovWindow, "o/r", 500, fanoutCoverageDeps{git: git.run, gh: gh})
	if err != nil {
		t.Fatalf("gatherFanoutCoverage: %v", err)
	}

	// The three witness invocations, in order, argv-exact. --since threads into
	// the first read and the SAME selector bounds the second, which is what makes
	// the two reads partition history instead of overlapping.
	want := [][]string{
		{"log", "--since=" + fanoutCovWindow, "--diff-filter=A", "--name-only", "--pretty=format:"},
		{"log", "--until=" + fanoutCovWindow, "--diff-filter=A", "--name-only", "--pretty=format:"},
		{"ls-files"},
	}
	if len(git.calls) != len(want) {
		t.Fatalf("git invocations = %d, want %d: %v", len(git.calls), len(want), git.calls)
	}
	for i, w := range want {
		if strings.Join(git.calls[i], "\x00") != strings.Join(w, "\x00") {
			t.Errorf("git call %d = %q, want %q", i, git.calls[i], w)
		}
	}

	// The --until read is subtracted, so gamma never reaches the denominator.
	if rep.NewLeaves != 2 {
		t.Fatalf("new leaves = %d, want 2 (alpha, beta; gamma predates the window): %+v", rep.NewLeaves, rep.Leaves)
	}
	got := map[string]issuefanout.LeafCoverage{}
	for _, l := range rep.Leaves {
		got[l.Leaf] = l
	}
	if _, ok := got["gamma"]; ok {
		t.Errorf("gamma counted as new — the --until witness was not subtracted: %+v", rep.Leaves)
	}
	if a := got["alpha"]; !a.HasTest || !a.HasVerb || !a.HasSpine || a.FanoutFiled != 3 || !a.ClearsFloor {
		t.Errorf("alpha row = %+v, want a full spine clearing the floor with 3 markers", a)
	}
	if b := got["beta"]; !b.HasTest || b.HasVerb || b.HasSpine {
		t.Errorf("beta row = %+v, want test-only (no verb, so no spine)", b)
	}
	if len(rep.SpineGaps) != 1 || rep.SpineGaps[0] != "beta" {
		t.Errorf("spine gaps = %v, want [beta]", rep.SpineGaps)
	}
	if rep.Spines != 1 || rep.FanoutCleared != 1 {
		t.Errorf("fan-out rate = %d/%d, want 1/1 (the fan-out denominator is the SPINE set)", rep.FanoutCleared, rep.Spines)
	}
	if len(rep.OrphanMarkers) != 1 || rep.OrphanMarkers[0] != "fanout-delta-x" {
		t.Errorf("orphan markers = %v, want [fanout-delta-x]", rep.OrphanMarkers)
	}

	// The tracker export runs under the coverage scan cap (not --live's dedupe
	// cap) and against the requested repo, and the scan size is reported so a
	// truncated export cannot pass as a measured rate.
	if len(ghCalls) != 1 {
		t.Fatalf("gh invocations = %d, want 1: %v", len(ghCalls), ghCalls)
	}
	wantGH := strings.Join(issuefanout.ListExistingArgs("o/r", 500), "\x00")
	if strings.Join(ghCalls[0], "\x00") != wantGH {
		t.Errorf("gh call = %q, want %q", ghCalls[0], issuefanout.ListExistingArgs("o/r", 500))
	}
	if rep.ScanCap != 500 || rep.ScannedIssues != 1 || rep.ScanTruncated {
		t.Errorf("scan provenance = %d/%d truncated=%v, want 1/500 truncated=false",
			rep.ScannedIssues, rep.ScanCap, rep.ScanTruncated)
	}
}

func TestFanoutCoverageScanCapDefaultsWhenUnset(t *testing.T) {
	git := fanoutCovOKGit()
	var ghCalls [][]string
	if _, err := gatherFanoutCoverage(fanoutCovWindow, "", 0, fanoutCoverageDeps{git: git.run, gh: fanoutCovGH(&ghCalls)}); err != nil {
		t.Fatalf("gatherFanoutCoverage: %v", err)
	}
	want := strings.Join(issuefanout.ListExistingArgs("", issuefanout.DefaultCoverageScanCap), "\x00")
	if len(ghCalls) != 1 || strings.Join(ghCalls[0], "\x00") != want {
		t.Fatalf("gh call = %v, want the DefaultCoverageScanCap bound %v",
			ghCalls, issuefanout.ListExistingArgs("", issuefanout.DefaultCoverageScanCap))
	}
}

func TestFanoutCoverageGitFailureNamesTheCommand(t *testing.T) {
	for _, tc := range []struct {
		surface string
		want    string
	}{
		{"since", "git log --since=" + fanoutCovWindow + " --diff-filter=A --name-only --pretty=format:"},
		{"until", "git log --until=" + fanoutCovWindow + " --diff-filter=A --name-only --pretty=format:"},
		{"ls-files", "git ls-files"},
	} {
		t.Run(tc.surface, func(t *testing.T) {
			git := fanoutCovOKGit()
			git.failOn = tc.surface
			var ghCalls [][]string

			_, err := gatherFanoutCoverage(fanoutCovWindow, "o/r", 500, fanoutCoverageDeps{git: git.run, gh: fanoutCovGH(&ghCalls)})
			if err == nil {
				t.Fatalf("a failing %s witness must be an error, got nil", tc.surface)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to name the command %q", err, tc.want)
			}
			if !strings.Contains(err.Error(), "wedged") {
				t.Errorf("error = %q, want it to carry git's stderr", err)
			}
			// A failed witness must stop the scan: reporting a rate computed
			// from a witness that never answered is the false-green this meter
			// exists to prevent.
			if len(ghCalls) != 0 {
				t.Errorf("gh ran %d time(s) after a failed git witness, want 0", len(ghCalls))
			}

			// The verb surfaces the same failure as a non-zero exit.
			var out, errOut bytes.Buffer
			code := emitFanoutCoverage(&out, &errOut, fanoutCovWindow, "o/r", 500, false, fanoutCoverageDeps{git: git.run, gh: fanoutCovGH(nil)})
			if code == 0 {
				t.Errorf("exit = 0 on a failing %s witness, want non-zero", tc.surface)
			}
			if !strings.Contains(errOut.String(), tc.want) {
				t.Errorf("stderr = %q, want it to name the command %q", errOut.String(), tc.want)
			}
			if out.Len() != 0 {
				t.Errorf("stdout = %q, want nothing rendered when a witness failed", out.String())
			}
		})
	}
}

func TestFanoutCoverageEmitExitsOneWhenARateIsShort(t *testing.T) {
	git := &fanoutCovGit{
		added:   "internal/alpha/alpha.go\ninternal/beta/beta.go\n",
		before:  "",
		tracked: fanoutCovTracked,
	}
	var out, errOut bytes.Buffer
	// alpha clears the floor; beta has no verb, so it is a spine gap.
	code := emitFanoutCoverage(&out, &errOut, fanoutCovWindow, "o/r", 500, false,
		fanoutCoverageDeps{git: git.run, gh: fanoutCovGH(nil, "fanout-alpha-a", "fanout-alpha-b", "fanout-alpha-c")})
	if code != 1 {
		t.Fatalf("exit = %d, want 1 when a rate is short\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "spine gaps") || !strings.Contains(out.String(), "beta") {
		t.Errorf("rendered scorecard must name the offending leaf:\n%s", out.String())
	}
}

func TestFanoutCoverageEmitExitsZeroWhenOK(t *testing.T) {
	git := fanoutCovOKGit()
	var out, errOut bytes.Buffer
	code := emitFanoutCoverage(&out, &errOut, fanoutCovWindow, "o/r", 500, false,
		fanoutCoverageDeps{git: git.run, gh: fanoutCovGH(nil, "fanout-alpha-a", "fanout-alpha-b", "fanout-alpha-c")})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 when both rates are clean\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
	}
	if strings.Contains(out.String(), "spine gaps") || strings.Contains(out.String(), "fan-out gaps") {
		t.Errorf("clean scorecard must report no gaps:\n%s", out.String())
	}
}

func TestFanoutCoverageEmitJSONCarriesTheSchema(t *testing.T) {
	git := fanoutCovOKGit()
	var out, errOut bytes.Buffer
	code := emitFanoutCoverage(&out, &errOut, fanoutCovWindow, "o/r", 500, true,
		fanoutCoverageDeps{git: git.run, gh: fanoutCovGH(nil, "fanout-alpha-a", "fanout-alpha-b", "fanout-alpha-c")})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "fak.issue-fanout-coverage.v1") {
		t.Fatalf("--json output must carry the schema id:\n%s", out.String())
	}
	var rep issuefanout.CoverageReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("decode --json: %v\n%s", err, out.String())
	}
	if rep.Schema != issuefanout.CoverageSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, issuefanout.CoverageSchema)
	}
	if !rep.OK || rep.NewLeaves != 1 || rep.SpineWitnessed != 1 || rep.FanoutCleared != 1 {
		t.Errorf("report = %+v, want OK with 1/1 spine and 1/1 fan-out", rep)
	}
	if rep.MinFanout != issuefanout.MinFanout || rep.ScanCap != 500 {
		t.Errorf("report floor/cap = %d/%d, want %d/500", rep.MinFanout, rep.ScanCap, issuefanout.MinFanout)
	}
}
