package main

// loop_drive_dosroot_test.go — the loop-gate witness must name the workspace it
// asks the kernel about.
//
// Part of #5933 (report the resolved lease root). The corelocks side of that
// ticket proves the general rule; this file pins the ONE call site in cmd/fak
// that broke it.
//
// ⛔ THE DEFECT. runDOSLoopGateWitness shelled out with
//
//	cmd := exec.CommandContext(ctx, "dos", req.Argv()...)
//
// and set neither cmd.Dir nor an explicit --workspace, so the child inherited
// whatever working directory the fak process happened to have. The kernel then
// resolves its lease journal by an UPWARD POSITIONAL WALK for a `.dos`
// directory — so the journal it reads is decided by the caller's position, not
// by the repository under test. Every other kernel shell-out in cmd/fak
// (dispatch_tick_lease_beat.go, dispatch_tick_witness.go, steer_prs.go,
// sync.go, knownbad.go, …) already pins one or the other; this was the last one
// that did not.
//
// ⛔ WHY IT MATTERS HERE SPECIFICALLY. This function's return value is a
// loop-gate WITNESS: it decides whether a loop iteration may proceed. A witness
// read against the wrong workspace is not a missing answer, it is a FABRICATED
// one — the gate reports a verdict this repository never produced, and the loop
// carries on believing it was checked. That is the fail-open direction, which is
// the whole subject of #5933.
//
// The assertion is source-level on purpose. The alternative — running the real
// `dos` binary from a decoy directory — would need the kernel installed, a
// second workspace, and a lease fixture, and it would prove the same one-line
// property far less directly. What must never regress is the PIN, and the pin is
// visible in the source.

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// dosShellOut matches a direct shell-out to the trust kernel by its literal
// program name.
//
// ⛔ The trailing delimiter is load-bearing and a `\b` here would be a BUG, not a
// stricter rule: `\b` asserts a word/non-word boundary, the literal ends in a
// quote, and every real call site continues with `,` or `)`. Quote and comma are
// both non-word characters, so there is no boundary between them and `"dos"\b`
// would match nothing at all — leaving this test green over every call site it
// exists to police. Requiring the delimiter explicitly keeps `"dossier"` out
// while keeping `"dos"` in.
var dosShellOut = regexp.MustCompile(`exec\.Command(?:Context)?\((?:[A-Za-z_][A-Za-z0-9_.]*,\s*)?"dos"\s*[,)]`)

// TestDosShellOutPatternMatchesTheRealCallShapes is the known-positive table
// that keeps the scanner below from silently becoming vacuous. A source scanner
// that matches nothing passes forever and reports itself satisfied, which is
// strictly worse than not having one.
func TestDosShellOutPatternMatchesTheRealCallShapes(t *testing.T) {
	for _, tc := range []struct {
		name string
		line string
		want bool
	}{
		{"context form, variadic argv", `	cmd := exec.CommandContext(ctx, "dos", req.Argv()...)`, true},
		{"context form, literal args", `	cmd := exec.CommandContext(ctx, "dos", "lease-lane", "live")`, true},
		{"plain form", `	cmd := exec.Command("dos", "commit-audit", "--json", ref)`, true},
		{"no arguments at all", `	cmd := exec.Command("dos")`, true},
		{"a longer program name is not the kernel", `	cmd := exec.Command("dossier", "x")`, false},
		{"a substring in prose is not a call", `	// we shell out to "dos" here`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := dosShellOut.MatchString(tc.line); got != tc.want {
				t.Errorf("dosShellOut.MatchString(%q) = %v, want %v — the scanner in this "+
					"file only protects call sites it can actually see", tc.line, got, tc.want)
			}
		})
	}
}

// TestLoopDriveKernelShellOutsPinTheWorkspace is the regression proper.
//
// It scans loop_drive.go for kernel shell-outs and requires each to pin the
// workspace within a short window after the call — either `cmd.Dir = …` or an
// explicit `--workspace` in the argv. The window is deliberately small so the
// failure message can name a file, a line, and a one-line fix.
func TestLoopDriveKernelShellOutsPinTheWorkspace(t *testing.T) {
	const rel = "loop_drive.go"
	// Keep the window bounded, but wide enough for the production call site's
	// security comment between exec.CommandContext and cmd.Dir.
	const pinWindow = 12
	body, err := os.ReadFile(rel)
	if err != nil {
		t.Fatalf("reading %s: %v", rel, err)
	}
	lines := strings.Split(string(body), "\n")

	var offenders []string
	for i, line := range lines {
		if !dosShellOut.MatchString(line) {
			continue
		}
		if strings.Contains(line, "--workspace") {
			continue
		}
		pinned := false
		for j := i; j < len(lines) && j < i+pinWindow; j++ {
			if strings.Contains(lines[j], ".Dir =") || strings.Contains(lines[j], "--workspace") {
				pinned = true
				break
			}
		}
		if !pinned {
			offenders = append(offenders, filepath.ToSlash(rel)+":"+loopDriveItoa(i+1)+": "+strings.TrimSpace(line))
		}
	}
	if len(offenders) > 0 {
		t.Errorf("these kernel shell-outs pin neither cmd.Dir nor --workspace, so the lease "+
			"journal the kernel reads is decided by whatever cwd this process happens to "+
			"have — and a loop-gate witness read against the wrong workspace is a verdict "+
			"this repository never produced (#5933):\n  %s\nFix: set `cmd.Dir = repoRoot()` "+
			"on the line after the exec.Command call.", strings.Join(offenders, "\n  "))
	}
}

// TestRunDOSLoopGateWitnessPinsTheRepoRoot pins the specific remedy, not just
// the absence of the defect: a future edit that swapped the pin for a relative
// path or a cwd read would keep the scanner above green while reintroducing the
// positional walk.
func TestRunDOSLoopGateWitnessPinsTheRepoRoot(t *testing.T) {
	body, err := os.ReadFile("loop_drive.go")
	if err != nil {
		t.Fatalf("reading loop_drive.go: %v", err)
	}
	const fn = "func runDOSLoopGateWitness("
	idx := strings.Index(string(body), fn)
	if idx < 0 {
		t.Fatalf("%s is gone — this test's subject no longer exists and its claim is stale", fn)
	}
	rest := string(body)[idx:]
	if end := strings.Index(rest, "\nfunc "); end > 0 {
		rest = rest[:end]
	}
	if !strings.Contains(rest, "cmd.Dir = repoRoot()") {
		t.Errorf("runDOSLoopGateWitness does not set `cmd.Dir = repoRoot()`. Its verdict gates "+
			"whether a loop iteration may proceed, so the workspace it asks about must be "+
			"THIS repository and not wherever the process was started (#5933). Body:\n%s", rest)
	}
}

// loopDriveItoa keeps the offender message free of a strconv import for one call
// without colliding with the package's other test-local integer helpers.
func loopDriveItoa(n int) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}
