package main

// guard_tempdir_convention_test.go — the durable half of #5527.
//
// The guard temp-dir family is reapable only when creator and reaper agree on
// TWO rungs, and guard_tempreap.go's own header calls that coupling out:
//
//	rung 1  the dir's hook token is in the CLOSED guardTempDirHooks set, and
//	rung 2  the dir's name carries a <pid> segment, so guardTempDirOwner can
//	        test the owning guard for liveness before removing anything.
//
// Both rungs were silently broken, in OPPOSITE directions, which is why neither
// read as a typo:
//
//	leak 1  startGuardLifecycleServer allocated through guardSessionTempDir, so
//	        its dirs were the right SHAPE, but "lifecycle" was missing from
//	        guardTempDirHooks — rung 1 failed, and no lifecycle dir was ever
//	        claimed.
//	leak 2  the task-handoff dir in cmdGuard used a raw os.MkdirTemp, so its
//	        token "handoff" WAS in the set but the name had no <pid> — rung 2
//	        failed, and no handoff dir was ever claimed either.
//
// A test that hardcoded today's call sites would have caught neither, and would
// miss tomorrow's. So this file DISCOVERS the call sites by parsing the package
// source and asserts the two rungs as properties over whatever it finds:
//
//	property A  a raw os.MkdirTemp with a "fak-guard-<token>-" prefix must NOT
//	            name a token the reaper claims. Such a dir looks reapable to a
//	            reader and is invisible to the reaper forever (leak 2). A token
//	            deliberately OUTSIDE the set is the sanctioned opt-out — that is
//	            how the fak-guard-reset-* forensic seeds stay unswept.
//	property B  every guardSessionTempDir("<token>") call site must name a token
//	            the reaper claims, or its correctly-shaped dirs are ignored
//	            forever (leak 1).
//
// LOCAL-WORKTREE vs CI-HEAD SEAM. A source-scanning test reads the WORKING TREE
// when run locally and HEAD when run in CI, and cmd/fak is a shared tree that
// routinely carries several lanes' uncommitted work. That divergence is handled
// three ways rather than wished away:
//   - the assertions are PROPERTIES of correctly-written code, not a snapshot of
//     today's file list, so a peer's new call site passes in both environments as
//     long as it is written correctly, and fails in both when it is not;
//   - every failure message names the offending FILE and LINE plus the exact
//     cure, so whoever hits it can act without knowing this ticket; and
//   - the one waiver below is keyed by file AND token and is self-policing, so a
//     lane that fixes the waived site is told to delete the waiver in the same
//     commit — the divergence announces itself instead of rotting.
//
// The scan covers non-test .go files only: test files deliberately construct
// malformed names as fixtures (guard_tempreap_test.go builds pid-less and
// unknown-hook names on purpose), and flagging those would be a false red.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// guardTempDirRawWaiver lists raw os.MkdirTemp call sites that name a REAPED
// token and are known-open rather than sanctioned. It exists so this test can
// land as a ratchet against NEW breakage without silently absorbing the one site
// #5527's scope fence put out of reach.
//
// installGuardStopHook allocates its settings dir with a raw
// os.MkdirTemp("", "fak-guard-stophook-*") while "stophook" is in
// guardTempDirHooks — the same defect as leak 2, at a third call site, and it
// needs the same one-line cure (route through guardSessionTempDir("stophook")).
// It is rare in practice, not merely theoretical: the raw call runs only when no
// earlier installer already wrote a settings file to merge into, so it fires far
// less often than the handoff path did.
//
// Keyed by file base name -> token. Removing a waiver is the LAST step of fixing
// its site; the stale-waiver check below fails if a waived site is gone, so the
// list cannot rot into a permanent blind spot.
var guardTempDirRawWaiver = map[string]string{
	"guard_stophook.go": "stophook",
}

// guardTempDirSite is one discovered temp-dir creation call site.
type guardTempDirSite struct {
	file    string // base name, e.g. "guard.go"
	line    int
	token   string // hook token parsed from the literal, "" when not a literal
	literal bool   // false when the argument was not a plain string literal
}

func (s guardTempDirSite) where() string {
	return s.file + ":" + strconv.Itoa(s.line)
}

// guardTempDirScanPackage parses every non-test .go file in the package
// directory and returns the raw os.MkdirTemp sites carrying a guardTempDirPrefix
// name literal, plus every guardSessionTempDir call site.
//
// The creation seam in guard_tempreap.go builds its prefix with fmt.Sprintf, so
// it is not a string literal and never appears as a raw site — the scanner sees
// only call sites that spell a fak-guard-* name out by hand, which is exactly
// the population property A governs.
func guardTempDirScanPackage(t *testing.T, dir string) (raw, seam []guardTempDirSite) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read package dir %q: %v", dir, err)
	}
	fset := token.NewFileSet()
	goFiles := 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		goFiles++
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, parser.SkipObjectResolution)
		if err != nil {
			// A peer lane's half-finished edit can leave a file unparseable. That is
			// not this convention's business, but skipping silently could hide a real
			// site, so say so loudly without failing an unrelated lane's work.
			t.Logf("guard temp-dir scan: skipping unparseable %s: %v", name, err)
			continue
		}
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			switch fun := call.Fun.(type) {
			case *ast.SelectorExpr:
				// os.MkdirTemp(dir, pattern)
				pkg, ok := fun.X.(*ast.Ident)
				if !ok || pkg.Name != "os" || fun.Sel.Name != "MkdirTemp" || len(call.Args) != 2 {
					return true
				}
				pattern, ok := guardTempDirStringLit(call.Args[1])
				if !ok || !strings.HasPrefix(pattern, guardTempDirPrefix) {
					return true
				}
				raw = append(raw, guardTempDirSite{
					file:    name,
					line:    fset.Position(call.Pos()).Line,
					token:   guardTempDirTokenOf(pattern),
					literal: true,
				})
			case *ast.Ident:
				if fun.Name != "guardSessionTempDir" || len(call.Args) != 1 {
					return true
				}
				site := guardTempDirSite{file: name, line: fset.Position(call.Pos()).Line}
				if lit, ok := guardTempDirStringLit(call.Args[0]); ok {
					site.token, site.literal = lit, true
				}
				seam = append(seam, site)
			}
			return true
		})
	}
	if goFiles == 0 {
		t.Fatalf("guard temp-dir scan found no non-test .go files under %q; the scan is vacuous", dir)
	}
	return raw, seam
}

// guardTempDirStringLit unwraps a plain string-literal argument.
func guardTempDirStringLit(e ast.Expr) (string, bool) {
	lit, ok := e.(*ast.BasicLit)
	if !ok || lit.Kind != token.STRING {
		return "", false
	}
	s, err := strconv.Unquote(lit.Value)
	if err != nil {
		return "", false
	}
	return s, true
}

// guardTempDirTokenOf pulls the hook token out of a "fak-guard-<token>-*" name
// pattern, mirroring the split guardTempDirOwner performs on a real basename.
func guardTempDirTokenOf(pattern string) string {
	rest := strings.TrimPrefix(pattern, guardTempDirPrefix)
	if i := strings.IndexByte(rest, '-'); i >= 0 {
		return rest[:i]
	}
	return rest
}

// TestGuardTempDirCreationConvention is the discovering half: it walks every
// guard temp-dir creation call site in the package as it exists on disk and
// holds each to the rung it is responsible for. New call sites are covered the
// day they are written, with no list here to update.
func TestGuardTempDirCreationConvention(t *testing.T) {
	raw, seam := guardTempDirScanPackage(t, ".")

	// Property A: a hand-written fak-guard-* name must not claim a REAPED token.
	// Such a dir carries no <pid>, so guardTempDirOwner rejects it and the reaper
	// skips it forever, while its name advertises membership in the reaped family.
	usedWaivers := map[string]bool{}
	for _, s := range raw {
		if !guardTempDirHooks[s.token] {
			continue // deliberate opt-out, e.g. the fak-guard-reset-* forensic seeds
		}
		if guardTempDirRawWaiver[s.file] == s.token {
			usedWaivers[s.file] = true
			t.Logf("known-open (#5527 follow-on): %s creates a pid-less %s%s-* dir; "+
				"cure is guardSessionTempDir(%q)", s.where(), guardTempDirPrefix, s.token, s.token)
			continue
		}
		t.Errorf("%s: os.MkdirTemp(\"\", %q...) names the REAPED token %q but emits no <pid> "+
			"segment, so guardTempDirOwner rejects the name and guardReapStaleTempDirs can never "+
			"claim the dir. Cure: call guardSessionTempDir(%q) instead (see guard_tempreap.go).",
			s.where(), guardTempDirPrefix+s.token, s.token, s.token)
	}

	// Property B: every seam call site must name a token the reaper claims, or its
	// correctly-shaped dirs are ignored forever. This is leak 1's regression gate.
	seenSeamToken := false
	for _, s := range seam {
		if !s.literal {
			t.Errorf("%s: guardSessionTempDir is called with a non-literal token, so this "+
				"convention test cannot prove it is present in guardTempDirHooks. Pass a string "+
				"literal, or the closed set stops being single-source-of-truth.", s.where())
			continue
		}
		seenSeamToken = true
		if !guardTempDirHooks[s.token] {
			t.Errorf("%s: guardSessionTempDir(%q) creates %s%s-<pid>-* dirs, but %q is absent from "+
				"guardTempDirHooks, so guardTempDirOwner rejects every one and the reaper never "+
				"claims them. Cure: add %q to guardTempDirHooks in guard_tempreap.go.",
				s.where(), s.token, guardTempDirPrefix, s.token, s.token, s.token)
		}
	}
	if !seenSeamToken {
		t.Fatalf("guard temp-dir scan found no guardSessionTempDir call site with a literal token; "+
			"the scan is vacuous (raw sites found: %d)", len(raw))
	}

	// Self-policing waivers: a waived site that no longer violates property A means
	// the fix landed, so the waiver must go with it. Local runs read the working
	// tree, so the lane that fixes the site sees this the moment it edits, before
	// committing — which is the point.
	for file, tok := range guardTempDirRawWaiver {
		if !usedWaivers[file] {
			t.Errorf("stale waiver: guardTempDirRawWaiver[%q]=%q no longer matches any raw "+
				"os.MkdirTemp site. If you just routed it through guardSessionTempDir, delete the "+
				"waiver entry in the same change so the list cannot rot.", file, tok)
		}
	}
}

// guardTempDirRedirectTemp points os.TempDir() at an isolated root for the
// duration of the test, on both the Windows (TMP/TEMP) and Unix (TMPDIR)
// lookups, so a real guardSessionTempDir call cannot litter the shared temp root.
func guardTempDirRedirectTemp(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	t.Setenv("TMPDIR", root)
	t.Setenv("TMP", root)
	t.Setenv("TEMP", root)
	if got := os.TempDir(); got != root {
		t.Fatalf("os.TempDir() = %q after redirect, want %q", got, root)
	}
	return root
}

// TestGuardLifecycleTempDirIsReaped is leak 1's behavioural witness: a dir the
// lifecycle IPC server allocates must be claimed by the parser AND removed by the
// dead-owner sweep. Before "lifecycle" was added to guardTempDirHooks this failed
// at the first assertion, and every lifecycle dir a killed guard left behind
// stayed on disk for the life of the machine.
func TestGuardLifecycleTempDirIsReaped(t *testing.T) {
	root := guardTempDirRedirectTemp(t)

	dir, err := guardSessionTempDir("lifecycle")
	if err != nil {
		t.Fatalf("guardSessionTempDir(lifecycle): %v", err)
	}
	base := filepath.Base(dir)

	// Rung 1 + rung 2: the parser claims the name and recovers this process's PID.
	hook, pid, ok := guardTempDirOwner(base)
	if !ok || hook != "lifecycle" || pid != os.Getpid() {
		t.Fatalf("guardTempDirOwner(%q) = (%q, %d, %v), want (\"lifecycle\", %d, true) — "+
			"is \"lifecycle\" present in guardTempDirHooks?", base, hook, pid, ok, os.Getpid())
	}

	// And the sweep actually removes it once the owner reads as dead. selfPID 0 is
	// never a real owner (guardTempDirOwner rejects pid <= 0), so nothing is
	// excluded as "this process's own dir".
	reaped := guardReapStaleTempDirs(root, 0, func(int) bool { return false })
	if len(reaped) != 1 || reaped[0] != dir {
		t.Fatalf("reaped = %v, want exactly [%s]", reaped, dir)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("expected %s removed, stat err = %v", base, err)
	}
}

// TestGuardPidlessHandoffDirIsNotClaimed is leak 2's regression witness. The old
// cmdGuard call site created "fak-guard-handoff-<random>" with a raw
// os.MkdirTemp: the token was already reaped-eligible, but with no <pid> segment
// the parser must refuse it and the sweep must leave it alone. That refusal is
// correct and stays correct — it is precisely why the raw call site was a leak,
// and why the cure had to be at the CREATION seam rather than in the parser.
func TestGuardPidlessHandoffDirIsNotClaimed(t *testing.T) {
	// The exact shape os.MkdirTemp("", "fak-guard-handoff-*") produces: prefix,
	// token, then the random suffix with no PID between them.
	const pidless = "fak-guard-handoff-3980821147"
	if hook, pid, ok := guardTempDirOwner(pidless); ok {
		t.Fatalf("guardTempDirOwner(%q) = (%q, %d, true), want ok=false: a pid-less name has no "+
			"owner to prove dead and must never be reaped", pidless, hook, pid)
	}

	root := t.TempDir()
	stale := filepath.Join(root, pidless)
	if err := os.Mkdir(stale, 0o700); err != nil {
		t.Fatalf("mkdir %s: %v", pidless, err)
	}
	// Even with every PID reading as dead, an unclaimable name is untouched.
	if reaped := guardReapStaleTempDirs(root, 0, func(int) bool { return false }); len(reaped) != 0 {
		t.Fatalf("reaped = %v, want none: %q is not claimable", reaped, pidless)
	}
	if _, err := os.Stat(stale); err != nil {
		t.Fatalf("expected %s kept, stat err = %v", pidless, err)
	}

	// The cure the fix applies: the same token through the creation seam IS claimed.
	guardTempDirRedirectTemp(t)
	dir, err := guardSessionTempDir("handoff")
	if err != nil {
		t.Fatalf("guardSessionTempDir(handoff): %v", err)
	}
	hook, pid, ok := guardTempDirOwner(filepath.Base(dir))
	if !ok || hook != "handoff" || pid != os.Getpid() {
		t.Fatalf("guardTempDirOwner(%q) = (%q, %d, %v), want (\"handoff\", %d, true)",
			filepath.Base(dir), hook, pid, ok, os.Getpid())
	}
}
