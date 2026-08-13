package architest

// The read-mostly steer-overlay fence (#5032, epic #5015). The overlay
// (internal/steerpr + `fak steer prs` + `fak release prplan`) rebuilds the
// operator's unit of ATTENTION over the PR-free trunk. Its entire thesis is
// that the PR's gate function and its observability/steering function are
// separable: you can have the second without the first. Every affordance
// asserts "must not block a merge" in prose — but prose is not a floor. As the
// overlay grows (a hook here, a --check there, a CI wiring), the pressure to
// "just fail the commit when it's RESIDUAL" is enormous; continuous merge
// would die by accretion, not decision. These gates are the durable guard for
// that regression class (refusal token: OVERLAY_WOULD_GATE in dos.toml).
//
// The floor has three planks, matching the three ways the overlay could grow
// a gate:
//
//  1. A steerpr code path mutates git (commit/push/reset/revert/merge/...):
//     TestSteerOverlayLeafStaysPureAndGitFree pins the fold leaf to a
//     no-subprocess, no-internal-import shape, and
//     TestSteerOverlayVerbsReadButNeverMutateGit proves every subprocess the
//     overlay verbs launch is a PROVABLY read-only invocation (literal tool +
//     literal verb from a closed read-only allowlist; anything unprovable is
//     refused — fail closed).
//  2. An overlay code path fails a commit: the commit path (cmd/fak
//     commit*.go), the hook layer (internal/hooks), and the git hook scripts
//     (tools/githooks) must never reference the overlay at all —
//     TestSteerOverlayCheckStaysOffCommitAndPromotionPaths.
//  3. --check's exit code leaks into a commit/promotion path: the same test
//     pins the ONLY reachable callers of the exit-code-bearing entry points
//     (runSteerPRs/runSteer/cmdSteer/runReleasePRPlan) to the operator CLI
//     dispatch. Fold/render helpers (steerpr.FoldUnits, renderPRPlanMarkdown)
//     stay importable everywhere — rendering is the overlay's JOB; only the
//     blocking exit is fenced.
//
// TestSteerOverlayGateDetectorRedsOnBlockingCall is the guard's own witness:
// a guard that has never been seen to fail is not a proven guard, so the
// detector is fed a deliberately-blocking fixture (exec git commit / seam git
// push) and must red on it, and a read-only fixture and must stay green.

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

// steerOverlayReadOnlyGitVerbs is the closed allowlist of git subcommands an
// overlay code path may invoke: verbs that READ history/refs and can never
// mutate the repository. Extend ONLY with provably read-only verbs; a
// mutating verb (commit, push, reset, revert, merge, rebase, tag, update-ref,
// apply, stash, checkout, restore, clean, cherry-pick, ...) must never be
// added — that is the OVERLAY_WOULD_GATE class this floor exists to refuse.
var steerOverlayReadOnlyGitVerbs = map[string]bool{
	"log": true, "rev-parse": true, "show": true, "diff": true,
	"ls-files": true, "status": true, "for-each-ref": true,
	"merge-base": true, "cat-file": true, "describe": true,
}

// steerOverlayDosVerbs is the closed allowlist of `dos` subcommands the
// overlay may shell: read-only audit queries over already-landed commits.
var steerOverlayDosVerbs = map[string]bool{"commit-audit": true}

// steerOverlayGitSeams are the local helper names the overlay verb files
// shell git through (`releasePRPlanGit(root, verb, args...)` and its default
// `releaseStatusGitOutput`). A call through a seam is scanned exactly like a
// direct exec.Command("git", ...): the verb argument must be a string literal
// from the read-only allowlist.
var steerOverlayGitSeams = map[string]bool{
	"releasePRPlanGit":       true,
	"releaseStatusGitOutput": true,
}

// steerOverlayEntryPoints are the exit-code-bearing overlay entry points (the
// functions whose return value / os.Exit carries the --check verdict), mapped
// to the ONLY non-test cmd/fak files allowed to reference them. This is plank
// 3: --check's exit code is reachable only from the operator CLI dispatch,
// never from a commit, hook, or promotion path.
var steerOverlayEntryPoints = map[string]map[string]bool{
	"runSteerPRs":      {"steer_prs.go": true},
	"runSteer":         {"steer_prs.go": true},
	"cmdSteer":         {"steer_prs.go": true, "main.go": true},
	"runReleasePRPlan": {"release_prplan.go": true, "release.go": true},
}

// steerOverlayGitVerbViolation returns the refusal reason for a git
// invocation (verb + the arguments after it), or "" when the invocation is
// provably read-only. `config` is the one verb that is only CONDITIONALLY
// read-only — bare `git config key value` WRITES repo config — so it is
// allowed solely when the argument immediately after the verb is the literal
// read selector `--get`.
func steerOverlayGitVerbViolation(verb string, rest []ast.Expr) string {
	if verb == "config" {
		if len(rest) == 0 {
			return "bare `config` with no arguments is unprovable — only a literal `config --get <key>` read is provably read-only (OVERLAY_WOULD_GATE)"
		}
		if sel, ok := steerOverlayLiteral(rest[0]); !ok || sel != "--get" {
			return "bare `config` can WRITE repo state (`git config key value`); only a literal `config --get <key>` read is provably read-only (OVERLAY_WOULD_GATE)"
		}
		return ""
	}
	if !steerOverlayReadOnlyGitVerbs[verb] {
		return "not in the read-only allowlist; the overlay must NEVER mutate git (OVERLAY_WOULD_GATE)"
	}
	return ""
}

// steerOverlayLiteral unwraps a string-literal argument. ok=false for
// anything else — a non-literal command name or verb is UNPROVABLE, and the
// fence fails closed on it.
func steerOverlayLiteral(e ast.Expr) (string, bool) {
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

// steerOverlayScanFile returns every OVERLAY_WOULD_GATE violation in one
// parsed overlay file: a subprocess or git-seam call that is not a provably
// read-only invocation. It is shared by the live floor
// (TestSteerOverlayVerbsReadButNeverMutateGit) and the detector self-witness
// (TestSteerOverlayGateDetectorRedsOnBlockingCall).
func steerOverlayScanFile(fset *token.FileSet, f *ast.File) []string {
	var violations []string
	report := func(pos token.Pos, msg string) {
		violations = append(violations, fset.Position(pos).String()+": "+msg)
	}
	ast.Inspect(f, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		var callee string
		switch fun := call.Fun.(type) {
		case *ast.Ident:
			callee = fun.Name
		case *ast.SelectorExpr:
			if x, ok := fun.X.(*ast.Ident); ok {
				callee = x.Name + "." + fun.Sel.Name
			}
		}
		switch {
		case callee == "exec.Command" || callee == "exec.CommandContext":
			args := call.Args
			if callee == "exec.CommandContext" {
				if len(args) < 1 {
					return true
				}
				args = args[1:] // drop ctx
			}
			if len(args) == 0 {
				return true
			}
			tool, ok := steerOverlayLiteral(args[0])
			if !ok {
				report(call.Pos(), "overlay launches a subprocess with a NON-LITERAL command name — unprovable, refused (OVERLAY_WOULD_GATE)")
				return true
			}
			switch tool {
			case "git":
				if len(args) < 2 {
					report(call.Pos(), "overlay execs bare `git` with no subcommand — unprovable, refused (OVERLAY_WOULD_GATE)")
					return true
				}
				verb, ok := steerOverlayLiteral(args[1])
				if !ok {
					report(call.Pos(), "overlay execs `git` with a NON-LITERAL subcommand — unprovable, refused (OVERLAY_WOULD_GATE)")
				} else if msg := steerOverlayGitVerbViolation(verb, args[2:]); msg != "" {
					report(call.Pos(), "overlay execs `git "+verb+"` — "+msg)
				}
			case "dos":
				if len(args) < 2 {
					report(call.Pos(), "overlay execs bare `dos` with no subcommand — unprovable, refused (OVERLAY_WOULD_GATE)")
					return true
				}
				verb, ok := steerOverlayLiteral(args[1])
				if !ok || !steerOverlayDosVerbs[verb] {
					report(call.Pos(), "overlay execs a `dos` subcommand outside the read-only audit allowlist (OVERLAY_WOULD_GATE)")
				}
			default:
				report(call.Pos(), "overlay execs unexpected tool `"+tool+"` — only read-only `git`/`dos` queries are allowed (OVERLAY_WOULD_GATE)")
			}
		case steerOverlayGitSeams[callee]:
			if len(call.Args) < 2 {
				report(call.Pos(), "overlay calls git seam "+callee+" with no subcommand — unprovable, refused (OVERLAY_WOULD_GATE)")
				return true
			}
			verb, ok := steerOverlayLiteral(call.Args[1])
			if !ok {
				report(call.Pos(), "overlay calls git seam "+callee+" with a NON-LITERAL subcommand — unprovable, refused (OVERLAY_WOULD_GATE)")
			} else if msg := steerOverlayGitVerbViolation(verb, call.Args[2:]); msg != "" {
				report(call.Pos(), "overlay calls git seam "+callee+" with `"+verb+"` — "+msg)
			}
		}
		return true
	})
	return violations
}

// steerOverlayVerbFiles returns the overlay's cmd/fak verb files: every
// non-test steer_*.go (the `fak steer *` surface — steer_prs.go today; the
// glob sweeps future subcommands in automatically) plus release_prplan.go
// (`fak release prplan`, the same fold's release-time twin, whose --check
// carries an exit code). steering.go (the Slack steerability surface) is a
// different feature and deliberately not matched.
func steerOverlayVerbFiles(t *testing.T, cmdFak string) []string {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(cmdFak, "steer_*.go"))
	if err != nil {
		t.Fatalf("glob steer_*.go: %v", err)
	}
	var out []string
	for _, f := range files {
		if !strings.HasSuffix(f, "_test.go") {
			out = append(out, f)
		}
	}
	out = append(out, filepath.Join(cmdFak, "release_prplan.go"))
	if len(out) < 2 {
		t.Fatalf("expected at least steer_prs.go + release_prplan.go under %s; the overlay verb surface moved — update steerOverlayVerbFiles", cmdFak)
	}
	return out
}

func steerOverlayCmdFakDir(t *testing.T) string {
	t.Helper()
	return filepath.Join(filepath.Dir(internalDir(t)), "cmd", "fak")
}

// TestSteerOverlayLeafStaysPureAndGitFree pins plank 1 at the leaf: the
// internal/steerpr fold can never reach git and can never fail the process
// hosting it, because it cannot launch a subprocess, touch the network,
// import another internal package, or call an exiting function AT ALL. The
// fold's own ledger file IO (ack/redirect/loop append their overlay-owned
// ledgers via plain "os") is deliberately allowed — appending a ledger row
// can neither mutate git nor fail a commit, and fencing it would outlaw the
// overlay's job rather than the gate class.
func TestSteerOverlayLeafStaysPureAndGitFree(t *testing.T) {
	dir := filepath.Join(internalDir(t), "steerpr")
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, dir,
		func(fi os.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}
	forbidden := map[string]string{
		"os/exec":  "subprocess launch (the git mutation vector)",
		"syscall":  "raw syscalls",
		"net":      "the network",
		"net/http": "the network",
	}
	// The exiting calls: any of these inside the fold would let a caller's
	// process die on the fold's verdict — the exact "fail the commit" vector
	// plank 2 exists to keep out of reach.
	exiting := map[string]bool{
		"os.Exit": true, "log.Fatal": true, "log.Fatalf": true, "log.Fatalln": true,
	}
	for _, p := range parsed {
		for name, f := range p.Files {
			for _, imp := range f.Imports {
				path, _ := strconv.Unquote(imp.Path.Value)
				if why, bad := forbidden[path]; bad {
					t.Errorf("%s imports %q — %s. internal/steerpr is the overlay's git-free fold; a leaf that can reach this can grow a gate (OVERLAY_WOULD_GATE). Keep the fold git-free and put subprocess needs in the cmd/fak verb shell, where TestSteerOverlayVerbsReadButNeverMutateGit fences them.", name, path, why)
				}
				if strings.HasPrefix(path, modPrefix) {
					t.Errorf("%s imports internal package %q — internal/steerpr must import NOTHING internal: verdicts are supplied by the caller so the band stays a VIEW over the kernel's witness oracle, never a second oracle wired where it could gate (OVERLAY_WOULD_GATE).", name, path)
				}
			}
			ast.Inspect(f, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
					if x, ok := sel.X.(*ast.Ident); ok && exiting[x.Name+"."+sel.Sel.Name] {
						t.Errorf("%s: %s calls %s.%s — the fold must never be able to kill its host process: a caller on the commit path that imports a fold which can exit has a gate by accretion (OVERLAY_WOULD_GATE). Return an error and let the operator CLI decide the exit.",
							name, fset.Position(call.Pos()), x.Name, sel.Sel.Name)
					}
				}
				return true
			})
		}
	}
}

// TestSteerOverlayVerbsReadButNeverMutateGit pins plank 1 at the verb shell:
// every subprocess the overlay verbs (`fak steer *`, `fak release prplan`)
// launch is a PROVABLY read-only invocation — a literal tool name with a
// literal subcommand from the closed read-only allowlist. commit / push /
// reset / revert / merge / rebase (or any unprovable call shape) reds this
// gate with OVERLAY_WOULD_GATE.
//
// Seeded GREEN at reality: steer_prs.go shells `dos commit-audit` (read-only
// audit), the `git log` seam, and `git config --get user.name` (the --get is
// what makes config provably a read); release_prplan.go shells `git log` /
// `git rev-parse` through releasePRPlanGit. The deliberate-red proof that
// this detector BITES is TestSteerOverlayGateDetectorRedsOnBlockingCall.
func TestSteerOverlayVerbsReadButNeverMutateGit(t *testing.T) {
	fset := token.NewFileSet()
	for _, file := range steerOverlayVerbFiles(t, steerOverlayCmdFakDir(t)) {
		f, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", file, err)
		}
		for _, v := range steerOverlayScanFile(fset, f) {
			t.Error(v)
		}
	}
}

// TestSteerOverlayGateDetectorRedsOnBlockingCall is the guard's own witness
// (#5032's acceptance gate): a structural test that passes but cannot be
// shown to fail is not a proven floor. The detector is fed the exact blocking
// shapes the fence exists to refuse — a direct `git commit` exec and a `git
// push` laundered through the seam — and MUST report both; and a read-only
// fixture on which it MUST stay silent (a detector that reds everything
// proves nothing).
func TestSteerOverlayGateDetectorRedsOnBlockingCall(t *testing.T) {
	const blocking = `package main

import "os/exec"

// A deliberately-introduced gate: the land-time hook "just fails the commit
// when it's RESIDUAL". This is the accretion #5032 fences out.
func landHook(residual bool) error {
	if residual {
		return exec.Command("git", "commit", "--amend", "--no-edit").Run()
	}
	releasePRPlanGit(".", "push", "--force")
	releasePRPlanGit(".", "config", "receive.denyCurrentBranch", "ignore")
	return nil
}
`
	const readOnly = `package main

import "os/exec"

func view() {
	_ = exec.Command("dos", "commit-audit", "a..b", "--json")
	_ = releasePRPlanGit(".", "log", "--no-merges")
	_ = releasePRPlanGit(".", "rev-parse", "HEAD")
	_ = releasePRPlanGit(".", "config", "--get", "user.name")
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "blocking_fixture.go", blocking, 0)
	if err != nil {
		t.Fatalf("parse blocking fixture: %v", err)
	}
	got := steerOverlayScanFile(fset, f)
	if len(got) != 3 {
		t.Fatalf("detector must red on ALL THREE deliberately-wired gates (direct `git commit` exec + seam `git push` + seam bare-`config` write); got %d violation(s): %v — a guard that cannot be seen to fail is not a proven guard", len(got), got)
	}
	joined := strings.Join(got, "\n")
	for _, verb := range []string{"git commit", "push", "config"} {
		if !strings.Contains(joined, verb) {
			t.Errorf("detector violations do not name the wired gate %q:\n%s", verb, joined)
		}
	}

	f, err = parser.ParseFile(fset, "readonly_fixture.go", readOnly, 0)
	if err != nil {
		t.Fatalf("parse read-only fixture: %v", err)
	}
	if got := steerOverlayScanFile(fset, f); len(got) != 0 {
		t.Errorf("detector reds on provably read-only overlay calls — it would fence the overlay's own job, not just gates: %v", got)
	}
}

// TestSteerOverlayCheckStaysOffCommitAndPromotionPaths pins planks 2 and 3:
//
//   - The exit-code-bearing overlay entry points are referenced ONLY from the
//     operator CLI dispatch (steerOverlayEntryPoints), so `--check`'s exit
//     code is reachable only from an operator/CI reporting invocation — never
//     consumed by a commit, hook, ship, or promotion code path. (The pure
//     fold/render helpers are NOT fenced: release-ship rendering a PR body
//     through steerpr.FoldUnits is the overlay doing its job.)
//   - The commit path (cmd/fak commit*.go), the hook layer (internal/hooks),
//     and the git hook scripts (tools/githooks/*) contain no overlay
//     reference at all — no steerpr import, no entry-point call, no
//     `fak steer prs`/`prplan` invocation string a hook could shell.
func TestSteerOverlayCheckStaysOffCommitAndPromotionPaths(t *testing.T) {
	cmdFak := steerOverlayCmdFakDir(t)
	fset := token.NewFileSet()

	// Plank 3: entry points only reachable from the CLI dispatch.
	goFiles, err := filepath.Glob(filepath.Join(cmdFak, "*.go"))
	if err != nil {
		t.Fatalf("glob cmd/fak: %v", err)
	}
	for _, file := range goFiles {
		base := filepath.Base(file)
		if strings.HasSuffix(base, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			// A syntactically-broken peer WIP file cannot compile, so it
			// cannot ship a gate; note it and move on rather than redding
			// the floor on someone else's half-typed edit.
			t.Logf("skipping unparseable %s: %v", base, err)
			continue
		}
		ast.Inspect(f, func(n ast.Node) bool {
			id, ok := n.(*ast.Ident)
			if !ok {
				return true
			}
			allowed, fenced := steerOverlayEntryPoints[id.Name]
			if fenced && !allowed[base] {
				t.Errorf("%s references overlay entry point %s — the --check exit code must be reachable ONLY from the operator CLI dispatch (%v), never from a commit/hook/ship/promotion path. Consume the pure fold (steerpr.FoldUnits / the render helpers) instead; a blocking wire is OVERLAY_WOULD_GATE.",
					fset.Position(id.Pos()), id.Name, steerOverlayEntryPoints[id.Name])
			}
			return true
		})
	}

	// Plank 2a: the commit path and the hook layer never touch the overlay.
	commitFiles, err := filepath.Glob(filepath.Join(cmdFak, "commit*.go"))
	if err != nil {
		t.Fatalf("glob commit*.go: %v", err)
	}
	hooksDir := filepath.Join(internalDir(t), "hooks")
	hookEntries, err := os.ReadDir(hooksDir)
	if err != nil {
		t.Fatalf("read %s: %v", hooksDir, err)
	}
	var guarded []string
	for _, f := range commitFiles {
		guarded = append(guarded, f)
	}
	for _, e := range hookEntries {
		if strings.HasSuffix(e.Name(), ".go") {
			guarded = append(guarded, filepath.Join(hooksDir, e.Name()))
		}
	}
	for _, file := range guarded {
		base := filepath.Base(file)
		if strings.HasSuffix(base, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Logf("skipping unparseable %s: %v", file, err)
			continue
		}
		for _, imp := range f.Imports {
			path, _ := strconv.Unquote(imp.Path.Value)
			if path == modPrefix+"steerpr" {
				t.Errorf("%s imports internal/steerpr — the commit/hook layer must never reference the overlay; observability lives beside the merge path, not in it (OVERLAY_WOULD_GATE).", file)
			}
		}
		ast.Inspect(f, func(n ast.Node) bool {
			switch node := n.(type) {
			case *ast.Ident:
				if _, fenced := steerOverlayEntryPoints[node.Name]; fenced || node.Name == "steerpr" {
					t.Errorf("%s: %s references overlay symbol %q — a commit/hook path that can see the overlay can grow a gate from it (OVERLAY_WOULD_GATE).",
						file, fset.Position(node.Pos()), node.Name)
					return false
				}
			case *ast.BasicLit:
				if node.Kind == token.STRING {
					if s, err := strconv.Unquote(node.Value); err == nil {
						low := strings.ToLower(s)
						if strings.Contains(low, "steer prs") || strings.Contains(low, "steerpr") || strings.Contains(low, "prplan") {
							t.Errorf("%s: %s embeds overlay invocation string %q — a commit/hook path must not shell the overlay (its --check exit would gate the commit: OVERLAY_WOULD_GATE).",
								file, fset.Position(node.Pos()), s)
						}
					}
				}
			}
			return true
		})
	}

	// Plank 2b: the git hook SCRIPTS never shell the overlay either.
	hookScriptsDir := filepath.Join(filepath.Dir(internalDir(t)), "tools", "githooks")
	scripts, err := os.ReadDir(hookScriptsDir)
	if err != nil {
		t.Fatalf("read %s: %v", hookScriptsDir, err)
	}
	for _, s := range scripts {
		if s.IsDir() {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(hookScriptsDir, s.Name()))
		if err != nil {
			t.Fatalf("read hook script %s: %v", s.Name(), err)
		}
		low := strings.ToLower(string(raw))
		for _, needle := range []string{"steer prs", "steerpr", "prplan"} {
			if strings.Contains(low, needle) {
				t.Errorf("tools/githooks/%s mentions %q — a git hook that shells the overlay puts --check's exit code in the commit path (OVERLAY_WOULD_GATE).", s.Name(), needle)
			}
		}
	}
}

// TestSteerOverlayRefusalReasonDeclared pins the vocabulary half of #5032:
// OVERLAY_WOULD_GATE is declared in dos.toml [reasons.*] with a summary and a
// fix, so a refusal of this class speaks the closed vocabulary
// (`dos man wedge OVERLAY_WOULD_GATE --explain` resolves) instead of free-text.
func TestSteerOverlayRefusalReasonDeclared(t *testing.T) {
	tomlPath := filepath.Join(filepath.Dir(internalDir(t)), "dos.toml")
	raw, err := os.ReadFile(tomlPath)
	if err != nil {
		t.Fatalf("read %s: %v", tomlPath, err)
	}
	lines := strings.Split(string(raw), "\n")
	inSection := false
	keys := map[string]bool{}
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "[") {
			inSection = trimmed == "[reasons.OVERLAY_WOULD_GATE]"
			continue
		}
		if !inSection || trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		if key, val, ok := strings.Cut(trimmed, "="); ok {
			if v := strings.TrimSpace(val); v != "" && v != `""` {
				keys[strings.TrimSpace(key)] = true
			}
		}
	}
	if len(keys) == 0 {
		t.Fatalf("dos.toml declares no [reasons.OVERLAY_WOULD_GATE] section — the overlay-gating regression class has no refusal token, so a refusal of it cannot speak the closed vocabulary. Declare it with category/refusal/summary/fix (#5032).")
	}
	for _, want := range []string{"summary", "fix", "category"} {
		if !keys[want] {
			t.Errorf("dos.toml [reasons.OVERLAY_WOULD_GATE] lacks a non-empty %q — the token must carry its own explanation and cure so `dos man wedge OVERLAY_WOULD_GATE --explain` teaches, not just names.", want)
		}
	}
}
