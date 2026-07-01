package main

// `fak test` -- the host-aware developer test runner. It is the thin convenience
// verb the developer-tooling guide (docs/dev-tooling.md) specs: one command that
// resolves the right `go test` invocation for the tier you ask for AND encodes the
// one piece of host knowledge that bites newcomers -- on Windows, native `go test`
// is blocked by an OS Application-Control policy, so the suite must run inside WSL.
// `fak test` routes there automatically (via the repo-root test.ps1 wrapper) instead
// of letting you discover the block by hitting it.
//
//	fak test                     run the fast smoke tier (go test -short ./...)
//	fak test full                the full suite (go test ./...)
//	fak test race                the race tier (go test -short -race ./...)
//	fak test affected            run fak affected for the changed package closure
//	fak test durations           fold go test -json into a duration ledger
//	fak test ./internal/ctxmmu/  one package (any ./... or import-path arg)
//	fak test fast -- -run TestX -count=1   pass extra flags through to go test
//	fak test --list              print the tiers and exit
//	fak test -n                  print the resolved command without running it
//
// The pure planTest below resolves (host, args) -> the exact command to exec, so the
// decision is testable and reproducible; cmdTest is the impure shell that prints the
// resolved command and runs it. The make targets (test-fast/test/test-affected/...)
// and `fak affected` remain the authoritative gates -- this verb is the host-routing
// convenience layer over the same `go test`, not a replacement.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

func cmdTest(argv []string) { os.Exit(runTest(os.Stdout, os.Stderr, argv)) }

// testPlan is the resolved, reproducible command a `fak test` invocation will run.
type testPlan struct {
	Tier   string   // the resolved tier name ("fast" | "full" | "race" | "package")
	GoArgs []string // the args handed to `go test`, e.g. ["-short", "./..."]
	Argv   []string // the actual command to exec, host-resolved
	ViaWSL bool     // true when routed through test.ps1 (Windows -> WSL)
	Note   string   // a one-line human note about the routing
}

// planTest is the pure resolver: given the host GOOS and the post-verb args, it
// returns the command to run. No I/O, no exec -- so a peer can reproduce the exact
// invocation from the same inputs, and the table test pins the routing.
func planTest(goos string, args []string) (testPlan, error) {
	// Split off any pass-through args after a literal "--".
	var passthrough []string
	for i, a := range args {
		if a == "--" {
			passthrough = append(passthrough, args[i+1:]...)
			args = args[:i]
			break
		}
	}

	// The first remaining token selects the tier or names a package target.
	tierArg := "fast"
	if len(args) > 0 && args[0] != "" {
		tierArg = args[0]
	}

	var p testPlan
	switch tierArg {
	case "fast", "smoke", "short":
		p.Tier, p.GoArgs = "fast", []string{"-short", "./..."}
	case "full", "all", "":
		p.Tier, p.GoArgs = "full", []string{"./..."}
	case "race":
		p.Tier, p.GoArgs = "race", []string{"-short", "-race", "./..."}
	default:
		// Treat anything else as a package target (./..., internal/foo, a path).
		if !looksLikePackage(tierArg) {
			return testPlan{}, fmt.Errorf("unknown tier or package %q (try: fast | full | race | a ./package/... path)", tierArg)
		}
		p.Tier, p.GoArgs = "package", []string{tierArg}
	}
	p.GoArgs = append(p.GoArgs, passthrough...)

	if goos == "windows" {
		// Native `go test` is OS-policy-blocked here; route through test.ps1, which
		// forwards every arg verbatim to `go test` inside WSL.
		p.ViaWSL = true
		p.Note = "windows host: routing go test to WSL via test.ps1 (native go test is OS-policy-blocked)"
		p.Argv = append([]string{"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "test.ps1"}, p.GoArgs...)
	} else {
		p.Note = goos + " host: running go test directly"
		p.Argv = append([]string{"go", "test"}, p.GoArgs...)
	}
	return p, nil
}

// looksLikePackage reports whether a token is plausibly a Go package target rather
// than a fat-fingered tier name, so a typo'd tier fails loudly instead of being
// silently handed to `go test` as a bogus package.
func looksLikePackage(s string) bool {
	return strings.HasPrefix(s, "./") || strings.HasPrefix(s, "internal/") ||
		strings.HasPrefix(s, "cmd/") || strings.Contains(s, "/...") ||
		strings.HasPrefix(s, "github.com/")
}

func runTest(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak test", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		list = fs.Bool("list", false, "print the available tiers and exit")
		dry  = fs.Bool("n", false, "print the resolved command without running it")
		dry2 = fs.Bool("print", false, "alias of -n")
	)
	fs.Usage = func() {
		fmt.Fprint(stderr, `fak test -- host-aware developer test runner

  fak test                     fast smoke tier (go test -short ./...)
  fak test full                full suite (go test ./...)
  fak test race                race tier (go test -short -race ./...)
  fak test affected            affected-package loop (delegates to fak affected)
  fak test durations           fold go test -json into a duration ledger
  fak test ./internal/ctxmmu/  one package (any ./... or import-path target)
  fak test fast -- -run TestX  pass extra flags through to go test
  fak test --list              list tiers
  fak test -n                  print the resolved command, do not run

On Windows, go test is routed to WSL via test.ps1 (native go test is OS-policy-blocked).
`)
	}
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *list {
		fmt.Fprint(stdout, "tiers:\n  fast       go test -short ./...   (default; pre-commit smoke)\n  full       go test ./...          (authoritative suite)\n  race       go test -short -race ./...\n  affected   fak affected ...       (changed packages plus importers)\n  durations  parse go test -json into a duration ledger\n  <pkg>      a ./... or import-path target\n")
		return 0
	}

	args := fs.Args()
	if len(args) > 0 {
		switch args[0] {
		case "affected":
			return runAffected(stdout, stderr, args[1:])
		case "durations", "duration", "duration-ledger":
			return runTestDurations(stdout, stderr, args[1:])
		}
	}

	p, err := planTest(runtime.GOOS, args)
	if err != nil {
		fmt.Fprintf(stderr, "fak test: %v\n", err)
		return 2
	}
	fmt.Fprintf(stdout, "# %s\n# tier=%s -> %s\n", p.Note, p.Tier, strings.Join(p.Argv, " "))
	if *dry || *dry2 {
		return 0
	}

	cmd := exec.Command(p.Argv[0], p.Argv[1:]...)
	cmd.Stdout, cmd.Stderr, cmd.Stdin = stdout, stderr, os.Stdin
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak test: %v\n", err)
		return 1
	}
	return 0
}
