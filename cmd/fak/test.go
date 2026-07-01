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
//	fak test build               run the build gate (go build ./...)
//	fak test vet                 run the vet gate (go vet ./...)
//	fak test gofmt               run the formatting gate (gofmt -l .)
//	fak test codelint <path>     run the agent-code lint packs
//	fak test full                the full suite (go test ./...)
//	fak test race                the race tier (go test -short -race ./...)
//	fak test affected            run fak affected for the changed package closure
//	fak test durations           fold go test -json into a duration ledger
//	fak test shards              balance packages from a duration ledger
//	fak test --json -n race      emit a machine-readable repair packet
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
	"runtime"
	"strings"
)

func cmdTest(argv []string) { os.Exit(runTest(os.Stdout, os.Stderr, argv)) }

// testPlan is the resolved, reproducible command a `fak test` invocation will run.
type testPlan struct {
	Tier         string   // the resolved tier name ("fast" | "full" | "race" | "package")
	GoArgs       []string // the args handed to `go test`, e.g. ["-short", "./..."]
	Argv         []string // the actual command to exec, host-resolved
	ViaWSL       bool     // true when routed through test.ps1 (Windows -> WSL)
	Note         string   // a one-line human note about the routing
	FailOnStdout bool     // true for list-style gates such as gofmt -l
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
		json = fs.Bool("json", false, "emit a JSON repair packet for the resolved or finished command")
	)
	fs.Usage = func() {
		fmt.Fprint(stderr, `fak test -- host-aware developer test runner

  fak test                     fast smoke tier (go test -short ./...)
  fak test build               build gate (go build ./...)
  fak test vet                 vet gate (go vet ./...)
  fak test gofmt               formatting gate (gofmt -l .)
  fak test codelint <path>     agent-code lint packs
  fak test full                full suite (go test ./...)
  fak test race                race tier (go test -short -race ./...)
  fak test affected            affected-package loop (delegates to fak affected)
  fak test durations           fold go test -json into a duration ledger
  fak test shards              balance packages from a duration ledger
  fak test --json -n race      emit a machine-readable repair packet
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
		if *json {
			return writeTestListJSON(stdout, stderr)
		}
		fmt.Fprint(stdout, "tiers:\n  fast       go test -short ./...   (default; pre-commit smoke)\n  build      go build ./...\n  vet        go vet ./...\n  gofmt      gofmt -l .\n  codelint   fak codelint ...\n  full       go test ./...          (authoritative suite)\n  race       go test -short -race ./...\n  affected   fak affected ...       (changed packages plus importers)\n  durations  parse go test -json into a duration ledger\n  shards     balance packages from a duration ledger\n  <pkg>      a ./... or import-path target\n")
		return 0
	}

	args := fs.Args()
	if len(args) > 0 {
		switch args[0] {
		case "build", "vet", "gofmt", "fmt":
			return runTestCheck(stdout, stderr, os.Stdin, args[0], args[1:], *json, *dry || *dry2)
		case "codelint":
			return runTestCodelint(stdout, stderr, args[1:], *json, *dry || *dry2)
		case "affected":
			subArgs := args[1:]
			if *json && !hasFlag(subArgs, "json") {
				subArgs = append([]string{"--json"}, subArgs...)
			}
			return runAffected(stdout, stderr, subArgs)
		case "durations", "duration", "duration-ledger":
			return runTestDurations(stdout, stderr, args[1:])
		case "shards", "shard", "shard-plan":
			return runTestShards(stdout, stderr, args[1:])
		}
	}

	p, err := planTest(runtime.GOOS, args)
	if err != nil {
		if *json {
			return writeTestRepairJSON(stdout, stderr, newTestUsageRepairPacket(err))
		}
		fmt.Fprintf(stderr, "fak test: %v\n", err)
		return 2
	}
	if *json {
		if *dry || *dry2 {
			return writeTestRepairJSON(stdout, stderr, newTestResolvedRepairPacket(p))
		}
		return runTestRepairJSON(stdout, stderr, os.Stdin, p)
	}
	fmt.Fprintf(stdout, "# %s\n# tier=%s -> %s\n", p.Note, p.Tier, strings.Join(p.Argv, " "))
	if *dry || *dry2 {
		return 0
	}

	return runTestPlan(stdout, stderr, os.Stdin, p)
}

func runTestCheck(stdout, stderr io.Writer, stdin io.Reader, name string, args []string, asJSON, dry bool) int {
	p, err := planTestCheck(name, args)
	if err != nil {
		if asJSON {
			return writeTestRepairJSON(stdout, stderr, newTestUsageRepairPacket(err))
		}
		fmt.Fprintf(stderr, "fak test: %v\n", err)
		return 2
	}
	if asJSON {
		if dry {
			return writeTestRepairJSON(stdout, stderr, newTestResolvedRepairPacket(p))
		}
		return runTestRepairJSON(stdout, stderr, stdin, p)
	}
	fmt.Fprintf(stdout, "# %s\n# tier=%s -> %s\n", p.Note, p.Tier, strings.Join(p.Argv, " "))
	if dry {
		return 0
	}
	return runTestPlan(stdout, stderr, stdin, p)
}

func planTestCheck(name string, args []string) (testPlan, error) {
	switch name {
	case "build":
		targets := defaultArgs(args, "./...")
		return testPlan{
			Tier: "build",
			Argv: append([]string{"go", "build"}, targets...),
			Note: "running build gate",
		}, nil
	case "vet":
		targets := defaultArgs(args, "./...")
		return testPlan{
			Tier: "vet",
			Argv: append([]string{"go", "vet"}, targets...),
			Note: "running vet gate",
		}, nil
	case "gofmt", "fmt":
		targets := defaultArgs(args, ".")
		return testPlan{
			Tier:         "gofmt",
			Argv:         append([]string{"gofmt", "-l"}, targets...),
			Note:         "running gofmt check",
			FailOnStdout: true,
		}, nil
	default:
		return testPlan{}, fmt.Errorf("unknown check %q", name)
	}
}

func defaultArgs(args []string, def string) []string {
	if len(args) == 0 {
		return []string{def}
	}
	return append([]string(nil), args...)
}
