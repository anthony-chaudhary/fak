package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/armbench"
	"github.com/anthony-chaudhary/fak/internal/committedbuildwitness"
	"github.com/anthony-chaudhary/fak/internal/committedtree"
	"github.com/anthony-chaudhary/fak/internal/studymonitor"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// cmd/fak/ci_preflight.go — `fak ci-preflight`: answer "is the COMMITTED trunk tip CI-buildable and
// gofmt-clean, and if not, exactly which files / why" WITHOUT trusting the working tree.
//
// Why a dedicated verb: on this shared multi-session checkout the working tree is permanently
// peer-dirty (hundreds of uncommitted files, half-wired untracked siblings). `go build ./...` /
// `gofmt -l .` run in place therefore report the WORKING tree, which is neither what CI gates nor a
// stable signal — a peer's uncommitted fix can MASK a real committed red, and a peer's broken WIP
// can FABRICATE a red that is not on trunk. The two failure modes that actually red-trunk the `ci`
// job (`build·vet·test·claims-lint`) this repo hits repeatedly are:
//   - a partial commit: a caller lands but its callee file stays untracked → `undefined: X`, and
//   - a committed gofmt violation: a struct/const block realigned after a field add/remove.
// Both are invisible-to-guess from the dirty tree and were, before this verb, rediscovered by hand
// each session via `git archive <tip> | tar -x | (cd tmp && go build ./... && gofmt -l .)`.
//
// This verb encodes exactly that: archive the committed tip to a throwaway checkout, run
// `go build ./...` and `gofmt -l` THERE, and report the failing step + exact files as JSON. It is
// read-only against the repo (the throwaway checkout is under os.TempDir) so it never touches the
// shared tree. Exit 0 = trunk clean, 1 = a preflight check fired, 2 = could-not-run.
//
//   fak ci-preflight            # human summary of the committed origin/main-style tip
//   fak ci-preflight --json     # machine result for dispatch loops / agents
//   fak ci-preflight --ref R    # check an explicit ref/sha instead of HEAD
//   fak ci-preflight --skip-build   # gofmt-only (fast; build dominates the runtime)

// ciPreflightFailure is one failing CI-relevant check against the committed tip.
type ciPreflightFailure struct {
	Step   string   `json:"step"`             // "build" | "gofmt"
	Detail string   `json:"detail,omitempty"` // compiler message for a build break
	Files  []string `json:"files,omitempty"`  // unformatted files for a gofmt break
}

// ciPreflightResult is the whole verdict, JSON-stable for agents/loops.
type ciPreflightResult struct {
	Ref      string               `json:"ref"`      // the ref we resolved
	Tip      string               `json:"tip"`      // its resolved sha
	OK       bool                 `json:"ok"`       // true iff no failures
	Failures []ciPreflightFailure `json:"failures"` // empty when OK
	Skipped  []string             `json:"skipped,omitempty"`
}

func RunCIPreflight(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("ci-preflight", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	ref := fs.String("ref", "HEAD", "ref or sha of the committed tip to check")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	skipBuild := fs.Bool("skip-build", false, "skip `go build ./...` (gofmt-only; much faster)")
	smoke := fs.Bool("smoke", false, "run real-world binary smoke tests against the committed tip")
	if !parseFlags(fs, argv) {
		return 2
	}

	r := resolveRoot(*root)
	if r == "" {
		fmt.Fprintln(stderr, "fak ci-preflight: not in a git repo (or git unavailable)")
		return 2
	}

	tip, err := committedtree.Resolve(r, *ref)
	if err != nil {
		fmt.Fprintf(stderr, "fak ci-preflight: cannot resolve ref %q: %v\n", *ref, err)
		return 2
	}

	// Extract the committed tip to a throwaway checkout so every content check reads the COMMITTED
	// bytes, immune to the peer-dirty working tree. Cleaned up on return.
	dir, err := committedtree.Extract(r, tip)
	if err != nil {
		fmt.Fprintf(stderr, "fak ci-preflight: cannot materialize tip %s: %v\n", short(tip), err)
		return 2
	}
	defer os.RemoveAll(dir)

	res := ciPreflightResult{Ref: *ref, Tip: tip, OK: true}

	// gofmt-check: the exact `gofmt -l .` CI runs, over the committed bytes.
	if files, gerr := gofmtList(dir); gerr != nil {
		res.Skipped = append(res.Skipped, "gofmt")
	} else if len(files) > 0 {
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "gofmt", Files: files})
	}

	// Generated disambiguation index: invoke the committed public writer inside
	// the extracted tip so neither this binary nor peer-dirty files can mask drift.
	detail, checked, ok := checkDisambiguationGenerated(dir)
	recordOptionalCIPreflightCheck(&res, "disambiguation-generated", detail, checked, ok)

	// The self-study inventory is default-on when the committed artifact exists. The check runs
	// against the already-extracted tip, never against the shared working tree.
	detail, checked, ok = checkStudySelfInventory(dir)
	recordOptionalCIPreflightCheck(&res, "study-self-inventory", detail, checked, ok)

	if checked, detail, werr := armbenchWitnessDrift(dir); werr != nil {
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "armbench-witness", Detail: werr.Error()})
	} else if checked && detail != "" {
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "armbench-witness", Detail: detail})
	}
	// build: `go build ./...` — catches the partial-commit `undefined: X` red.
	if *skipBuild {
		res.Skipped = append(res.Skipped, "build")
	} else if detail, ok := goBuildAll(dir); !ok {
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "build", Detail: detail})
	} else {
		committedbuildwitness.Record(r, tip, "ci-preflight", time.Now())
	}

	// smoke: run real-world hermetic CLI smoke tests against the freshly compiled fak binary in the extracted tip
	if *smoke && res.OK {
		if detail, ok := goSmokeCheck(dir); !ok {
			res.OK = false
			res.Failures = append(res.Failures, ciPreflightFailure{Step: "smoke", Detail: detail})
		}
	} else if !*smoke {
		res.Skipped = append(res.Skipped, "smoke")
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
	} else {
		renderCIPreflight(stdout, res)
	}
	if !res.OK {
		return 1
	}
	return 0
}

// gofmtList returns the unformatted .go files (relative to dir) that `gofmt -l .` reports, matching
// the CI gofmt-check step exactly.
func armbenchWitnessDrift(dir string) (bool, string, error) {
	path := filepath.Join(dir, "docs", "_witnesses", "armbench-selfcheck-2026-08-13.json")
	want, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, "", nil
	}
	if err != nil {
		return true, "", fmt.Errorf("read committed armbench witness: %w", err)
	}
	res, err := armbench.Selfcheck()
	if err != nil {
		return true, "", fmt.Errorf("run armbench selfcheck: %w", err)
	}
	got, err := armbench.MarshalSelfcheck(res)
	if err != nil {
		return true, "", fmt.Errorf("marshal armbench selfcheck: %w", err)
	}
	if bytes.Equal(got, want) {
		return true, "", nil
	}
	return true, "committed armbench witness is stale; regenerate with `fak armbench selfcheck --json`", nil
}
func gofmtList(dir string) ([]string, error) {
	cmd := windowgate.Command("gofmt", "-l", ".")
	cmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	var files []string
	for _, ln := range strings.Split(string(out), "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		files = append(files, filepath.ToSlash(ln))
	}
	return files, nil
}

func recordOptionalCIPreflightCheck(res *ciPreflightResult, step string, detail string, checked, ok bool) {
	if !checked {
		res.Skipped = append(res.Skipped, step)
		return
	}
	if !ok {
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: step, Detail: detail})
	}
}

func checkDisambiguationGenerated(dir string) (detail string, checked, ok bool) {
	artifact := filepath.Join(dir, "docs", "generated", "disambiguation-index.json")
	if _, err := os.Stat(artifact); errors.Is(err, os.ErrNotExist) {
		return "", false, true
	} else if err != nil {
		return err.Error(), true, false
	}
	cmd := windowgate.Command("go", ciPreflightDisambiguationArgs()...)
	cmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return strings.TrimSpace(string(out)), true, false
	}
	return "", true, true
}

func checkStudySelfInventory(dir string) (detail string, checked, ok bool) {
	artifact := filepath.Join(dir, filepath.FromSlash(studymonitor.DefaultSelfInventoryPath))
	if _, err := os.Stat(artifact); errors.Is(err, os.ErrNotExist) {
		return "", false, true
	} else if err != nil {
		return err.Error(), true, false
	}
	result, err := studymonitor.VerifySelfInventory(dir, studymonitor.DefaultSelfInventoryPath, "anthony-chaudhary/fak")
	if err != nil {
		return err.Error(), true, false
	}
	if result.OK {
		return "", true, true
	}
	var lines []string
	for _, drift := range result.Drift {
		line := fmt.Sprintf("[%s] %s", drift.Kind, drift.Path)
		if drift.Expected != "" || drift.Actual != "" {
			line += fmt.Sprintf(" expected=%s actual=%s", drift.Expected, drift.Actual)
		}
		lines = append(lines, line)
	}
	lines = append(lines, "refresh explicitly with `fak study-inventory --self --refresh`")
	return strings.Join(lines, "\n"), true, false
}

// goBuildAll runs `go build ./...` in dir. Returns (detail, ok); on failure detail is the trimmed
// compiler output so an agent sees the exact `undefined: X` without re-running anything.
func goBuildAll(dir string) (string, bool) {
	cmd := windowgate.Command("go", ciPreflightBuildArgs()...)
	cmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return strings.TrimSpace(string(out)), false
	}
	return "", true
}

func ciPreflightDisambiguationArgs() []string {
	return []string{"run", "-trimpath", "./cmd/fak", "disambiguation", "generate", "--check", "--json"}
}

func ciPreflightBuildArgs() []string {
	return []string{"build", "-trimpath", "./..."}
}

func renderCIPreflight(w io.Writer, res ciPreflightResult) {
	if res.OK {
		fmt.Fprintf(w, "ci-preflight OK — committed tip %s builds and is gofmt-clean", short(res.Tip))
		if len(res.Skipped) > 0 {
			fmt.Fprintf(w, " (skipped: %s)", strings.Join(res.Skipped, ", "))
		}
		fmt.Fprintln(w)
		return
	}
	fmt.Fprintf(w, "ci-preflight FAILED — committed tip %s is red:\n", short(res.Tip))
	for _, f := range res.Failures {
		switch f.Step {
		case "gofmt":
			fmt.Fprintf(w, "  [gofmt-check] %d file(s) not formatted (run `gofmt -w`):\n", len(f.Files))
			for _, file := range f.Files {
				fmt.Fprintf(w, "    %s\n", file)
			}
		case "disambiguation-generated":
			fmt.Fprintln(w, "  [disambiguation-generated] tracked index is stale (run `fak disambiguation generate`):")
			for _, ln := range strings.Split(f.Detail, "\n") {
				fmt.Fprintf(w, "    %s\n", ln)
			}
		case "study-self-inventory":
			fmt.Fprintln(w, "  [study-self-inventory] committed self inventory is stale:")
			for _, ln := range strings.Split(f.Detail, "\n") {
				fmt.Fprintf(w, "    %s\n", ln)
			}
		case "build":
			fmt.Fprintln(w, "  [build] `go build ./...` failed:")
			for _, ln := range strings.Split(f.Detail, "\n") {
				fmt.Fprintf(w, "    %s\n", ln)
			}
		case "smoke":
			fmt.Fprintln(w, "  [smoke] real-world CLI smoke test failed:")
			for _, ln := range strings.Split(f.Detail, "\n") {
				fmt.Fprintf(w, "    %s\n", ln)
			}
		}
	}
}

func goSmokeCheck(dir string) (string, bool) {
	smokeBin := "fak_ci_smoke"
	if runtime.GOOS == "windows" {
		smokeBin = "fak_ci_smoke.exe"
	}
	smokePath := filepath.Join(dir, smokeBin)
	defer os.Remove(smokePath)

	buildCmd := windowgate.Command("go", "build", "-o", smokeBin, "./cmd/fak")
	buildCmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(buildCmd)
	if out, err := buildCmd.CombinedOutput(); err != nil {
		return fmt.Sprintf("compile smoke binary: %v: %s", err, strings.TrimSpace(string(out))), false
	}

	// 1. Version check
	verCmd := windowgate.Command(smokePath, "version")
	verCmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(verCmd)
	if out, err := verCmd.CombinedOutput(); err != nil {
		return fmt.Sprintf("smoke 'fak version': %v: %s", err, strings.TrimSpace(string(out))), false
	}

	// 2. Preflight DENY check
	denyCmd := windowgate.Command(smokePath, "preflight", "--policy", "examples/customer-support-readonly-policy.json", "--tool", "refund_payment", "--args", "{}")
	denyCmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(denyCmd)
	denyOut, err := denyCmd.CombinedOutput()
	if err != nil || !strings.Contains(string(denyOut), "verdict=DENY") {
		return fmt.Sprintf("smoke 'fak preflight' DENY failed: %v: %s", err, strings.TrimSpace(string(denyOut))), false
	}

	// 3. Preflight ALLOW check
	allowCmd := windowgate.Command(smokePath, "preflight", "--policy", "examples/customer-support-readonly-policy.json", "--tool", "search_kb", "--args", "{}")
	allowCmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(allowCmd)
	allowOut, err := allowCmd.CombinedOutput()
	if err != nil || !strings.Contains(string(allowOut), "verdict=ALLOW") {
		return fmt.Sprintf("smoke 'fak preflight' ALLOW failed: %v: %s", err, strings.TrimSpace(string(allowOut))), false
	}

	return "", true
}
