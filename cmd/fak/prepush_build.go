package main

import (
	"archive/tar"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// cmd/fak/prepush_build.go — `fak hooks pre-push`: the push-seam compile gate
// TRUNK_WOULD_NOT_COMPILE. Answer "would the commits this push ADDS to the trunk still
// compile for every OTHER worker's build graph?" WITHOUT trusting the peer-dirty working
// tree.
//
// Why this exists (the gap it closes). `dos arbitrate` gives workers DISJOINT FILE leases,
// but Go compiles at PACKAGE granularity and the whole module shares one build graph. So two
// workers on the shared trunk holding disjoint file leases — even in different packages — can
// still red each other's `go build`: a caller lands referencing a symbol whose callee edit
// is not yet committed (`undefined: X`), or a callee removes an exported symbol an importer
// still calls. Nothing at the commit or push seam builds anything today (the git hooks and
// `fak commit` run only content/structural gates; `make ci`'s `go build ./...` is CI/manual),
// so such a break lands on origin and poisons every peer that fetches — the witnessed #1338
// class (docs/notes/SHARED-TRUNK-COMPILE-INTEGRITY-GAP-2026-07-06.md). This verb is that
// note's "Rung B": at the push boundary, build the pushed state in ISOLATION and refuse if
// it would not compile.
//
// How (reuses the ci-preflight / affected machinery, nothing new invented):
//   - resolve the range this push adds (origin/<branch> → origin/main → origin/master), take
//     its committed changed .go files via `git diff --name-only <base>...HEAD` (three-dot:
//     the merge-base range = "commits this push adds", NEVER the dirty working tree);
//   - materialize the committed tip in a throwaway checkout (`git archive HEAD` untarred
//     IN-PROCESS via extractArchive), immune to peer WIP, under a hard deadline so a stalled
//     `git archive` can never wedge the push (the #3432 fix — see prepushArchiveTimeout);
//   - build the import graph THERE (goListGraph in the archive dir) and select the changed
//     packages PLUS their importer closure (affectedtests.Select) — the importer build is what
//     surfaces the cross-package symbol break;
//   - `go build <selected>` natively in the archive dir (native go build works on win32;
//     only `go test` exec is OS-blocked).
//
// This verb is a DETECTOR, mode-agnostic like `fak hygiene` behind the TIER_DECLARED rung:
// exit 0 = clean/NOOP, 1 = TRUNK_WOULD_NOT_COMPILE, 2 = could-not-run (fail-open). The
// tools/githooks/pre-push shell owns the FLEET_BUILD_GUARD block|warn|off mode and the
// FLEET_ALLOW_BUILD_BREAK one-shot escape — one place, mirroring the other push-seam gates.
// A slow-but-GREEN build is never a block: a build over FLEET_BUILD_BUDGET reports
// GATE_LATENCY_REGRESSION but still exits 0 (penalizing a push for touching a popular leaf
// would be worse than the latency).
//
// Honest limit: this gates a single clone's push and covers mutation/addition breaks (the
// #1338 shape). A change that ONLY deletes a file with no other edit in its package, or two
// clones racing the same trunk, is narrowed but not fully closed — `make ci` on the merged
// tree stays the authoritative oracle.
//
//	fak hooks pre-push               # human summary; exit 1 iff the pushed tip would not build
//	fak hooks pre-push --json        # machine result (schema fak.trunk_build.v1)
//	fak hooks pre-push --base R      # override the base ref the range is computed against
//	fak hooks pre-push --budget 60s  # GATE_LATENCY_REGRESSION advisory over this build time
//	fak hooks pre-push --advisory    # warn-mode single-flight: SKIPPED_CONTENDED (exit 0) when a
//	                                 # peer build is already running on this host (skip == allow)

// Seams (overridable in tests) over the impure git/go steps. Helpers are defined LOCAL to this
// file (prepushGitRevParse / prepushArchiveTip) rather than borrowed from ci_preflight.go so the
// gate stays self-contained on the shared trunk — it must not depend on a sibling file that may
// be a peer's uncommitted WIP (the very build-integrity hazard this gate exists to catch).
var (
	prepushRevParse     = prepushGitRevParse
	prepushResolveBase  = resolvePrepushBase
	prepushChangedFiles = gitChangedGoFilesRange
	prepushExtractTip   = prepushArchiveTip
	prepushListGraph    = goListGraph
	prepushBuild        = goBuildPackages
	prepushNow          = time.Now
)

// prepushBaselineTolerance arms the pre-existing-red attribution (#3618): when the tip's cone
// fails to build, re-build each failing package against the base trunk to tell a peer's already-
// published red (TRUNK_ALREADY_RED, allow) from a break THIS push introduced (TRUNK_WOULD_NOT_
// COMPILE, refuse). Default on — it only ever RELAXES a would-be block and is fail-safe (any
// package it cannot PROVE was already red on the base stays a regression). Off restores the
// pre-#3618 whole-tip verdict. A seam so a test can force the legacy path; operators keep the
// FLEET_BUILD_GUARD block/warn/off dial at the shell.
var prepushBaselineTolerance = envFlagDefault("FLEET_BUILD_BASELINE_TOLERANCE", true)

// buildPkgHeaderRE matches the `# <import-path>` header `go build`/`go vet` prints before a
// package's own compile diagnostics. A package that fails to LOAD (absent, no non-test files, an
// unresolved import path) errors WITHOUT this header, so it never counts as a compile failure of
// the base — the fail-safe direction: an unprovable pre-existing red stays this push's regression.
var buildPkgHeaderRE = regexp.MustCompile(`(?m)^#\s+(\S+)`)

// failingPackagesFromBuild extracts the import paths of packages that produced COMPILE errors from
// `go build` output, deduped and sorted. Pure.
func failingPackagesFromBuild(out string) []string {
	seen := map[string]bool{}
	var pkgs []string
	for _, m := range buildPkgHeaderRE.FindAllStringSubmatch(out, -1) {
		p := m[1]
		if !seen[p] {
			seen[p] = true
			pkgs = append(pkgs, p)
		}
	}
	sort.Strings(pkgs)
	return pkgs
}

// envFlagDefault reads an on/off env var, returning def when unset/unrecognized.
func envFlagDefault(name string, def bool) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "0", "false", "no", "off":
		return false
	case "1", "true", "yes", "on":
		return true
	default:
		return def
	}
}

// prepushArchiveTimeout bounds the whole-repo `git archive` + in-process untar (extractArchive).
// The gate is fail-open BY DESIGN (see the file header), but the archive step had NO deadline, so
// a `git archive` that stalled — the witnessed #3432 wedge: it blocked writing to a full stdout
// pipe that a dead/early-exiting external `tar` stopped draining — hung every agent's trunk push
// for 14+ hours at ~0 CPU with no bound. A deadline turns that silent forever-wedge into a bounded
// COULD_NOT_RUN (exit 2 → push allowed), honoring the fail-open contract. A seam so a test can
// shrink it. (goBuildPackages is deliberately left UNbounded: its (detail, ok) contract maps a
// false to TRUNK_WOULD_NOT_COMPILE — a hard block — so a build timeout there would false-block a
// slow-but-correct push, the opposite of fail-open. Bounding it needs a separate fail-open state.)
var prepushArchiveTimeout = 2 * time.Minute

// Host-wide advisory single-flight for the expensive trunk-build gate.
//
// The gate's whole-repo `git archive` + in-process untar + `go list` + importer-cone `go build` is
// the heaviest push-seam step, and in the default FLEET_BUILD_GUARD=warn mode its verdict is
// ADVISORY — it never blocks the push (warn = allow). When many peers push at once, running
// that full build concurrently in every clone reproduces the SAME advisory N times while piling
// onto the CPU/disk/git contention that is what actually slows the fleet. So under advisory mode
// exactly one gate builds and the rest report SKIPPED_CONTENDED (exit 0 = allowed, IDENTICAL to
// warn's own outcome — no verdict is lost that warn would have acted on). The marker lives in the
// host TempDir because the contention is host-wide, not per-clone. O_EXCL makes the winner unique;
// a marker older than prepushBuildSlotStale (a crashed gate) is stolen so the check can never be
// wedged off. Block mode (advisory=false) NEVER skips — it always builds, so a hard-enforced push
// is never let through unbuilt.
var (
	prepushBuildSlotPath    = filepath.Join(os.TempDir(), "fak-prepush-build.slot")
	prepushBuildSlotStale   = 3 * time.Minute
	prepushAcquireBuildSlot = tryAcquireBuildSlot

	prepushSlotStat   = os.Stat
	prepushSlotRemove = os.Remove
	prepushSlotNow    = time.Now
	prepushSlotCreate = func(path string) error {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		fmt.Fprintf(f, "pid=%d\n", os.Getpid())
		return f.Close()
	}
)

// tryAcquireBuildSlot decides whether THIS gate runs the expensive build now.
//   - advisory=false (FLEET_BUILD_GUARD=block): always run — a blocking gate is never skipped, so
//     a hard-enforced push is never allowed through unbuilt. The marker seams are not touched.
//   - advisory=true (warn, the default): non-blocking single-flight. run=true iff this process wins
//     the O_EXCL marker (no fresh peer build in flight); otherwise run=false and the caller reports
//     SKIPPED_CONTENDED. A marker older than prepushBuildSlotStale is stolen first so a crashed gate
//     cannot disable the check permanently.
//
// The returned release removes the marker; it is a no-op when run=false or in block mode.
func tryAcquireBuildSlot(advisory bool) (run bool, release func()) {
	noop := func() {}
	if !advisory {
		return true, noop
	}
	if fi, err := prepushSlotStat(prepushBuildSlotPath); err == nil {
		if prepushSlotNow().Sub(fi.ModTime()) >= prepushBuildSlotStale {
			_ = prepushSlotRemove(prepushBuildSlotPath) // stale holder (crashed gate) → steal it
		}
	}
	if err := prepushSlotCreate(prepushBuildSlotPath); err != nil {
		return false, noop // a peer holds a fresh marker → skip the redundant advisory build
	}
	return true, func() { _ = prepushSlotRemove(prepushBuildSlotPath) }
}

// trunkBuildResult is the JSON-stable verdict for agents / the shell rung.
type trunkBuildResult struct {
	Schema           string   `json:"schema"`           // fak.trunk_build.v1
	Reason           string   `json:"reason,omitempty"` // "TRUNK_WOULD_NOT_COMPILE" | "" (empty when it builds/NOOP)
	OK               bool     `json:"ok"`               // true iff the push may proceed on build grounds
	Ref              string   `json:"ref"`              // resolved HEAD sha (the pushed tip)
	Base             string   `json:"base"`             // the base ref the range was computed against
	ChangedPackages  []string `json:"changed_packages"`
	SelectedPackages []string `json:"selected_packages"` // changed + importer closure — what was built
	Detail           string   `json:"detail,omitempty"`  // trimmed `go build` output on a break
	ElapsedMS        int64    `json:"elapsed_ms"`
	Verdict          string   `json:"verdict"` // OK | NOOP | TRUNK_WOULD_NOT_COMPILE | TRUNK_ALREADY_RED | GATE_LATENCY_REGRESSION | COULD_NOT_RUN | SKIPPED_CONTENDED
	// Pre-existing-red tolerance (#3618). When the tip's cone fails to build, each failing
	// package is re-built against the base trunk (origin/main) to attribute the break. BaseSha is
	// the resolved base commit built against; PreExistingRed are packages red at BOTH tip and base
	// (a peer's already-published break — not this push); Regressions are packages that build at
	// base but fail at the tip (introduced by this push). A failure with only PreExistingRed and no
	// Regressions is TRUNK_ALREADY_RED (exit 0, push allowed): a clean delta must not be false-
	// blocked by a peer's red trunk.
	BaseSha        string   `json:"base_sha,omitempty"`
	PreExistingRed []string `json:"pre_existing_red,omitempty"`
	Regressions    []string `json:"regressions,omitempty"`
}

func runHooksPrePush(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hooks pre-push", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit the result as JSON (schema fak.trunk_build.v1)")
	base := fs.String("base", "", "override the base ref the push range is computed against (default: origin/<branch>→origin/main→origin/master)")
	tip := fs.String("tip", "", "exact local object SHA supplied by pre-push stdin (default: HEAD for manual invocation)")
	budget := fs.Duration("budget", 60*time.Second, "report GATE_LATENCY_REGRESSION if the build exceeds this (still exits 0 when green)")
	report := fs.String("report", "", "write the JSON result to this path in addition to stdout")
	advisory := fs.Bool("advisory", false, "advisory mode (FLEET_BUILD_GUARD=warn): single-flight the build — skip with SKIPPED_CONTENDED when a peer build is already running on this host, rather than run a redundant concurrent full build (skip == push allowed)")
	if !parseFlags(fs, argv) {
		return 2
	}

	r := resolveRoot(*root)
	if r == "" {
		// could-not-run: not in a repo (or git unavailable) → fail open so the shell allows.
		fmt.Fprintln(stderr, "fak hooks pre-push: not in a git repo (or git unavailable); build gate skipped")
		return 2
	}

	res, code := evaluatePrePushBuildAt(r, *base, *tip, *budget, *advisory)
	// Repeat the earliest commit admission decision over immutable base..tip
	// objects. This protects direct pushes and is the CI-consumed committed-diff seam.
	if res.BaseSha != "" && res.Ref != "" {
		if d, err := hooks.ReadRangeDiff(r, res.BaseSha, res.Ref); err == nil {
			if findings, err := hooks.CheckConceptAdmission(d); err == nil && len(findings) > 0 {
				for _, f := range findings {
					fmt.Fprintf(stderr, "CONCEPT_ADMISSION %s:%d: %s\n", f.File, f.Line, f.Detail)
				}
				if code == 0 {
					code = 1
				}
				res.Verdict = "CONCEPT_ADMISSION"
				res.Detail = findings[0].Detail
			}
		}
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
	} else {
		renderPrePushBuild(stdout, res)
	}
	// Best-effort fleet witness for a pre-existing trunk red this push was admitted over
	// (#3618 TRUNK_ALREADY_RED): fold this clone onto the shared class so `fak trunk-red`
	// shows the whole fleet stuck on ONE break instead of each clone re-discovering it. To
	// stderr so --json stdout stays pure; fail-open, never changes the push decision.
	if res.Verdict == "TRUNK_ALREADY_RED" {
		w := emitTrunkRedWitness(stderr, r, "pre-push", res.BaseSha, res.PreExistingRed, extractUndefinedSymbol(res.Detail))
		fmt.Fprint(stderr, trunkRedWitnessNote(w))
	}
	// Test-quality is advisory at the push seam: the baseline absorbs existing debt,
	// and only growth is surfaced. Never turn scanner failure into an unrelated refusal.
	if tqCode := runTestQuality(r, io.Discard, stderr, nil); tqCode != 0 {
		fmt.Fprintln(stderr, "fak hooks pre-push: WARNING: test-quality ratchet reported growth or could not run (advisory)")
	}
	if *report != "" {
		if err := writeIndentedJSONFile(*report, res); err != nil {
			fmt.Fprintf(stderr, "fak hooks pre-push: write report: %v\n", err)
		}
	}
	return code
}

// evaluatePrePushBuild runs the gate over repo r and returns the verdict plus the DETECTOR
// exit code: 0 clean/NOOP/latency-advisory/skipped, 1 TRUNK_WOULD_NOT_COMPILE, 2 could-not-run
// (fail-open). Mode (block/warn/off) is the shell's job, not this function's, but `advisory`
// (true iff the shell is in FLEET_BUILD_GUARD=warn) lets the gate single-flight under host
// contention: when a peer's build is already running, an advisory invocation returns
// SKIPPED_CONTENDED instead of running a redundant concurrent full build. Block mode
// (advisory=false) never skips.
func evaluatePrePushBuild(r, baseOverride string, budget time.Duration, advisory bool) (trunkBuildResult, int) {
	return evaluatePrePushBuildAt(r, baseOverride, "", budget, advisory)
}

func evaluatePrePushBuildAt(r, baseOverride, tipOverride string, budget time.Duration, advisory bool) (trunkBuildResult, int) {
	res := trunkBuildResult{Schema: "fak.trunk_build.v1"}

	tipRef := strings.TrimSpace(tipOverride)
	if tipRef == "" {
		tipRef = "HEAD"
	}
	tip, err := prepushRevParse(r, tipRef)
	if err != nil {
		res.Verdict, res.Detail = "COULD_NOT_RUN", fmt.Sprintf("cannot resolve pushed tip %s: %v", tipRef, err)
		return res, 2
	}
	res.Ref = tip

	base := baseOverride
	if strings.TrimSpace(base) == "" {
		base = prepushResolveBase(r)
	}
	res.Base = base

	changed, err := prepushChangedFiles(r, base, tip)
	if err != nil {
		res.Verdict, res.Detail = "COULD_NOT_RUN", fmt.Sprintf("cannot diff %s...%s: %v", base, tip, err)
		return res, 2
	}
	if len(changed) == 0 {
		// No committed .go delta in the range → nothing this push can break at build time.
		res.OK, res.Verdict = true, "NOOP"
		return res, 0
	}

	// A real Go delta needs the EXPENSIVE steps below (archive tip + go list + importer-cone
	// build). In advisory mode, single-flight them: if a peer's build is already running on this
	// host, skip rather than pile a redundant concurrent full build onto the contention. Skip ==
	// warn's own outcome (exit 0, push allowed), so no verdict warn would have enforced is lost.
	// Block mode always runs (advisory=false → prepushAcquireBuildSlot returns run=true).
	run, releaseSlot := prepushAcquireBuildSlot(advisory)
	if !run {
		res.OK, res.Verdict = true, "SKIPPED_CONTENDED"
		return res, 0
	}
	defer releaseSlot()

	dir, err := prepushExtractTip(r, tip)
	if err != nil {
		res.Verdict, res.Detail = "COULD_NOT_RUN", fmt.Sprintf("cannot materialize tip %s: %v", short(tip), err)
		return res, 2
	}
	defer os.RemoveAll(dir)

	fileToPkg, edges, _, err := prepushListGraph(dir)
	if err != nil {
		res.Verdict, res.Detail = "COULD_NOT_RUN", fmt.Sprintf("go list in archive tip: %v", err)
		return res, 2
	}

	res.ChangedPackages = affectedtests.ChangedPackages(fileToPkg, changed)
	res.SelectedPackages = affectedtests.Select(edges, res.ChangedPackages)
	if len(res.SelectedPackages) == 0 {
		// Changed .go files map to no package in the pushed tip (e.g. an only-deletion or a
		// non-package file) → nothing to build. See the honest-limit note in the file header.
		res.OK, res.Verdict = true, "NOOP"
		return res, 0
	}

	start := prepushNow()
	detail, ok := prepushBuild(dir, res.SelectedPackages)
	res.ElapsedMS = prepushNow().Sub(start).Milliseconds()
	if !ok {
		// The tip's cone did not build. Attribute the break before blaming this push: a package
		// red at BOTH the tip AND the base trunk is a peer's already-published break, not this
		// delta (#3618). Only a package that builds at the base but fails at the tip is refused.
		res.Detail = detail
		return res, resolveTipBuildFailure(r, base, &res)
	}

	res.OK = true
	if budget > 0 && time.Duration(res.ElapsedMS)*time.Millisecond > budget {
		// Green but slow: advise, never block. A popular leaf's importer cone is large by
		// nature; refusing the push for a slow-but-correct build would be worse than the wait.
		res.Verdict = "GATE_LATENCY_REGRESSION"
		return res, 0
	}
	res.Verdict = "OK"
	return res, 0
}

// resolveTipBuildFailure attributes a tip-cone build failure to either this push or a peer (#3618),
// mutating res and returning the exit code. It is fail-safe and only ever RELAXES a block:
//
//   - baseline tolerance off, base unresolvable/unarchivable, or no compile-failing package parsed
//     from the tip output → the conservative legacy verdict TRUNK_WOULD_NOT_COMPILE (exit 1);
//   - otherwise each failing package is re-built against the BASE trunk (origin/main). A package
//     that builds at the base but failed at the tip is a REGRESSION this push introduced; one that
//     also fails to COMPILE at the base (a `# pkg` header there) is a peer's PRE-EXISTING red; one
//     that fails at the base without a compile header (absent/new/load error) stays a regression —
//     never excuse a break we cannot PROVE pre-existed;
//   - zero regressions ⇒ the whole failure pre-existed on the base ⇒ TRUNK_ALREADY_RED (exit 0,
//     push allowed): a clean delta must not be false-blocked by a peer's red trunk.
func resolveTipBuildFailure(r, base string, res *trunkBuildResult) int {
	block := func() int {
		res.OK, res.Reason, res.Verdict = false, "TRUNK_WOULD_NOT_COMPILE", "TRUNK_WOULD_NOT_COMPILE"
		return 1
	}
	tipFailures := failingPackagesFromBuild(res.Detail)
	if !prepushBaselineTolerance || len(tipFailures) == 0 {
		return block()
	}
	baseSha, err := prepushRevParse(r, base)
	if err != nil || strings.TrimSpace(baseSha) == "" {
		return block()
	}
	res.BaseSha = baseSha
	baseDir, err := prepushExtractTip(r, baseSha)
	if err != nil {
		return block()
	}
	defer os.RemoveAll(baseDir)

	var regressions, preExisting []string
	for _, pkg := range tipFailures {
		d, okp := prepushBuild(baseDir, []string{pkg})
		switch {
		case okp:
			regressions = append(regressions, pkg) // green at base, red at tip → this push broke it
		case len(failingPackagesFromBuild(d)) > 0:
			preExisting = append(preExisting, pkg) // genuine compile-red at base too → a peer's red
		default:
			regressions = append(regressions, pkg) // absent/new/load-error at base → fail-safe: this push's
		}
	}
	sort.Strings(regressions)
	sort.Strings(preExisting)
	res.Regressions = regressions
	res.PreExistingRed = preExisting

	if len(regressions) == 0 {
		res.OK, res.Reason, res.Verdict = true, "", "TRUNK_ALREADY_RED"
		return 0
	}
	return block()
}

// resolvePrepushBase resolves the base ref the push range is measured from, mirroring the
// tools/githooks/pre-push ladder: this branch's own upstream, then origin/main, then the
// legacy origin/master, then a bounded HEAD~20 fallback for a fresh/detached clone.
func resolvePrepushBase(r string) string {
	branch, err := gitOut(r, "rev-parse", "--abbrev-ref", "HEAD")
	branch = strings.TrimSpace(branch)
	candidates := []string{}
	if err == nil && branch != "" && branch != "HEAD" {
		candidates = append(candidates, "origin/"+branch)
	}
	candidates = append(candidates, "origin/main", "origin/master")
	for _, ref := range candidates {
		if _, verr := gitOut(r, "rev-parse", "--verify", "--quiet", ref); verr == nil {
			return ref
		}
	}
	return "HEAD~20"
}

// gitChangedGoFilesRange returns the repo-relative, slash-separated .go files the commits in
// base..HEAD add or change, via three-dot `git diff` (merge-base range = exactly what the
// push adds), reading COMMITTED bytes only — never the peer-dirty working tree.
func gitChangedGoFilesRange(r, base, tip string) ([]string, error) {
	return gitChangedFilesRange(r, base, tip, ".go")
}

// gitChangedFilesRange is the general form of gitChangedGoFilesRange: it returns the
// repo-relative, slash-separated files the commits in base...tip add or change whose name
// ends in ANY of suffixes, via three-dot `git diff` (merge-base range = exactly what the
// push adds), reading COMMITTED bytes only — never the peer-dirty working tree. Passing no
// suffixes returns every changed path. Shared by the .go-only prepush build gate and the
// .ps1/.py/.go pre-push popup gate (#5145) so the range walk lives in exactly one place.
func gitChangedFilesRange(r, base, tip string, suffixes ...string) ([]string, error) {
	out, err := gitOut(r, "diff", "--name-only", base+"..."+tip)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		if len(suffixes) > 0 && !hasAnySuffix(ln, suffixes) {
			continue
		}
		files = append(files, filepath.ToSlash(ln))
	}
	sort.Strings(files)
	return files, nil
}

// hasAnySuffix reports whether s ends in any of the given suffixes.
func hasAnySuffix(s string, suffixes []string) bool {
	for _, suf := range suffixes {
		if strings.HasSuffix(s, suf) {
			return true
		}
	}
	return false
}

// prepushGitRevParse resolves ref to a full sha in repo r via the committed `gitOut` helper.
func prepushGitRevParse(r, ref string) (string, error) {
	out, err := gitOut(r, "rev-parse", ref)
	return strings.TrimSpace(out), err
}

// prepushArchiveTip archives sha from repo r into a fresh temp dir (committed bytes only — `git
// archive` never reads the working tree or index) and returns its path. A local twin of
// ci_preflight's extractCommittedTip, kept here so the gate carries no cross-file dependency.
//
// Deadlock-proofing (#3432): the archive+extract runs under prepushArchiveTimeout, so `git
// archive` can no longer wedge the push forever — the deadline kills it and the gate fails open.
// On any error the temp dir is removed so a partial extraction never leaks.
func prepushArchiveTip(r, sha string) (string, error) {
	dir, err := os.MkdirTemp("", "fak-prepush-build-*")
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), prepushArchiveTimeout)
	defer cancel()
	if err := extractArchive(ctx, r, sha, dir); err != nil {
		os.RemoveAll(dir)
		return "", err
	}
	return dir, nil
}

// extractArchive streams `git archive <sha>` from repo r and untars it into dir IN-PROCESS (Go's
// archive/tar), under ctx. This is the load-bearing #3432 fix, and it closes the wedge in two
// ways at once:
//
//   - No external `tar`. The old pipe shelled out to `tar -x`; on this host the first `tar` on
//     PATH is MSYS GNU tar, which cannot open a native Windows `-C` path and exits immediately.
//     A consumer that dies early stops draining, so `git archive` blocks writing to a full 64 KiB
//     pipe at ~0 CPU — the witnessed 14-hour wedge (a small archive that fits the buffer merely
//     surfaces the error; a real repo tip hangs). Draining the stream in-process removes both the
//     tar-flavor fragility AND the "nobody drains the producer" deadlock class entirely, on every
//     OS.
//   - Hard deadline. Even a `git archive` that stalls on its own is now bounded: ctx expiry kills
//     it and we return a diagnostic error, so the gate fails open (COULD_NOT_RUN, push allowed)
//     rather than hanging. Factored out so the no-wedge contract is unit-testable with a
//     pre-expired deadline.
//
// The producer's stdout is always read to EOF (or it is killed) before Wait, so Wait never races a
// live read. git-only: no `tar` binary is required to push anymore.
// prepushArchiveCommand builds the `git archive` producer. It is a seam so a test can substitute a
// controllable producer (one that streams valid data then stalls, or emits garbage) and thereby
// witness the load-bearing #3432 property deterministically: a LIVE producer that stalls is killed
// — by the deadline (the ctx.Err() branch) or by an untar error (the Process.Kill() branch) — so
// extractArchive always returns bounded and the gate fails open, never wedging the push.
var prepushArchiveCommand = func(ctx context.Context, r, sha string) *exec.Cmd {
	return exec.CommandContext(ctx, "git", "-C", r, "archive", "--format=tar", sha)
}

func extractArchive(ctx context.Context, r, sha, dir string) error {
	ar := prepushArchiveCommand(ctx, r, sha)
	windowgate.ConfigureBackgroundCommand(ar)
	stdout, err := ar.StdoutPipe()
	if err != nil {
		return err
	}
	var stderr strings.Builder
	ar.Stderr = &stderr

	if err := ar.Start(); err != nil {
		return fmt.Errorf("git archive start: %w", err)
	}
	// Drain and extract in-process — this NEVER lets the producer block on a full pipe.
	untarErr := untarInto(stdout, dir)
	if untarErr != nil {
		// Extraction bailed before EOF; the producer may still be writing, so kill it rather than
		// let ar.Wait() block on a stream nobody is reading.
		_ = ar.Process.Kill()
	}
	// Drain to EOF before Wait: tar.Reader.Next() stops at the logical end-of-archive marker, but
	// `git archive` writes record padding AFTER it. Leaving that in the pipe lets git block on the
	// write and never exit, so ar.Wait() would hang forever — the same #3432 wedge in a new guise.
	// Reading to EOF also satisfies StdoutPipe's "all reads complete before Wait" contract.
	_, _ = io.Copy(io.Discard, stdout)
	waitErr := ar.Wait()

	switch {
	case ctx.Err() != nil:
		return fmt.Errorf("archive timed out after %s (failing open): %w", prepushArchiveTimeout, ctx.Err())
	case untarErr != nil:
		return fmt.Errorf("untar archive: %w", untarErr)
	case waitErr != nil:
		return fmt.Errorf("git archive: %w (%s)", waitErr, strings.TrimSpace(stderr.String()))
	}
	return nil
}

// untarInto extracts a `git archive --format=tar` stream into dir using Go's archive/tar — no
// external `tar` binary, so it is immune to the MSYS-vs-bsdtar path ambiguity that triggered the
// #3432 wedge on Windows. It materializes directories and regular files (all a `go build`/`go
// list` of the tip needs); pax global headers and non-regular entries (symlinks — absent from this
// Go source tree) are skipped. Each entry name is confined under dir to refuse any `../` traversal.
func untarInto(r io.Reader, dir string) error {
	tr := tar.NewReader(r)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		target, err := safeArchiveJoin(dir, hdr.Name)
		if err != nil {
			return err
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.FileMode(hdr.Mode)&0o777|0o600)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				_ = f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
		default:
			// pax_global_header, symlinks, etc. — nothing a compile check of a Go tip requires.
			continue
		}
	}
}

// safeArchiveJoin joins a tar entry name under root and REFUSES any name that would escape it —
// an absolute path, a Windows volume-relative name (`C:..\..\x`), or one whose cleaned form climbs
// past root via `..`. A defensive guard: `git archive` emits clean repo-relative names, so any such
// entry is anomalous and a build gate must reject it loudly rather than write outside its throwaway
// dir.
func safeArchiveJoin(root, name string) (string, error) {
	rel := filepath.Clean(filepath.FromSlash(name))
	// Refuse an absolute path, a `..` climb past root, OR a Windows volume-relative name such as
	// `C:..\..\x`: filepath.IsAbs reports false for the drive-relative form (it has a volume but no
	// leading separator), yet filepath.Join(root, `C:..`) honors the drive and escapes root. A bare
	// VolumeName check closes that gap — git archive never emits a volume-prefixed name, so this
	// only ever fires on an anomalous entry the gate must reject loudly.
	if filepath.IsAbs(rel) || filepath.VolumeName(rel) != "" || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("archive entry %q escapes extraction root", name)
	}
	return filepath.Join(root, rel), nil
}

// goBuildPackages runs `go build <pkgs>` in dir (the archive tip). Returns (detail, ok); on
// failure detail is the trimmed compiler output so the exact `undefined: X` is visible without
// re-running anything. The package-list generalization of ci_preflight's goBuildAll("./...").
func goBuildPackages(dir string, pkgs []string) (string, bool) {
	cmd := exec.Command("go", append([]string{"build"}, pkgs...)...)
	cmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return strings.TrimSpace(string(out)), false
	}
	return "", true
}

func renderPrePushBuild(w io.Writer, res trunkBuildResult) {
	switch res.Verdict {
	case "NOOP":
		fmt.Fprintln(w, "trunk-build-gate: NOOP — the push adds no Go package changes to build")
	case "OK":
		fmt.Fprintf(w, "trunk-build-gate OK — pushed tip %s builds (%d package(s) checked, %dms)\n",
			short(res.Ref), len(res.SelectedPackages), res.ElapsedMS)
	case "GATE_LATENCY_REGRESSION":
		fmt.Fprintf(w, "trunk-build-gate OK (slow) — pushed tip %s builds but took %dms over budget (%d package(s))\n",
			short(res.Ref), res.ElapsedMS, len(res.SelectedPackages))
	case "SKIPPED_CONTENDED":
		fmt.Fprintln(w, "trunk-build-gate: SKIPPED_CONTENDED — a peer build is already running on this host; redundant advisory build skipped (push allowed)")
	case "COULD_NOT_RUN":
		fmt.Fprintf(w, "trunk-build-gate could-not-run (push allowed): %s\n", res.Detail)
	case "TRUNK_ALREADY_RED":
		fmt.Fprintf(w, "trunk-build-gate OK — pushed tip %s builds YOUR delta; the trunk is ALREADY red on base %s at %s from earlier commit(s), not this push (#3618). Push allowed; the peer red still needs fixing at its source.\n",
			short(res.Ref), short(res.BaseSha), strings.Join(res.PreExistingRed, " "))
	case "TRUNK_WOULD_NOT_COMPILE":
		fmt.Fprintf(w, "TRUNK_WOULD_NOT_COMPILE — pushed tip %s would not build for peers:\n", short(res.Ref))
		for _, ln := range strings.Split(res.Detail, "\n") {
			fmt.Fprintf(w, "    %s\n", ln)
		}
		if len(res.Regressions) > 0 {
			fmt.Fprintf(w, "  YOUR delta regressed: %s\n", strings.Join(res.Regressions, " "))
		}
		if len(res.PreExistingRed) > 0 {
			fmt.Fprintf(w, "  (already red on base %s, not yours: %s)\n", short(res.BaseSha), strings.Join(res.PreExistingRed, " "))
		}
		fmt.Fprintf(w, "  built: %s\n", strings.Join(res.SelectedPackages, " "))
	}
}
