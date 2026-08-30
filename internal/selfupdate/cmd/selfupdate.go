package selfupdatecmd

import (
	"context"
	"crypto/rand"
	"debug/buildinfo"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
	"github.com/anthony-chaudhary/fak/internal/selfupdate"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

// cmdSelfUpdate — `fak self-update`: converge THIS install on the latest VERIFIED fak.
//
// It is the durable answer to "the running fak/guard is stale": instead of waiting for a
// crash-and-relaunch (which on its own changes nothing, because relaunch re-execs the SAME
// stale file), a watchdog tick — or an operator — runs this. It:
//  1. reads the repo HEAD and compares it to THIS binary's embedded VCS revision
//     (binstamp): nothing to do unless the binary is provably Stale;
//  2. rebuilds fak, then GATES the candidate (vet + a `version` smoke run) and only then
//     atomically swaps it over this binary's path (selfinstall). A tree that is not green
//     is never installed.
//
// --check reports the freshness verdict and exits without building. --force installs even
// when freshness is Unknown/Fresh (e.g. to pick up an uncommitted local build) but STILL
// runs the full gate — force bypasses the staleness check, never the green gate.
func Run(argv []string) {
	fs := flag.NewFlagSet("self-update", flag.ExitOnError)
	verbFlagUsage(fs, "self-update") // #2232: overview verb -> deep help above the flag dump
	check := fs.Bool("check", false, "report whether this binary is stale vs HEAD and exit (no build)")
	manifestURL := fs.String("manifest-url", "", "opt in to signed conditional update selection from this HTTPS endpoint")
	manifestID := fs.String("manifest-id", "fak-stable", "expected signed manifest identity")
	manifestStatePath := fs.String("manifest-cache", defaultSelfUpdateManifestStatePath(), "authenticated manifest cache path")
	manifestChannel := fs.String("manifest-channel", "stable", "signed manifest channel identity")
	manifestCohort := fs.String("manifest-cohort", "default", "signed manifest cohort identity")
	offline := fs.Bool("offline", false, "use only a valid authenticated manifest cache; perform no manifest HTTP request")
	installer := fs.String("installer", "", "update installer: native (default) or msix; FAK_SELF_UPDATE_INSTALLER is used when omitted")
	msixURI := fs.String("msix-appinstaller-uri", "", "signed HTTPS .appinstaller URI (requires --installer msix)")
	msixPackage := fs.String("msix-package", "", "MSIX package identity Name from AppxManifest.xml")
	msixPublisher := fs.String("msix-publisher", "", "expected signed package publisher")
	msixArtifact := fs.String("msix-artifact-digest", "", "FAK artifact digest carried by signed package provenance")
	msixFullArtifact := fs.String("msix-full-artifact-digest", "", "full-fallback artifact digest carried by signed package provenance")
	msixSource := fs.String("msix-source-revision", "", "FAK source revision carried by signed package provenance")
	msixInstalledVersion := fs.String("msix-installed-version", "", "installed package version used for downgrade policy")
	msixTargetVersion := fs.String("msix-target-version", "", "signed target package version used for downgrade policy")
	msixFullFallback := fs.String("msix-full-fallback-uri", "", "signed HTTPS full-package URI when differential delivery is unavailable")
	msixRepair := fs.Bool("msix-repair", false, "repair the installed MSIX package instead of updating it")
	msixUninstall := fs.Bool("msix-uninstall", false, "uninstall the installed MSIX package")
	msixAllowDowngrade := fs.Bool("msix-allow-downgrade", false, "allow an explicitly signed MSIX downgrade")
	force := fs.Bool("force", false, "build+gate+install even if not provably stale (still runs the green gate)")
	jsonMode := fs.Bool("json", false, "emit one versioned JSON receipt")
	handoffSession := fs.String("handoff-session", "", "after installation, launch the successor with this stable session identity")
	handoffTimeout := fs.Duration("handoff-timeout", 30*time.Second, "maximum time for graceful handoff (requires --handoff-session)")
	root := fs.String("root", "", "repo root to build from (default: discover from cwd)")
	target := fs.String("target", "", "binary path to replace (default: this binary's own path). Lets a scheduler update the FLEET binary regardless of which fak it invokes.")
	pinnedBin := fs.String("pinned-bin", "", "the binary path a scheduled-task registration REVIEWED and pinned; refuse to run when the executing binary has drifted from it (#6508)")
	_ = fs.Parse(argv)
	beginSelfUpdateOutput(*jsonMode)
	if handled, err := runSelfUpdateMSIX(selfUpdateMSIXOptions{
		CLIInstaller: *installer, ConfigInstaller: os.Getenv("FAK_SELF_UPDATE_INSTALLER"),
		AppInstallerURI: *msixURI, FullFallbackURI: *msixFullFallback,
		PackageIdentity: *msixPackage, Publisher: *msixPublisher,
		ArtifactIdentity: *msixArtifact, FullArtifactIdentity: *msixFullArtifact, SourceIdentity: *msixSource,
		InstalledVersion: *msixInstalledVersion, TargetVersion: *msixTargetVersion,
		Repair: *msixRepair, Uninstall: *msixUninstall, Check: *check,
		AllowDowngrade: *msixAllowDowngrade, Offline: *offline, JSON: *jsonMode,
	}); handled {
		if err != nil {
			fmt.Fprintln(os.Stderr, "self-update: MSIX adapter refused:", err)
		}
		return
	}
	reportSelfUpdateProgress(5, "checking installed binary provenance")

	// Scheduled-task provenance skew, checked BEFORE anything else: the task pinned one
	// reviewed absolute path at registration, and a tick executing anything else — or executing
	// that path after it turned into an unattestable / `+uncommitted` build — is adjudicating
	// the fleet's binary with code nobody reviewed. Refuse instead of converging from it.
	if strings.TrimSpace(*pinnedBin) != "" {
		if skew, why := selfinstall.PinSkew(selfinstall.Pin{Path: *pinnedBin}, selfUpdateInvoker()); skew {
			fmt.Fprintln(os.Stderr, "self-update: PIN_SKEW —", why)
			emitSelfUpdateOutcome(outcomePinSkew, *pinnedBin, why)
			os.Exit(2)
		}
	}

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}
	if repoRoot == "" {
		fmt.Fprintln(os.Stderr, "self-update: could not resolve a git repo root (pass --root)")
		os.Exit(2)
	}

	// Conditional selection is intentionally before any git fetch/build/install. With no
	// manifest URL configured this block is skipped, preserving the legacy path byte-for-byte.
	var manifestSelection selfUpdateManifestSelection
	if strings.TrimSpace(*manifestURL) != "" {
		installed := selfUpdateInstalledIdentity(strings.TrimSpace(*target))
		installedState, err := selfinstall.ReadInstallIdentity(selfinstall.IdentityStatePath(installTargetOr(*target)))
		if err != nil {
			fmt.Fprintln(os.Stderr, "self-update: installed identity refused:", err)
			return
		}
		installedVersion := installedState.AppVersion
		if strings.TrimSpace(installedVersion) == "" {
			installedVersion = selfUpdateBinaryVersion(installTargetOr(*target))
		}
		manifestSelection, err = selfUpdateManifestSelect(context.Background(), selfUpdateManifestRequest{
			URL: *manifestURL, ManifestID: *manifestID, CachePath: *manifestStatePath, Channel: *manifestChannel,
			Cohort: *manifestCohort, Platform: runtime.GOOS, Architecture: runtime.GOARCH,
			InstalledIdentity: installed, InstalledVersion: installedVersion,
			InstalledGeneration: installedState.SignedMetadataGeneration, Offline: *offline, Force: *force,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "self-update: signed manifest refused:", err)
			return
		}
		if manifestSelection.Disposition != "update" {
			fmt.Fprintln(os.Stderr, "self-update:", manifestSelection.Disposition)
			return
		}
	}

	// Compare against origin/main, not local HEAD: on a permanently-dirty shared trunk the
	// local tree is always ahead-or-behind with peer WIP, and origin/main is the verified
	// line we actually want guards converged on.
	reportSelfUpdateProgress(10, "reading selected revision")
	headRev := ""
	if strings.TrimSpace(*manifestURL) != "" && manifestSelection.Disposition == "update" {
		headRev = manifestSelection.TargetRevision
		if manifestSelection.Artifact == nil {
			selfUpdateFetchOrigin(context.Background(), selfinstall.RealRunner, repoRoot, *check)
		}
	} else {
		selfUpdateFetchOrigin(context.Background(), selfinstall.RealRunner, repoRoot, *check)
		headRev = repoRevOf(repoRoot, "origin/main")
	}

	// Whose freshness are we judging? When --target names a DIFFERENT binary (the scheduler
	// case: a dev fak invokes self-update to converge the FLEET binary), read the TARGET's
	// embedded stamp — not this invoking binary's. Otherwise the dirty dev binary's "Unknown"
	// would short-circuit the update and the stale fleet binary would never get replaced.
	selfStamp := binstamp.Self() // ALWAYS the invoking binary's own stamp (see convergeInvoker)
	stamp := selfStamp
	subject := "running"
	fleetTarget := false // --target names a DIFFERENT binary (the scheduler/fleet case)
	if t := strings.TrimSpace(*target); t != "" && !sameBinary(t) {
		subject = "target"
		fleetTarget = true
		if ts, ok := stampOfBinary(t); ok {
			stamp = ts
		}
		// If the target can't self-report, stamp stays as Self() but fleetTarget forces the
		// build below regardless — a fleet binary we cannot prove current gets refreshed.
	}
	verdict := binstamp.Compare(stamp, headRev)

	// Beyond the coarse Fresh/Stale/Unknown, classify the stamp by git ANCESTRY vs origin/main.
	// This is what lets SELF mode tell a binary that is provably BEHIND (Skewed — worth
	// rebuilding) apart from one that is merely AHEAD (a fresh local build not yet pushed):
	// binstamp.Compare collapses BOTH into Stale, so the old `verdict == Stale` rule would
	// rebuild origin/main straight OVER a newer dev binary. AssessStamp reuses the stamp we
	// already resolved (the target's, in fleet mode) and the origin/main we just fetched.
	reportSelfUpdateProgress(15, "comparing installed and origin/main revisions")
	skew := versionskew.AssessStamp(context.Background(), selfinstall.RealRunner, repoRoot, "origin/main", stamp)

	stampRev := stamp.Revision
	selfUpdateReceiptOldRevision = stampRev
	selfUpdateReceiptNewRevision = headRev
	if stampRev == "" {
		stampRev = "(unstamped)"
	} else if len(stampRev) > 12 {
		stampRev = stampRev[:12]
	}
	head := headRev
	if len(head) > 12 {
		head = head[:12]
	}
	fmt.Fprintf(selfUpdateProgress, "%s: %s%s   origin/main: %s   => %s (skew: %s)\n",
		subject, stampRev, dirtyMark(stamp.Dirty), head, verdict, skew.Verdict)

	if *check {
		// Observability: also report the swap-aside footprint next to the target binary, so a
		// leak of "<binary>.old.<pid>.<i>" files (the 9 GB class this reaper exists to prevent)
		// is a one-line signal here instead of an invisible pile only `ls` would reveal.
		reportAsideFootprint(installTargetOr(*target))
		// The whole role table, not just this one binary: a host is converged only when EVERY
		// declared hot copy holds origin/main, and --check is where an operator asks that.
		audit := selfUpdateAudit(repoRoot, headRev)
		printHotCopyAudit(audit)
		emitSelfUpdateCheckOutcome(installTargetOr(*target), fmt.Sprintf("%s/%s", verdict, skew.Verdict), verdict, audit.Partition())
		return
	}
	// Decide whether to build (see selfUpdateShouldBuild for the SELF/FLEET asymmetry). An
	// already-current fak must still run the cycle when its installed fak-dev companion lags.
	companionPaths := selfUpdateFakDevTargets(repoRoot, installTargetOr(*target))
	selectedArtifact := usableSelfUpdateArtifact(manifestSelection, companionPaths)
	if manifestSelection.Artifact != nil && selectedArtifact == nil {
		fmt.Fprintln(selfUpdateProgress, "self-update: signed catalog has no fak-dev companion target; using full source-build fallback")
		selectedArtifact = nil
		selfUpdateFetchOrigin(context.Background(), selfinstall.RealRunner, repoRoot, *check)
	}
	companionStale := selfUpdateFakDevNeedsConverge(companionPaths, headRev, selfUpdateProbe)
	manifestSelectedUpdate := strings.TrimSpace(*manifestURL) != "" && manifestSelection.Disposition == "update"
	proceed := manifestSelectedUpdate || selfUpdateShouldBuild(*force, fleetTarget, verdict, skew.Verdict) || companionStale
	if !proceed {
		emitSelfUpdateOutcome(selfUpdateSkipOutcome(fleetTarget, skew.Verdict), installTargetOr(*target), fmt.Sprintf("%s", skew.Verdict))
		switch {
		case fleetTarget:
			fmt.Fprintln(selfUpdateProgress, "self-update: target already current — nothing to do.")
		case skew.Verdict == versionskew.Ahead:
			fmt.Fprintln(selfUpdateProgress, "self-update: running binary is AHEAD of origin/main (a local build not yet pushed) — not rebuilding (pass --force to build+gate+install origin/main anyway).")
		case skew.Verdict == versionskew.Fresh:
			fmt.Fprintln(selfUpdateProgress, "self-update: already current — nothing to do.")
		case skew.Verdict == versionskew.Dirty, skew.Verdict == versionskew.Unstamped, skew.Verdict == versionskew.Diverged:
			fmt.Fprintf(selfUpdateProgress, "self-update: running binary is %s vs origin/main — not auto-rebuilding a local/off-trunk build (pass --force to build+gate+install origin/main).\n", skew.Verdict)
		default:
			fmt.Fprintln(selfUpdateProgress, "self-update: freshness unknown — not rebuilding (pass --force to build+gate+install anyway).")
		}
		return
	}

	performSelfUpdate(repoRoot, headRev, target, companionPaths, selectedArtifact, manifestSelection.MetadataGeneration, manifestSelection.TargetVersion, strings.TrimSpace(*handoffSession), *handoffTimeout, fs.Args())
}

func usableSelfUpdateArtifact(selection selfUpdateManifestSelection, companionPaths []string) *selfUpdateArtifactTarget {
	if selection.Disposition != "update" || selection.Artifact == nil || len(companionPaths) != 0 {
		return nil
	}
	return selection.Artifact
}

func selfUpdateInstalledIdentity(target string) string {
	if target != "" && !sameBinary(target) {
		if stamp, ok := stampOfBinary(target); ok && strings.TrimSpace(stamp.Revision) != "" {
			return stamp.Revision
		}
	}
	return binstamp.Self().Revision
}

func selfUpdateBinaryVersion(target string) string {
	if strings.TrimSpace(target) == "" {
		return ""
	}
	out, ok := selfinstall.RealRunner(context.Background(), "", target, "version", "--json")
	if !ok {
		return ""
	}
	var identity struct {
		AppVersion string `json:"app_version"`
	}
	if json.Unmarshal([]byte(out), &identity) != nil {
		return ""
	}
	return strings.TrimSpace(identity.AppVersion)
}

// selfUpdateFetchOrigin refreshes the remote-tracking ref only for an update run. A fetch writes
// Git metadata even when the worktree is untouched, so --check deliberately observes the current
// origin/main ref without fetching and remains strictly non-mutating.
func selfUpdateFetchOrigin(ctx context.Context, runner selfinstall.Runner, repoRoot string, checkOnly bool) {
	if checkOnly {
		return
	}
	_, _ = runner(ctx, repoRoot, "git", "fetch", "origin", "--quiet")
}

// selfUpdateHost roots the hot-copy role table on this host: the checkout we build from, the
// user home dir the installed/Go-bin copies live under, and the binary actually running this
// tick (which under a scheduled task is the path its registration pinned).
func selfUpdateHost(repoRoot string) selfinstall.Host {
	exe, _ := os.Executable()
	home, _ := os.UserHomeDir()
	return selfinstall.Host{RepoRoot: repoRoot, Home: home, Scheduled: exe}
}

// selfUpdateProbe reads one deployed binary's VCS provenance from its path. This includes our
// own path: after a Windows swap the mapped process still carries the old stamp while the file
// already contains the new build, so an in-process answer would fabricate post-update skew.
func selfUpdateProbe(path string) (string, bool, bool) {
	s, ok := stampOfBinary(path)
	return s.Revision, s.Dirty, ok
}

// selfUpdateInvoker is the hot copy for the binary running this tick — what the scheduled-task
// pin is checked against.
func selfUpdateInvoker() selfinstall.HotCopy {
	exe, err := os.Executable()
	c := selfinstall.HotCopy{Role: selfinstall.RoleScheduled, Path: exe}
	if err != nil || strings.TrimSpace(exe) == "" {
		c.Err = "cannot resolve this binary's own path"
		return c
	}
	if st, serr := os.Stat(exe); serr == nil && !st.IsDir() {
		c.Present = true
	}
	s := binstamp.Self()
	c.Build, c.Dirty = s.Revision, s.Dirty
	c.Attested = s.HasVCS && strings.TrimSpace(s.Revision) != ""
	return c
}

// selfUpdateAudit grades every declared hot copy against origin/main.
func selfUpdateAudit(repoRoot, headRev string) selfinstall.Audit {
	return selfinstall.AuditCopies(selfinstall.Census(selfUpdateHost(repoRoot), selfUpdateProbe), headRev)
}

// printHotCopyAudit prints one greppable line per hot copy plus the verdict.
func printHotCopyAudit(a selfinstall.Audit) {
	for _, line := range a.Lines() {
		fmt.Fprintln(selfUpdateProgress, "self-update: "+line)
	}
}

// selfUpdateSiblings lists the OTHER fak binaries on this host that a fleet consumer resolves
// but no updater targets — the binary running this command, the in-tree
// <root>/tools/.bin/fak[.exe] that dispatch_worker.resolve_fak_bin prefers ahead of PATH, the
// installed <home>/bin copy, and the <home>/go/bin copy scheduled tasks execute.
//
// The set is the CONVERGEABLE half of internal/selfinstall's declared role table (#6508), not an
// ad-hoc list here: one place declares which binary fills which role, and the repo-root gate
// binary is deliberately excluded from unattended swaps (it is a hand-build in a shared dirty
// checkout, audited every tick instead). Paths that do not exist are skipped — we converge
// binaries, never create new install locations — as is anything equal to the primary target.
// Order is stable and deduped case-insensitively so a Windows host does not swap the same file
// twice.
func selfUpdateSiblings(repoRoot, target string) []string {
	return selfinstall.ConvergeTargets(selfinstall.Roles(selfUpdateHost(repoRoot)), target)
}

// selfUpdateFakDevTargets returns existing fak-dev companions beside converged fak copies.
// Absence is intentional: self-update maintains an installed dev artifact but never creates one
// on product-only hosts.
func selfUpdateFakDevTargets(repoRoot, target string) []string {
	name := "fak-dev"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	seen := map[string]bool{}
	out := []string{}
	for _, fakPath := range append([]string{target}, selfUpdateSiblings(repoRoot, target)...) {
		candidate := filepath.Join(filepath.Dir(fakPath), name)
		key := strings.ToLower(filepath.Clean(candidate))
		if seen[key] {
			continue
		}
		seen[key] = true
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			out = append(out, candidate)
		}
	}
	return out
}

func selfUpdateFakDevNeedsConverge(targets []string, headRev string, probe selfinstall.StampProbe) bool {
	for _, target := range targets {
		revision, dirty, attested := probe(target)
		if !attested || dirty || !strings.EqualFold(strings.TrimSpace(revision), strings.TrimSpace(headRev)) {
			return true
		}
	}
	return false
}

// prepareFakDevUpdate builds and smokes the dev companion before the primary fak swap. This
// keeps the update fail-closed: a broken fak-dev build cannot leave fak current while the
// already-installed repository tool remains on an older local revision.
func prepareFakDevUpdate(ctx context.Context, runner selfinstall.Runner, buildDir string, targets []string, headRev string) (string, []string, error) {
	if len(targets) == 0 {
		return "", nil, nil
	}
	candidate := filepath.Join(os.TempDir(), fmt.Sprintf("fak-dev-selfupdate-%d%s", os.Getpid(), filepath.Ext(targets[0])))
	_ = os.Remove(candidate)
	stamp := "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildCommit=" + headRev
	stopHeartbeat := startSelfUpdateHeartbeat(40, "building fak-dev companion")
	if out, ok := runner(ctx, buildDir, "go", "build", "-trimpath", "-buildvcs=true", "-ldflags", stamp, "-o", candidate, "./cmd/fak-dev"); !ok {
		stopHeartbeat()
		return "", nil, fmt.Errorf("build fak-dev companion: %s", strings.TrimSpace(out))
	}
	stopHeartbeat()
	stopHeartbeat = startSelfUpdateHeartbeat(46, "verifying fak-dev companion")
	out, ok := runner(ctx, buildDir, candidate, "version")
	stopHeartbeat()
	if !ok || !strings.Contains(out, "build: "+headRev) {
		_ = os.Remove(candidate)
		return "", nil, fmt.Errorf("smoke fak-dev companion: expected build %s, got %q", headRev, strings.TrimSpace(out))
	}
	return candidate, targets, nil
}

// selfUpdateGateRunner exposes the gated install ladder's actual blocking operation instead
// of flattening build, vet, and smoke into one long ambiguous wait.
func selfUpdateGateRunner(runner selfinstall.Runner) selfinstall.Runner {
	return func(ctx context.Context, dir, name string, args ...string) (string, bool) {
		percent := 0
		operation := ""
		phase := selfUpdatePhase("")
		switch {
		case name == "go" && len(args) > 0 && args[0] == "build":
			percent, operation = 55, "building fak candidate"
			phase = selfUpdatePhaseBuild
		case name == "go" && len(args) > 0 && args[0] == "vet":
			percent, operation = 70, "vetting fak candidate"
			phase = selfUpdatePhaseVet
		case len(args) >= 1 && args[0] == "version":
			percent, operation = 78, "smoke-verifying fak candidate"
			phase = selfUpdatePhaseSmoke
		}
		if percent == 0 {
			return runner(ctx, dir, name, args...)
		}
		startSelfUpdatePhase(phase)
		stopHeartbeat := startSelfUpdateHeartbeat(percent, operation)
		out, ok := runner(ctx, dir, name, args...)
		stopHeartbeat()
		return out, ok
	}
}

// selfUpdateOutcome is the closed vocabulary of self-update tick outcomes.
//
// The scheduler observes ONLY the process exit code, and rc=0 is identical for "installed a
// fresh build", "target already current", "another build was in flight", and "--check". So a
// host whose binary has not advanced in nine hours is indistinguishable from one that is
// perfectly converged — the success code is decoupled from whether an update happened. Every
// exit path therefore emits exactly one greppable `self-update: outcome=<cause>` line, so the
// tick leaves a durable, machine-readable record of WHICH of those four things it did.
type selfUpdateOutcome string

const (
	outcomeInstalled      selfUpdateOutcome = "installed"      // target swapped to a fresh gated build
	outcomeMetadataOnly   selfUpdateOutcome = "metadata_only"  // selected provenance advanced; binary bytes did not
	outcomeTargetCurrent  selfUpdateOutcome = "target-current" // --target already at origin/main
	outcomeSelfFresh      selfUpdateOutcome = "self-fresh"     // SELF mode, running binary is trunk tip
	outcomeSelfAhead      selfUpdateOutcome = "self-ahead"     // SELF mode, local build newer than trunk
	outcomeSelfLocal      selfUpdateOutcome = "self-local"     // SELF mode, dirty/unstamped/diverged
	outcomeSelfUnknown    selfUpdateOutcome = "self-unknown"   // SELF mode, freshness unresolvable
	outcomeBusy           selfUpdateOutcome = "busy"           // single-flight lock held by a live build
	outcomeCheckOnly      selfUpdateOutcome = "check-only"     // --check reported and exited
	outcomeGateFailed     selfUpdateOutcome = "gate-failed"    // build/vet/smoke refused the candidate
	outcomePrepareFailed  selfUpdateOutcome = "prepare-failed" // could not stage the origin/main worktree
	outcomeRolledBack     selfUpdateOutcome = "rolled-back"    // activation failed and all changed targets were restored
	outcomeRollbackFailed selfUpdateOutcome = "rollback-failed"
	outcomeHandoffRefused selfUpdateOutcome = "handoff-refused" // activation failed and at least one restore failed
	// outcomeHotCopyDivergent: everything this tick was allowed to swap landed, but the role
	// census still shows a declared hot copy on another build (typically the audit-only repo-root
	// gate binary). Distinct from sibling-stale, which is a FAILED swap (#6508).
	outcomeHotCopyDivergent selfUpdateOutcome = "hot-copy-divergent"
	// outcomePinSkew: the binary executing this tick is not the reviewed one the scheduled task
	// pinned, so it refused to adjudicate the fleet's binary at all (#6508).
	outcomePinSkew selfUpdateOutcome = "pin-skew"
)

func selfUpdateTransactionDetail(err error, rollbackErrors []error) string {
	detail := "transaction failed"
	if err != nil {
		detail = err.Error()
	}
	if len(rollbackErrors) == 0 {
		return detail
	}
	parts := make([]string, 0, len(rollbackErrors))
	for _, rollbackErr := range rollbackErrors {
		if rollbackErr != nil {
			parts = append(parts, rollbackErr.Error())
		}
	}
	if len(parts) == 0 {
		return detail
	}
	return detail + "; rollback: " + strings.Join(parts, "; ")
}

// emitSelfUpdateOutcome prints the single machine-readable outcome line for this tick. It
// carries its own UTC timestamp so the record is self-describing wherever it is captured — the
// scheduler's "Last Result" is one overwritten integer with no history, so a tick that leaves
// only an exit code cannot answer "when did the fleet binary last actually advance?".
func emitSelfUpdateOutcome(cause selfUpdateOutcome, target, detail string) {
	finishSelfUpdateProgress(cause)
	timing := finishSelfUpdateTiming()
	reportSelfUpdateTiming(timing)
	if selfUpdateJSON != nil {
		receipt := newSelfUpdateReceiptWithTiming(cause, target, detail, timing)
		_ = json.NewEncoder(selfUpdateJSON).Encode(receipt)
		selfUpdateJSON = nil
		return
	}
	line := fmt.Sprintf("self-update: at=%s outcome=%s target=%s",
		time.Now().UTC().Format(time.RFC3339), cause, target)
	if d := strings.TrimSpace(detail); d != "" {
		line += " detail=" + strconv.Quote(d)
	}
	fmt.Fprintln(selfUpdateProgress, line)
}

// emitSelfUpdateCheckOutcome keeps the human outcome vocabulary stable while making the JSON
// receipt stateful: "current" is reserved for a fresh target and a converged hot-copy audit.
func emitSelfUpdateCheckOutcome(target, detail string, freshness binstamp.Freshness, audit selfinstall.AuditPartition) {
	if selfUpdateJSON == nil {
		emitSelfUpdateOutcome(outcomeCheckOnly, target, detail)
		return
	}
	finishSelfUpdateProgress(outcomeCheckOnly)
	timing := finishSelfUpdateTiming()
	reportSelfUpdateTiming(timing)
	posture := selfupdate.ClassifyCheck(freshness, audit)
	receipt := newSelfUpdateReceiptWithTiming(outcomeCheckOnly, target, detail, timing)
	receipt.Status = string(posture.Status)
	receipt.NextCommand = posture.NextCommand
	_ = json.NewEncoder(selfUpdateJSON).Encode(receipt)
	selfUpdateJSON = nil
}

const selfUpdateReceiptSchema = "fak.self-update.receipt/v1"

type selfUpdateReceipt struct {
	Schema          string                     `json:"schema"`
	SchemaVersion   int                        `json:"schema_version"`
	CorrelationID   string                     `json:"correlation_id"`
	Status          string                     `json:"status"`
	OldRevision     *string                    `json:"old_revision"`
	NewRevision     *string                    `json:"new_revision"`
	Targets         []selfUpdateReceiptTarget  `json:"targets"`
	Attempted       int                        `json:"attempted"`
	Changed         int                        `json:"changed"`
	RollbackStatus  string                     `json:"rollback_status"`
	RollbackErrors  []string                   `json:"rollback_errors"`
	RestartRequired bool                       `json:"restart_required"`
	NextCommand     string                     `json:"next_command"`
	Detail          string                     `json:"detail,omitempty"`
	BuildProvenance *selfUpdateBuildProvenance `json:"build_provenance,omitempty"`
	Transfer        *selfUpdateTransferReceipt `json:"transfer,omitempty"`
	Handoff         *selfUpdateHandoffReceipt  `json:"handoff,omitempty"`
	TotalMS         int64                      `json:"total_ms"`
	PhaseMS         selfUpdatePhaseMS          `json:"phase_ms"`
}

type selfUpdateReceiptTarget struct {
	Role                    string `json:"role"`
	Path                    string `json:"path"`
	CompatibilityGroup      string `json:"compatibility_group,omitempty"`
	DesiredArtifactDigest   string `json:"desired_artifact_digest,omitempty"`
	InstalledArtifactDigest string `json:"installed_artifact_digest,omitempty"`
	Acquisition             string `json:"acquisition,omitempty"`
	Activation              string `json:"activation,omitempty"`
	Rollback                string `json:"rollback,omitempty"`
}

type selfUpdateBuildProvenance struct {
	SourceCommit         string            `json:"source_commit"`
	ArtifactSourceCommit string            `json:"artifact_source_commit"`
	BuildInputDigest     string            `json:"build_input_digest"`
	BuildEnvelope        map[string]string `json:"build_envelope"`
	ArtifactDigest       string            `json:"artifact_digest"`
	ArtifactSize         int64             `json:"artifact_size"`
	AppVersion           string            `json:"app_version"`
	Reused               bool              `json:"reused"`
}

// selfUpdatePhaseMS is a fixed-shape object rather than a map so receipt consumers always see
// the same phase vocabulary, including zeroes for phases an early exit did not reach.
type selfUpdatePhaseMS struct {
	Check     int64 `json:"check"`
	Lock      int64 `json:"lock"`
	Cleanup   int64 `json:"cleanup"`
	Prepare   int64 `json:"prepare"`
	Companion int64 `json:"companion"`
	Build     int64 `json:"build"`
	Vet       int64 `json:"vet"`
	Smoke     int64 `json:"smoke"`
	Install   int64 `json:"install"`
	Verify    int64 `json:"verify"`
	Handoff   int64 `json:"handoff"`
}

type selfUpdatePhase string

const (
	selfUpdatePhaseCheck     selfUpdatePhase = "check"
	selfUpdatePhaseLock      selfUpdatePhase = "lock"
	selfUpdatePhaseCleanup   selfUpdatePhase = "cleanup"
	selfUpdatePhasePrepare   selfUpdatePhase = "prepare"
	selfUpdatePhaseCompanion selfUpdatePhase = "companion"
	selfUpdatePhaseBuild     selfUpdatePhase = "build"
	selfUpdatePhaseVet       selfUpdatePhase = "vet"
	selfUpdatePhaseSmoke     selfUpdatePhase = "smoke"
	selfUpdatePhaseInstall   selfUpdatePhase = "install"
	selfUpdatePhaseVerify    selfUpdatePhase = "verify"
	selfUpdatePhaseHandoff   selfUpdatePhase = "handoff"
)

var selfUpdatePhaseOrder = [...]selfUpdatePhase{
	selfUpdatePhaseCheck,
	selfUpdatePhaseLock,
	selfUpdatePhaseCleanup,
	selfUpdatePhasePrepare,
	selfUpdatePhaseCompanion,
	selfUpdatePhaseBuild,
	selfUpdatePhaseVet,
	selfUpdatePhaseSmoke,
	selfUpdatePhaseInstall,
	selfUpdatePhaseVerify,
	selfUpdatePhaseHandoff,
}

func (p *selfUpdatePhaseMS) set(phase selfUpdatePhase, value int64) {
	switch phase {
	case selfUpdatePhaseCheck:
		p.Check = value
	case selfUpdatePhaseLock:
		p.Lock = value
	case selfUpdatePhaseCleanup:
		p.Cleanup = value
	case selfUpdatePhasePrepare:
		p.Prepare = value
	case selfUpdatePhaseCompanion:
		p.Companion = value
	case selfUpdatePhaseBuild:
		p.Build = value
	case selfUpdatePhaseVet:
		p.Vet = value
	case selfUpdatePhaseSmoke:
		p.Smoke = value
	case selfUpdatePhaseInstall:
		p.Install = value
	case selfUpdatePhaseVerify:
		p.Verify = value
	case selfUpdatePhaseHandoff:
		p.Handoff = value
	}
}

type selfUpdateTimingSnapshot struct {
	totalMS       int64
	phaseMS       selfUpdatePhaseMS
	dominantPhase selfUpdatePhase
	dominantMS    int64
}

var selfUpdateProgress io.Writer = os.Stderr
var selfUpdateJSON io.Writer
var selfUpdateCorrelationID = randomSelfUpdateCorrelationID
var selfUpdateReceiptOldRevision string
var selfUpdateReceiptNewRevision string
var selfUpdateReceiptTargets []selfUpdateReceiptTarget
var selfUpdateReceiptAttempted int
var selfUpdateReceiptChanged int
var selfUpdateReceiptHandoff *selfUpdateHandoffReceipt
var selfUpdateReceiptBuildProvenance *selfUpdateBuildProvenance
var selfUpdateReceiptTransfer *selfUpdateTransferReceipt

// selfUpdateTimingNow is the deterministic wall-clock seam for timing receipts. It is kept
// separate from outcome timestamps and cleanup-age decisions so tests can control cost
// attribution without changing production semantics elsewhere in the command.
var selfUpdateTimingNow = time.Now

type selfUpdateTimingTracker struct {
	sync.Mutex
	initialized  bool
	finished     bool
	started      time.Time
	phaseStarted time.Time
	active       selfUpdatePhase
	elapsed      map[selfUpdatePhase]time.Duration
	snapshot     selfUpdateTimingSnapshot
}

var selfUpdateTimingState selfUpdateTimingTracker

var selfUpdateProgressState struct {
	sync.Mutex
	percent   int
	operation string
}

const selfUpdateHeartbeatInterval = 15 * time.Second

// selfUpdateHeartbeatWait is a seam for deterministic captured tests. Production waits at
// most once per interval; tests can release exact ticks without sleeping.
var selfUpdateHeartbeatWait = func(stop <-chan struct{}, interval time.Duration) bool {
	timer := time.NewTimer(interval)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-stop:
		return false
	}
}

func beginSelfUpdateTiming() {
	now := selfUpdateTimingNow()
	selfUpdateTimingState.Lock()
	selfUpdateTimingState.initialized = true
	selfUpdateTimingState.finished = false
	selfUpdateTimingState.started = now
	selfUpdateTimingState.phaseStarted = now
	selfUpdateTimingState.active = selfUpdatePhaseCheck
	selfUpdateTimingState.elapsed = make(map[selfUpdatePhase]time.Duration, len(selfUpdatePhaseOrder))
	selfUpdateTimingState.snapshot = selfUpdateTimingSnapshot{}
	selfUpdateTimingState.Unlock()
}

// startSelfUpdatePhase closes the previous phase and starts the named one. Re-entering a phase
// accumulates its duration, which lets "prepare" cover both worktree preparation and the later
// target census without inventing an unstable one-off phase name.
func startSelfUpdatePhase(phase selfUpdatePhase) {
	now := selfUpdateTimingNow()
	selfUpdateTimingState.Lock()
	defer selfUpdateTimingState.Unlock()
	if !selfUpdateTimingState.initialized || selfUpdateTimingState.finished {
		return
	}
	selfUpdateTimingState.recordActive(now)
	selfUpdateTimingState.active = phase
}

func (s *selfUpdateTimingTracker) recordActive(now time.Time) {
	if s.active != "" && !now.Before(s.phaseStarted) {
		s.elapsed[s.active] += now.Sub(s.phaseStarted)
	}
	s.phaseStarted = now
}

func finishSelfUpdateTiming() selfUpdateTimingSnapshot {
	now := selfUpdateTimingNow()
	selfUpdateTimingState.Lock()
	defer selfUpdateTimingState.Unlock()
	if !selfUpdateTimingState.initialized {
		return selfUpdateTimingSnapshot{}
	}
	if selfUpdateTimingState.finished {
		return selfUpdateTimingState.snapshot
	}
	selfUpdateTimingState.recordActive(now)

	var phaseMS selfUpdatePhaseMS
	dominant := selfUpdatePhaseOrder[0]
	dominantDuration := selfUpdateTimingState.elapsed[dominant]
	for _, phase := range selfUpdatePhaseOrder {
		elapsed := selfUpdateTimingState.elapsed[phase]
		phaseMS.set(phase, elapsed.Milliseconds())
		if elapsed > dominantDuration {
			dominant = phase
			dominantDuration = elapsed
		}
	}
	total := time.Duration(0)
	if !now.Before(selfUpdateTimingState.started) {
		total = now.Sub(selfUpdateTimingState.started)
	}
	selfUpdateTimingState.snapshot = selfUpdateTimingSnapshot{
		totalMS:       total.Milliseconds(),
		phaseMS:       phaseMS,
		dominantPhase: dominant,
		dominantMS:    dominantDuration.Milliseconds(),
	}
	selfUpdateTimingState.finished = true
	return selfUpdateTimingState.snapshot
}

func reportSelfUpdateTiming(timing selfUpdateTimingSnapshot) {
	fmt.Fprintf(selfUpdateProgress, "self-update: timing total_ms=%d dominant_phase=%s dominant_ms=%d\n",
		timing.totalMS, timing.dominantPhase, timing.dominantMS)
}

// reportSelfUpdateProgress emits a useful current operation with an honest overall estimate.
// Ordinary work is capped below 100; only finishSelfUpdateProgress may publish completion.
func reportSelfUpdateProgress(percent int, operation string) {
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()
	if percent > 99 {
		percent = 99
	}
	if percent < selfUpdateProgressState.percent {
		return
	}
	operation = strings.TrimSpace(operation)
	if percent == selfUpdateProgressState.percent && operation == selfUpdateProgressState.operation {
		return
	}
	selfUpdateProgressState.percent = percent
	selfUpdateProgressState.operation = operation
	fmt.Fprintf(selfUpdateProgress, "self-update: progress=%d%% operation=%q\n", percent, operation)
}

func finishSelfUpdateProgress(cause selfUpdateOutcome) {
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()
	if selfUpdateProgressState.percent == 100 {
		return
	}
	selfUpdateProgressState.percent = 100
	selfUpdateProgressState.operation = "terminal outcome: " + string(cause)
	fmt.Fprintf(selfUpdateProgress, "self-update: progress=100%% operation=%q\n", selfUpdateProgressState.operation)
}

// startSelfUpdateHeartbeat holds the current percent steady while a blocking operation runs.
// The fixed 15-second lower bound keeps the signal useful without turning slow builds into spam.
func startSelfUpdateHeartbeat(percent int, operation string) func() {
	reportSelfUpdateProgress(percent, operation)
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for selfUpdateHeartbeatWait(stop, selfUpdateHeartbeatInterval) {
			selfUpdateProgressState.Lock()
			currentPercent := selfUpdateProgressState.percent
			currentOperation := selfUpdateProgressState.operation
			if currentPercent == percent && currentOperation == strings.TrimSpace(operation) {
				fmt.Fprintf(selfUpdateProgress, "self-update: progress=%d%% operation=%q heartbeat=true\n", currentPercent, currentOperation)
			}
			selfUpdateProgressState.Unlock()
		}
	}()
	var once sync.Once
	return func() {
		once.Do(func() {
			close(stop)
			<-done
		})
	}
}

func beginSelfUpdateOutput(enabled bool) {
	selfUpdateProgress = os.Stderr
	selfUpdateJSON = nil
	selfUpdateProgressState.Lock()
	selfUpdateProgressState.percent = 0
	selfUpdateProgressState.operation = ""
	selfUpdateProgressState.Unlock()
	selfUpdateReceiptOldRevision = ""
	selfUpdateReceiptNewRevision = ""
	selfUpdateReceiptTargets = nil
	selfUpdateReceiptAttempted = 0
	selfUpdateReceiptChanged = 0
	selfUpdateReceiptHandoff = nil
	selfUpdateReceiptBuildProvenance = nil
	selfUpdateReceiptTransfer = nil
	beginSelfUpdateTiming()
	if enabled {
		selfUpdateJSON = os.Stdout
	}
}

func randomSelfUpdateCorrelationID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err == nil {
		return hex.EncodeToString(b[:])
	}
	return fmt.Sprintf("pid-%d", os.Getpid())
}

func newSelfUpdateReceipt(cause selfUpdateOutcome, target, detail string) selfUpdateReceipt {
	return newSelfUpdateReceiptWithTiming(cause, target, detail, selfUpdateTimingSnapshot{})
}

func newSelfUpdateReceiptWithTiming(cause selfUpdateOutcome, target, detail string, timing selfUpdateTimingSnapshot) selfUpdateReceipt {
	status := "current"
	rollbackStatus := "not_attempted"
	restartRequired := false
	nextCommand := "fak version"
	switch cause {
	case outcomeInstalled:
		status = "updated"
	case outcomeMetadataOnly:
		status = "current"
	case outcomeGateFailed:
		status, nextCommand = "gate_failed", "fak self-update"
	case outcomePrepareFailed:
		status, nextCommand = "prepare_failed", "fak self-update"
	case outcomePinSkew:
		status, nextCommand = "pin_skew", "fak self-update --check"
	case outcomeRolledBack:
		status, rollbackStatus, nextCommand = "rolled_back", "succeeded", "fak self-update"
	case outcomeRollbackFailed:
		status, rollbackStatus, nextCommand = "rollback_failed", "failed", "fak self-update --check"
	case outcomeBusy:
		status, nextCommand = "busy", "fak self-update"
	case outcomeCheckOnly:
		// A check-only receipt describes freshness rather than an installation effect.
		// Comparing the two revisions keeps the JSON contract honest for operators and
		// automation: a stale, non-mutating check must not claim the binary is current.
		if oldRevision, newRevision := strings.TrimSpace(selfUpdateReceiptOldRevision), strings.TrimSpace(selfUpdateReceiptNewRevision); oldRevision != "" && newRevision != "" && oldRevision != newRevision {
			status, nextCommand = string(selfupdate.StatusStale), "fak self-update"
		}
	case outcomeHotCopyDivergent:
		status, nextCommand = string(selfupdate.StatusDivergent), "fak self-update"
	case outcomeHandoffRefused:
		status, nextCommand = "handoff_refused", "fak self-update --check"
	case selfUpdateOutcome("restart_required"):
		status, restartRequired, nextCommand = "restart_required", true, "fak self-update --check"
	}
	rollbackErrors := []string{}
	if status == "rollback_failed" && strings.TrimSpace(detail) != "" {
		rollbackErrors = append(rollbackErrors, detail)
	}
	targets := append([]selfUpdateReceiptTarget(nil), selfUpdateReceiptTargets...)
	if len(targets) == 0 && strings.TrimSpace(target) != "" && target != "<self>" {
		targets = append(targets, selfUpdateReceiptTarget{Role: "primary", Path: filepath.Clean(target)})
	}
	if targets == nil {
		targets = []selfUpdateReceiptTarget{}
	}
	return selfUpdateReceipt{
		Schema: selfUpdateReceiptSchema, SchemaVersion: 1, CorrelationID: selfUpdateCorrelationID(), Status: status,
		OldRevision: optionalRevision(selfUpdateReceiptOldRevision), NewRevision: optionalRevision(selfUpdateReceiptNewRevision),
		Targets: targets, Attempted: selfUpdateReceiptAttempted, Changed: selfUpdateReceiptChanged, RollbackStatus: rollbackStatus,
		RollbackErrors: rollbackErrors, RestartRequired: restartRequired, NextCommand: nextCommand, Handoff: selfUpdateReceiptHandoff,
		BuildProvenance: selfUpdateReceiptBuildProvenance,
		Transfer:        selfUpdateReceiptTransfer,
		Detail:          strings.TrimSpace(detail), TotalMS: timing.totalMS, PhaseMS: timing.phaseMS,
	}
}

func optionalRevision(revision string) *string {
	if strings.TrimSpace(revision) == "" {
		return nil
	}
	return &revision
}

// selfUpdateSkipOutcome names WHY a tick decided not to build, mirroring the branches of the
// message switch so the greppable outcome and the human sentence can never drift apart.
func selfUpdateSkipOutcome(fleetTarget bool, skew versionskew.Verdict) selfUpdateOutcome {
	if fleetTarget {
		return outcomeTargetCurrent
	}
	switch skew {
	case versionskew.Ahead:
		return outcomeSelfAhead
	case versionskew.Fresh:
		return outcomeSelfFresh
	case versionskew.Dirty, versionskew.Unstamped, versionskew.Diverged:
		return outcomeSelfLocal
	default:
		return outcomeSelfUnknown
	}
}

// selfUpdateShouldBuild decides whether self-update proceeds to build+gate+install. The two
// modes consult DIFFERENT witnesses on purpose:
//   - FLEET (--target names a different binary): rebuild unless binstamp can PROVE the target
//     is the trunk tip (Fresh). Anything short of proof — including an unresolvable Unknown —
//     rebuilds, because a cheap gated swap is the right default for a binary the fleet runs.
//   - SELF (updating our own binary): rebuild ONLY when versionskew proves the running binary
//     is a strict ANCESTOR of origin/main (Skewed). This is the fix for the case binstamp
//     collapses: a clean local build that is AHEAD of origin/main reads as binstamp.Stale
//     (rev differs) and the old `verdict == Stale` rule would rebuild origin/main straight
//     OVER — downgrading — that newer binary. versionskew keeps Ahead (and Diverged / Dirty /
//     Unstamped) distinct from Skewed, so SELF mode never downgrades a developer's build out
//     from under them; --force is the deliberate escape for those cases.
//
// --force builds in either mode (force bypasses only the staleness check, never the green gate).
func selfUpdateShouldBuild(force, fleetTarget bool, bin binstamp.Freshness, skew versionskew.Verdict) bool {
	if force {
		return true
	}
	if fleetTarget {
		return bin != binstamp.Fresh
	}
	return skew == versionskew.Skewed
}

// installTargetOr resolves the binary path --check should inspect: the explicit --target, or
// this binary's own path. Mirrors the resolution in the main flow so --check reports the same
// target a real self-update would swap (and thus the same install dir the reaper cleans).
func installTargetOr(target string) string {
	if t := strings.TrimSpace(target); t != "" {
		return t
	}
	if exe, err := os.Executable(); err == nil {
		return exe
	}
	return ""
}

// reportAsideFootprint prints a one-line summary of the swap-aside files sitting next to the
// target binary, so `self-update --check` surfaces a leak while it is 5 files instead of 500.
// A clean install dir prints nothing (no noise on the healthy path).
func reportAsideFootprint(target string) {
	if strings.TrimSpace(target) == "" {
		return
	}
	fp := selfinstall.MeasureAsides(target, os.Getpid(), safecommit.ProcessAlive)
	if fp.Count == 0 {
		return
	}
	fmt.Fprintf(selfUpdateProgress, "self-update: swap-aside footprint next to %s — %d file(s), %s (%d reclaimable, %s); the next self-update reaps the reclaimable ones\n",
		filepath.Base(target), fp.Count, humanBytes(fp.Bytes), fp.DeadCount, humanBytes(fp.DeadBytes))
}

// sameBinary reports whether path refers to this running executable (so --target pointing
// at ourselves falls back to the in-process binstamp.Self()).
func sameBinary(path string) bool {
	exe, err := os.Executable()
	if err != nil {
		return false
	}
	a, _ := filepath.Abs(filepath.Clean(path))
	b, _ := filepath.Abs(filepath.Clean(exe))
	return strings.EqualFold(a, b)
}

// stampOfBinary reads another fak binary's embedded VCS stamp by running `<bin> version` and
// parsing the "build: <rev>[ +uncommitted]" line cmdVersion prints. A binary too old to have
// that line (or that errors) yields ok=false; the caller treats that as "cannot prove fresh"
// which, in --target mode, correctly lets the update proceed (a fak that cannot self-report
// its build is by definition not the current one).
func stampOfBinary(path string) (binstamp.Stamp, bool) {
	bi, err := buildinfo.ReadFile(path)
	if err != nil {
		return binstamp.Stamp{}, false
	}
	stamp := binstamp.FromBuildInfo(bi)
	return stamp, stamp.HasVCS
}

// repoRevOf returns the full SHA a ref resolves to in the repo at root, or "" on error.
func repoRevOf(root, ref string) string {
	out, ok := selfinstall.RealRunner(context.Background(), root, "git", "rev-parse", ref)
	if !ok {
		return ""
	}
	return strings.TrimSpace(out)
}

func dirtyMark(dirty bool) string {
	if dirty {
		return " +uncommitted"
	}
	return ""
}
