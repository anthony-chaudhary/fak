package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
	"github.com/anthony-chaudhary/fak/internal/selfupdate"
)

func performSelfUpdate(repoRoot, headRev string, target *string, companionPaths []string, artifact *selfUpdateArtifactTarget, metadataGeneration uint64, signedVersion string, handoffSession string, handoffTimeout time.Duration, successorArgs []string) {
	reportSelfUpdateProgress(20, "acquiring self-update lock")
	installTarget := strings.TrimSpace(*target)
	if installTarget == "" {
		exe, err := os.Executable()
		if err != nil || strings.TrimSpace(exe) == "" {
			fmt.Fprintln(os.Stderr, "self-update: cannot resolve this binary's own path (pass --target):", err)
			os.Exit(1)
		}
		installTarget = exe
	}

	// Single-flight: only one self-update may BUILD at a time on this host. A second
	// invocation (e.g. the scheduled tick firing while a slow build from the last tick is
	// still going on a saturated box) exits immediately rather than stacking another
	// expensive origin checkout + build. The lock is released on return (or process exit).
	release, lerr := selfinstall.TrySingleFlight("")
	if lerr != nil {
		if lerr == selfinstall.ErrBusy {
			fmt.Fprintln(selfUpdateProgress, "self-update: another self-update is already building — skipping this run.")
			emitSelfUpdateOutcome(outcomeBusy, installTarget, "single-flight lock held")
			return
		}
		fmt.Fprintln(os.Stderr, "self-update: lock error:", lerr)
		os.Exit(1)
	}
	defer release()

	// Build from a PRISTINE detached origin/main checkout, never the live (peer-dirty)
	// tree: that gives a clean VCS stamp on the installed binary and guarantees we install
	// exactly verified origin/main, not a build contaminated with peers' work-in-progress.
	ctx := context.Background()
	var stopHeartbeat func()
	reportSelfUpdateProgress(25, "cleaning stale self-update artifacts")

	// Self-heal first: collect build worktrees leaked by PRIOR self-update ticks that
	// were killed before their source cleanup ran. This is an explicit apply of a
	// dry-run-by-default GC: exact temp-root shape + old age + dead owner + no process
	// command line + clean tree + no unpushed commit are all rechecked immediately
	// before directory-first removal. That preserves #6510's external-process rule:
	// an active or undeleted tree is never unregistered.
	buildGC := selfinstall.GarbageCollectStaleBuilds(ctx, selfinstall.RealRunner, repoRoot, selfinstall.BuildGCOptions{
		Now:          time.Now(),
		MinAge:       selfinstall.DefaultBuildGCMinAge,
		Apply:        true,
		SelfPID:      os.Getpid(),
		TempRoot:     os.TempDir(),
		BaseRef:      "origin/main",
		ProcessAlive: safecommit.ProcessAlive,
	})
	if buildGC.Reaped > 0 {
		fmt.Fprintf(selfUpdateProgress, "self-update: reaped %d stale build worktree(s) leaked by killed prior runs\n", buildGC.Reaped)
	}
	if len(buildGC.Failures) > 0 {
		fmt.Fprintf(selfUpdateProgress, "self-update: kept %d stale-build candidate(s) after apply-time revalidation/removal failure\n", len(buildGC.Failures))
	}

	// Also reap the "<binary>.old.<pid>.<i>" swap-aside files OSSwap leaks on Windows when the
	// old binary was still handle-locked at swap time. Nothing else reclaims them, so one leaks
	// per tick (a real host accumulated 211 of them, ~9 GB). We delete only asides whose owning
	// PID is provably dead — so the old .exe is no longer mapped and the file is safe to remove.
	if reaped := selfinstall.ReapStaleAsides(installTarget, os.Getpid(), safecommit.ProcessAlive); len(reaped) > 0 {
		fmt.Fprintf(selfUpdateProgress, "self-update: reaped %d stale swap-aside binary file(s) leaked by prior swaps\n", len(reaped))
	}
	// After reaping, report any REMAINING footprint: asides pinned by still-live owners that we
	// could not reclaim. A large surviving count is the early signal that swaps are outrunning
	// exits (the leak's leading edge) — visible now instead of after it reaches gigabytes.
	if fp := selfinstall.MeasureAsides(installTarget, os.Getpid(), safecommit.ProcessAlive); fp.Count >= 8 {
		fmt.Fprintf(selfUpdateProgress, "self-update: NOTE — %d swap-aside file(s) still next to the binary (%s); %d reclaimable once their owners exit\n",
			fp.Count, humanBytes(fp.Bytes), fp.DeadCount)
	}

	// The build worktree must live OUTSIDE .git (git refuses `worktree add` to a path
	// inside the git dir) and outside the live tree (so it never shows up as peer churn).
	// A per-invocation temp dir under the OS temp root satisfies both. BuildDirName encodes
	// our PID so a future run's reaper can tell a live build from a leaked corpse.
	buildDir := ""
	cleanup := func() {}
	buildRunner := selfinstall.RealRunner
	cleanupBuildCache := func() {}
	if artifact == nil {
		buildDir = filepath.Join(os.TempDir(), selfinstall.BuildDirName(os.Getpid()))
		stopHeartbeat := startSelfUpdateHeartbeat(30, "preparing pristine selected-commit worktree")
		_, preparedCleanup, perr := prepareSelfUpdateAttempt(ctx, selfinstall.RealRunner, repoRoot, headRev, buildDir)
		stopHeartbeat()
		if perr != nil {
			fmt.Fprintln(os.Stderr, "self-update:", perr)
			emitSelfUpdateOutcome(outcomePrepareFailed, installTarget, perr.Error())
			os.Exit(1)
		}
		cleanup = preparedCleanup
		defer cleanup()

		var cacheErr error
		buildRunner, cleanupBuildCache, cacheErr = selfinstall.NewSelfUpdateRunner()
		if cacheErr != nil {
			detail := "prepare update-owned Go build cache: " + cacheErr.Error()
			fmt.Fprintln(os.Stderr, "self-update:", detail)
			emitSelfUpdateOutcome(outcomeGateFailed, installTarget, detail)
			cleanup()
			os.Exit(1)
		}
		defer cleanupBuildCache()
	}
	cleanupAttempt := func() {
		cleanupBuildCache()
		cleanup()
	}

	companionBinary, companionPaths, companionErr := prepareFakDevUpdate(ctx, buildRunner, buildDir, companionPaths, headRev)
	if companionErr != nil {
		fmt.Fprintln(os.Stderr, "self-update:", companionErr)
		emitSelfUpdateOutcome(outcomeGateFailed, installTarget, companionErr.Error())
		cleanupAttempt() // os.Exit skips deferred functions; owned cache/source cleanup must run first.
		os.Exit(1)
	}
	if companionBinary != "" {
		defer os.Remove(companionBinary)
	} else {
		reportSelfUpdateProgress(46, "no fak-dev companion installed; continuing")
	}

	// Select every hot-copy target before changing any deployed bytes. The transaction must use
	// one pre-update census or a successful early swap can hide a stale later target.
	census := selfinstall.Census(selfUpdateHost(repoRoot), selfUpdateProbe)
	reportSelfUpdateProgress(48, "selecting installed targets")
	staleSiblings := []string{}
	for _, sib := range selfUpdateSiblings(repoRoot, installTarget) {
		if selfinstall.NeedsConverge(census, sib, headRev) {
			staleSiblings = append(staleSiblings, sib)
		}
	}

	candidate := ""
	candidateEphemeral := false
	var res selfinstall.Result
	if artifact != nil {
		fmt.Fprintf(selfUpdateProgress, "self-update: acquiring signed artifact for %d target(s) …\n", 1+len(staleSiblings))
		stopHeartbeat = startSelfUpdateHeartbeat(55, "acquiring signed fak artifact")
		downloaded, transfer, err := acquireSelfUpdateArtifact(ctx, selfinstall.RealRunner, installTarget, *artifact, os.TempDir())
		stopHeartbeat()
		selfUpdateReceiptTransfer = &transfer
		if err == nil {
			defer os.Remove(downloaded)
			stopHeartbeat = startSelfUpdateHeartbeat(78, "provenance-verifying signed fak artifact")
			err = selfinstall.VerifyTarget(ctx, selfinstall.RealRunner, downloaded, repoRoot, selfinstall.VerifiedTarget{
				MetadataGeneration: metadataGeneration, SourceCommit: artifact.SourceRevision,
				ArtifactDigest: artifact.SHA256, ArtifactSize: artifact.Size, AppVersion: artifact.AppVersion,
			})
			stopHeartbeat()
		}
		if err == nil {
			candidate, err = selfinstall.StoreVerifiedSlot(installTarget, downloaded, selfinstall.VerifiedTarget{
				MetadataGeneration: metadataGeneration, SourceCommit: artifact.SourceRevision,
				ArtifactDigest: artifact.SHA256, ArtifactSize: artifact.Size, AppVersion: artifact.AppVersion,
			})
		}
		if err != nil {
			res = selfinstall.Result{Stage: selfinstall.StageSmoke, Detail: err.Error()}
		} else {
			res = selfinstall.Result{
				Installed: true, Stage: selfinstall.StageSmoke, Detail: "verified signed artifact target",
				SourceCommit: artifact.SourceRevision, ArtifactSourceCommit: artifact.SourceRevision,
				BuildInputDigest: selfUpdateArtifactBindingDigest(*artifact, metadataGeneration),
				BuildEnvelope:    map[string]string{"acquisition": "signed_artifact_" + transfer.ChosenPath, "metadata_generation": fmt.Sprint(metadataGeneration)},
				ArtifactDigest:   artifact.SHA256, ArtifactSize: artifact.Size, AppVersion: artifact.AppVersion,
			}
		}
	} else {
		fmt.Fprintf(selfUpdateProgress, "self-update: building and gating origin/main for %d target(s) …\n", 1+len(staleSiblings)+len(companionPaths))
		res = selfinstall.Install(ctx, selfUpdateGateRunner(buildRunner), func(source, _ string) error {
			candidate = source
			return nil
		}, selfUpdateAttemptOptions(buildDir, installTarget, headRev))
		candidateEphemeral = true
	}
	if res.Installed {
		if metadataGeneration != 0 && strings.TrimSpace(res.AppVersion) != strings.TrimSpace(signedVersion) {
			res = selfinstall.Result{Stage: selfinstall.StageSmoke, Detail: fmt.Sprintf("signed app version mismatch: candidate reports %q want %q", res.AppVersion, signedVersion)}
		}
	}
	if res.Installed {
		selfUpdateReceiptBuildProvenance = &selfUpdateBuildProvenance{
			SourceCommit: res.SourceCommit, ArtifactSourceCommit: res.ArtifactSourceCommit,
			BuildInputDigest: res.BuildInputDigest, BuildEnvelope: res.BuildEnvelope,
			ArtifactDigest: res.ArtifactDigest, ArtifactSize: res.ArtifactSize, AppVersion: res.AppVersion, Reused: res.Reused,
		}
	}
	if !res.Installed || candidate == "" {
		detail := string(res.Stage) + ": " + res.Detail
		if candidate == "" && res.Installed {
			detail = "swap: gated candidate was not captured"
		}
		emitSelfUpdateOutcome(outcomeGateFailed, installTarget, detail)
		cleanupAttempt() // os.Exit skips deferred functions; owned cache/source cleanup must run first.
		os.Exit(1)
	}
	if candidateEphemeral {
		defer os.Remove(candidate)
	}
	removeCandidate := func() {
		if candidateEphemeral {
			_ = os.Remove(candidate)
		}
	}
	identityPath := selfinstall.IdentityStatePath(installTarget)
	priorIdentity, identityErr := selfinstall.ReadInstallIdentity(identityPath)
	if identityErr != nil {
		detail := "identity: " + identityErr.Error()
		emitSelfUpdateOutcome(outcomeGateFailed, installTarget, detail)
		cleanupAttempt()
		os.Exit(1)
	}
	primaryEqual, identityErr := selfinstall.ArtifactsEqual(candidate, installTarget)
	if identityErr != nil {
		detail := "identity: compare current artifact: " + identityErr.Error()
		emitSelfUpdateOutcome(outcomeGateFailed, installTarget, detail)
		cleanupAttempt()
		os.Exit(1)
	}

	components := make([]selfinstall.Component, 0, 1+len(staleSiblings)+len(companionPaths))
	components = append(components, selfinstall.Component{Name: "primary", Source: candidate, Target: installTarget, CompatibilityGroup: "launcher", Acquisition: selfinstall.ComponentTransferOrBuild})
	for _, target := range staleSiblings {
		components = append(components, selfinstall.Component{Name: "hot_copy", Source: candidate, Target: target, CompatibilityGroup: "launcher", Acquisition: selfinstall.ComponentReuse})
	}
	for _, target := range companionPaths {
		components = append(components, selfinstall.Component{Name: "companion", Source: companionBinary, Target: target, CompatibilityGroup: "launcher", Acquisition: selfinstall.ComponentTransferOrBuild})
	}
	componentPlan, planErr := selfinstall.PlanComponents(components)
	if planErr != nil {
		emitSelfUpdateOutcome(outcomeGateFailed, installTarget, "component plan: "+planErr.Error())
		cleanupAttempt()
		os.Exit(1)
	}
	copies := selfinstall.CopiesForActivation(components, componentPlan)
	selfUpdateReceiptTargets = make([]selfUpdateReceiptTarget, 0, len(componentPlan))
	for _, plan := range componentPlan {
		selfUpdateReceiptTargets = append(selfUpdateReceiptTargets, selfUpdateReceiptTarget{
			Role: plan.Name, Path: filepath.Clean(plan.Target), CompatibilityGroup: plan.CompatibilityGroup,
			DesiredArtifactDigest: plan.DesiredArtifactDigest, InstalledArtifactDigest: plan.InstalledArtifactDigest,
			Acquisition: string(plan.Acquisition), Activation: string(plan.Activation), Rollback: plan.Rollback,
		})
	}
	selfUpdateReceiptAttempted = len(componentPlan)
	stopHeartbeat = startSelfUpdateHeartbeat(82, "installing verified binaries")
	transactionLaunchTarget := installTarget
	if len(copies) > 0 && primaryEqual {
		transactionLaunchTarget = copies[0].Target
	}
	var transaction selfinstall.TransactionResult
	if len(copies) == 0 {
		transaction = selfinstall.Updated{Attempted: len(componentPlan)}
	} else {
		transaction = selfinstall.RunLaunchTransaction(copies, transactionLaunchTarget, selfinstall.OSSwap)
	}
	stopHeartbeat()
	switch result := transaction.(type) {
	case selfinstall.Updated:
		selfUpdateReceiptAttempted, selfUpdateReceiptChanged = result.Attempted, result.Changed
		fmt.Fprintf(selfUpdateProgress, "self-update: updated %d target(s)\n", result.Changed)
	case selfinstall.RolledBack:
		selfUpdateReceiptAttempted, selfUpdateReceiptChanged = result.Attempted, result.Changed
		detail := selfUpdateTransactionDetail(result.Err, result.RollbackErrors)
		emitSelfUpdateOutcome(outcomeRolledBack, installTarget, detail)
		removeCandidate()
		if companionBinary != "" {
			_ = os.Remove(companionBinary)
		}
		cleanupAttempt()
		os.Exit(1)
	case selfinstall.RollbackFailed:
		selfUpdateReceiptAttempted, selfUpdateReceiptChanged = result.Attempted, result.Changed
		detail := selfUpdateTransactionDetail(result.Err, result.RollbackErrors)
		emitSelfUpdateOutcome(outcomeRollbackFailed, installTarget, detail)
		removeCandidate()
		if companionBinary != "" {
			_ = os.Remove(companionBinary)
		}
		cleanupAttempt()
		os.Exit(1)
	default:
		emitSelfUpdateOutcome(outcomeRollbackFailed, installTarget, "unknown transaction result")
		removeCandidate()
		if companionBinary != "" {
			_ = os.Remove(companionBinary)
		}
		cleanupAttempt()
		os.Exit(1)
	}
	activeIdentityPath := installTarget
	if artifact != nil {
		activeIdentityPath = candidate
	}
	_, identityErr = selfinstall.AdvanceInstallIdentity(identityPath, priorIdentity, selfinstall.StateUpdate{
		SignedMetadataGeneration: metadataGeneration,
		SelectedSourceCommit:     res.SourceCommit, ArtifactSourceCommit: res.ArtifactSourceCommit,
		BuildInputDigest: res.BuildInputDigest, ArtifactDigest: res.ArtifactDigest,
		ArtifactSize: res.ArtifactSize, AppVersion: res.AppVersion,
	}, activeIdentityPath, selfinstall.LaunchPriorPath(installTarget), !primaryEqual)
	if identityErr != nil {
		emitSelfUpdateOutcome(outcomeHotCopyDivergent, installTarget, "binary transaction completed but identity persistence failed: "+identityErr.Error())
		return
	}
	if primaryEqual {
		res.Detail = strings.TrimSpace(res.Detail + "; selected-source metadata advanced without app-version change or primary activation")
	}
	// Re-census and AUDIT: every configured hot copy is either converged above or named here with
	// the build it is stuck on. The audit-only role (the repo-root gate binary, a hand-build in a
	// shared dirty checkout that may be held live) is never swapped unattended — so an explicit,
	// greppable audit line is the only thing that keeps it from drifting unnoticed.
	stopHeartbeat = startSelfUpdateHeartbeat(92, "verifying installed hot copies")
	audit := selfUpdateAudit(repoRoot, headRev)
	stopHeartbeat()
	printHotCopyAudit(audit)
	posture := selfupdate.ClassifyInstall(audit.Partition())
	if !posture.Completed {
		emitSelfUpdateOutcome(outcomeHotCopyDivergent, installTarget,
			"target installed; convergeable hot copies still need repair: "+strings.Join(audit.Lines(), " | "))
		return
	}
	if posture.AuditOnlyAttention {
		attention := "automatic targets converged; audit-only hot copies need manual attention: " + strings.Join(audit.Lines(), " | ")
		if strings.TrimSpace(res.Detail) == "" {
			res.Detail = attention
		} else {
			res.Detail = strings.TrimSpace(res.Detail) + "; " + attention
		}
	}
	if handoffSession != "" && !primaryEqual {
		ctx, cancel := context.WithTimeout(context.Background(), handoffTimeout)
		defer cancel()
		stopHeartbeat = startSelfUpdateHeartbeat(97, "handing off to updated binary")
		handoff := runSelfUpdateHandoff(ctx, installTarget, handoffSession, headRev, successorArgs)
		stopHeartbeat()
		selfUpdateReceiptHandoff = &handoff
		if handoff.State == selfinstall.HandoffRefused {
			emitSelfUpdateOutcome(outcomeHandoffRefused, installTarget, handoff.Detail)
			return
		}
		res.Detail = strings.TrimSpace(res.Detail + "; session " + handoffSession + " handed off to " + headRev)
	}
	if selfUpdateReceiptChanged == 0 {
		emitSelfUpdateOutcome(outcomeMetadataOnly, installTarget, res.Detail)
	} else {
		emitSelfUpdateOutcome(outcomeInstalled, installTarget, res.Detail)
	}

}

// prepareSelfUpdateAttempt materializes the immutable commit selected by the admission
// fetch. Passing origin/main here would recreate the TOCTOU window where that mutable ref
// can advance between selection and build; a full commit also tells PrepareOrigin that no
// second fetch is needed for this transaction.
func prepareSelfUpdateAttempt(ctx context.Context, runner selfinstall.Runner, repoRoot, expectedCommit, buildDir string) (string, func(), error) {
	expectedCommit = strings.TrimSpace(expectedCommit)
	if !isFullGitCommit(expectedCommit) {
		return "", func() {}, fmt.Errorf("self-update: selected revision is not a full 40-hex commit")
	}
	return selfinstall.PrepareOrigin(ctx, runner, repoRoot, expectedCommit, buildDir)
}

func selfUpdateAttemptOptions(buildDir, installTarget, expectedCommit string) selfinstall.Options {
	return selfinstall.Options{
		RepoRoot:       buildDir,
		Target:         installTarget,
		CacheDir:       selfinstall.CandidateCacheDir(discoverGitCommonDir(buildDir)),
		ExpectedCommit: strings.TrimSpace(expectedCommit),
	}
}
