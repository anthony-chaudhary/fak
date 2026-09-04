package preflight

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"path"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/testroute"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

const (
	// WorkspacePreflightSchema identifies version 1 of the workspace preflight report structure.
	WorkspacePreflightSchema = "fak.preflight.workspace/1"

	// WorkspaceVerdictReady indicates that all workspace preflight checks passed and execution may proceed.
	WorkspaceVerdictReady = "READY"
	// WorkspaceVerdictBlockedByLease indicates that a conflicting live lease prevents workspace execution.
	WorkspaceVerdictBlockedByLease = "BLOCKED_BY_LEASE"
	// WorkspaceVerdictWouldBeRed indicates a missing test route, broken build, or unviable preparation step.
	WorkspaceVerdictWouldBeRed = "WOULD_BE_RED"

	// ReasonCollisionRisk names a concurrency collision against another active worker tree.
	ReasonCollisionRisk = dispatchorder.ReasonCollisionRisk
	// ReasonTestRouteUnavailable indicates no suitable test harness or runner is reachable on the host.
	ReasonTestRouteUnavailable = "TEST_ROUTE_UNAVAILABLE"
	// ReasonGoBuildFailed indicates that invoking go build failed with a non-zero exit status.
	ReasonGoBuildFailed = "GO_BUILD_FAILED"
	// ReasonGoBuildRunnerMissing indicates that no implementation of GoBuildRunner was supplied.
	ReasonGoBuildRunnerMissing = "GO_BUILD_RUNNER_MISSING"
	// ReasonGoBuildWarmThreshold indicates that a warm go build invocation exceeded the target time threshold.
	ReasonGoBuildWarmThreshold = "GO_BUILD_WARM_THRESHOLD_EXCEEDED"
	// ReasonLeaseAcquireFailed indicates that the workspace preparer failed to acquire the requested write lease.
	ReasonLeaseAcquireFailed = "LEASE_ACQUIRE_FAILED"
	// ReasonWorkspacePreparerMissing indicates that nil was passed where a WorkspacePreparer was required.
	ReasonWorkspacePreparerMissing = "WORKSPACE_PREPARER_MISSING"
	// ReasonDevIndexWarmFailed indicates that warming the developer search index timed out or failed.
	ReasonDevIndexWarmFailed = "DEVINDEX_WARM_FAILED"
	// ReasonBuildCacheWarmFailed indicates that warming the compiler build cache encountered an error.
	ReasonBuildCacheWarmFailed = "BUILD_CACHE_WARM_FAILED"
	// DefaultWarmBuildThresholdMS is the default latency limit in milliseconds for a warm go build invocation.
	DefaultWarmBuildThresholdMS int64 = 2000
)

const (
	// StepAcquireWriteLease identifies the workspace preparation step for acquiring exclusive write access.
	StepAcquireWriteLease = "acquire_write_lease"
	// StepWarmGoBuildCache identifies the workspace preparation step for populating the compiler build cache.
	StepWarmGoBuildCache = "warm_go_build_cache"
	// StepResolveTestRoute identifies the workspace preparation step for locating an executable test command.
	StepResolveTestRoute = "resolve_test_route"
	// StepWarmDevIndex identifies the workspace preparation step for warming file indexers and search caches.
	StepWarmDevIndex = "warm_devindex"
)

// PackageGraph is the go-list evidence the workspace preflight needs, supplied by
// the caller so this package stays deterministic and shell-free.
type PackageGraph struct {
	FileToPackage map[string]string   `json:"file_to_package,omitempty"`
	Edges         map[string][]string `json:"edges,omitempty"`
	TotalPackages int                 `json:"total_packages,omitempty"`
}

// WorkspacePreflightInput is a task's declared workspace footprint plus host
// facts. It contains data only; no field causes I/O by itself.
type WorkspacePreflightInput struct {
	TaskID          string             `json:"task_id,omitempty"`
	Actor           string             `json:"actor,omitempty"`
	LeaseID         string             `json:"lease_id,omitempty"`
	LeaseTTLSeconds int64              `json:"lease_ttl_seconds,omitempty"`
	ReadGlobs       []string           `json:"read_globs,omitempty"`
	WriteGlobs      []string           `json:"write_globs,omitempty"`
	TaskPackages    []string           `json:"task_packages,omitempty"`
	PackageGraph    PackageGraph       `json:"package_graph,omitempty"`
	LiveLeases      []LeaseObservation `json:"live_leases,omitempty"`
	TestProbe       testroute.Probe    `json:"test_probe,omitempty"`
	TestArgs        []string           `json:"test_args,omitempty"`
	VerifyWarmBuild bool               `json:"verify_warm_build,omitempty"`
	WarmThresholdMS int64              `json:"warm_threshold_ms,omitempty"`
}

// LeaseObservation is the preflight projection of a live workspace lease.
type LeaseObservation struct {
	ID     string   `json:"id,omitempty"`
	Holder string   `json:"holder,omitempty"`
	Tree   []string `json:"tree,omitempty"`
}

// LeaseRequest is the write lease the preflight intends to acquire before the
// first edit.
type LeaseRequest struct {
	ID         string   `json:"id"`
	Holder     string   `json:"holder,omitempty"`
	Tree       []string `json:"tree"`
	TTLSeconds int64    `json:"ttl_seconds,omitempty"`
}

// DevIndexWarmRequest specifies directory globs to prime in the search indexing layer.
type DevIndexWarmRequest struct {
	Globs []string `json:"globs"`
}

// GoBuildWarmRequest configures a compiler cache warming pass for a set of Go packages.
type GoBuildWarmRequest struct {
	Packages        []string `json:"packages"`
	Command         []string `json:"command"`
	Env             []string `json:"env,omitempty"`
	VerifyWarm      bool     `json:"verify_warm,omitempty"`
	WarmThresholdMS int64    `json:"warm_threshold_ms,omitempty"`
}

// PreparationStep represents a single discrete setup action planned during workspace preflight.
type PreparationStep struct {
	Kind     string               `json:"kind"`
	Lease    *LeaseRequest        `json:"lease,omitempty"`
	Build    *GoBuildWarmRequest  `json:"build,omitempty"`
	DevIndex *DevIndexWarmRequest `json:"devindex,omitempty"`
	Route    *testroute.Route     `json:"route,omitempty"`
	Command  []string             `json:"command,omitempty"`
}

// WorkspacePreflight is the deterministic plan and first verdict for a declared
// read/write set. READY means side effects may run; BLOCKED_BY_LEASE names a live
// conflict before any edit; WOULD_BE_RED means the plan lacks an executable route
// or a later injected prewarm action failed.
type WorkspacePreflight struct {
	Schema  string `json:"schema"`
	Verdict string `json:"verdict"`
	Reason  string `json:"reason,omitempty"`
	Detail  string `json:"detail,omitempty"`

	TaskID  string `json:"task_id,omitempty"`
	Actor   string `json:"actor,omitempty"`
	LeaseID string `json:"lease_id,omitempty"`

	ReadGlobs  []string `json:"read_globs,omitempty"`
	WriteGlobs []string `json:"write_globs,omitempty"`

	DirectPackages   []string `json:"direct_packages,omitempty"`
	AffectedPackages []string `json:"affected_packages,omitempty"`
	TotalPackages    int      `json:"total_packages,omitempty"`

	LeaseRequest *LeaseRequest        `json:"lease_request,omitempty"`
	Conflict     *LeaseObservation    `json:"conflict,omitempty"`
	GoBuild      *GoBuildWarmRequest  `json:"go_build,omitempty"`
	DevIndex     *DevIndexWarmRequest `json:"devindex,omitempty"`
	TestRoute    testroute.Route      `json:"test_route,omitempty"`
	TestCommand  []string             `json:"test_command,omitempty"`
	Steps        []PreparationStep    `json:"steps,omitempty"`
}

// PlanWorkspacePreflight folds declared read/write globs, package graph
// evidence, live lease observations, and host test-route facts into one stable
// preparation plan. It never shells out or reads the filesystem. A live lease
// carrying this task's own lease id (in.LeaseID, or the id derived for the write
// set) is not a conflict: the preflight is re-runnable, so an agent that already
// holds its write lease never blocks itself on a resume or retry.
func PlanWorkspacePreflight(in WorkspacePreflightInput) WorkspacePreflight {
	readGlobs := normalizeList(in.ReadGlobs)
	writeGlobs := normalizeList(in.WriteGlobs)
	direct := taskPackagesForGlobs(in.PackageGraph, in.TaskPackages, writeGlobs)
	affected := affectedtests.Select(copyEdges(in.PackageGraph.Edges), direct)
	leaseID := strings.TrimSpace(in.LeaseID)
	if leaseID == "" && len(writeGlobs) > 0 {
		leaseID = defaultWorkspaceLeaseID(in.TaskID, writeGlobs)
	}
	actor := strings.TrimSpace(in.Actor)
	if actor == "" {
		actor = "preflight"
	}

	out := WorkspacePreflight{
		Schema:           WorkspacePreflightSchema,
		Verdict:          WorkspaceVerdictReady,
		TaskID:           strings.TrimSpace(in.TaskID),
		Actor:            actor,
		LeaseID:          leaseID,
		ReadGlobs:        readGlobs,
		WriteGlobs:       writeGlobs,
		DirectPackages:   direct,
		AffectedPackages: affected,
		TotalPackages:    in.PackageGraph.TotalPackages,
	}

	if conflict := firstLeaseConflict(writeGlobs, in.LiveLeases, leaseID); conflict != nil {
		out.Verdict = WorkspaceVerdictBlockedByLease
		out.Reason = ReasonCollisionRisk
		out.Detail = fmt.Sprintf("declared write set %v overlaps live lease %s held by %s", writeGlobs, conflict.ID, conflict.Holder)
		out.Conflict = conflict
		return out
	}

	if len(writeGlobs) > 0 {
		out.LeaseRequest = &LeaseRequest{
			ID:         leaseID,
			Holder:     actor,
			Tree:       append([]string(nil), writeGlobs...),
			TTLSeconds: in.LeaseTTLSeconds,
		}
		out.Steps = append(out.Steps, PreparationStep{Kind: StepAcquireWriteLease, Lease: out.LeaseRequest})
	}

	if len(direct) > 0 {
		req := PlanGoBuildWarm(direct, in.VerifyWarmBuild, in.WarmThresholdMS)
		out.GoBuild = &req
		out.Steps = append(out.Steps, PreparationStep{Kind: StepWarmGoBuildCache, Build: &req, Command: req.Command})
	}

	if len(affected) > 0 {
		route := testroute.Decide(in.TestProbe)
		out.TestRoute = route
		if route.Kind == testroute.KindUnavailable {
			out.Verdict = WorkspaceVerdictWouldBeRed
			out.Reason = ReasonTestRouteUnavailable
			out.Detail = route.Reason
			return out
		}
		args := append([]string(nil), in.TestArgs...)
		args = append(args, affected...)
		out.TestCommand = testroute.Command(route.CommandTemplate, args)
		routeCopy := route
		out.Steps = append(out.Steps, PreparationStep{Kind: StepResolveTestRoute, Route: &routeCopy, Command: out.TestCommand})
	}

	if len(readGlobs) > 0 {
		req := DevIndexWarmRequest{Globs: append([]string(nil), readGlobs...)}
		out.DevIndex = &req
		out.Steps = append(out.Steps, PreparationStep{Kind: StepWarmDevIndex, DevIndex: &req})
	}

	return out
}

// PlanGoBuildWarm constructs a compilation warmup plan and argument list for targeted packages.
func PlanGoBuildWarm(packages []string, verify bool, thresholdMS int64) GoBuildWarmRequest {
	pkgs := sortedUnique(packages)
	if thresholdMS <= 0 {
		thresholdMS = DefaultWarmBuildThresholdMS
	}
	cmd := append([]string{"go", "build"}, pkgs...)
	return GoBuildWarmRequest{
		Packages:        pkgs,
		Command:         cmd,
		Env:             []string{"GOTOOLCHAIN=auto"},
		VerifyWarm:      verify,
		WarmThresholdMS: thresholdMS,
	}
}

// GoBuildRun records timing, output tail, and exit status metrics from a single compiler execution.
type GoBuildRun struct {
	Command    []string `json:"command,omitempty"`
	ExitCode   int      `json:"exit_code"`
	ElapsedMS  int64    `json:"elapsed_ms,omitempty"`
	OutputTail string   `json:"output_tail,omitempty"`
}

// GoBuildWarmReport summarizes cold vs warm build latencies and evaluates against cache targets.
type GoBuildWarmReport struct {
	Request         GoBuildWarmRequest `json:"request"`
	Verdict         string             `json:"verdict"`
	Reason          string             `json:"reason,omitempty"`
	Detail          string             `json:"detail,omitempty"`
	Runs            []GoBuildRun       `json:"runs,omitempty"`
	ColdElapsedMS   int64              `json:"cold_elapsed_ms,omitempty"`
	WarmElapsedMS   int64              `json:"warm_elapsed_ms,omitempty"`
	DeltaMS         int64              `json:"delta_ms,omitempty"`
	WarmThresholdMS int64              `json:"warm_threshold_ms,omitempty"`
	Warm            bool               `json:"warm"`
}

// GoBuildRunner abstracts external compiler invocations so testing remains shell-free and deterministic.
type GoBuildRunner interface {
	RunGoBuild(context.Context, GoBuildWarmRequest) (GoBuildRun, error)
}

// WarmGoBuildCache executes the planned go-build cache warm through an injected
// runner. With VerifyWarm set it runs a second build to witness the warm-cache
// elapsed time and reports the cold-vs-warm delta without reading a clock itself.
func WarmGoBuildCache(ctx context.Context, runner GoBuildRunner, req GoBuildWarmRequest) GoBuildWarmReport {
	if req.WarmThresholdMS <= 0 {
		req.WarmThresholdMS = DefaultWarmBuildThresholdMS
	}
	rep := GoBuildWarmReport{
		Request:         req,
		Verdict:         WorkspaceVerdictReady,
		WarmThresholdMS: req.WarmThresholdMS,
	}
	if len(req.Packages) == 0 {
		rep.Warm = true
		return rep
	}
	if runner == nil {
		rep.Verdict = WorkspaceVerdictWouldBeRed
		rep.Reason = ReasonGoBuildRunnerMissing
		return rep
	}
	first, err := runner.RunGoBuild(ctx, req)
	if len(first.Command) == 0 {
		first.Command = append([]string(nil), req.Command...)
	}
	rep.Runs = append(rep.Runs, first)
	rep.ColdElapsedMS = first.ElapsedMS
	if err != nil || first.ExitCode != 0 {
		rep.Verdict = WorkspaceVerdictWouldBeRed
		rep.Reason = ReasonGoBuildFailed
		rep.Detail = errorDetail(err, first.OutputTail)
		return rep
	}
	if !req.VerifyWarm {
		rep.Warm = true
		rep.WarmElapsedMS = first.ElapsedMS
		return rep
	}
	second, err := runner.RunGoBuild(ctx, req)
	if len(second.Command) == 0 {
		second.Command = append([]string(nil), req.Command...)
	}
	rep.Runs = append(rep.Runs, second)
	rep.WarmElapsedMS = second.ElapsedMS
	rep.DeltaMS = first.ElapsedMS - second.ElapsedMS
	if err != nil || second.ExitCode != 0 {
		rep.Verdict = WorkspaceVerdictWouldBeRed
		rep.Reason = ReasonGoBuildFailed
		rep.Detail = errorDetail(err, second.OutputTail)
		return rep
	}
	rep.Warm = second.ElapsedMS <= req.WarmThresholdMS
	if !rep.Warm {
		rep.Verdict = WorkspaceVerdictWouldBeRed
		rep.Reason = ReasonGoBuildWarmThreshold
		rep.Detail = fmt.Sprintf("warm go build took %dms over threshold %dms", second.ElapsedMS, req.WarmThresholdMS)
	}
	return rep
}

// LeaseAcquireResult details whether a requested write lock was granted or refused.
type LeaseAcquireResult struct {
	Held     bool              `json:"held"`
	Reason   string            `json:"reason,omitempty"`
	Detail   string            `json:"detail,omitempty"`
	LeaseID  string            `json:"lease_id,omitempty"`
	Conflict *LeaseObservation `json:"conflict,omitempty"`
}

// WarmStepResult conveys the success or failure status and duration of an index warming action.
type WarmStepResult struct {
	OK        bool   `json:"ok"`
	Reason    string `json:"reason,omitempty"`
	Detail    string `json:"detail,omitempty"`
	ElapsedMS int64  `json:"elapsed_ms,omitempty"`
}

// WorkspacePreparer defines lifecycle methods for coordinating lease acquisition and cache warming.
type WorkspacePreparer interface {
	AcquireWriteLease(context.Context, LeaseRequest) (LeaseAcquireResult, error)
	WarmGoBuild(context.Context, GoBuildWarmRequest) (GoBuildWarmReport, error)
	WarmDevIndex(context.Context, DevIndexWarmRequest) (WarmStepResult, error)
}

// WorkspacePreparationReport captures step-by-step outcomes from executing workspace preparation hooks.
type WorkspacePreparationReport struct {
	Schema   string                 `json:"schema"`
	Verdict  string                 `json:"verdict"`
	Reason   string                 `json:"reason,omitempty"`
	Detail   string                 `json:"detail,omitempty"`
	Plan     WorkspacePreflight     `json:"plan"`
	Lease    *LeaseAcquireResult    `json:"lease,omitempty"`
	Build    *GoBuildWarmReport     `json:"build,omitempty"`
	DevIndex *WarmStepResult        `json:"devindex,omitempty"`
	Steps    []WorkspaceStepOutcome `json:"steps,omitempty"`
}

// WorkspaceStepOutcome provides granular status, timing, and error information for an individual setup step.
type WorkspaceStepOutcome struct {
	Kind      string `json:"kind"`
	OK        bool   `json:"ok"`
	Reason    string `json:"reason,omitempty"`
	Detail    string `json:"detail,omitempty"`
	ElapsedMS int64  `json:"elapsed_ms,omitempty"`
}

// PrepareWorkspace runs the side-effectful part of a READY workspace preflight
// through injected seams. Lease acquisition happens before build/devindex warming,
// so tests can prove the first edit would see an already-held lease.
func PrepareWorkspace(ctx context.Context, plan WorkspacePreflight, prep WorkspacePreparer) WorkspacePreparationReport {
	rep := WorkspacePreparationReport{Schema: plan.Schema, Verdict: plan.Verdict, Reason: plan.Reason, Detail: plan.Detail, Plan: plan}
	if plan.Verdict != WorkspaceVerdictReady {
		return rep
	}
	if prep == nil {
		rep.Verdict = WorkspaceVerdictWouldBeRed
		rep.Reason = ReasonWorkspacePreparerMissing
		return rep
	}
	if plan.LeaseRequest != nil {
		lease, err := prep.AcquireWriteLease(ctx, *plan.LeaseRequest)
		rep.Lease = &lease
		out := WorkspaceStepOutcome{Kind: StepAcquireWriteLease, OK: lease.Held, Reason: lease.Reason, Detail: errorDetail(err, lease.Detail)}
		rep.Steps = append(rep.Steps, out)
		if err != nil || !lease.Held {
			rep.Verdict = WorkspaceVerdictBlockedByLease
			rep.Reason = strmatch.FirstNonBlank(lease.Reason, ReasonLeaseAcquireFailed)
			rep.Detail = out.Detail
			return rep
		}
	}
	if plan.GoBuild != nil {
		build, err := prep.WarmGoBuild(ctx, *plan.GoBuild)
		rep.Build = &build
		out := WorkspaceStepOutcome{Kind: StepWarmGoBuildCache, OK: err == nil && build.Verdict == WorkspaceVerdictReady, Reason: build.Reason, Detail: errorDetail(err, build.Detail)}
		if len(build.Runs) > 0 {
			out.ElapsedMS = build.Runs[len(build.Runs)-1].ElapsedMS
		}
		rep.Steps = append(rep.Steps, out)
		if err != nil || build.Verdict != WorkspaceVerdictReady {
			rep.Verdict = WorkspaceVerdictWouldBeRed
			rep.Reason = strmatch.FirstNonBlank(build.Reason, ReasonBuildCacheWarmFailed)
			rep.Detail = out.Detail
			return rep
		}
	}
	if plan.DevIndex != nil {
		dev, err := prep.WarmDevIndex(ctx, *plan.DevIndex)
		rep.DevIndex = &dev
		out := WorkspaceStepOutcome{Kind: StepWarmDevIndex, OK: err == nil && dev.OK, Reason: dev.Reason, Detail: errorDetail(err, dev.Detail), ElapsedMS: dev.ElapsedMS}
		rep.Steps = append(rep.Steps, out)
		if err != nil || !dev.OK {
			rep.Verdict = WorkspaceVerdictWouldBeRed
			rep.Reason = strmatch.FirstNonBlank(dev.Reason, ReasonDevIndexWarmFailed)
			rep.Detail = out.Detail
			return rep
		}
	}
	rep.Verdict = WorkspaceVerdictReady
	rep.Reason = ""
	rep.Detail = ""
	return rep
}

func taskPackagesForGlobs(graph PackageGraph, explicit, globs []string) []string {
	if len(explicit) > 0 {
		return sortedUnique(explicit)
	}
	if moduleWide(globs) {
		return allGraphPackages(graph)
	}
	set := map[string]bool{}
	for file, pkg := range graph.FileToPackage {
		file = normalizePath(file)
		for _, g := range globs {
			if globMatches(g, file) {
				set[pkg] = true
				break
			}
		}
	}
	return sortedKeys(set)
}

// firstLeaseConflict returns the lexically-first live lease whose tree overlaps
// the declared write set, or nil when the write set is clear. A live lease whose
// ID equals ownLeaseID is the caller's own — the lease this plan would acquire
// under, or one it already holds when the preflight is re-run (a resume after
// compaction, a retry) — and never conflicts, mirroring internal/laneadmit's
// admission contract ("a live lease with this id is the caller's own"). Without
// this skip, re-running the preflight while already holding the write lease would
// self-block with BLOCKED_BY_LEASE. The authoritative serialization against real
// peers stays the atomic acquire step (PrepareWorkspace's AcquireWriteLease); this
// plan-time check only refuses declarations that visibly collide with a peer.
func firstLeaseConflict(writeGlobs []string, live []LeaseObservation, ownLeaseID string) *LeaseObservation {
	if len(writeGlobs) == 0 {
		return nil
	}
	ownLeaseID = strings.TrimSpace(ownLeaseID)
	leases := append([]LeaseObservation(nil), live...)
	sort.SliceStable(leases, func(i, j int) bool {
		if leases[i].ID != leases[j].ID {
			return leases[i].ID < leases[j].ID
		}
		return leases[i].Holder < leases[j].Holder
	})
	for _, lease := range leases {
		if ownLeaseID != "" && strings.TrimSpace(lease.ID) == ownLeaseID {
			continue
		}
		tree := normalizeList(lease.Tree)
		if dispatchorder.TreesOverlap(writeGlobs, tree) {
			l := lease
			l.Tree = tree
			return &l
		}
	}
	return nil
}

func globMatches(glob, file string) bool {
	glob = normalizePath(glob)
	file = normalizePath(file)
	if glob == "" || file == "" {
		return false
	}
	if glob == file {
		return true
	}
	if strings.HasSuffix(glob, "/**") {
		prefix := strings.TrimSuffix(glob, "/**")
		return file == prefix || strings.HasPrefix(file, prefix+"/")
	}
	if strings.HasSuffix(glob, "/*") {
		prefix := strings.TrimSuffix(glob, "/*")
		rest := strings.TrimPrefix(file, prefix+"/")
		return rest != file && rest != "" && !strings.Contains(rest, "/")
	}
	if ok, err := path.Match(glob, file); err == nil && ok {
		return true
	}
	if !strings.ContainsAny(glob, "*?[") {
		prefix := strings.TrimSuffix(glob, "/")
		return file == prefix || strings.HasPrefix(file, prefix+"/")
	}
	return false
}

func moduleWide(globs []string) bool {
	for _, g := range globs {
		switch normalizePath(g) {
		case "go.mod", "go.sum":
			return true
		}
	}
	return false
}

func allGraphPackages(graph PackageGraph) []string {
	set := map[string]bool{}
	for _, pkg := range graph.FileToPackage {
		set[pkg] = true
	}
	for pkg, deps := range graph.Edges {
		set[pkg] = true
		for _, dep := range deps {
			set[dep] = true
		}
	}
	return sortedKeys(set)
}

func defaultWorkspaceLeaseID(taskID string, writeGlobs []string) string {
	token := cleanLeaseToken(taskID)
	if token != "" {
		return "preflight-" + token
	}
	h := sha1.Sum([]byte(strings.Join(writeGlobs, "\n")))
	return "preflight-" + hex.EncodeToString(h[:])[:12]
}

func cleanLeaseToken(s string) string {
	s = strings.TrimSpace(s)
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-' || r == '_' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-.")
}

func normalizeList(xs []string) []string {
	set := map[string]bool{}
	for _, x := range xs {
		if n := normalizePath(x); n != "" {
			set[n] = true
		}
	}
	return sortedKeys(set)
}

func normalizePath(p string) string {
	p = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
	p = strings.TrimPrefix(p, "./")
	for strings.Contains(p, "//") {
		p = strings.ReplaceAll(p, "//", "/")
	}
	return strings.TrimSuffix(p, "/")
}

func sortedUnique(xs []string) []string {
	set := map[string]bool{}
	for _, x := range xs {
		x = strings.TrimSpace(x)
		if x != "" {
			set[x] = true
		}
	}
	return sortedKeys(set)
}

func sortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func copyEdges(edges map[string][]string) map[string][]string {
	out := make(map[string][]string, len(edges))
	for k, v := range edges {
		out[k] = append([]string(nil), v...)
	}
	return out
}

func errorDetail(err error, fallback string) string {
	if err != nil {
		if fallback != "" {
			return fallback + ": " + err.Error()
		}
		return err.Error()
	}
	return fallback
}
