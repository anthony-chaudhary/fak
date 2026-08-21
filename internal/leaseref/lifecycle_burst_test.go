package leaseref

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	burstPatternDisjoint  = "disjoint"
	burstPatternOverlap25 = "overlap25"
	burstPatternHotspot   = "hotspot"
)

type leaseLifecycleBurstCase struct {
	Children int    `json:"children"`
	Pattern  string `json:"pattern"`
}

func leaseLifecycleBurstCases() []leaseLifecycleBurstCase {
	widths := []int{1, 8, 32, 128}
	patterns := []string{burstPatternDisjoint, burstPatternOverlap25, burstPatternHotspot}
	out := make([]leaseLifecycleBurstCase, 0, len(widths)*len(patterns))
	for _, n := range widths {
		for _, pattern := range patterns {
			out = append(out, leaseLifecycleBurstCase{Children: n, Pattern: pattern})
		}
	}
	return out
}

func TestLeaseLifecycleBurstMatrix(t *testing.T) {
	got := leaseLifecycleBurstCases()
	wantWidths := []int{1, 8, 32, 128}
	wantPatterns := []string{burstPatternDisjoint, burstPatternOverlap25, burstPatternHotspot}
	if len(got) != len(wantWidths)*len(wantPatterns) {
		t.Fatalf("matrix has %d cells, want %d widths x %d patterns", len(got), len(wantWidths), len(wantPatterns))
	}
	for wi, width := range wantWidths {
		for pi, pattern := range wantPatterns {
			cell := got[wi*len(wantPatterns)+pi]
			if cell.Children != width || cell.Pattern != pattern {
				t.Fatalf("cell %d = %+v, want children=%d pattern=%s", wi*3+pi, cell, width, pattern)
			}
		}
	}
}

// TestLeaseLifecycleBurstSpine is the fast failure-before/pass-after witness for the
// benchmark harness. It drives the production acquire/sync/release/reap methods through
// a real two-clone/bare-origin topology, requires a collision refusal, exercises both
// cancellation cleanup and crash reaping, and proves the three ref stores finish empty.
func TestLeaseLifecycleBurstSpine(t *testing.T) {
	h, err := newLeaseLifecycleBurstHarness()
	if err != nil {
		if exec.ErrNotFound == err {
			t.Skip("git not available")
		}
		t.Fatal(err)
	}
	defer h.Close()

	got, err := runLeaseLifecycleBurst(context.Background(), h, leaseLifecycleBurstCase{
		Children: 8,
		Pattern:  burstPatternOverlap25,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.AcquireAdmitted == 0 || got.AcquireRefused == 0 {
		t.Fatalf("acquire admitted=%d refused=%d, want both outcomes under overlap", got.AcquireAdmitted, got.AcquireRefused)
	}
	if got.ReleaseOK == 0 {
		t.Fatalf("release_ok=%d, want at least one explicit release before reap", got.ReleaseOK)
	}
	if got.Cancelled != 1 || got.Crashed != 1 {
		t.Fatalf("cancelled=%d crashed=%d, want one of each", got.Cancelled, got.Crashed)
	}
	if got.ReapedRefs == 0 {
		t.Fatal("reaped_refs=0, want cleanup of cancellation/crash or converged remote refs")
	}
	if got.ResidualRefs != 0 {
		t.Fatalf("residual_refs=%d, want zero", got.ResidualRefs)
	}
	if got.GitProcesses == 0 {
		t.Fatal("git_processes=0: benchmark bypassed the production git-backed path")
	}
}

// BenchmarkLeaseLifecycleBurst measures the complete ref-backed lifecycle under a
// bounded N-child burst. Each cell creates two real clones and one bare origin, races
// AcquireFenced, converges admitted leases with Sync, explicitly releases healthy
// children, releases one cancelled child from a detached cleanup context, leaves one
// crashed child when possible, then Reaps all three stores at an advanced TTL instant
// and asserts zero residual refs.
//
// The harness deliberately preserves leaseref's documented boundary: same-clone CAS is
// atomic, while two clones may each admit the same id before either sees the other's
// ref. Results report that outcome; they do not claim cross-machine atomic acquisition.
func BenchmarkLeaseLifecycleBurst(b *testing.B) {
	for _, cell := range leaseLifecycleBurstCases() {
		cell := cell
		b.Run(fmt.Sprintf("N=%03d/%s", cell.Children, cell.Pattern), func(b *testing.B) {
			b.ReportAllocs()
			b.StopTimer()
			h, err := newLeaseLifecycleBurstHarness()
			if err != nil {
				if exec.ErrNotFound == err {
					b.Skip("git not available")
				}
				b.Fatal(err)
			}
			defer h.Close()
			results := make([]leaseLifecycleBurstResult, 0, b.N)
			b.StartTimer()
			for range b.N {
				got, runErr := runLeaseLifecycleBurst(context.Background(), h, cell)
				if os.Getenv("FAK_LEASEREF_BURST_JSON") == "1" {
					emitLeaseLifecycleBurstArtifact(got, h.Provenance)
				}
				if runErr != nil {
					b.Fatal(runErr)
				}
				results = append(results, got)
			}
			b.StopTimer()
			reportLeaseLifecycleBurstMetrics(b, results)
		})
	}
}

type leaseLifecycleBurstHarness struct {
	root        string
	origin      string
	clones      [2]string
	stores      [2]*Store
	originStore *Store
	metrics     *burstGitMetrics
	Provenance  leaseLifecycleBurstProvenance
}

func newLeaseLifecycleBurstHarness() (*leaseLifecycleBurstHarness, error) {
	if _, err := exec.LookPath("git"); err != nil {
		return nil, exec.ErrNotFound
	}
	root, err := os.MkdirTemp("", "fak-leaseref-burst-*")
	if err != nil {
		return nil, err
	}
	h := &leaseLifecycleBurstHarness{
		root:    root,
		origin:  filepath.Join(root, "origin.git"),
		clones:  [2]string{filepath.Join(root, "clone-a"), filepath.Join(root, "clone-b")},
		metrics: &burstGitMetrics{},
	}
	fail := func(err error) (*leaseLifecycleBurstHarness, error) {
		h.Close()
		return nil, err
	}
	if err := runBurstSetupGit(root, "init", "--bare", "-q", h.origin); err != nil {
		return fail(err)
	}
	for _, clone := range h.clones {
		if err := runBurstSetupGit(root, "clone", "-q", h.origin, clone); err != nil {
			return fail(err)
		}
	}
	for i, clone := range h.clones {
		h.stores[i] = NewWithStdinRunner(h.metrics.run, h.metrics.runStdin, clone)
	}
	h.originStore = NewWithStdinRunner(h.metrics.run, h.metrics.runStdin, h.origin)
	h.Provenance = collectLeaseLifecycleBurstProvenance(h.clones[0])
	return h, nil
}

func (h *leaseLifecycleBurstHarness) Close() {
	if h != nil && h.root != "" {
		_ = os.RemoveAll(h.root)
		h.root = ""
	}
}

func runBurstSetupGit(dir string, args ...string) error {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git %s in %s: %w\n%s", strings.Join(args, " "), dir, err, out)
	}
	return nil
}

type burstGitMetrics struct {
	mu              sync.Mutex
	processes       int
	stdoutBytes     int64
	stdinBytes      int64
	recordJSONBytes int64
}

func (m *burstGitMetrics) run(ctx context.Context, dir string, args ...string) (string, int, error) {
	out, code, err := gitRunner(ctx, dir, args...)
	m.mu.Lock()
	m.processes++
	m.stdoutBytes += int64(len(out))
	m.mu.Unlock()
	return out, code, err
}

func (m *burstGitMetrics) runStdin(ctx context.Context, dir, stdin string, args ...string) (string, int, error) {
	out, code, err := gitStdinRunner(ctx, dir, stdin, args...)
	m.mu.Lock()
	m.processes++
	m.stdinBytes += int64(len(stdin))
	m.stdoutBytes += int64(len(out))
	m.mu.Unlock()
	return out, code, err
}

func (m *burstGitMetrics) addRecordJSONBytes(n int) {
	m.mu.Lock()
	m.recordJSONBytes += int64(n)
	m.mu.Unlock()
}

func (m *burstGitMetrics) snapshot() (processes int, stdoutBytes, stdinBytes, recordJSONBytes int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.processes, m.stdoutBytes, m.stdinBytes, m.recordJSONBytes
}

type burstAcquireOutcome struct {
	StoreIndex int
	Record     Record
	Verdict    FenceVerdict
	Err        error
	Duration   time.Duration
}

type burstSyncOutcome struct {
	Err      error
	Duration time.Duration
}

type burstReleaseOutcome struct {
	Verdict  FenceVerdict
	Err      error
	Duration time.Duration
}

type leaseLifecycleBurstResult struct {
	Case                leaseLifecycleBurstCase
	Elapsed             time.Duration
	AcquireDurations    []time.Duration
	ConvergeDurations   []time.Duration
	ReleaseDurations    []time.Duration
	ReapDurations       []time.Duration
	AcquireAdmitted     int
	AcquireRefused      int
	AcquireErrors       int
	AcquireReasons      map[string]int
	ConvergeOK          int
	ConvergeErrors      int
	ConvergeReasons     map[string]int
	ReleaseOK           int
	ReleaseRefused      int
	ReleaseErrors       int
	ReleaseReasons      map[string]int
	Cancelled           int
	Crashed             int
	ReapedRefs          int
	ResidualRefs        int
	GitProcesses        int
	GitStdoutBytes      int64
	GitStdinBytes       int64
	RecordJSONBytes     int64
	VisibilityOnly      bool
	TransportBytesKnown bool
}

func runLeaseLifecycleBurst(ctx context.Context, h *leaseLifecycleBurstHarness, cell leaseLifecycleBurstCase) (leaseLifecycleBurstResult, error) {
	result := leaseLifecycleBurstResult{
		Case:                cell,
		AcquireReasons:      map[string]int{},
		ConvergeReasons:     map[string]int{},
		ReleaseReasons:      map[string]int{},
		VisibilityOnly:      true,
		TransportBytesKnown: false,
	}
	processesBefore, stdoutBefore, stdinBefore, recordJSONBefore := h.metrics.snapshot()
	started := time.Now()
	now := time.Now().UTC().Truncate(time.Second)
	acquired := make([]burstAcquireOutcome, cell.Children)
	startAcquire := make(chan struct{})
	var acquireWG sync.WaitGroup
	for i := range cell.Children {
		i := i
		acquireWG.Add(1)
		go func() {
			defer acquireWG.Done()
			<-startAcquire
			rec, storeIndex := leaseLifecycleBurstRequest(cell, i)
			if payload, err := json.Marshal(rec); err == nil {
				h.metrics.addRecordJSONBytes(len(payload))
			}
			phaseStart := time.Now()
			written, verdict, err := h.stores[storeIndex].AcquireFenced(ctx, rec, now)
			acquired[i] = burstAcquireOutcome{
				StoreIndex: storeIndex,
				Record:     written,
				Verdict:    verdict,
				Err:        err,
				Duration:   time.Since(phaseStart),
			}
		}()
	}
	close(startAcquire)
	acquireWG.Wait()

	admitted := make([]int, 0, cell.Children)
	for i, outcome := range acquired {
		result.AcquireDurations = append(result.AcquireDurations, outcome.Duration)
		switch {
		case outcome.Err != nil:
			result.AcquireErrors++
		case outcome.Verdict.OK:
			result.AcquireAdmitted++
			admitted = append(admitted, i)
		default:
			result.AcquireRefused++
			result.AcquireReasons[outcome.Verdict.Reason]++
		}
	}

	// Select failures only after the acquire burst so every pattern deterministically
	// exercises the same cleanup semantics without depending on which goroutine won.
	cancelled, crashed := -1, -1
	var cancelledContext context.Context
	if cell.Children >= 8 {
		switch {
		case len(admitted) >= 3:
			cancelled, crashed = admitted[0], admitted[1]
		case len(admitted) == 2:
			crashed = admitted[0]
		case len(admitted) == 1:
			crashed = admitted[0]
		}
	}
	if cancelled >= 0 {
		childContext, cancelChild := context.WithCancel(ctx)
		cancelChild()
		if !errors.Is(childContext.Err(), context.Canceled) {
			return result, fmt.Errorf("cancel child context: got %v, want context canceled", childContext.Err())
		}
		cancelledContext = childContext
		result.Cancelled = 1
	}
	if crashed >= 0 {
		result.Crashed = 1
	}

	syncOutcomes := make([]burstSyncOutcome, cell.Children)
	startSync := make(chan struct{})
	var syncWG sync.WaitGroup
	for _, i := range admitted {
		if i == cancelled {
			continue
		}
		i := i
		syncWG.Add(1)
		go func() {
			defer syncWG.Done()
			<-startSync
			phaseStart := time.Now()
			_, err := h.stores[acquired[i].StoreIndex].Sync(ctx, "origin", true, true)
			syncOutcomes[i] = burstSyncOutcome{Err: err, Duration: time.Since(phaseStart)}
		}()
	}
	close(startSync)
	syncWG.Wait()
	for _, i := range admitted {
		if i == cancelled {
			continue
		}
		outcome := syncOutcomes[i]
		result.ConvergeDurations = append(result.ConvergeDurations, outcome.Duration)
		if outcome.Err != nil {
			result.ConvergeErrors++
			result.ConvergeReasons[leaseLifecycleBurstSyncFailure(outcome.Err)]++
		} else {
			result.ConvergeOK++
		}
	}

	released := make([]burstReleaseOutcome, cell.Children)
	startRelease := make(chan struct{})
	var releaseWG sync.WaitGroup
	for _, i := range admitted {
		if i == crashed {
			continue
		}
		i := i
		releaseWG.Add(1)
		go func() {
			defer releaseWG.Done()
			<-startRelease
			rec := acquired[i].Record
			releaseContext := ctx
			if i == cancelled {
				releaseContext = context.WithoutCancel(cancelledContext)
			}
			phaseStart := time.Now()
			verdict, err := h.stores[acquired[i].StoreIndex].ReleaseFenced(releaseContext, rec.ID, rec.Holder, rec.Generation, now)
			released[i] = burstReleaseOutcome{Verdict: verdict, Err: err, Duration: time.Since(phaseStart)}
		}()
	}
	close(startRelease)
	releaseWG.Wait()
	for _, i := range admitted {
		if i == crashed {
			continue
		}
		outcome := released[i]
		result.ReleaseDurations = append(result.ReleaseDurations, outcome.Duration)
		switch {
		case outcome.Err != nil:
			result.ReleaseErrors++
		case outcome.Verdict.OK:
			result.ReleaseOK++
		default:
			result.ReleaseRefused++
			result.ReleaseReasons[outcome.Verdict.Reason]++
		}
	}

	future := now.Add(2 * time.Second)
	reapStores := []*Store{h.stores[0], h.stores[1], h.originStore}
	reapDurations := make([]time.Duration, len(reapStores))
	reapCounts := make([]int, len(reapStores))
	reapErrors := make([]error, len(reapStores))
	startReap := make(chan struct{})
	var reapWG sync.WaitGroup
	for i, store := range reapStores {
		i, store := i, store
		reapWG.Add(1)
		go func() {
			defer reapWG.Done()
			<-startReap
			phaseStart := time.Now()
			reaped, err := store.Reap(ctx, future)
			reapDurations[i] = time.Since(phaseStart)
			reapCounts[i] = len(reaped)
			reapErrors[i] = err
		}()
	}
	close(startReap)
	reapWG.Wait()
	for i, err := range reapErrors {
		result.ReapDurations = append(result.ReapDurations, reapDurations[i])
		result.ReapedRefs += reapCounts[i]
		if err != nil {
			return result, fmt.Errorf("reap store %d: %w", i, err)
		}
	}

	for _, dir := range []string{h.clones[0], h.clones[1], h.origin} {
		refs, err := listLeaseLifecycleBurstRefs(ctx, h.metrics, dir)
		if err != nil {
			return result, err
		}
		result.ResidualRefs += len(refs)
		if len(refs) != 0 {
			return result, fmt.Errorf("cleanup left refs in %s: %v", dir, refs)
		}
	}
	result.Elapsed = time.Since(started)
	result.GitProcesses, result.GitStdoutBytes, result.GitStdinBytes, result.RecordJSONBytes = h.metrics.snapshot()
	result.GitProcesses -= processesBefore
	result.GitStdoutBytes -= stdoutBefore
	result.GitStdinBytes -= stdinBefore
	result.RecordJSONBytes -= recordJSONBefore
	return result, nil
}

func leaseLifecycleBurstSyncFailure(err error) string {
	message := err.Error()
	switch {
	case strings.Contains(message, "push origin"):
		return "SYNC_PUSH_ERROR"
	case strings.Contains(message, "fetch origin"):
		return "SYNC_FETCH_ERROR"
	default:
		return "SYNC_INFRA_ERROR"
	}
}

func leaseLifecycleBurstRequest(cell leaseLifecycleBurstCase, child int) (Record, int) {
	store := child % 2
	id := fmt.Sprintf("disjoint-%03d", child)
	tree := fmt.Sprintf("internal/pkg%03d/**", child)
	switch cell.Pattern {
	case burstPatternOverlap25:
		overlap := (cell.Children + 3) / 4
		if child < overlap {
			id = "overlap-quarter"
			tree = "internal/shared-quarter/**"
			store = 0 // make the declared 25% collide on one CAS-visible clone
		} else {
			id = fmt.Sprintf("overlap-disjoint-%03d", child)
		}
	case burstPatternHotspot:
		id = "hotspot"
		tree = "internal/hotspot/**"
	}
	return Record{
		ID:          id,
		TreeGlobs:   []string{tree},
		Holder:      fmt.Sprintf("clone-%c:child-%03d", 'a'+rune(store), child),
		TTLSeconds:  1,
		Description: "leaseref burst benchmark",
	}, store
}

func listLeaseLifecycleBurstRefs(ctx context.Context, metrics *burstGitMetrics, dir string) ([]string, error) {
	out, code, err := metrics.run(ctx, dir, "for-each-ref", "--format=%(refname)", refPrefix)
	if err != nil {
		return nil, fmt.Errorf("list residual refs in %s: %w", dir, err)
	}
	if code != 0 {
		return nil, fmt.Errorf("list residual refs in %s exited %d", dir, code)
	}
	if strings.TrimSpace(out) == "" {
		return nil, nil
	}
	return strings.Fields(out), nil
}

func reportLeaseLifecycleBurstMetrics(b *testing.B, results []leaseLifecycleBurstResult) {
	var acquire, converge, release, reap []time.Duration
	var children, admitted, acquireRefused, acquireErrors int
	var convergeOK, convergeErrors, releaseOK, releaseRefused, releaseErrors int
	var cancelled, crashed, reaped, residual, gitProcesses int
	acquireReasons := map[string]int{}
	convergeReasons := map[string]int{}
	releaseReasons := map[string]int{}
	var elapsed time.Duration
	var gitStdout, gitStdin, recordJSON int64
	for _, result := range results {
		children += result.Case.Children
		elapsed += result.Elapsed
		acquire = append(acquire, result.AcquireDurations...)
		converge = append(converge, result.ConvergeDurations...)
		release = append(release, result.ReleaseDurations...)
		reap = append(reap, result.ReapDurations...)
		admitted += result.AcquireAdmitted
		acquireRefused += result.AcquireRefused
		acquireErrors += result.AcquireErrors
		for reason, count := range result.AcquireReasons {
			acquireReasons[reason] += count
		}
		convergeOK += result.ConvergeOK
		convergeErrors += result.ConvergeErrors
		for reason, count := range result.ConvergeReasons {
			convergeReasons[reason] += count
		}
		releaseOK += result.ReleaseOK
		releaseRefused += result.ReleaseRefused
		releaseErrors += result.ReleaseErrors
		for reason, count := range result.ReleaseReasons {
			releaseReasons[reason] += count
		}
		cancelled += result.Cancelled
		crashed += result.Crashed
		reaped += result.ReapedRefs
		residual += result.ResidualRefs
		gitProcesses += result.GitProcesses
		gitStdout += result.GitStdoutBytes
		gitStdin += result.GitStdinBytes
		recordJSON += result.RecordJSONBytes
	}
	reportBurstPhase := func(name string, samples []time.Duration) {
		b.ReportMetric(durationMillis(burstPercentile(samples, 0.50)), name+"_p50_ms")
		b.ReportMetric(durationMillis(burstPercentile(samples, 0.95)), name+"_p95_ms")
		b.ReportMetric(durationMillis(burstPercentile(samples, 0.99)), name+"_p99_ms")
	}
	reportBurstPhase("acquire", acquire)
	reportBurstPhase("converge", converge)
	reportBurstPhase("release", release)
	reportBurstPhase("reap", reap)
	perRun := float64(len(results))
	b.ReportMetric(float64(children)/elapsed.Seconds(), "children_per_s")
	b.ReportMetric(float64(admitted)/elapsed.Seconds(), "admitted_per_s")
	b.ReportMetric(float64(admitted)/perRun, "acquire_admitted")
	b.ReportMetric(float64(acquireRefused)/perRun, "acquire_refused")
	b.ReportMetric(float64(acquireErrors)/perRun, "acquire_errors")
	b.ReportMetric(float64(acquireReasons[ReasonLeaseHeld])/perRun, "acquire_lease_held")
	b.ReportMetric(float64(acquireReasons[ReasonLeaseContended])/perRun, "acquire_lease_contended")
	b.ReportMetric(float64(convergeOK)/perRun, "converge_ok")
	b.ReportMetric(float64(convergeOK)/elapsed.Seconds(), "converged_per_s")
	b.ReportMetric(float64(convergeErrors)/perRun, "converge_errors")
	b.ReportMetric(float64(convergeReasons["SYNC_PUSH_ERROR"])/perRun, "converge_push_errors")
	b.ReportMetric(float64(convergeReasons["SYNC_FETCH_ERROR"])/perRun, "converge_fetch_errors")
	b.ReportMetric(float64(releaseOK)/perRun, "release_ok")
	b.ReportMetric(float64(releaseOK)/elapsed.Seconds(), "released_per_s")
	b.ReportMetric(float64(releaseRefused)/perRun, "release_refused")
	b.ReportMetric(float64(releaseErrors)/perRun, "release_errors")
	b.ReportMetric(float64(releaseReasons[ReasonStaleLease])/perRun, "release_stale_lease")
	b.ReportMetric(float64(releaseReasons[ReasonLeaseContended])/perRun, "release_lease_contended")
	b.ReportMetric(float64(cancelled)/perRun, "cancelled")
	b.ReportMetric(float64(crashed)/perRun, "crashed")
	b.ReportMetric(float64(reaped)/perRun, "reaped_refs")
	b.ReportMetric(float64(residual)/perRun, "residual_refs")
	b.ReportMetric(float64(gitProcesses)/perRun, "git_processes")
	b.ReportMetric(float64(gitStdout)/perRun, "git_stdout_B")
	b.ReportMetric(float64(gitStdin)/perRun, "git_stdin_B")
	b.ReportMetric(float64(recordJSON)/perRun, "record_json_B")
}

func burstPercentile(samples []time.Duration, percentile float64) time.Duration {
	if len(samples) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), samples...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	index := int(float64(len(sorted))*percentile + 0.999999999)
	if index < 1 {
		index = 1
	}
	if index > len(sorted) {
		index = len(sorted)
	}
	return sorted[index-1]
}

func durationMillis(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

type leaseLifecycleBurstProvenance struct {
	Commit              string `json:"commit"`
	SourcePath          string `json:"source_path"`
	SourceSHA256        string `json:"source_sha256"`
	SourceWorkingStatus string `json:"source_working_status"`
	GoVersion           string `json:"go_version"`
	GitVersion          string `json:"git_version"`
	OS                  string `json:"os"`
	Architecture        string `json:"architecture"`
	CPU                 string `json:"cpu"`
	TempRoot            string `json:"temp_root"`
	Storage             string `json:"storage"`
	GitObjectFormat     string `json:"git_object_format"`
}

func collectLeaseLifecycleBurstProvenance(clone string) leaseLifecycleBurstProvenance {
	const sourcePath = "internal/leaseref/lifecycle_burst_test.go"
	repoRoot := burstCommandOutput("", "git", "rev-parse", "--show-toplevel")
	return leaseLifecycleBurstProvenance{
		Commit:              burstCommandOutput(repoRoot, "git", "rev-parse", "HEAD"),
		SourcePath:          sourcePath,
		SourceSHA256:        burstFileSHA256(filepath.Join(repoRoot, filepath.FromSlash(sourcePath))),
		SourceWorkingStatus: burstCommandOutput(repoRoot, "git", "status", "--short", "--", sourcePath),
		GoVersion:           runtime.Version(),
		GitVersion:          burstCommandOutput("", "git", "--version"),
		OS:                  runtime.GOOS,
		Architecture:        runtime.GOARCH,
		CPU:                 leaseLifecycleBurstCPU(),
		TempRoot:            os.TempDir(),
		Storage:             "local temporary filesystem; two working clones plus one bare origin over file transport",
		GitObjectFormat:     burstCommandOutput(clone, "git", "rev-parse", "--show-object-format"),
	}
}

func burstFileSHA256(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return "unavailable"
	}
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

func burstCommandOutput(dir, name string, args ...string) string {
	cmd := exec.Command(name, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.Output()
	if err != nil {
		return "unavailable"
	}
	return strings.TrimSpace(string(out))
}

func leaseLifecycleBurstCPU() string {
	if data, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			key, value, ok := strings.Cut(line, ":")
			if ok && strings.TrimSpace(key) == "model name" {
				return strings.TrimSpace(value)
			}
		}
	}
	if value := strings.TrimSpace(os.Getenv("PROCESSOR_IDENTIFIER")); value != "" {
		return value
	}
	return runtime.GOARCH
}

type burstPhaseArtifact struct {
	P50Milliseconds float64 `json:"p50_ms"`
	P95Milliseconds float64 `json:"p95_ms"`
	P99Milliseconds float64 `json:"p99_ms"`
}

type leaseLifecycleBurstArtifact struct {
	Schema               string                        `json:"schema"`
	CapturedAt           string                        `json:"captured_at"`
	Case                 leaseLifecycleBurstCase       `json:"case"`
	Provenance           leaseLifecycleBurstProvenance `json:"provenance"`
	Acquire              burstPhaseArtifact            `json:"acquire"`
	Converge             burstPhaseArtifact            `json:"converge"`
	Release              burstPhaseArtifact            `json:"release"`
	Reap                 burstPhaseArtifact            `json:"reap"`
	ElapsedMilliseconds  float64                       `json:"elapsed_ms"`
	ChildrenPerSecond    float64                       `json:"children_per_second"`
	AdmittedPerSecond    float64                       `json:"admitted_per_second"`
	ConvergedPerSecond   float64                       `json:"converged_per_second"`
	ReleasedPerSecond    float64                       `json:"released_per_second"`
	AcquireAdmitted      int                           `json:"acquire_admitted"`
	AcquireRefused       int                           `json:"acquire_refused"`
	AcquireErrors        int                           `json:"acquire_errors"`
	AcquireReasons       map[string]int                `json:"acquire_reasons"`
	ConvergeOK           int                           `json:"converge_ok"`
	ConvergeErrors       int                           `json:"converge_errors"`
	ConvergeReasons      map[string]int                `json:"converge_reasons"`
	ReleaseOK            int                           `json:"release_ok"`
	ReleaseRefused       int                           `json:"release_refused"`
	ReleaseErrors        int                           `json:"release_errors"`
	ReleaseReasons       map[string]int                `json:"release_reasons"`
	Cancelled            int                           `json:"cancelled"`
	Crashed              int                           `json:"crashed"`
	ReapedRefs           int                           `json:"reaped_refs"`
	ResidualRefs         int                           `json:"residual_refs"`
	GitProcesses         int                           `json:"git_processes"`
	GitStdoutBytes       int64                         `json:"git_stdout_bytes"`
	GitStdinBytes        int64                         `json:"git_stdin_bytes"`
	RecordJSONBytes      int64                         `json:"record_json_bytes"`
	TransportBytes       *int64                        `json:"transport_bytes"`
	TransportBytesReason string                        `json:"transport_bytes_reason"`
	CrossMachineBoundary string                        `json:"cross_machine_boundary"`
}

func emitLeaseLifecycleBurstArtifact(result leaseLifecycleBurstResult, provenance leaseLifecycleBurstProvenance) {
	phase := func(samples []time.Duration) burstPhaseArtifact {
		return burstPhaseArtifact{
			P50Milliseconds: durationMillis(burstPercentile(samples, 0.50)),
			P95Milliseconds: durationMillis(burstPercentile(samples, 0.95)),
			P99Milliseconds: durationMillis(burstPercentile(samples, 0.99)),
		}
	}
	row := leaseLifecycleBurstArtifact{
		Schema:               "fak-leaseref-burst/1",
		CapturedAt:           time.Now().UTC().Format(time.RFC3339Nano),
		Case:                 result.Case,
		Provenance:           provenance,
		Acquire:              phase(result.AcquireDurations),
		Converge:             phase(result.ConvergeDurations),
		Release:              phase(result.ReleaseDurations),
		Reap:                 phase(result.ReapDurations),
		ElapsedMilliseconds:  durationMillis(result.Elapsed),
		ChildrenPerSecond:    float64(result.Case.Children) / result.Elapsed.Seconds(),
		AdmittedPerSecond:    float64(result.AcquireAdmitted) / result.Elapsed.Seconds(),
		ConvergedPerSecond:   float64(result.ConvergeOK) / result.Elapsed.Seconds(),
		ReleasedPerSecond:    float64(result.ReleaseOK) / result.Elapsed.Seconds(),
		AcquireAdmitted:      result.AcquireAdmitted,
		AcquireRefused:       result.AcquireRefused,
		AcquireErrors:        result.AcquireErrors,
		AcquireReasons:       result.AcquireReasons,
		ConvergeOK:           result.ConvergeOK,
		ConvergeErrors:       result.ConvergeErrors,
		ConvergeReasons:      result.ConvergeReasons,
		ReleaseOK:            result.ReleaseOK,
		ReleaseRefused:       result.ReleaseRefused,
		ReleaseErrors:        result.ReleaseErrors,
		ReleaseReasons:       result.ReleaseReasons,
		Cancelled:            result.Cancelled,
		Crashed:              result.Crashed,
		ReapedRefs:           result.ReapedRefs,
		ResidualRefs:         result.ResidualRefs,
		GitProcesses:         result.GitProcesses,
		GitStdoutBytes:       result.GitStdoutBytes,
		GitStdinBytes:        result.GitStdinBytes,
		RecordJSONBytes:      result.RecordJSONBytes,
		TransportBytes:       nil,
		TransportBytesReason: "the production Runner exposes command stdout and stdin, not Git file-transport payload bytes; process I/O bytes are reported separately and no network-byte estimate is inferred",
		CrossMachineBoundary: "visibility and convergence only; same-clone update-ref CAS is atomic, but two clones can each admit the same id before sync",
	}
	data, err := json.Marshal(row)
	if err == nil {
		fmt.Printf("LEASEREF_BURST_JSON %s\n", data)
	}
}
