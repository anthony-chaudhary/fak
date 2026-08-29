package armbench

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// RunSchema tags the run artifact (the raw trial ledger + per-arm rollup).
const RunSchema = "fak.armbench.run/1"

// Task is one corpus item. The runner never interprets Input beyond handing it
// to the provider — task semantics belong to the corpus importer (#6677) and the
// judge adapters (#6678), not to the harness that pairs them.
type Task struct {
	ID     string `json:"id"`
	Input  string `json:"input"`
	Expect string `json:"expect,omitempty"`
}

// Request is exactly what an arm asks the provider for one trial. It carries the
// arm's identity so a provider can install the arm's treatment, and the trial
// index so a provider that seeds per-trial can be reproducible.
type Request struct {
	ManifestIdentity string   `json:"manifest_identity"`
	ArmID            string   `json:"arm_id"`
	ArmKind          ArmKind  `json:"arm_kind"`
	Capabilities     []string `json:"capabilities,omitempty"`
	TaskID           string   `json:"task_id"`
	Trial            int      `json:"trial"`
	Position         int      `json:"position"`
	Wave             int      `json:"wave,omitempty"`
	GPUIndex         *int     `json:"gpu_index,omitempty"`
	// CUDAVisibleDevices is the exact value a process-backed provider must set
	// for CUDA_VISIBLE_DEVICES in the child environment.
	CUDAVisibleDevices string `json:"cuda_visible_devices,omitempty"`
	Input              string `json:"input"`
	PromptHash         string `json:"prompt_hash"`
	Model              Model  `json:"model"`
}

// Usage is the provider-reported accounting. Input and output tokens are kept
// SEPARATE all the way through the runner and the report: collapsing them into
// one "tokens saved" number is the single most common way a context-compression
// claim becomes untrue, since the two have different prices and different causes.
type Usage struct {
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	CostUSD      float64 `json:"cost_usd"`
}

// TotalTokens is the labelled sum. It exists so callers never hand-add the two
// and lose the label.
func (u Usage) TotalTokens() int { return u.InputTokens + u.OutputTokens }

// Latency separates the three timings a streaming provider can report, each with
// an explicit availability bit. A provider that cannot measure TTFT reports
// TTFTAvailable=false rather than 0, so the report can say "unavailable" instead
// of averaging zeros into a fictional speedup.
type Latency struct {
	WallMS              float64 `json:"wall_ms"`
	TTFTMS              float64 `json:"ttft_ms"`
	TTFTAvailable       bool    `json:"ttft_available"`
	InterTokenMS        float64 `json:"inter_token_ms"`
	InterTokenAvailable bool    `json:"inter_token_available"`
}

// CacheCounters is the provider/kernel cache accounting for one trial.
type CacheCounters struct {
	ReadTokens  int `json:"read_tokens"`
	WriteTokens int `json:"write_tokens"`
	Hits        int `json:"hits"`
	Misses      int `json:"misses"`
}

// Response is what a provider returns for one trial. RawRequest and RawResponse
// are the evidence: the runner refuses a trial that reports usage or latency
// without them (see checkEvidence).
type Response struct {
	RawRequest  string `json:"raw_request"`
	RawResponse string `json:"raw_response"`
	Text        string `json:"text"`
	Usage       Usage  `json:"usage"`
	// Accounting is the authority/completeness receipt for Usage and Cache.
	// Legacy numeric fields remain the provider adapter surface; publishable
	// totals and comparisons use this receipt so an absent field is not zero.
	Accounting AccountingReceipt `json:"accounting"`
	Latency    Latency           `json:"latency"`
	Cache      CacheCounters     `json:"cache"`
	Retries    int               `json:"retries"`
	// Failure, when non-empty, marks a trial the provider could not complete.
	// A failed trial still owes a raw request and a reason — it is counted in
	// the report's failure column, never dropped.
	Failure string `json:"failure,omitempty"`
}

// Judgment is the grader's verdict for one trial. RawJudgment is the judge's own
// evidence and is required for a graded trial, for the same reason the provider
// owes a raw response.
type Judgment struct {
	Pass        bool    `json:"pass"`
	Score       float64 `json:"score"`
	Reason      string  `json:"reason,omitempty"`
	RawJudgment string  `json:"raw_judgment"`
}

// SetupCost is an arm's one-time cost — installing a skill, warming a cache,
// building an index. It is charged to the arm and amortized across its trials in
// the report, because an arm that saves 5% per turn and costs a minute to set up
// is not a win at three turns.
type SetupCost struct {
	WallMS  float64 `json:"wall_ms"`
	Tokens  int     `json:"tokens"`
	CostUSD float64 `json:"cost_usd"`
	Note    string  `json:"note,omitempty"`
}

// Provider executes one trial for one arm.
type Provider interface {
	Complete(ctx context.Context, req Request) (Response, error)
}

// ReceiptProvider is the optional process-aware provider seam. Implementations
// return only after the launched child has been reaped and report its actual
// timing and exit outcome. Providers that implement only Provider receive a
// conservative wrapper receipt around their Complete call.
type ReceiptProvider interface {
	CompleteWithReceipt(ctx context.Context, req Request) (Response, LaunchReceipt, error)
}

// ArmSetup is the optional half of Provider: a provider that has a real per-arm
// setup cost implements it and the runner charges what it reports. A provider
// that does not is charged zero — an honest zero, not a hidden one.
type ArmSetup interface {
	SetupArm(ctx context.Context, arm Arm) (SetupCost, error)
}

// Grader grades one completed trial.
type Grader interface {
	Grade(ctx context.Context, req Request, resp Response) (Judgment, error)
}

// TrialResult is one (arm, task, trial) row: the whole evidence chain for a
// single measurement.
type TrialResult struct {
	ManifestIdentity string   `json:"manifest_identity"`
	ArmID            string   `json:"arm_id"`
	ArmKind          ArmKind  `json:"arm_kind"`
	TaskID           string   `json:"task_id"`
	Trial            int      `json:"trial"`
	Position         int      `json:"position"`
	Response         Response `json:"response"`
	Judgment         Judgment `json:"judgment"`
	// Launch records the bounded-wave execution envelope. It is optional only so
	// run/1 ledgers written before bounded waves remain readable and reportable.
	Launch *LaunchReceipt `json:"launch,omitempty"`
	// Resumed marks a row carried over from a prior run's ledger rather than
	// re-executed. It is recorded so a resumed report can never be mistaken for
	// a fresh full run.
	Resumed bool `json:"resumed"`
}

// LaunchReceipt proves where and when one arm trial ran and that its provider
// invocation returned through the reap boundary.
type LaunchReceipt struct {
	Wave               int     `json:"wave"`
	GPUIndex           *int    `json:"gpu_index,omitempty"`
	CUDAVisibleDevices string  `json:"cuda_visible_devices,omitempty"`
	StartedAt          string  `json:"started_at"`
	EndedAt            string  `json:"ended_at"`
	WallMS             float64 `json:"wall_ms"`
	ExitCode           int     `json:"exit_code"`
	Reaped             bool    `json:"reaped"`
	ReapOutcome        string  `json:"reap_outcome"`
}

// Key is the trial's resume identity. Every field in it is part of what makes
// the row unique; two rows with the same key are the same measurement, which is
// exactly what resume must not duplicate.
func (t TrialResult) Key() string {
	return strings.Join([]string{t.ManifestIdentity, t.ArmID, t.TaskID, fmt.Sprint(t.Trial)}, "|")
}

// Run is the run artifact: the manifest it came from, its identity, every raw
// trial row, and the per-arm setup costs.
type Run struct {
	Schema           string               `json:"schema"`
	ManifestIdentity string               `json:"manifest_identity"`
	Manifest         *Manifest            `json:"manifest"`
	Setup            map[string]SetupCost `json:"setup"`
	Trials           []TrialResult        `json:"trials"`
	Executed         int                  `json:"executed"`
	ResumedCount     int                  `json:"resumed"`
	// MaxParallel is the global launch bound used by this run. Zero is accepted
	// only when reading a legacy run/1 ledger and means the historical serial
	// default of one.
	MaxParallel int `json:"max_parallel,omitempty"`
}

// Options configures one Run call.
type Options struct {
	// Resume, when non-nil, supplies a prior run whose completed trials are
	// carried over instead of re-executed.
	Resume *Run
	// MaxParallel bounds provider launches across all paired units. Zero keeps
	// source compatibility with older callers and selects the serial default 1.
	MaxParallel int
	// Now is an injectable receipt clock. Nil uses time.Now.
	Now func() time.Time
}

// PairUnit is one paired execution unit: the same (task, trial) run through
// EVERY arm, back to back, in the unit's counterbalanced or seeded-random arm
// order. Pairing at this granularity is what makes the comparison paired — arms
// see the same task under the same conditions, and no arm is systematically
// advantaged by always going first (a warm-cache and a rate-limit artifact both
// favour position).
type PairUnit struct {
	TaskIndex int
	TaskID    string
	Trial     int
	ArmOrder  []string
}

// PlanUnits derives the deterministic execution plan from the manifest and the
// corpus. It is exported and pure so the ordering can be asserted in a test
// without running a provider.
func PlanUnits(m *Manifest, tasks []Task) []PairUnit {
	units := make([]PairUnit, 0, len(tasks)*m.Trials.Count)
	armIDs := make([]string, 0, len(m.Arms))
	for _, a := range m.Arms {
		armIDs = append(armIDs, a.ID)
	}
	sort.Strings(armIDs)
	for ti, task := range tasks {
		for trial := 0; trial < m.Trials.Count; trial++ {
			units = append(units, PairUnit{
				TaskIndex: ti,
				TaskID:    task.ID,
				Trial:     trial,
				ArmOrder:  armOrder(m, armIDs, ti, task.ID, trial),
			})
		}
	}
	return units
}

// armOrder implements the two order strategies. Both are deterministic given the
// manifest: counterbalanced rotates so each arm occupies each position equally,
// randomized shuffles with a PRNG seeded from (seed, task id, trial) so the
// order is unpredictable across pairs yet reproducible from the manifest alone.
func armOrder(m *Manifest, armIDs []string, taskIndex int, taskID string, trial int) []string {
	n := len(armIDs)
	out := make([]string, n)
	if n == 0 {
		return out
	}
	switch m.Trials.Order {
	case OrderRandomized:
		copy(out, armIDs)
		h := fnv.New64a()
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], uint64(m.Trials.Seed))
		_, _ = h.Write(buf[:])
		_, _ = h.Write([]byte(taskID))
		binary.LittleEndian.PutUint64(buf[:], uint64(trial))
		_, _ = h.Write(buf[:])
		rng := rand.New(rand.NewSource(int64(h.Sum64()))) //nolint:gosec // reproducibility, not secrecy
		rng.Shuffle(n, func(i, j int) { out[i], out[j] = out[j], out[i] })
	default: // OrderCounterbalanced
		shift := (taskIndex + trial) % n
		for i := 0; i < n; i++ {
			out[i] = armIDs[(i+shift)%n]
		}
	}
	return out
}

// Execute runs the manifest's arms over the corpus and returns the raw ledger.
// It fails closed: a validation refusal, a provider error, a grader error, or a
// trial with missing raw evidence aborts the run rather than returning a
// partially-evidenced report.
func Execute(ctx context.Context, m *Manifest, tasks []Task, prov Provider, grader Grader, opts Options) (*Run, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if err := validateTasksAgainstManifest(m, tasks); err != nil {
		return nil, err
	}
	if prov == nil || grader == nil {
		return nil, refuse(ReasonManifestInvalid, "a provider and a grader are both required")
	}
	maxParallel, err := normalizeMaxParallel(opts.MaxParallel)
	if err != nil {
		return nil, err
	}
	// Admission precedes setup: a duplicate or unknown assignment must not run
	// even a provider's setup child before the campaign is refused.
	if err := validateParallelAssignments(m, maxParallel); err != nil {
		return nil, err
	}
	receiptMode := opts.MaxParallel != 0 || hasGPUAssignments(m)
	identity := m.Identity()

	carried, err := carryOver(identity, m, tasks, maxParallel, opts.Resume)
	if err != nil {
		return nil, err
	}

	setup, err := chargeSetup(ctx, m, prov)
	if err != nil {
		return nil, err
	}

	units := PlanUnits(m, tasks)
	byID := map[string]Task{}
	for _, t := range tasks {
		byID[t.ID] = t
	}

	now := opts.Now
	if now == nil {
		now = time.Now
	}
	rows, executed, err := executeUnits(ctx, m, identity, units, byID, carried, prov, grader, maxParallel, receiptMode, now)
	if err != nil {
		return nil, err
	}
	sortTrials(rows)

	return &Run{
		Schema:           RunSchema,
		ManifestIdentity: identity,
		Manifest:         m,
		Setup:            setup,
		Trials:           rows,
		Executed:         executed,
		ResumedCount:     len(rows) - executed,
		MaxParallel:      recordedMaxParallel(maxParallel, receiptMode),
	}, nil
}

func hasGPUAssignments(m *Manifest) bool {
	for _, arm := range m.Arms {
		if arm.GPUIndex != nil {
			return true
		}
	}
	return false
}

func recordedMaxParallel(maxParallel int, receiptMode bool) int {
	if !receiptMode {
		return 0
	}
	return maxParallel
}

func normalizeMaxParallel(n int) (int, error) {
	if n == 0 {
		return 1, nil
	}
	if n < 0 {
		return 0, refuse(ReasonManifestInvalid, "max_parallel is %d, want >= 1", n)
	}
	return n, nil
}

func effectiveRunMaxParallel(r *Run) int {
	if r == nil || r.MaxParallel == 0 {
		return 1
	}
	return r.MaxParallel
}

// CheckRunsComparable extends manifest comparability with the additive runtime
// launch bound. Legacy ledgers normalize an omitted max_parallel to one.
func CheckRunsComparable(a, b *Run) ([]ComparabilityField, error) {
	if a == nil || b == nil || a.Manifest == nil || b.Manifest == nil {
		return nil, refuse(ReasonIncomparableManifest, "a nil run or manifest is comparable to nothing")
	}
	fields, manifestErr := CheckComparable(a.Manifest, b.Manifest)
	aMax, bMax := effectiveRunMaxParallel(a), effectiveRunMaxParallel(b)
	if aMax != bMax {
		fields = append(fields, ComparabilityField{Field: "run.max_parallel", A: strconv.Itoa(aMax), B: strconv.Itoa(bMax)})
	}
	if len(fields) == 0 {
		return nil, nil
	}
	if manifestErr != nil && aMax == bMax {
		return fields, manifestErr
	}
	names := make([]string, 0, len(fields))
	for _, field := range fields {
		names = append(names, fmt.Sprintf("%s (%q vs %q)", field.Field, field.A, field.B))
	}
	return fields, refuse(ReasonIncomparableManifest, "%d term(s) decide what was measured and disagree: %s", len(fields), strings.Join(names, "; "))
}

func validateParallelAssignments(m *Manifest, maxParallel int) error {
	if maxParallel == 1 {
		return nil
	}
	for i, arm := range m.Arms {
		if arm.GPUIndex == nil {
			return refuse(ReasonGPUAssignmentUnknown,
				"arms[%d] (%s) has no gpu_index; --max-parallel %d requires one explicit host-local GPU per arm", i, arm.ID, maxParallel)
		}
	}
	return nil
}

// carryOver indexes a resume ledger by trial key. A ledger from a DIFFERENT
// manifest identity is refused outright: silently mixing two experiments is the
// exact failure resume is supposed to prevent, and it is invisible in the
// output once it happens.
func carryOver(identity string, m *Manifest, tasks []Task, maxParallel int, resume *Run) (map[string]TrialResult, error) {
	carried := map[string]TrialResult{}
	if resume == nil {
		return carried, nil
	}
	if resume.Schema != RunSchema {
		return nil, refuse(ReasonResumeIdentityMismatch, "resume ledger schema %q is not %q", resume.Schema, RunSchema)
	}
	if resume.ManifestIdentity != identity {
		return nil, refuse(ReasonResumeIdentityMismatch, "resume ledger was produced under identity %s but this manifest is %s — resuming would mix two experiments", resume.ManifestIdentity, identity)
	}
	if resume.Manifest == nil || resume.Manifest.Identity() != identity {
		return nil, refuse(ReasonResumeIdentityMismatch, "resume ledger's embedded manifest does not hash to %s", identity)
	}
	if prior := effectiveRunMaxParallel(resume); prior != maxParallel {
		return nil, refuse(ReasonResumeIdentityMismatch, "resume ledger used max_parallel %d but this run requested %d", prior, maxParallel)
	}
	taskIDs := make(map[string]bool, len(tasks))
	for _, task := range tasks {
		taskIDs[task.ID] = true
	}
	seen := make(map[string]bool, len(resume.Trials))
	for _, t := range resume.Trials {
		if t.ManifestIdentity != identity {
			return nil, refuse(ReasonResumeIdentityMismatch, "resume row %s declares foreign identity %s", t.Key(), t.ManifestIdentity)
		}
		arm, ok := m.ArmByID(t.ArmID)
		if !ok || arm.Kind != t.ArmKind || !taskIDs[t.TaskID] || t.Trial < 0 || t.Trial >= m.Trials.Count || t.Position < 0 || t.Position >= len(m.Arms) {
			return nil, refuse(ReasonResumeIdentityMismatch, "resume row %s is outside the manifest/corpus execution plan", t.Key())
		}
		key := t.Key()
		if seen[key] {
			return nil, refuse(ReasonDuplicateTrial, "resume ledger contains trial key %s more than once", key)
		}
		seen[key] = true
		if t.Response.Failure != "" {
			// A failed trial is NOT carried over: resume re-runs it. Carrying a
			// failure forward would freeze a transient provider error into the
			// published result forever.
			continue
		}
		if err := checkEvidence(t.Response, t.Judgment); err != nil {
			return nil, fmt.Errorf("resume ledger row %s: %w", t.Key(), err)
		}
		t.Resumed = true
		carried[key] = t
	}
	return carried, nil
}

// chargeSetup charges each arm's one-time cost exactly once, before any trial
// runs. A provider with no setup pays a recorded zero.
func chargeSetup(ctx context.Context, m *Manifest, prov Provider) (map[string]SetupCost, error) {
	setup := map[string]SetupCost{}
	su, ok := prov.(ArmSetup)
	for _, arm := range m.Arms {
		if !ok {
			setup[arm.ID] = SetupCost{Note: "provider reports no setup cost"}
			continue
		}
		cost, err := su.SetupArm(ctx, arm)
		if err != nil {
			return nil, fmt.Errorf("arm %s setup: %w", arm.ID, err)
		}
		setup[arm.ID] = cost
	}
	return setup, nil
}

type executionState struct {
	ctx      context.Context
	cancel   context.CancelFunc
	now      func() time.Time
	global   chan struct{}
	devices  map[int]chan struct{}
	errOnce  sync.Once
	mu       sync.Mutex
	firstErr error
}

func newExecutionState(ctx context.Context, m *Manifest, launchCapacity int, now func() time.Time) *executionState {
	runCtx, cancel := context.WithCancel(ctx)
	s := &executionState{
		ctx: runCtx, cancel: cancel, now: now,
		global:  make(chan struct{}, launchCapacity),
		devices: map[int]chan struct{}{},
	}
	for _, arm := range m.Arms {
		if arm.GPUIndex != nil {
			s.devices[*arm.GPUIndex] = make(chan struct{}, 1)
		}
	}
	return s
}

func (s *executionState) fail(err error) {
	if err == nil {
		return
	}
	s.errOnce.Do(func() {
		s.mu.Lock()
		s.firstErr = err
		s.mu.Unlock()
		s.cancel()
	})
}

func (s *executionState) err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.firstErr
}

func (s *executionState) acquire(gpu *int) (func(), error) {
	releaseDevice := func() {}
	if gpu != nil {
		device := s.devices[*gpu]
		select {
		case device <- struct{}{}:
			releaseDevice = func() { <-device }
		case <-s.ctx.Done():
			return nil, s.ctx.Err()
		}
	}
	select {
	case s.global <- struct{}{}:
		return func() {
			<-s.global
			releaseDevice()
		}, nil
	case <-s.ctx.Done():
		releaseDevice()
		return nil, s.ctx.Err()
	}
}

// executeUnits keeps the manifest's paired-unit admission bound while applying
// one global arm-launch bound and one lock per explicitly assigned GPU.
func executeUnits(ctx context.Context, m *Manifest, identity string, units []PairUnit, byID map[string]Task, carried map[string]TrialResult, prov Provider, grader Grader, maxParallel int, receiptMode bool, now func() time.Time) ([]TrialResult, int, error) {
	var (
		mu       sync.Mutex
		rows     []TrialResult
		executed int
		wg       sync.WaitGroup
	)
	waveWidth := usefulParallelism(maxParallel, len(m.Arms))
	state := newExecutionState(ctx, m, waveWidth, now)
	defer state.cancel()
	unitConcurrency := m.Trials.Concurrency
	if unitConcurrency > len(units) {
		unitConcurrency = len(units)
	}
	if maxParallel == 1 {
		// The command's compatibility default is truly serial, including across
		// paired units. This also makes the first refusal deterministic.
		unitConcurrency = 1
	}
	sem := make(chan struct{}, unitConcurrency)

	launching := true
	for _, unit := range units {
		if !launching {
			break
		}
		select {
		case sem <- struct{}{}:
		case <-state.ctx.Done():
			launching = false
			continue
		}
		wg.Add(1)
		go func(unit PairUnit) {
			defer wg.Done()
			defer func() { <-sem }()
			got, ran, err := runUnit(state, m, identity, unit, byID[unit.TaskID], carried, prov, grader, waveWidth, receiptMode)
			if err != nil {
				state.fail(err)
				return
			}
			mu.Lock()
			defer mu.Unlock()
			rows = append(rows, got...)
			executed += ran
		}(unit)
	}
	wg.Wait()
	if err := state.err(); err != nil {
		return nil, 0, err
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	return rows, executed, nil
}

// runUnit partitions one paired arm order into deterministic, barrier-separated
// waves. Completion order inside a wave never affects ledger order.
func runUnit(state *executionState, m *Manifest, identity string, unit PairUnit, task Task, carried map[string]TrialResult, prov Provider, grader Grader, waveWidth int, receiptMode bool) ([]TrialResult, int, error) {
	out := make([]TrialResult, 0, len(unit.ArmOrder))
	executed := 0
	for start := 0; start < len(unit.ArmOrder); {
		width := waveWidth
		if remaining := len(unit.ArmOrder) - start; width > remaining {
			width = remaining
		}
		end := start + width
		wave := start/waveWidth + 1
		results := make([]*TrialResult, end-start)
		var wg sync.WaitGroup
		for pos := start; pos < end; pos++ {
			armID := unit.ArmOrder[pos]
			arm, ok := m.ArmByID(armID)
			if !ok {
				err := refuse(ReasonManifestInvalid, "planned arm %q is not declared in the manifest", armID)
				state.fail(err)
				wg.Wait()
				return nil, 0, err
			}
			key := strings.Join([]string{identity, armID, unit.TaskID, fmt.Sprint(unit.Trial)}, "|")
			if prior, ok := carried[key]; ok {
				if prior.Position != pos {
					err := refuse(ReasonResumeIdentityMismatch,
						"resume row %s records position %d but the manifest plan requires %d", key, prior.Position, pos)
					state.fail(err)
					wg.Wait()
					return nil, 0, err
				}
				priorCopy := prior
				results[pos-start] = &priorCopy
				continue
			}
			slot := pos - start
			wg.Add(1)
			go func(arm Arm, pos, slot int) {
				defer wg.Done()
				row, err := runArmTrial(state, m, identity, unit, task, arm, pos, wave, receiptMode, prov, grader)
				if err != nil {
					state.fail(err)
					return
				}
				results[slot] = &row
			}(arm, pos, slot)
		}
		wg.Wait()
		if err := state.err(); err != nil {
			return nil, 0, err
		}
		for _, row := range results {
			if row != nil {
				out = append(out, *row)
				if !row.Resumed {
					executed++
				}
			}
		}
		start = end
	}
	return out, executed, nil
}

func usefulParallelism(requested, arms int) int {
	if requested < arms {
		return requested
	}
	return arms
}

func runArmTrial(state *executionState, m *Manifest, identity string, unit PairUnit, task Task, arm Arm, pos, wave int, receiptMode bool, prov Provider, grader Grader) (TrialResult, error) {
	release, err := state.acquire(arm.GPUIndex)
	if err != nil {
		return TrialResult{}, err
	}
	defer release()

	req := Request{
		ManifestIdentity: identity,
		ArmID:            arm.ID, ArmKind: arm.Kind, Capabilities: arm.Capabilities,
		TaskID: task.ID, Trial: unit.Trial, Position: pos,
		Input: task.Input, PromptHash: arm.PromptHash, Model: m.Model,
	}
	var (
		resp   Response
		launch *LaunchReceipt
	)
	if receiptMode {
		gpu := cloneInt(arm.GPUIndex)
		req.Wave = wave
		req.GPUIndex = gpu
		if gpu != nil {
			req.CUDAVisibleDevices = strconv.Itoa(*gpu)
		}
		var receipt LaunchReceipt
		resp, receipt, err = completeWithReceipt(state.ctx, state.now, prov, req)
		if err == nil {
			receipt.Wave = wave
			receipt.GPUIndex = cloneInt(gpu)
			receipt.CUDAVisibleDevices = req.CUDAVisibleDevices
			if err = validateLaunchReceipt(receipt); err == nil {
				launch = &receipt
			}
		}
	} else {
		resp, err = prov.Complete(state.ctx, req)
	}
	if err != nil {
		return TrialResult{}, fmt.Errorf("arm %s task %s trial %d: %w", arm.ID, task.ID, unit.Trial, err)
	}
	var judgment Judgment
	if resp.Failure == "" {
		judgment, err = grader.Grade(state.ctx, req, resp)
		if err != nil {
			return TrialResult{}, fmt.Errorf("grade arm %s task %s trial %d: %w", arm.ID, task.ID, unit.Trial, err)
		}
	}
	if err := checkEvidence(resp, judgment); err != nil {
		return TrialResult{}, fmt.Errorf("arm %s task %s trial %d: %w", arm.ID, task.ID, unit.Trial, err)
	}
	return TrialResult{
		ManifestIdentity: identity, ArmID: arm.ID, ArmKind: arm.Kind,
		TaskID: task.ID, Trial: unit.Trial, Position: pos,
		Response: resp, Judgment: judgment, Launch: launch,
	}, nil
}

func completeWithReceipt(ctx context.Context, now func() time.Time, prov Provider, req Request) (Response, LaunchReceipt, error) {
	if rp, ok := prov.(ReceiptProvider); ok {
		return rp.CompleteWithReceipt(ctx, req)
	}
	started := now().UTC()
	resp, err := prov.Complete(ctx, req)
	ended := now().UTC()
	return resp, LaunchReceipt{
		StartedAt: started.Format(time.RFC3339Nano), EndedAt: ended.Format(time.RFC3339Nano),
		WallMS:   float64(ended.Sub(started)) / float64(time.Millisecond),
		ExitCode: 0, Reaped: true, ReapOutcome: "provider_returned",
	}, err
}

func validateLaunchReceipt(r LaunchReceipt) error {
	if r.Wave <= 0 {
		return refuse(ReasonManifestInvalid, "receipt wave is %d, want >= 1", r.Wave)
	}
	started, err := time.Parse(time.RFC3339Nano, r.StartedAt)
	if err != nil {
		return refuse(ReasonManifestInvalid, "receipt started_at %q is not RFC3339Nano", r.StartedAt)
	}
	ended, err := time.Parse(time.RFC3339Nano, r.EndedAt)
	if err != nil {
		return refuse(ReasonManifestInvalid, "receipt ended_at %q is not RFC3339Nano", r.EndedAt)
	}
	if ended.Before(started) || r.WallMS < 0 {
		return refuse(ReasonManifestInvalid, "receipt timing is negative (started=%s ended=%s wall_ms=%v)", r.StartedAt, r.EndedAt, r.WallMS)
	}
	if !r.Reaped || strings.TrimSpace(r.ReapOutcome) == "" {
		return refuse(ReasonManifestInvalid, "provider returned without a witnessed reap outcome")
	}
	return nil
}

func cloneInt(p *int) *int {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

// checkEvidence is the fail-closed evidence fence. A number with no raw artifact
// behind it is not a measurement, so the run is refused rather than reported
// with a gap: an absent raw response is indistinguishable, downstream, from one
// that was never inspected.
//
// A FAILED trial still owes its raw request and a failure reason — that is the
// evidence that the failure happened — but owes no response or judgment.
func checkEvidence(resp Response, j Judgment) error {
	if strings.TrimSpace(resp.RawRequest) == "" {
		return refuse(ReasonMissingRawEvidence, "trial recorded no raw request")
	}
	if resp.Failure != "" {
		return nil
	}
	if strings.TrimSpace(resp.RawResponse) == "" {
		return refuse(ReasonMissingRawEvidence, "trial recorded no raw response and no failure reason — a usage/latency row with no response behind it is not evidence")
	}
	if resp.Accounting.Schema != "" {
		if err := resp.Accounting.Validate(); err != nil {
			return fmt.Errorf("invalid accounting receipt: %w", err)
		}
	}
	if strings.TrimSpace(j.RawJudgment) == "" {
		return refuse(ReasonMissingRawEvidence, "trial recorded no raw judgment — a pass/fail with no grader evidence behind it is not evidence")
	}
	return nil
}

// sortTrials puts the ledger in a stable, diffable order regardless of the
// concurrency the run happened to use.
func sortTrials(rows []TrialResult) {
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.TaskID != b.TaskID {
			return a.TaskID < b.TaskID
		}
		if a.Trial != b.Trial {
			return a.Trial < b.Trial
		}
		return a.ArmID < b.ArmID
	})
}

// MarshalRun renders a run as strict, stable JSON.
func MarshalRun(r *Run) ([]byte, error) {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// UnmarshalRun parses a run artifact, refusing an unknown schema tag rather than
// best-effort decoding a shape it does not understand.
func UnmarshalRun(b []byte) (*Run, error) {
	var r Run
	if err := decodeStrict(b, &r); err != nil {
		return nil, err
	}
	if r.Schema != RunSchema {
		return nil, refuse(ReasonManifestInvalid, "run schema %q is not %q", r.Schema, RunSchema)
	}
	return &r, nil
}

// UnmarshalManifest parses a manifest, refusing unknown fields so a typo'd
// provenance key is a refusal rather than a silently-unpinned term.
func UnmarshalManifest(b []byte) (*Manifest, error) {
	var m Manifest
	if err := decodeStrict(b, &m); err != nil {
		return nil, refuse(ReasonManifestInvalid, "decode: %v", err)
	}
	return &m, nil
}
