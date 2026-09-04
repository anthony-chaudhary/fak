package agenttopo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Structured refusal reasons for reflex dispatch.
const (
	RefuseTreeCollision = "REFUSE_TREE_COLLISION"
	RefuseInvalidS0S1   = "REFUSE_INVALID_S0S1"
	RefuseTimeout       = "REFUSE_TIMEOUT"
)

var (
	// ErrTreeCollision is returned when a task collides with an active lane lease or file tree.
	ErrTreeCollision = errors.New("agenttopo: REFUSE_TREE_COLLISION: tree-disjoint lease conflict")
	// ErrInvalidS0S1 is returned when a candidate task violates atomic S0/S1 leaf boundaries.
	ErrInvalidS0S1 = errors.New("agenttopo: REFUSE_INVALID_S0S1: task violates atomic S0/S1 leaf bounds")
	// ErrReflexTimeout is returned when a micro-agent worker exceeds its sub-second execution budget.
	ErrReflexTimeout = errors.New("agenttopo: REFUSE_TIMEOUT: reflex worker exceeded execution budget")
)

// ReflexWorkerProfile represents the fast-spawn execution profile for atomic leaf tasks.
// It enforces sub-second execution budgets, atomic S0/S1 file-touch boundaries, and failure
// quarantine to prevent coordinator context pollution.
type ReflexWorkerProfile struct {
	Name               string        `json:"name"`
	MaxDuration        time.Duration `json:"max_duration"`
	SpawnOverheadLimit time.Duration `json:"spawn_overhead_limit"`
	MaxTouchedFiles    int           `json:"max_touched_files"`
	QuarantineFailures bool          `json:"quarantine_failures"`
}

// DefaultReflexWorkerProfile returns the canonical fast-spawn profile for atomic leaf tasks:
// sub-second execution ceiling (800ms), 3-file atomic boundary, and isolated failure quarantine.
func DefaultReflexWorkerProfile() ReflexWorkerProfile {
	return ReflexWorkerProfile{
		Name:               "reflex-micro-agent",
		MaxDuration:        800 * time.Millisecond,
		SpawnOverheadLimit: 50 * time.Millisecond,
		MaxTouchedFiles:    3,
		QuarantineFailures: true,
	}
}

// NewReflexWorkerProfile constructs a custom fast-spawn profile with sub-second bounds.
func NewReflexWorkerProfile(name string, maxDuration time.Duration) ReflexWorkerProfile {
	if maxDuration <= 0 || maxDuration > time.Second {
		maxDuration = 800 * time.Millisecond
	}
	return ReflexWorkerProfile{
		Name:               name,
		MaxDuration:        maxDuration,
		SpawnOverheadLimit: 50 * time.Millisecond,
		MaxTouchedFiles:    3,
		QuarantineFailures: true,
	}
}

// SubSecond reports whether the profile enforces a sub-second execution bound.
func (p ReflexWorkerProfile) SubSecond() bool {
	return p.MaxDuration > 0 && p.MaxDuration <= time.Second
}

// ExecuteLeaf executes task in an isolated worker context with a sub-second timeout.
func (p ReflexWorkerProfile) ExecuteLeaf(ctx context.Context, task ReflexTask) (int, string, error) {
	if p.MaxDuration > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, p.MaxDuration)
		defer cancel()
	}

	if task.ExecuteFn == nil {
		return 0, "", nil
	}

	type execResult struct {
		code int
		out  string
		err  error
	}
	done := make(chan execResult, 1)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				done <- execResult{
					code: 2,
					out:  fmt.Sprintf("panic: %v", r),
					err:  fmt.Errorf("worker panic: %v", r),
				}
			}
		}()
		code, out, err := task.ExecuteFn(ctx)
		done <- execResult{code: code, out: out, err: err}
	}()

	select {
	case res := <-done:
		return res.code, res.out, res.err
	case <-ctx.Done():
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return 124, "", ErrReflexTimeout
		}
		return 1, "", ctx.Err()
	}
}

// ReflexTask models an atomic S0/S1 candidate task evaluated and dispatched
// by ReflexDispatcher.
type ReflexTask struct {
	ID             string                                         `json:"id"`
	Lane           string                                         `json:"lane"`
	TreePatterns   []string                                       `json:"tree_patterns,omitempty"`
	TouchedFiles   []string                                       `json:"touched_files,omitempty"`
	WitnessCommand string                                         `json:"witness_command"`
	Description    string                                         `json:"description,omitempty"`
	ExecuteFn      func(ctx context.Context) (int, string, error) `json:"-"`
}

// ValidateS0S1 checks that the task adheres to atomic S0/S1 invariants:
// 1. Non-empty ID and Lane.
// 2. Non-empty single witness command (no chained pipelines &&, ;, ||, \n).
// 3. File touch surface bounded to maxFiles (1 to 3).
func (t ReflexTask) ValidateS0S1(maxFiles int) error {
	if strings.TrimSpace(t.ID) == "" {
		return fmt.Errorf("%w: task ID is required", ErrInvalidS0S1)
	}
	if strings.TrimSpace(t.Lane) == "" {
		return fmt.Errorf("%w: lane is required", ErrInvalidS0S1)
	}
	cmd := strings.TrimSpace(t.WitnessCommand)
	if cmd == "" {
		return fmt.Errorf("%w: witness command is required", ErrInvalidS0S1)
	}
	if strings.Contains(cmd, "&&") || strings.Contains(cmd, ";") || strings.Contains(cmd, "||") || strings.Contains(cmd, "\n") {
		return fmt.Errorf("%w: witness command must be a single command, not chained", ErrInvalidS0S1)
	}
	if maxFiles <= 0 {
		maxFiles = 3
	}
	fileCount := len(t.TouchedFiles)
	if fileCount == 0 {
		fileCount = len(t.TreePatterns)
	}
	if fileCount > maxFiles {
		return fmt.Errorf("%w: file count %d exceeds atomic S0/S1 ceiling of %d", ErrInvalidS0S1, fileCount, maxFiles)
	}
	return nil
}

// SpeculativeReceipt captures the compact, witnessed result of an atomic leaf task.
// It deliberately omits voluminous compiler stderr, panic traces, or command logs
// to prevent coordinator context bloat.
type SpeculativeReceipt struct {
	TaskID         string        `json:"task_id"`
	Lane           string        `json:"lane"`
	WitnessCommand string        `json:"witness_command"`
	ExitCode       int           `json:"exit_code"`
	ElapsedTime    time.Duration `json:"elapsed_time"`
	Status         string        `json:"status,omitempty"`
	TouchedFiles   []string      `json:"touched_files,omitempty"`
	Summary        string        `json:"summary,omitempty"`
	Error          string        `json:"error,omitempty"`
}

// CompactJSON serializes the receipt to a single compact line for coordinator ingestion.
func (r SpeculativeReceipt) CompactJSON() string {
	b, err := json.Marshal(r)
	if err != nil {
		return fmt.Sprintf(`{"task_id":%q,"lane":%q,"exit_code":%d,"witness_command":%q}`,
			r.TaskID, r.Lane, r.ExitCode, r.WitnessCommand)
	}
	return string(b)
}

// IsSuccess reports whether the receipt represents a successful, zero-exit execution.
func (r SpeculativeReceipt) IsSuccess() bool {
	return r.ExitCode == 0 && r.Error == ""
}

// LaneLease represents an active tree-disjoint lease held by a reflex worker.
type LaneLease struct {
	TaskID       string    `json:"task_id"`
	Lane         string    `json:"lane"`
	TreePatterns []string  `json:"tree_patterns"`
	WorkerID     string    `json:"worker_id"`
	AcquiredAt   time.Time `json:"acquired_at"`
}

// ReflexDispatcher evaluates candidate tasks against tree-disjoint lane leases,
// dispatches workers for atomic S0/S1 leaf tasks under a fast-spawn profile,
// and ingests compact, witnessed receipts without polluting coordinator context.
type ReflexDispatcher struct {
	mu             sync.Mutex
	profile        ReflexWorkerProfile
	activeLeases   map[string]LaneLease
	receipts       map[string]SpeculativeReceipt
	receiptOrder   []string
	coordinatorLog []string
	rawBytesSeen   int
	receiptBytes   int
	spawnCounter   int64
}

// NewReflexDispatcher creates a new dispatcher configured with the given profile.
func NewReflexDispatcher(profile ...ReflexWorkerProfile) *ReflexDispatcher {
	p := DefaultReflexWorkerProfile()
	if len(profile) > 0 {
		p = profile[0]
		if p.MaxDuration <= 0 {
			p.MaxDuration = 800 * time.Millisecond
		}
		if p.MaxTouchedFiles <= 0 {
			p.MaxTouchedFiles = 3
		}
	}
	return &ReflexDispatcher{
		profile:      p,
		activeLeases: make(map[string]LaneLease),
		receipts:     make(map[string]SpeculativeReceipt),
	}
}

// Profile returns the active ReflexWorkerProfile.
func (d *ReflexDispatcher) Profile() ReflexWorkerProfile {
	return d.profile
}

// Evaluate evaluates whether a candidate task can be admitted under S0/S1 invariants
// and tree-disjoint lane leases without mutating dispatcher state.
func (d *ReflexDispatcher) Evaluate(task ReflexTask) error {
	if err := task.ValidateS0S1(d.profile.MaxTouchedFiles); err != nil {
		return err
	}

	patterns := d.resolveTreePatterns(task)

	d.mu.Lock()
	defer d.mu.Unlock()

	lane := strings.TrimSpace(task.Lane)
	for _, active := range d.activeLeases {
		if strings.EqualFold(active.Lane, lane) {
			return fmt.Errorf("%w: lane %q is currently held by task %q", ErrTreeCollision, lane, active.TaskID)
		}
		if treesCollide(active.TreePatterns, patterns) {
			return fmt.Errorf("%w: task %q tree patterns %v collide with active task %q on lane %q (%v)",
				ErrTreeCollision, task.ID, patterns, active.TaskID, active.Lane, active.TreePatterns)
		}
	}
	return nil
}

// AcquireLease attempts to acquire a tree-disjoint lane lease for the task.
func (d *ReflexDispatcher) AcquireLease(task ReflexTask) (LaneLease, error) {
	if err := task.ValidateS0S1(d.profile.MaxTouchedFiles); err != nil {
		return LaneLease{}, err
	}
	patterns := d.resolveTreePatterns(task)
	workerID := fmt.Sprintf("reflex-worker-%d", atomic.AddInt64(&d.spawnCounter, 1))

	d.mu.Lock()
	defer d.mu.Unlock()

	lane := strings.TrimSpace(task.Lane)
	for _, active := range d.activeLeases {
		if strings.EqualFold(active.Lane, lane) {
			return LaneLease{}, fmt.Errorf("%w: lane %q is held by task %q", ErrTreeCollision, lane, active.TaskID)
		}
		if treesCollide(active.TreePatterns, patterns) {
			return LaneLease{}, fmt.Errorf("%w: task %q trees %v collide with active task %q on lane %q (%v)",
				ErrTreeCollision, task.ID, patterns, active.TaskID, active.Lane, active.TreePatterns)
		}
	}

	lease := LaneLease{
		TaskID:       task.ID,
		Lane:         lane,
		TreePatterns: append([]string(nil), patterns...),
		WorkerID:     workerID,
		AcquiredAt:   time.Now().UTC(),
	}
	d.activeLeases[task.ID] = lease
	return lease, nil
}

// ReleaseLease releases an active lane lease by task ID.
func (d *ReflexDispatcher) ReleaseLease(taskID string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	if _, ok := d.activeLeases[taskID]; ok {
		delete(d.activeLeases, taskID)
		return true
	}
	return false
}

// ActiveLeases returns a snapshot of all currently held lane leases.
func (d *ReflexDispatcher) ActiveLeases() []LaneLease {
	d.mu.Lock()
	defer d.mu.Unlock()
	res := make([]LaneLease, 0, len(d.activeLeases))
	for _, l := range d.activeLeases {
		res = append(res, l)
	}
	return res
}

// Dispatch executes an atomic leaf task using a fast-spawn reflex micro-agent.
// It acquires a tree-disjoint lane lease, executes the leaf task under sub-second bounds,
// releases the lease, and ingests a compact SpeculativeReceipt without coordinator context bloat.
func (d *ReflexDispatcher) Dispatch(ctx context.Context, task ReflexTask) (SpeculativeReceipt, error) {
	lease, err := d.AcquireLease(task)
	if err != nil {
		receipt := SpeculativeReceipt{
			TaskID:         task.ID,
			Lane:           task.Lane,
			WitnessCommand: task.WitnessCommand,
			ExitCode:       1,
			Status:         "REFUSED",
			Error:          err.Error(),
		}
		_ = d.IngestReceipt(receipt)
		return receipt, err
	}
	defer d.ReleaseLease(task.ID)

	start := time.Now()
	exitCode, output, runErr := d.profile.ExecuteLeaf(ctx, task)
	elapsed := time.Since(start)

	status := "COMPLETED"
	var errStr string
	if exitCode != 0 || runErr != nil {
		status = "FAILED"
		if runErr != nil {
			errStr = runErr.Error()
		} else {
			errStr = fmt.Sprintf("witness exit code %d", exitCode)
		}
	}

	receipt := SpeculativeReceipt{
		TaskID:         task.ID,
		Lane:           lease.Lane,
		WitnessCommand: task.WitnessCommand,
		ExitCode:       exitCode,
		ElapsedTime:    elapsed,
		Status:         status,
		TouchedFiles:   append([]string(nil), task.TouchedFiles...),
		Summary:        task.Description,
		Error:          errStr,
	}

	_ = d.IngestReceipt(receipt, output)
	return receipt, runErr
}

// DispatchParallel executes candidate tasks concurrently across tree-disjoint lanes.
func (d *ReflexDispatcher) DispatchParallel(ctx context.Context, tasks []ReflexTask) ([]SpeculativeReceipt, error) {
	results := make([]SpeculativeReceipt, len(tasks))
	errs := make([]error, len(tasks))
	var wg sync.WaitGroup

	for i, task := range tasks {
		wg.Add(1)
		go func(idx int, t ReflexTask) {
			defer wg.Done()
			r, err := d.Dispatch(ctx, t)
			results[idx] = r
			errs[idx] = err
		}(i, task)
	}

	wg.Wait()

	var firstErr error
	for _, err := range errs {
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return results, firstErr
}

// IngestReceipt accepts a SpeculativeReceipt and optional raw transcripts.
// Raw transcripts are quarantined and never enter coordinator context.
func (d *ReflexDispatcher) IngestReceipt(receipt SpeculativeReceipt, rawTranscript ...string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	var rawText string
	if len(rawTranscript) > 0 {
		rawText = strings.Join(rawTranscript, "\n")
	}
	d.rawBytesSeen += len(rawText)

	d.receipts[receipt.TaskID] = receipt
	d.receiptOrder = append(d.receiptOrder, receipt.TaskID)

	compact := receipt.CompactJSON()
	d.receiptBytes += len(compact)

	// Ingest compact representation into coordinator context log:
	// If QuarantineFailures is enabled and the task did not succeed, do NOT append to coordinator log.
	// Otherwise, append ONLY the compact receipt, NEVER the raw transcript.
	if receipt.ExitCode == 0 && receipt.Error == "" {
		d.coordinatorLog = append(d.coordinatorLog, compact)
	} else if !d.profile.QuarantineFailures {
		d.coordinatorLog = append(d.coordinatorLog, compact)
	}

	return nil
}

// Receipts returns all ingested receipts in receipt order.
func (d *ReflexDispatcher) Receipts() []SpeculativeReceipt {
	d.mu.Lock()
	defer d.mu.Unlock()
	res := make([]SpeculativeReceipt, 0, len(d.receiptOrder))
	for _, id := range d.receiptOrder {
		res = append(res, d.receipts[id])
	}
	return res
}

// GetReceipt retrieves a receipt by task ID.
func (d *ReflexDispatcher) GetReceipt(taskID string) (SpeculativeReceipt, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	r, ok := d.receipts[taskID]
	return r, ok
}

// CoordinatorLog returns all entries in the coordinator's message log.
// It is guaranteed to contain zero raw compiler/panic traces.
func (d *ReflexDispatcher) CoordinatorLog() []string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return append([]string(nil), d.coordinatorLog...)
}

// CoordinatorMessages is an alias for CoordinatorLog.
func (d *ReflexDispatcher) CoordinatorMessages() []string {
	return d.CoordinatorLog()
}

// CompletedTasks returns all receipts that completed with exit code 0.
func (d *ReflexDispatcher) CompletedTasks() []SpeculativeReceipt {
	d.mu.Lock()
	defer d.mu.Unlock()
	var res []SpeculativeReceipt
	for _, id := range d.receiptOrder {
		if r := d.receipts[id]; r.ExitCode == 0 && r.Error == "" {
			res = append(res, r)
		}
	}
	return res
}

// FailedTasks returns all receipts that failed or were refused.
func (d *ReflexDispatcher) FailedTasks() []SpeculativeReceipt {
	d.mu.Lock()
	defer d.mu.Unlock()
	var res []SpeculativeReceipt
	for _, id := range d.receiptOrder {
		if r := d.receipts[id]; r.ExitCode != 0 || r.Error != "" {
			res = append(res, r)
		}
	}
	return res
}

// RawBytesQuarantined returns the count of raw transcript bytes withheld from coordinator context.
func (d *ReflexDispatcher) RawBytesQuarantined() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.rawBytesSeen
}

// ReceiptBytes returns the count of compact receipt bytes admitted to coordinator context.
func (d *ReflexDispatcher) ReceiptBytes() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.receiptBytes
}

// ContextReductionRatio returns (rawBytes - receiptBytes) / rawBytes.
func (d *ReflexDispatcher) ContextReductionRatio() float64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.rawBytesSeen == 0 {
		return 0.0
	}
	saved := d.rawBytesSeen - d.receiptBytes
	if saved < 0 {
		return 0.0
	}
	return float64(saved) / float64(d.rawBytesSeen)
}

func (d *ReflexDispatcher) resolveTreePatterns(task ReflexTask) []string {
	if len(task.TreePatterns) > 0 {
		return task.TreePatterns
	}
	if len(task.TouchedFiles) > 0 {
		return task.TouchedFiles
	}
	switch strings.ToLower(task.Lane) {
	case "gateway":
		return []string{"internal/gateway/**"}
	case "engine":
		return []string{"internal/engine/**"}
	case "docs":
		return []string{"docs/**"}
	default:
		return []string{"internal/" + task.Lane + "/**"}
	}
}

func isUniversalPattern(p string) bool {
	p = strings.TrimSpace(filepath.ToSlash(p))
	return p == "**" || p == "*" || p == "." || p == "..." || p == "./"
}

func normalizePattern(p string) string {
	p = strings.TrimSpace(p)
	p = filepath.ToSlash(p)
	p = strings.TrimPrefix(p, "./")
	p = strings.TrimPrefix(p, "/")
	p = filepath.Clean(p)
	p = filepath.ToSlash(p)
	p = strings.TrimPrefix(p, "./")
	p = strings.TrimPrefix(p, "/")
	return p
}

func extractBase(p string) (string, bool) {
	if strings.HasSuffix(p, "/**") {
		return strings.TrimSuffix(p, "/**"), true
	}
	if strings.HasSuffix(p, "/*") {
		return strings.TrimSuffix(p, "/*"), true
	}
	if strings.HasSuffix(p, "/...") {
		return strings.TrimSuffix(p, "/..."), true
	}
	if strings.HasSuffix(p, "...") {
		return strings.TrimSuffix(p, "..."), true
	}
	if strings.Contains(p, "*") {
		return p, true
	}
	return p, false
}

func patternsCollide(p1, p2 string) bool {
	if isUniversalPattern(p1) || isUniversalPattern(p2) {
		return true
	}

	n1 := normalizePattern(p1)
	n2 := normalizePattern(p2)

	if n1 == "" || n2 == "" {
		return false
	}
	if n1 == n2 {
		return true
	}

	base1, _ := extractBase(n1)
	base2, _ := extractBase(n2)

	if base1 == base2 {
		return true
	}
	if base1 != "" && strings.HasPrefix(base2, base1+"/") {
		return true
	}
	if base2 != "" && strings.HasPrefix(base1, base2+"/") {
		return true
	}

	if matched, err := filepath.Match(n1, n2); err == nil && matched {
		return true
	}
	if matched, err := filepath.Match(n2, n1); err == nil && matched {
		return true
	}

	return segmentsCollide(strings.Split(n1, "/"), strings.Split(n2, "/"))
}

func segmentsCollide(s1, s2 []string) bool {
	if len(s1) == 0 && len(s2) == 0 {
		return true
	}
	if len(s1) == 0 || len(s2) == 0 {
		return false
	}

	head1 := s1[0]
	head2 := s2[0]

	if head1 == "**" || head1 == "..." {
		for i := 0; i <= len(s2); i++ {
			if segmentsCollide(s1[1:], s2[i:]) {
				return true
			}
		}
		return false
	}
	if head2 == "**" || head2 == "..." {
		for i := 0; i <= len(s1); i++ {
			if segmentsCollide(s1[i:], s2[1:]) {
				return true
			}
		}
		return false
	}

	if head1 == head2 || head1 == "*" || head2 == "*" {
		return segmentsCollide(s1[1:], s2[1:])
	}
	if matched, err := filepath.Match(head1, head2); err == nil && matched {
		return segmentsCollide(s1[1:], s2[1:])
	}
	if matched, err := filepath.Match(head2, head1); err == nil && matched {
		return segmentsCollide(s1[1:], s2[1:])
	}

	return false
}

func treesCollide(pats1, pats2 []string) bool {
	if len(pats1) == 0 || len(pats2) == 0 {
		return false
	}
	for _, p1 := range pats1 {
		p1 = strings.TrimSpace(p1)
		if p1 == "" {
			continue
		}
		for _, p2 := range pats2 {
			p2 = strings.TrimSpace(p2)
			if p2 == "" {
				continue
			}
			if patternsCollide(p1, p2) {
				return true
			}
		}
	}
	return false
}
