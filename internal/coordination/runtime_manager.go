//go:build wip_coordination

package coordination

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// RuntimeManager orchestrates workers across roles according to a coordination manifest,
// evaluating escalation gates, enforcing receipt policies, and executing coordination strategies.
type RuntimeManager struct {
	manifest    harnesskit.CoordinationManifest
	pool        harnesskit.WorkerPool
	foldHandler harnesskit.ReceiptFoldHandler
	taskCounter uint64
}

var _ harnesskit.Manager = (*RuntimeManager)(nil)

// NewRuntimeManager constructs an initialized manager instance.
func NewRuntimeManager(
	manifest harnesskit.CoordinationManifest,
	pool harnesskit.WorkerPool,
	foldHandler harnesskit.ReceiptFoldHandler,
) (*RuntimeManager, error) {
	if err := manifest.Validate(); err != nil {
		return nil, err
	}
	if pool == nil {
		return nil, &harnesskit.Error{
			Code: harnesskit.CodeInvalid,
			Op:   "manager.new",
			Err:  errors.New("worker pool is required"),
		}
	}
	if foldHandler == nil {
		foldHandler = NewCompactFoldHandler()
	}

	return &RuntimeManager{
		manifest:    manifest,
		pool:        pool,
		foldHandler: foldHandler,
	}, nil
}

// Manifest returns the active coordination manifest.
func (m *RuntimeManager) Manifest() harnesskit.CoordinationManifest {
	return m.manifest
}

// Dispatch evaluates escalation gates, acquires a worker from the pool,
// executes the invocation, enforces receipt policies, and folds the receipt.
func (m *RuntimeManager) Dispatch(ctx context.Context, roleID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error) {
	if err := ctx.Err(); err != nil {
		return harnesskit.WorkerReceipt{}, &harnesskit.Error{
			Code: harnesskit.CodeCanceled,
			Op:   "manager.dispatch",
			Err:  err,
		}
	}

	if _, exists := m.manifest.Workers[roleID]; !exists {
		return harnesskit.WorkerReceipt{}, &harnesskit.Error{
			Code: harnesskit.CodeInvalid,
			Op:   "manager.dispatch",
			Err:  fmt.Errorf("unknown role %q not declared in manifest", roleID),
		}
	}

	// 1. Evaluate Escalation Gates
	effectiveRole, triggeredGate, shouldAbstain := m.evaluateEscalation(roleID, input)
	if shouldAbstain && triggeredGate != nil {
		taskID := m.generateTaskID(effectiveRole)
		receipt := harnesskit.WorkerReceipt{
			TaskID:          taskID,
			WorkerID:        "coordinator-gate",
			RoleID:          effectiveRole,
			Status:          harnesskit.StatusAbstain,
			Summary:         fmt.Sprintf("execution halted by escalation gate %q", triggeredGate.Name),
			TouchedFiles:    []string{},
			ExecutionTimeMS: 0,
			Diagnosis: &harnesskit.FailureDiagnosis{
				ReasonCategory:   "escalation_gate",
				Message:          fmt.Sprintf("escalation gate %q triggered action %s", triggeredGate.Name, triggeredGate.Action),
				FailingSeam:      triggeredGate.Name,
				SuggestedRole:    triggeredGate.TargetRole,
				UnmetAssumptions: []string{fmt.Sprintf("escalation criteria triggered gate %s", triggeredGate.Name)},
			},
		}

		if m.foldHandler != nil {
			_, _ = m.foldHandler.Fold(ctx, receipt)
		}
		return receipt, nil
	}

	// 2. Acquire worker from pool
	worker, err := m.pool.Acquire(ctx, effectiveRole)
	if err != nil {
		return harnesskit.WorkerReceipt{}, err
	}
	defer func() {
		// Use Background context to ensure release succeeds even if call context was canceled
		_ = m.pool.Release(context.Background(), effectiveRole, worker)
	}()

	// 3. Dispatch task to worker instance
	taskID := m.generateTaskID(effectiveRole)
	receipt, err := worker.Dispatch(ctx, taskID, input)
	if err != nil {
		if ctx.Err() != nil {
			_ = worker.Cancel(taskID)
			return receipt, &harnesskit.Error{
				Code: harnesskit.CodeCanceled,
				Op:   "manager.dispatch",
				Err:  ctx.Err(),
			}
		}
		return receipt, err
	}

	// 4. Enforce ReceiptPolicy
	policy := m.manifest.Manager.ReceiptPolicy

	if policy.StrictReceipt {
		if valErr := receipt.Validate(); valErr != nil {
			return receipt, &harnesskit.Error{
				Code: harnesskit.CodeInvalid,
				Op:   "manager.strict_receipt",
				Err:  fmt.Errorf("strict receipt validation failed: %w", valErr),
			}
		}
	}

	if policy.RequireWitnessPass {
		if receipt.Status == harnesskit.StatusCompleted && !receipt.Witness.Passed {
			if policy.QuarantineFailures {
				receipt.Status = harnesskit.StatusFailed
				if receipt.Diagnosis == nil {
					receipt.Diagnosis = &harnesskit.FailureDiagnosis{
						ReasonCategory: "witness_failed",
						Message:        fmt.Sprintf("witness verification failed with exit code %d", receipt.Witness.ExitCode),
						FailingSeam:    "witness",
					}
				}
			}
			return receipt, &harnesskit.Error{
				Code: harnesskit.CodeDenied,
				Op:   "manager.witness_pass",
				Err:  fmt.Errorf("receipt failed required witness pass (command %q exit code %d)", receipt.Witness.Command, receipt.Witness.ExitCode),
			}
		}
	}

	// 5. Fold receipt into compact audit store
	if m.foldHandler != nil {
		if _, foldErr := m.foldHandler.Fold(ctx, receipt); foldErr != nil {
			return receipt, &harnesskit.Error{
				Code: harnesskit.CodeInternal,
				Op:   "manager.fold",
				Err:  foldErr,
			}
		}
	}

	return receipt, nil
}

// ExecuteStrategy executes worker dispatches across inputs according to the requested coordination strategy.
func (m *RuntimeManager) ExecuteStrategy(
	ctx context.Context,
	strategy harnesskit.StrategyKind,
	inputs map[string]harnesskit.Invocation,
) (map[string]harnesskit.WorkerReceipt, error) {
	if strategy == "" {
		strategy = m.manifest.Manager.DefaultStrategy
	}
	if strategy == "" {
		strategy = harnesskit.StrategyFanOutFanIn
	}
	if !strategy.IsValid() {
		return nil, &harnesskit.Error{
			Code: harnesskit.CodeInvalid,
			Op:   "manager.strategy",
			Err:  fmt.Errorf("invalid strategy %q", strategy),
		}
	}

	if len(inputs) == 0 {
		return make(map[string]harnesskit.WorkerReceipt), nil
	}

	switch strategy {
	case harnesskit.StrategyFanOutFanIn:
		return m.executeFanOutFanIn(ctx, inputs)
	case harnesskit.StrategySequential:
		return m.executeSequential(ctx, inputs)
	case harnesskit.StrategySpeculative:
		return m.executeSpeculative(ctx, inputs)
	case harnesskit.StrategyAdaptiveDAG:
		return m.executeAdaptiveDAG(ctx, inputs)
	default:
		return nil, &harnesskit.Error{
			Code: harnesskit.CodeUnsupported,
			Op:   "manager.strategy",
			Err:  fmt.Errorf("unsupported strategy %q", strategy),
		}
	}
}

// executeFanOutFanIn concurrently executes all input roles via goroutines and sync.WaitGroup.
func (m *RuntimeManager) executeFanOutFanIn(
	ctx context.Context,
	inputs map[string]harnesskit.Invocation,
) (map[string]harnesskit.WorkerReceipt, error) {
	receipts := make(map[string]harnesskit.WorkerReceipt, len(inputs))
	var (
		mu       sync.Mutex
		wg       sync.WaitGroup
		firstErr error
	)

	for roleID, inv := range inputs {
		wg.Add(1)
		go func(rID string, invocation harnesskit.Invocation) {
			defer wg.Done()
			rec, err := m.Dispatch(ctx, rID, invocation)

			mu.Lock()
			defer mu.Unlock()
			receipts[rID] = rec
			if err != nil && firstErr == nil {
				firstErr = err
			}
		}(roleID, inv)
	}

	wg.Wait()
	return receipts, firstErr
}

// executeSequential executes worker roles in ordered sequential turns according to manifest.Topology or sorted key order.
func (m *RuntimeManager) executeSequential(
	ctx context.Context,
	inputs map[string]harnesskit.Invocation,
) (map[string]harnesskit.WorkerReceipt, error) {
	orderedRoles := m.orderSequentialRoles(inputs)
	receipts := make(map[string]harnesskit.WorkerReceipt, len(inputs))

	for _, roleID := range orderedRoles {
		inv, exists := inputs[roleID]
		if !exists {
			continue
		}

		rec, err := m.Dispatch(ctx, roleID, inv)
		receipts[roleID] = rec
		if err != nil {
			return receipts, err
		}

		if m.manifest.Manager.ReceiptPolicy.QuarantineFailures && rec.Status == harnesskit.StatusFailed {
			// Stop sequential execution on quarantined failure
			return receipts, nil
		}
	}

	return receipts, nil
}

// executeSpeculative races multiple worker implementations; the first valid/completed receipt wins,
// and remaining tasks are promptly canceled.
func (m *RuntimeManager) executeSpeculative(
	ctx context.Context,
	inputs map[string]harnesskit.Invocation,
) (map[string]harnesskit.WorkerReceipt, error) {
	specCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	type specResult struct {
		roleID  string
		receipt harnesskit.WorkerReceipt
		err     error
	}

	resCh := make(chan specResult, len(inputs))
	for roleID, inv := range inputs {
		go func(rID string, invocation harnesskit.Invocation) {
			rec, err := m.Dispatch(specCtx, rID, invocation)
			resCh <- specResult{roleID: rID, receipt: rec, err: err}
		}(roleID, inv)
	}

	receipts := make(map[string]harnesskit.WorkerReceipt)
	var firstErr error

	for i := 0; i < len(inputs); i++ {
		select {
		case <-ctx.Done():
			return receipts, &harnesskit.Error{
				Code: harnesskit.CodeCanceled,
				Op:   "manager.speculative",
				Err:  ctx.Err(),
			}
		case res := <-resCh:
			if res.err == nil && res.receipt.Status == harnesskit.StatusCompleted {
				// Winning receipt
				receipts[res.roleID] = res.receipt
				cancel()
				return receipts, nil
			}
			if res.receipt.TaskID != "" {
				receipts[res.roleID] = res.receipt
			}
			if res.err != nil && firstErr == nil {
				firstErr = res.err
			}
		}
	}

	if firstErr != nil {
		return receipts, firstErr
	}
	return receipts, &harnesskit.Error{
		Code: harnesskit.CodeInternal,
		Op:   "manager.speculative",
		Err:  errors.New("all speculative workers failed or returned non-completed receipts"),
	}
}

// executeAdaptiveDAG performs a topological sort of tasks based on manifest.Topology rules
// and executes independent nodes in parallel waves.
func (m *RuntimeManager) executeAdaptiveDAG(
	ctx context.Context,
	inputs map[string]harnesskit.Invocation,
) (map[string]harnesskit.WorkerReceipt, error) {
	// Build graph restricted to inputs
	inDegree := make(map[string]int, len(inputs))
	adj := make(map[string][]string, len(inputs))
	for r := range inputs {
		inDegree[r] = 0
	}

	for _, rule := range m.manifest.Topology {
		_, fromOk := inputs[rule.FromRole]
		_, toOk := inputs[rule.ToRole]
		if fromOk && toOk {
			adj[rule.FromRole] = append(adj[rule.FromRole], rule.ToRole)
			inDegree[rule.ToRole]++
		}
	}

	// Upfront cycle check using Kahn's algorithm
	tempInDegree := make(map[string]int, len(inputs))
	for k, v := range inDegree {
		tempInDegree[k] = v
	}
	reachableCount := 0
	checkQueue := make([]string, 0, len(inputs))
	for r, deg := range tempInDegree {
		if deg == 0 {
			checkQueue = append(checkQueue, r)
		}
	}
	for len(checkQueue) > 0 {
		curr := checkQueue[0]
		checkQueue = checkQueue[1:]
		reachableCount++
		for _, next := range adj[curr] {
			tempInDegree[next]--
			if tempInDegree[next] == 0 {
				checkQueue = append(checkQueue, next)
			}
		}
	}
	if reachableCount < len(inputs) {
		return nil, &harnesskit.Error{
			Code: harnesskit.CodeInvalid,
			Op:   "manager.dag",
			Err:  errors.New("cycle detected in coordination topology"),
		}
	}

	receipts := make(map[string]harnesskit.WorkerReceipt, len(inputs))
	remaining := make(map[string]bool, len(inputs))
	for r := range inputs {
		remaining[r] = true
	}

	for len(remaining) > 0 {
		// Identify independent nodes for current wave
		wave := make([]string, 0)
		for r := range remaining {
			if inDegree[r] == 0 {
				wave = append(wave, r)
			}
		}
		sort.Strings(wave)

		if len(wave) == 0 {
			return receipts, &harnesskit.Error{
				Code: harnesskit.CodeConflict,
				Op:   "manager.dag",
				Err:  errors.New("deadlock or cycle in DAG execution"),
			}
		}

		// Execute independent nodes in parallel wave
		var (
			waveWg  sync.WaitGroup
			waveMu  sync.Mutex
			waveErr error
		)

		for _, roleID := range wave {
			waveWg.Add(1)
			go func(rID string) {
				defer waveWg.Done()
				rec, err := m.Dispatch(ctx, rID, inputs[rID])

				waveMu.Lock()
				defer waveMu.Unlock()
				receipts[rID] = rec
				if err != nil && waveErr == nil {
					waveErr = err
				}
			}(roleID)
		}

		waveWg.Wait()
		if waveErr != nil {
			return receipts, waveErr
		}

		// Process completion of wave
		for _, rID := range wave {
			delete(remaining, rID)
			rec := receipts[rID]

			if m.manifest.Manager.ReceiptPolicy.QuarantineFailures && rec.Status == harnesskit.StatusFailed {
				// Stop subsequent downstream waves on failure
				return receipts, nil
			}

			// Decrement in-degree of downstream children
			for _, child := range adj[rID] {
				inDegree[child]--
			}
		}
	}

	return receipts, nil
}

func (m *RuntimeManager) evaluateEscalation(
	initialRole string,
	input harnesskit.Invocation,
) (effectiveRole string, triggeredGate *harnesskit.EscalationGate, shouldAbstain bool) {
	currentRole := initialRole
	visited := map[string]bool{currentRole: true}
	appliedGates := make(map[string]bool)

	for {
		var matched *harnesskit.EscalationGate
		for i := range m.manifest.Manager.EscalationGates {
			gate := &m.manifest.Manager.EscalationGates[i]
			if appliedGates[gate.Name] {
				continue
			}
			if gate.Action == harnesskit.EscalateRerouteRole && gate.TargetRole == currentRole {
				continue
			}
			if matchEscalationGate(*gate, input) {
				matched = gate
				break
			}
		}

		if matched == nil {
			return currentRole, nil, false
		}

		appliedGates[matched.Name] = true

		switch matched.Action {
		case harnesskit.EscalateAbstain, harnesskit.EscalatePromptHuman:
			return currentRole, matched, true

		case harnesskit.EscalateRerouteRole:
			target := matched.TargetRole
			if target == "" || target == currentRole || visited[target] {
				return currentRole, matched, true
			}
			if _, exists := m.manifest.Workers[target]; !exists {
				return currentRole, matched, true
			}
			visited[target] = true
			currentRole = target

		default:
			return currentRole, matched, true
		}
	}
}

func matchEscalationGate(gate harnesskit.EscalationGate, input harnesskit.Invocation) bool {
	// 1. Check RiskKeywords
	if len(gate.RiskKeywords) > 0 {
		toolLower := strings.ToLower(input.Tool)
		argsLower := strings.ToLower(string(input.Arguments))

		for _, kw := range gate.RiskKeywords {
			kwLower := strings.ToLower(kw)
			if strings.Contains(toolLower, kwLower) || strings.Contains(argsLower, kwLower) {
				return true
			}
		}
	}

	// 2. Check PathPatterns
	if len(gate.PathPatterns) > 0 {
		var stringValues []string
		if len(input.Arguments) > 0 {
			var raw any
			if json.Unmarshal(input.Arguments, &raw) == nil {
				collectStringValues(raw, &stringValues)
			}
		}
		stringValues = append(stringValues, input.Tool)

		for _, pat := range gate.PathPatterns {
			cleanPat := strings.ReplaceAll(pat, "\\", "/")
			cleanPatNoWildcard := strings.Trim(cleanPat, "*")

			for _, s := range stringValues {
				cleanS := strings.ReplaceAll(s, "\\", "/")

				if matched, _ := filepath.Match(cleanPat, cleanS); matched {
					return true
				}
				if matched, _ := filepath.Match(cleanPat, filepath.Base(cleanS)); matched {
					return true
				}
				if cleanPatNoWildcard != "" && strings.Contains(cleanS, cleanPatNoWildcard) {
					return true
				}
			}
		}
	}

	return false
}

func collectStringValues(val any, out *[]string) {
	switch v := val.(type) {
	case string:
		*out = append(*out, v)
	case []any:
		for _, item := range v {
			collectStringValues(item, out)
		}
	case map[string]any:
		for k, item := range v {
			*out = append(*out, k)
			collectStringValues(item, out)
		}
	}
}

func (m *RuntimeManager) orderSequentialRoles(inputs map[string]harnesskit.Invocation) []string {
	roles := make([]string, 0, len(inputs))
	for r := range inputs {
		roles = append(roles, r)
	}
	sort.Strings(roles)

	if len(m.manifest.Topology) == 0 {
		return roles
	}

	inDegree := make(map[string]int, len(roles))
	adj := make(map[string][]string, len(roles))
	for _, r := range roles {
		inDegree[r] = 0
	}

	for _, rule := range m.manifest.Topology {
		_, okFrom := inputs[rule.FromRole]
		_, okTo := inputs[rule.ToRole]
		if okFrom && okTo {
			adj[rule.FromRole] = append(adj[rule.FromRole], rule.ToRole)
			inDegree[rule.ToRole]++
		}
	}

	ordered := make([]string, 0, len(roles))
	queue := make([]string, 0)
	for _, r := range roles {
		if inDegree[r] == 0 {
			queue = append(queue, r)
		}
	}
	sort.Strings(queue)

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		ordered = append(ordered, curr)

		for _, next := range adj[curr] {
			inDegree[next]--
			if inDegree[next] == 0 {
				queue = append(queue, next)
				sort.Strings(queue)
			}
		}
	}

	// Append any remaining disconnected or cycle nodes
	if len(ordered) < len(roles) {
		seen := make(map[string]bool, len(ordered))
		for _, o := range ordered {
			seen[o] = true
		}
		for _, r := range roles {
			if !seen[r] {
				ordered = append(ordered, r)
			}
		}
	}

	return ordered
}

func (m *RuntimeManager) generateTaskID(roleID string) string {
	seq := atomic.AddUint64(&m.taskCounter, 1)
	return fmt.Sprintf("task-%s-%d", roleID, seq)
}
