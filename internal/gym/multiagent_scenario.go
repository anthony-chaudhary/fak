// Package gym coordinates agent evaluation environments, sub-10ms CoW
// snapshot lifecycles, and isolated execution trajectories.
package gym

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// The out-of-tree refusal code for circular query dependency / deadlock in multi-agent simulations.
const (
	// ReasonCircularDependency is the registered refusal code for detected cycles
	// in cross-agent context search and peer dependency graphs.
	ReasonCircularDependency     abi.ReasonCode = 1100
	ReasonCircularDependencyName                = "CIRCULAR_DEPENDENCY"
)

func init() {
	abi.RegisterReason(ReasonCircularDependency, ReasonCircularDependencyName)
}

// MultiAgentReceiptSchema defines the canonical schema identifier for multi-agent simulation receipts.
const MultiAgentReceiptSchema = "fak.gym.multiagent.v1"

// QueryRefusal represents a structured refusal when a cross-agent context query
// induces a cycle in the active peer dependency / wait-for graph.
type QueryRefusal struct {
	Status     string         `json:"status"` // "refused"
	Refusal    bool           `json:"refusal"` // true
	Reason     string         `json:"reason"` // "CIRCULAR_DEPENDENCY"
	ReasonCode abi.ReasonCode `json:"reason_code"` // 1100
	Cycle      []string       `json:"cycle"` // e.g. ["worker-A", "worker-B", "worker-A"]
	Detail     string         `json:"detail"`
}

func (r *QueryRefusal) Error() string {
	return fmt.Sprintf("refusal: %s (code=%d, cycle=%s): %s", r.Reason, r.ReasonCode, strings.Join(r.Cycle, " -> "), r.Detail)
}

// PeerContextItem represents a discrete fragment of context/memory stored in a peer's local knowledge base.
type PeerContextItem struct {
	ID          string            `json:"id"`
	Key         string            `json:"key"`
	Content     string            `json:"content"`
	Taint       abi.TaintLabel    `json:"taint"`
	Quarantined bool              `json:"quarantined"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// SimulatedPeer represents a concurrent simulated autonomous worker or coordinator in the simulation mesh.
type SimulatedPeer struct {
	mu          sync.RWMutex
	ID          string                     `json:"id"`
	Role        string                     `json:"role"`
	Taint       abi.TaintLabel             `json:"taint"`
	Quarantined bool                       `json:"quarantined"`
	Contexts    map[string]PeerContextItem `json:"contexts"`
}

// NewSimulatedPeer initializes a simulated agent peer with initial taint and quarantine status.
func NewSimulatedPeer(id, role string, initialTaint abi.TaintLabel) *SimulatedPeer {
	quarantined := initialTaint == abi.TaintQuarantined
	return &SimulatedPeer{
		ID:          id,
		Role:        role,
		Taint:       initialTaint,
		Quarantined: quarantined,
		Contexts:    make(map[string]PeerContextItem),
	}
}

// StoreContext seeds or mutates a context item in the peer's store, updating local taint if needed.
func (p *SimulatedPeer) StoreContext(item PeerContextItem) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.Contexts[item.Key] = item
	if item.Taint == abi.TaintQuarantined || item.Quarantined {
		p.Taint = abi.TaintQuarantined
		p.Quarantined = true
	} else if item.Taint == abi.TaintTainted && p.Taint != abi.TaintQuarantined {
		p.Taint = abi.TaintTainted
	}
}

// SearchContext finds all items whose key or content matches the query string.
func (p *SimulatedPeer) SearchContext(query string) []PeerContextItem {
	p.mu.RLock()
	defer p.mu.RUnlock()
	var matches []PeerContextItem
	q := strings.ToLower(strings.TrimSpace(query))
	for _, item := range p.Contexts {
		if q == "" || strings.Contains(strings.ToLower(item.Key), q) || strings.Contains(strings.ToLower(item.Content), q) {
			matches = append(matches, item)
		}
	}
	return matches
}

// IngestContext stores incoming context from a peer, enforcing strict taint and quarantine preservation.
func (p *SimulatedPeer) IngestContext(item PeerContextItem) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.Contexts[item.Key] = item

	// Taint preservation invariant: if received context is Quarantined, the receiving peer's
	// state is unconditionally elevated to Quarantined.
	if item.Taint == abi.TaintQuarantined || item.Quarantined {
		p.Taint = abi.TaintQuarantined
		p.Quarantined = true
	} else if item.Taint == abi.TaintTainted && p.Taint != abi.TaintQuarantined {
		p.Taint = abi.TaintTainted
	}
}

// Status returns the current taint label and quarantine flag for the peer.
func (p *SimulatedPeer) Status() (abi.TaintLabel, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.Taint, p.Quarantined
}

// ContextCount returns the number of context fragments held by the peer.
func (p *SimulatedPeer) ContextCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.Contexts)
}

// DeadlockDetector tracks in-flight dependency edges (caller -> target) and intercepts
// any query that would induce a cycle in the wait-for graph, preventing livelock/deadlock.
type DeadlockDetector struct {
	mu        sync.Mutex
	waitGraph map[string]map[string]struct{}
	deadlocks int
}

// NewDeadlockDetector initializes an empty DeadlockDetector.
func NewDeadlockDetector() *DeadlockDetector {
	return &DeadlockDetector{
		waitGraph: make(map[string]map[string]struct{}),
	}
}

// CheckAndRegister verifies if caller querying target would form a cycle in the active wait graph.
// If a cycle is detected, it returns a structured QueryRefusal (reason: CIRCULAR_DEPENDENCY)
// and refuses to register the edge.
// If acyclic, it registers the wait edge caller -> target and returns nil.
func (d *DeadlockDetector) CheckAndRegister(caller, target string) (*QueryRefusal, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Direct self-dependency cycle
	if caller == target {
		d.deadlocks++
		cycle := []string{caller, caller}
		return &QueryRefusal{
			Status:     "refused",
			Refusal:    true,
			Reason:     ReasonCircularDependencyName,
			ReasonCode: ReasonCircularDependency,
			Cycle:      cycle,
			Detail:     fmt.Sprintf("self-dependency cycle detected: %s -> %s", caller, caller),
		}, nil
	}

	// Check if path already exists from target to caller: target -> ... -> caller
	// If so, adding caller -> target would create caller -> target -> ... -> caller (cycle!)
	if cyclePath, hasPath := d.findPathLocked(target, caller); hasPath {
		d.deadlocks++
		fullCycle := append([]string{caller}, cyclePath...)
		return &QueryRefusal{
			Status:     "refused",
			Refusal:    true,
			Reason:     ReasonCircularDependencyName,
			ReasonCode: ReasonCircularDependency,
			Cycle:      fullCycle,
			Detail:     fmt.Sprintf("circular query dependency detected in wait graph: %s", strings.Join(fullCycle, " -> ")),
		}, nil
	}

	// Safe: register wait edge
	if d.waitGraph[caller] == nil {
		d.waitGraph[caller] = make(map[string]struct{})
	}
	d.waitGraph[caller][target] = struct{}{}
	return nil, nil
}

// Release removes the wait edge caller -> target upon query completion or error.
func (d *DeadlockDetector) Release(caller, target string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if targets, ok := d.waitGraph[caller]; ok {
		delete(targets, target)
		if len(targets) == 0 {
			delete(d.waitGraph, caller)
		}
	}
}

// findPathLocked performs BFS from start to end in the waitGraph.
func (d *DeadlockDetector) findPathLocked(start, end string) ([]string, bool) {
	if start == end {
		return []string{start}, true
	}

	visited := make(map[string]bool)
	parent := make(map[string]string)
	queue := []string{start}
	visited[start] = true

	found := false
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		if curr == end {
			found = true
			break
		}

		for next := range d.waitGraph[curr] {
			if !visited[next] {
				visited[next] = true
				parent[next] = curr
				queue = append(queue, next)
			}
		}
	}

	if !found {
		return nil, false
	}

	// Reconstruct path start -> ... -> end
	var path []string
	curr := end
	for curr != "" {
		path = append(path, curr)
		if curr == start {
			break
		}
		curr = parent[curr]
	}

	// Reverse path to be start -> ... -> end
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}

	return path, true
}

// DeadlocksCaught returns the total count of circular dependency deadlocks intercepted.
func (d *DeadlockDetector) DeadlocksCaught() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.deadlocks
}

// ActiveWaitsCount returns the current number of active waiting caller nodes.
func (d *DeadlockDetector) ActiveWaitsCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.waitGraph)
}

// ContextQueryResult records the outcome of a cross-agent context query.
type ContextQueryResult struct {
	QueryID     string            `json:"query_id"`
	CallerID    string            `json:"caller_id"`
	TargetID    string            `json:"target_id"`
	Query       string            `json:"query"`
	Status      string            `json:"status"` // "OK", "REFUSED"
	Matches     []PeerContextItem `json:"matches,omitempty"`
	Taint       abi.TaintLabel    `json:"taint"`
	Quarantined bool              `json:"quarantined"`
	Refusal     *QueryRefusal     `json:"refusal,omitempty"`
	Duration    time.Duration     `json:"duration_ns"`
}

// FanOutResult records the aggregated outcome of a concurrent multi-target context search.
type FanOutResult struct {
	QueryID          string               `json:"query_id"`
	CallerID         string               `json:"caller_id"`
	Query            string               `json:"query"`
	TotalTargets     int                  `json:"total_targets"`
	Successful       int                  `json:"successful"`
	Refused          int                  `json:"refused"`
	WorkerResponses  []ContextQueryResult `json:"worker_responses"`
	MergedMatches    []PeerContextItem    `json:"merged_matches"`
	MaxTaintObserved abi.TaintLabel       `json:"max_taint_observed"`
	QuarantineActive bool                 `json:"quarantine_active"`
	Duration         time.Duration        `json:"duration"`
}

// MultiAgentMesh coordinates concurrent simulated peers, cross-agent context search,
// deadlock cycle detection, and taint quarantine preservation.
type MultiAgentMesh struct {
	mu             sync.RWMutex
	peers          map[string]*SimulatedPeer
	detector       *DeadlockDetector
	queriesCount   int64
	taintsPreserve int64
}

// NewMultiAgentMesh constructs a new MultiAgentMesh.
func NewMultiAgentMesh() *MultiAgentMesh {
	return &MultiAgentMesh{
		peers:    make(map[string]*SimulatedPeer),
		detector: NewDeadlockDetector(),
	}
}

// RegisterPeer registers an autonomous peer in the simulation mesh.
func (m *MultiAgentMesh) RegisterPeer(peer *SimulatedPeer) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.peers[peer.ID]; exists {
		return fmt.Errorf("peer %q already registered in mesh", peer.ID)
	}
	m.peers[peer.ID] = peer
	return nil
}

// Peer retrieves a registered peer by ID.
func (m *MultiAgentMesh) Peer(id string) (*SimulatedPeer, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	p, ok := m.peers[id]
	return p, ok
}

// QueryContext executes a cross-agent context query with cycle deadlock detection and taint quarantine preservation.
func (m *MultiAgentMesh) QueryContext(ctx context.Context, callerID, targetID, query string) (*ContextQueryResult, error) {
	start := time.Now()
	atomic.AddInt64(&m.queriesCount, 1)

	m.mu.RLock()
	caller, callerOk := m.peers[callerID]
	target, targetOk := m.peers[targetID]
	m.mu.RUnlock()

	if !callerOk {
		return nil, fmt.Errorf("caller %q not registered in mesh", callerID)
	}
	if !targetOk {
		return nil, fmt.Errorf("target %q not registered in mesh", targetID)
	}

	// 1. Cycle deadlock check before initiating wait
	refusal, err := m.detector.CheckAndRegister(callerID, targetID)
	if err != nil {
		return nil, err
	}
	if refusal != nil {
		return &ContextQueryResult{
			CallerID: callerID,
			TargetID: targetID,
			Query:    query,
			Status:   "REFUSED",
			Refusal:  refusal,
			Duration: time.Since(start),
		}, nil
	}
	defer m.detector.Release(callerID, targetID)

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// 2. Perform search inside target peer's context store
	matches := target.SearchContext(query)

	// 3. Inspect taint on matches
	resTaint := abi.TaintTrusted
	resQuarantined := false
	for _, match := range matches {
		if match.Taint == abi.TaintQuarantined || match.Quarantined {
			resTaint = abi.TaintQuarantined
			resQuarantined = true
			break
		} else if match.Taint == abi.TaintTainted && resTaint != abi.TaintQuarantined {
			resTaint = abi.TaintTainted
		}
	}

	// 4. Ingest matches into caller, preserving taint & quarantine status
	if resQuarantined {
		atomic.AddInt64(&m.taintsPreserve, 1)
	}
	for _, match := range matches {
		caller.IngestContext(match)
	}

	return &ContextQueryResult{
		CallerID:    callerID,
		TargetID:    targetID,
		Query:       query,
		Status:      "OK",
		Matches:     matches,
		Taint:       resTaint,
		Quarantined: resQuarantined,
		Duration:    time.Since(start),
	}, nil
}

// FanOutSearch performs parallel concurrent context searches across targetIDs, fan-in aggregating results.
func (m *MultiAgentMesh) FanOutSearch(ctx context.Context, callerID string, targetIDs []string, query string) (*FanOutResult, error) {
	start := time.Now()
	if len(targetIDs) == 0 {
		return &FanOutResult{
			CallerID:     callerID,
			Query:        query,
			TotalTargets: 0,
			Duration:     time.Since(start),
		}, nil
	}

	var wg sync.WaitGroup
	type queryOut struct {
		res *ContextQueryResult
		err error
	}
	ch := make(chan queryOut, len(targetIDs))

	for _, target := range targetIDs {
		wg.Add(1)
		go func(t string) {
			defer wg.Done()
			r, err := m.QueryContext(ctx, callerID, t, query)
			ch <- queryOut{res: r, err: err}
		}(target)
	}

	wg.Wait()
	close(ch)

	fanOut := &FanOutResult{
		CallerID:         callerID,
		Query:            query,
		TotalTargets:     len(targetIDs),
		MaxTaintObserved: abi.TaintTrusted,
	}

	for out := range ch {
		if out.err != nil {
			fanOut.Refused++
			continue
		}
		if out.res.Refusal != nil {
			fanOut.Refused++
			fanOut.WorkerResponses = append(fanOut.WorkerResponses, *out.res)
			continue
		}

		fanOut.Successful++
		fanOut.WorkerResponses = append(fanOut.WorkerResponses, *out.res)
		fanOut.MergedMatches = append(fanOut.MergedMatches, out.res.Matches...)
		if out.res.Quarantined {
			fanOut.QuarantineActive = true
			fanOut.MaxTaintObserved = abi.TaintQuarantined
		} else if out.res.Taint == abi.TaintTainted && fanOut.MaxTaintObserved != abi.TaintQuarantined {
			fanOut.MaxTaintObserved = abi.TaintTainted
		}
	}

	fanOut.Duration = time.Since(start)
	return fanOut, nil
}

// DeadlocksCaught returns the total count of circular dependency deadlocks intercepted.
func (m *MultiAgentMesh) DeadlocksCaught() int {
	return m.detector.DeadlocksCaught()
}

// TaintsPreserved returns the count of queries where quarantined taint was preserved.
func (m *MultiAgentMesh) TaintsPreserved() int {
	return int(atomic.LoadInt64(&m.taintsPreserve))
}

// TotalQueries returns the total queries dispatched across the mesh.
func (m *MultiAgentMesh) TotalQueries() int {
	return int(atomic.LoadInt64(&m.queriesCount))
}

// MultiAgentReceipt records execution metrics, deadlock prevention, and taint preservation
// for a multi-agent gym simulation scenario.
type MultiAgentReceipt struct {
	Schema           string    `json:"schema"`
	ScenarioID       string    `json:"scenario_id"`
	Timestamp        time.Time `json:"timestamp"`
	WorkersCount     int       `json:"workers_count"`
	QueriesExecuted  int       `json:"queries_executed"`
	FanOutCalls      int       `json:"fan_out_calls"`
	DeadlocksCaught  int       `json:"deadlocks_caught"`
	TaintsPreserved  int       `json:"taints_preserved"`
	Outcome          string    `json:"outcome"` // "PASS", "FAIL"
	FailureReason    string    `json:"failure_reason,omitempty"`
	TranscriptDigest string    `json:"transcript_digest"`
}

// VerifyReceipt verifies that the multi-agent receipt adheres to the canonical schema,
// observed non-zero queries and passes the required scenario checks.
func (r *MultiAgentReceipt) VerifyReceipt(expectedScenario string) (bool, string) {
	if r == nil {
		return false, "receipt is nil"
	}
	if r.Schema != MultiAgentReceiptSchema {
		return false, fmt.Sprintf("invalid schema: expected %q, got %q", MultiAgentReceiptSchema, r.Schema)
	}
	if expectedScenario != "" && r.ScenarioID != expectedScenario {
		return false, fmt.Sprintf("scenario mismatch: expected %q, got %q", expectedScenario, r.ScenarioID)
	}
	if r.QueriesExecuted <= 0 {
		return false, fmt.Sprintf("zero queries executed (%d)", r.QueriesExecuted)
	}
	if r.Outcome != OutcomePass {
		reason := r.FailureReason
		if reason == "" {
			reason = "unspecified failure"
		}
		return false, fmt.Sprintf("outcome not PASS: %s (%s)", r.Outcome, reason)
	}
	return true, ""
}

// MultiAgentScenarioConfig configures a closed-loop multi-agent simulation scenario.
type MultiAgentScenarioConfig struct {
	ID             string
	Name           string
	WorkerCount    int
	SimulateFanOut bool
	SimulateCycle  bool
	SimulateTaint  bool
}

// MultiAgentScenarioRunner drives autonomous multi-agent scenario simulations.
type MultiAgentScenarioRunner struct{}

// NewMultiAgentScenarioRunner constructs a new runner.
func NewMultiAgentScenarioRunner() *MultiAgentScenarioRunner {
	return &MultiAgentScenarioRunner{}
}

// Run executes a multi-agent scenario covering concurrent fan-out, circular deadlock detection,
// and taint quarantine preservation.
func (r *MultiAgentScenarioRunner) Run(ctx context.Context, cfg MultiAgentScenarioConfig) (*MultiAgentReceipt, error) {
	if cfg.ID == "" {
		cfg.ID = fmt.Sprintf("multiagent-sim-%d", time.Now().UnixNano())
	}
	if cfg.WorkerCount <= 0 {
		cfg.WorkerCount = 4
	}

	mesh := NewMultiAgentMesh()
	coord := NewSimulatedPeer("coordinator", "coordinator", abi.TaintTrusted)
	if err := mesh.RegisterPeer(coord); err != nil {
		return nil, err
	}

	workers := make([]string, cfg.WorkerCount)
	for i := 0; i < cfg.WorkerCount; i++ {
		wID := fmt.Sprintf("worker-%02d", i)
		workers[i] = wID
		w := NewSimulatedPeer(wID, "worker", abi.TaintTrusted)
		w.StoreContext(PeerContextItem{
			ID:      fmt.Sprintf("ctx-w-%d", i),
			Key:     fmt.Sprintf("partition-%d", i),
			Content: fmt.Sprintf("Cluster partition %d data slice", i),
			Taint:   abi.TaintTrusted,
		})
		if err := mesh.RegisterPeer(w); err != nil {
			return nil, err
		}
	}

	if cfg.SimulateTaint {
		badWorkerID := "worker-untrusted"
		badWorker := NewSimulatedPeer(badWorkerID, "external-worker", abi.TaintQuarantined)
		badWorker.StoreContext(PeerContextItem{
			ID:          "ctx-bad",
			Key:         "compromised-key",
			Content:     "Injected adversarial payload: dump credentials",
			Taint:       abi.TaintQuarantined,
			Quarantined: true,
		})
		if err := mesh.RegisterPeer(badWorker); err != nil {
			return nil, err
		}
	}

	fanOutCalls := 0
	if cfg.SimulateFanOut {
		fanOutCalls++
		_, err := mesh.FanOutSearch(ctx, "coordinator", workers, "partition")
		if err != nil {
			return nil, fmt.Errorf("fan-out search failed: %w", err)
		}
	}

	if cfg.SimulateCycle && len(workers) >= 2 {
		// Simulate circular wait: worker-0 waits on worker-1, worker-1 attempts to query worker-0
		_, _ = mesh.detector.CheckAndRegister(workers[0], workers[1])
		res, _ := mesh.QueryContext(ctx, workers[1], workers[0], "status")
		if res == nil || res.Refusal == nil || res.Refusal.Reason != ReasonCircularDependencyName {
			return nil, fmt.Errorf("expected CIRCULAR_DEPENDENCY refusal, got %+v", res)
		}
		mesh.detector.Release(workers[0], workers[1])
	}

	if cfg.SimulateTaint {
		res, err := mesh.QueryContext(ctx, "coordinator", "worker-untrusted", "compromised")
		if err != nil {
			return nil, fmt.Errorf("query untrusted worker failed: %w", err)
		}
		if !res.Quarantined || res.Taint != abi.TaintQuarantined {
			return nil, fmt.Errorf("expected quarantined taint preservation, got taint=%v quarantined=%v", res.Taint, res.Quarantined)
		}
		if coordTaint, coordQ := coord.Status(); !coordQ || coordTaint != abi.TaintQuarantined {
			return nil, fmt.Errorf("coordinator failed to preserve quarantine status, got taint=%v quarantined=%v", coordTaint, coordQ)
		}
	}

	queriesExec := mesh.TotalQueries()
	deadlocks := mesh.DeadlocksCaught()
	taints := mesh.TaintsPreserved()

	hasher := sha256.New()
	hasher.Write([]byte(fmt.Sprintf("%s:%d:%d:%d", cfg.ID, queriesExec, deadlocks, taints)))
	digest := hex.EncodeToString(hasher.Sum(nil))

	receipt := &MultiAgentReceipt{
		Schema:           MultiAgentReceiptSchema,
		ScenarioID:       cfg.ID,
		Timestamp:        time.Now().UTC(),
		WorkersCount:     cfg.WorkerCount,
		QueriesExecuted:  queriesExec,
		FanOutCalls:      fanOutCalls,
		DeadlocksCaught:  deadlocks,
		TaintsPreserved:  taints,
		Outcome:          OutcomePass,
		TranscriptDigest: digest,
	}

	return receipt, nil
}
