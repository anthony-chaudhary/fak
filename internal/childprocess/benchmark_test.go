package childprocess

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
)

var (
	benchPIDListSink []int
	benchIntSink     int
	benchErrSink     error
)

// ProcessTree represents an in-memory process hierarchy for testing and benchmarking.
type ProcessTree struct {
	PID      int
	PPID     int
	Command  string
	Children []*ProcessTree
}

// FlatProcess represents a single row in a flat process table.
type FlatProcess struct {
	PID     int
	PPID    int
	Command string
}

// buildSampleTree creates a deterministic process tree of specified depth and fanout.
func buildSampleTree(depth, fanout int, currentPID *int) *ProcessTree {
	*currentPID++
	pid := *currentPID
	node := &ProcessTree{
		PID:     pid,
		Command: fmt.Sprintf("proc-%d", pid),
	}
	if depth <= 1 {
		return node
	}
	for i := 0; i < fanout; i++ {
		child := buildSampleTree(depth-1, fanout, currentPID)
		child.PPID = pid
		node.Children = append(node.Children, child)
	}
	return node
}

// flattenTree converts a hierarchical ProcessTree into a flat process table slice.
func flattenTree(root *ProcessTree) []FlatProcess {
	var list []FlatProcess
	var walk func(n *ProcessTree)
	walk = func(n *ProcessTree) {
		list = append(list, FlatProcess{PID: n.PID, PPID: n.PPID, Command: n.Command})
		for _, c := range n.Children {
			walk(c)
		}
	}
	walk(root)
	return list
}

// scanDescendants finds all descendant PIDs of rootPID in a flat process table.
func scanDescendants(table []FlatProcess, rootPID int) []int {
	byParent := make(map[int][]int, len(table))
	for _, p := range table {
		byParent[p.PPID] = append(byParent[p.PPID], p.PID)
	}
	var result []int
	queue := []int{rootPID}
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		for _, child := range byParent[curr] {
			result = append(result, child)
			queue = append(queue, child)
		}
	}
	return result
}

// collectTreePostOrder traverses the tree bottom-up to collect leaf-first teardown order.
func collectTreePostOrder(root *ProcessTree) []int {
	var result []int
	var walk func(n *ProcessTree)
	walk = func(n *ProcessTree) {
		for _, c := range n.Children {
			walk(c)
		}
		result = append(result, n.PID)
	}
	walk(root)
	return result
}

// ProcessState denotes the lifecycle phase of a monitored subprocess.
type ProcessState int

const (
	StateCreated ProcessState = iota
	StateStarting
	StateRunning
	StateExited
	StateFailed
)

// MonitoredProcess tracks runtime state and exit code normalization for one process.
type MonitoredProcess struct {
	PID      int
	State    ProcessState
	ExitCode int
	Err      error
	mu       sync.RWMutex
}

// LifecycleMonitor coordinates lifecycle state transitions for child processes.
type LifecycleMonitor struct {
	mu        sync.RWMutex
	processes map[int]*MonitoredProcess
}

func newLifecycleMonitor() *LifecycleMonitor {
	return &LifecycleMonitor{
		processes: make(map[int]*MonitoredProcess),
	}
}

func (m *LifecycleMonitor) Register(pid int) *MonitoredProcess {
	m.mu.Lock()
	defer m.mu.Unlock()
	p := &MonitoredProcess{PID: pid, State: StateCreated}
	m.processes[pid] = p
	return p
}

func (m *LifecycleMonitor) Transition(pid int, nextState ProcessState) error {
	m.mu.RLock()
	p, ok := m.processes[pid]
	m.mu.RUnlock()
	if !ok {
		return errors.New("process not found")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.State = nextState
	return nil
}

func (m *LifecycleMonitor) Complete(pid int, err error, launchFailure int) int {
	m.mu.RLock()
	p, ok := m.processes[pid]
	m.mu.RUnlock()
	if !ok {
		return launchFailure
	}
	code := ExitCode(err, launchFailure)
	p.mu.Lock()
	defer p.mu.Unlock()
	p.ExitCode = code
	p.Err = err
	if code == 0 {
		p.State = StateExited
	} else {
		p.State = StateFailed
	}
	return code
}

func (m *LifecycleMonitor) CountActive() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	active := 0
	for _, p := range m.processes {
		p.mu.RLock()
		if p.State == StateRunning || p.State == StateStarting {
			active++
		}
		p.mu.RUnlock()
	}
	return active
}

// SignalType abstracts process termination and control signals.
type SignalType int

const (
	SigTerm SignalType = iota
	SigKill
	SigInt
	SigHup
)

// SignalDispatchQueue coordinates targeted and tree-wide signal delivery.
type SignalDispatchQueue struct {
	mu       sync.RWMutex
	handlers map[int]func(SignalType) error
	queued   int64
}

func newSignalDispatchQueue() *SignalDispatchQueue {
	return &SignalDispatchQueue{
		handlers: make(map[int]func(SignalType) error),
	}
}

func (q *SignalDispatchQueue) RegisterHandler(pid int, fn func(SignalType) error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.handlers[pid] = fn
}

func (q *SignalDispatchQueue) UnregisterHandler(pid int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	delete(q.handlers, pid)
}

func (q *SignalDispatchQueue) Dispatch(pid int, sig SignalType) error {
	q.mu.RLock()
	fn, ok := q.handlers[pid]
	q.mu.RUnlock()
	if !ok {
		return errors.New("no signal handler")
	}
	atomic.AddInt64(&q.queued, 1)
	return fn(sig)
}

func (q *SignalDispatchQueue) BroadcastTree(pids []int, sig SignalType) int {
	dispatched := 0
	for _, pid := range pids {
		if err := q.Dispatch(pid, sig); err == nil {
			dispatched++
		}
	}
	return dispatched
}

// TestBenchmarkProcessTreeSanity confirms process tree traversal and flattening correctness.
func TestBenchmarkProcessTreeSanity(t *testing.T) {
	pidCounter := 0
	root := buildSampleTree(3, 2, &pidCounter)
	if root == nil || root.PID != 1 {
		t.Fatalf("expected root PID 1, got %+v", root)
	}
	flat := flattenTree(root)
	if len(flat) != 7 {
		t.Fatalf("expected 7 processes in 3-deep binary tree, got %d", len(flat))
	}
	descendants := scanDescendants(flat, 1)
	if len(descendants) != 6 {
		t.Fatalf("expected 6 descendants, got %d", len(descendants))
	}
	postOrder := collectTreePostOrder(root)
	if len(postOrder) != 7 || postOrder[len(postOrder)-1] != 1 {
		t.Fatalf("post-order must end at root PID 1, got %v", postOrder)
	}
}

// TestBenchmarkLifecycleSanity validates state progression and ExitCode normalization.
func TestBenchmarkLifecycleSanity(t *testing.T) {
	mon := newLifecycleMonitor()
	p := mon.Register(101)
	if p.State != StateCreated {
		t.Fatalf("initial state not created: %v", p.State)
	}
	if err := mon.Transition(101, StateRunning); err != nil {
		t.Fatalf("transition failed: %v", err)
	}
	if mon.CountActive() != 1 {
		t.Fatalf("expected 1 active process, got %d", mon.CountActive())
	}
	code := mon.Complete(101, nil, 127)
	if code != 0 || p.State != StateExited {
		t.Fatalf("clean exit mismatch: code=%d state=%v", code, p.State)
	}
	if mon.CountActive() != 0 {
		t.Fatalf("expected 0 active processes after exit, got %d", mon.CountActive())
	}

	mon.Register(102)
	failCode := mon.Complete(102, errors.New("exec missing"), 127)
	if failCode != 127 {
		t.Fatalf("expected launchFailure 127, got %d", failCode)
	}
}

// TestBenchmarkSignalDispatchSanity verifies dispatch queues and tree broadcast routing.
func TestBenchmarkSignalDispatchSanity(t *testing.T) {
	queue := newSignalDispatchQueue()
	var receivedSignal SignalType
	receivedPID := 0

	queue.RegisterHandler(42, func(sig SignalType) error {
		receivedSignal = sig
		receivedPID = 42
		return nil
	})

	if err := queue.Dispatch(42, SigTerm); err != nil {
		t.Fatalf("dispatch error: %v", err)
	}
	if receivedPID != 42 || receivedSignal != SigTerm {
		t.Fatalf("unexpected signal delivery: pid=%d sig=%v", receivedPID, receivedSignal)
	}

	queue.RegisterHandler(43, func(sig SignalType) error {
		return nil
	})
	broadcastCount := queue.BroadcastTree([]int{42, 43, 999}, SigKill)
	if broadcastCount != 2 {
		t.Fatalf("expected 2 dispatched signals out of 3, got %d", broadcastCount)
	}
}

// TestBenchmarkExitCodeSanity checks normalization of nil and non-nil launch errors.
func TestBenchmarkExitCodeSanity(t *testing.T) {
	if got := ExitCode(nil, 1); got != 0 {
		t.Fatalf("nil = %d", got)
	}
	if got := ExitCode(errors.New("spawn failed"), 42); got != 42 {
		t.Fatalf("spawn failed = %d", got)
	}
}

// BenchmarkProcessTreeScanning measures process tree collection, scanning, and traversal.
func BenchmarkProcessTreeScanning(b *testing.B) {
	pidCounter100 := 0
	tree100 := buildSampleTree(4, 3, &pidCounter100) // 1 + 3 + 9 + 27 = 40 nodes
	table100 := flattenTree(tree100)

	pidCounterLarge := 0
	treeLarge := buildSampleTree(5, 4, &pidCounterLarge) // 1 + 4 + 16 + 64 + 256 = 341 nodes
	tableLarge := flattenTree(treeLarge)

	b.Run("FlatScan_40Nodes", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			desc := scanDescendants(table100, 1)
			benchPIDListSink = desc
		}
	})

	b.Run("FlatScan_341Nodes", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			desc := scanDescendants(tableLarge, 1)
			benchPIDListSink = desc
		}
	})

	b.Run("PostOrderTeardownCollection", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			order := collectTreePostOrder(treeLarge)
			benchPIDListSink = order
		}
	})

	b.Run("FlattenTreeHierarchy", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			flat := flattenTree(treeLarge)
			benchIntSink = len(flat)
		}
	})
}

// BenchmarkLifecycleMonitoring evaluates process registration, state transitions, and exit tracking.
func BenchmarkLifecycleMonitoring(b *testing.B) {
	b.Run("SequentialStateTransitions", func(b *testing.B) {
		mon := newLifecycleMonitor()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pid := i + 1
			mon.Register(pid)
			_ = mon.Transition(pid, StateStarting)
			_ = mon.Transition(pid, StateRunning)
			code := mon.Complete(pid, nil, 1)
			benchIntSink = code
		}
	})

	b.Run("ExitStatusReconciliation", func(b *testing.B) {
		mon := newLifecycleMonitor()
		for i := 0; i < 1000; i++ {
			mon.Register(i)
		}
		errSample := errors.New("cannot allocate memory")

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pid := i % 1000
			var err error
			if i%2 == 1 {
				err = errSample
			}
			code := mon.Complete(pid, err, 127)
			benchIntSink = code
		}
	})

	b.Run("ActiveProcessSweep", func(b *testing.B) {
		mon := newLifecycleMonitor()
		for i := 0; i < 200; i++ {
			mon.Register(i)
			if i%3 != 0 {
				_ = mon.Transition(i, StateRunning)
			}
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			active := mon.CountActive()
			benchIntSink = active
		}
	})

	b.Run("ConcurrentMonitoring", func(b *testing.B) {
		mon := newLifecycleMonitor()
		var pidGen int64
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				pid := int(atomic.AddInt64(&pidGen, 1))
				mon.Register(pid)
				_ = mon.Transition(pid, StateRunning)
				code := mon.Complete(pid, nil, 1)
				benchIntSink = code
			}
		})
	})
}

// BenchmarkSignalDispatch measures targeted and broadcast signal routing through queues.
func BenchmarkSignalDispatch(b *testing.B) {
	b.Run("DirectDispatch", func(b *testing.B) {
		queue := newSignalDispatchQueue()
		for i := 0; i < 100; i++ {
			pid := i
			queue.RegisterHandler(pid, func(sig SignalType) error {
				benchIntSink = int(sig)
				return nil
			})
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pid := i % 100
			err := queue.Dispatch(pid, SigTerm)
			benchErrSink = err
		}
	})

	b.Run("TreeBroadcast", func(b *testing.B) {
		queue := newSignalDispatchQueue()
		pidCounter := 0
		tree := buildSampleTree(4, 3, &pidCounter) // 40 nodes
		order := collectTreePostOrder(tree)

		for _, pid := range order {
			queue.RegisterHandler(pid, func(sig SignalType) error {
				return nil
			})
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			count := queue.BroadcastTree(order, SigTerm)
			benchIntSink = count
		}
	})

	b.Run("ConcurrentSignalDispatch", func(b *testing.B) {
		queue := newSignalDispatchQueue()
		for i := 0; i < 50; i++ {
			queue.RegisterHandler(i, func(sig SignalType) error {
				return nil
			})
		}

		var counter int64
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				idx := int(atomic.AddInt64(&counter, 1) % 50)
				err := queue.Dispatch(idx, SigKill)
				benchErrSink = err
			}
		})
	})
}

// BenchmarkExitCode evaluates exit status resolution for nil and failure conditions.
func BenchmarkExitCode(b *testing.B) {
	errGeneric := errors.New("permission denied")

	b.Run("NilError", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			code := ExitCode(nil, 127)
			benchIntSink = code
		}
	})

	b.Run("LaunchFailure", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			code := ExitCode(errGeneric, 127)
			benchIntSink = code
		}
	})
}
