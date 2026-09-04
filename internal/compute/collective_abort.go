package compute

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

var (
	// ErrCollectiveAborted is returned when a collective operation is terminated due to fabric or peer failure.
	ErrCollectiveAborted = errors.New("collective operation aborted due to GPU fabric failure")

	// ErrOutputFenced is returned when attempting to read or consume an output tensor that has been fenced.
	ErrOutputFenced = errors.New("collective output is fenced and cannot be consumed")

	// ErrRankNotFound is returned for an invalid rank ID.
	ErrRankNotFound = errors.New("rank not found in collective communicator")

	// ErrRankDead is returned when dispatching on a dead rank.
	ErrRankDead = errors.New("collective rank is dead")
)

// RankStatus indicates the operational state of a rank within the communicator.
type RankStatus string

const (
	RankStatusActive   RankStatus = "ACTIVE"
	RankStatusDead     RankStatus = "DEAD"
	RankStatusTimedOut RankStatus = "TIMED_OUT"
	RankStatusAborted  RankStatus = "ABORTED"
)

// FabricStatus indicates the overall health of the GPU interconnect fabric.
type FabricStatus string

const (
	FabricHealthy   FabricStatus = "HEALTHY"
	FabricAborted   FabricStatus = "ABORTED"
	FabricFenced    FabricStatus = "FENCED"
	FabricRecovered FabricStatus = "RECOVERED"
)

// CollectiveOutput represents an output tensor or reduction buffer produced by a collective operation (#10443).
// If a mid-flight failure occurs, the output is permanently fenced to prevent downstream kernels
// from reading stale, partial, or corrupted bytes.
type CollectiveOutput struct {
	mu          sync.RWMutex
	Generation  uint64
	Rank        int
	Data        []float32
	fenced      bool
	fenceReason error
	completedAt time.Time
}

// Fenced reports whether this output has been fenced.
func (o *CollectiveOutput) Fenced() bool {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.fenced
}

// FenceReason returns the underlying error if the output is fenced.
func (o *CollectiveOutput) FenceReason() error {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.fenceReason
}

// CompletedAt returns the completion timestamp if completed.
func (o *CollectiveOutput) CompletedAt() time.Time {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.completedAt
}

// Read returns the output data or fails closed if the output has been fenced.
func (o *CollectiveOutput) Read() ([]float32, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()

	if o.fenced {
		return nil, fmt.Errorf("%w (gen=%d, rank=%d): %v", ErrOutputFenced, o.Generation, o.Rank, o.fenceReason)
	}

	res := make([]float32, len(o.Data))
	copy(res, o.Data)
	return res, nil
}

func (o *CollectiveOutput) fence(reason error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.fenced = true
	o.fenceReason = reason
}

// CollectiveCommunicatorConfig configures fault detection and fencing.
type CollectiveCommunicatorConfig struct {
	MeshID           string
	WorldSize        int
	HeartbeatTimeout time.Duration
}

// CollectiveMeshCommunicator manages GPU fabric collective operations, detecting rank death/timeout,
// aborting all peer ranks, and fencing outputs (#10443).
type CollectiveMeshCommunicator struct {
	mu               sync.RWMutex
	meshID           string
	worldSize        int
	generation       uint64
	fabricStatus     FabricStatus
	heartbeatTimeout time.Duration

	rankStatuses map[int]RankStatus
	heartbeats   map[int]time.Time
	abortCh      chan struct{}
	abortOnce    sync.Once
	abortReason  error
	aborted      int32

	trackedOutputs []*CollectiveOutput
}

// NewCollectiveMeshCommunicator constructs a communicator bound to initial generation 1.
func NewCollectiveMeshCommunicator(cfg CollectiveCommunicatorConfig) *CollectiveMeshCommunicator {
	if cfg.WorldSize <= 0 {
		cfg.WorldSize = 1
	}
	if cfg.HeartbeatTimeout <= 0 {
		cfg.HeartbeatTimeout = 2 * time.Second
	}

	c := &CollectiveMeshCommunicator{
		meshID:           cfg.MeshID,
		worldSize:        cfg.WorldSize,
		generation:       1,
		fabricStatus:     FabricHealthy,
		heartbeatTimeout: cfg.HeartbeatTimeout,
		rankStatuses:     make(map[int]RankStatus, cfg.WorldSize),
		heartbeats:       make(map[int]time.Time, cfg.WorldSize),
		abortCh:          make(chan struct{}),
	}

	now := time.Now()
	for r := 0; r < cfg.WorldSize; r++ {
		c.rankStatuses[r] = RankStatusActive
		c.heartbeats[r] = now
	}

	return c
}

// MeshID returns the mesh identifier for this communicator.
func (c *CollectiveMeshCommunicator) MeshID() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.meshID
}

// Generation returns the current generation/epoch of the communicator.
func (c *CollectiveMeshCommunicator) Generation() uint64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.generation
}

// WorldSize returns the rank count of the communicator.
func (c *CollectiveMeshCommunicator) WorldSize() int {
	return c.worldSize
}

// FabricStatus returns the current fabric health status.
func (c *CollectiveMeshCommunicator) FabricStatus() FabricStatus {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.fabricStatus
}

// IsAborted reports whether the current communicator generation is in an aborted state.
func (c *CollectiveMeshCommunicator) IsAborted() bool {
	return atomic.LoadInt32(&c.aborted) != 0
}

// AbortChan returns a channel closed when an abort is triggered, unblocking all peers.
func (c *CollectiveMeshCommunicator) AbortChan() <-chan struct{} {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.abortCh
}

// RankStatus returns the current status of a given rank.
func (c *CollectiveMeshCommunicator) RankStatus(rank int) (RankStatus, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	st, ok := c.rankStatuses[rank]
	if !ok {
		return "", ErrRankNotFound
	}
	return st, nil
}

// Heartbeat updates the liveness timestamp for a rank.
func (c *CollectiveMeshCommunicator) Heartbeat(rank int, now time.Time) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if rank < 0 || rank >= c.worldSize {
		return ErrRankNotFound
	}

	if c.rankStatuses[rank] == RankStatusDead {
		return ErrRankDead
	}

	c.heartbeats[rank] = now
	return nil
}

// CheckLiveness evaluates rank heartbeats against the timeout threshold.
// If any rank has timed out, an abort is immediately triggered across all peers.
func (c *CollectiveMeshCommunicator) CheckLiveness(now time.Time) error {
	c.mu.Lock()
	if c.fabricStatus == FabricAborted || c.fabricStatus == FabricFenced {
		reason := c.abortReason
		c.mu.Unlock()
		return reason
	}

	var timedOutRank = -1
	for r := 0; r < c.worldSize; r++ {
		if c.rankStatuses[r] == RankStatusActive {
			hb := c.heartbeats[r]
			if now.Sub(hb) > c.heartbeatTimeout {
				timedOutRank = r
				c.rankStatuses[r] = RankStatusTimedOut
				break
			}
		}
	}
	c.mu.Unlock()

	if timedOutRank >= 0 {
		err := fmt.Errorf("rank %d heartbeat timeout after %v", timedOutRank, c.heartbeatTimeout)
		c.Abort(err)
		return err
	}

	return nil
}

// ReportRankDeath reports a rank death (e.g. CUDA Xid fault, SIGKILL, PCI error)
// and immediately triggers an abort across all peers (#10443).
func (c *CollectiveMeshCommunicator) ReportRankDeath(rank int, err error) {
	c.mu.Lock()
	if rank >= 0 && rank < c.worldSize {
		c.rankStatuses[rank] = RankStatusDead
	}
	c.mu.Unlock()

	c.Abort(fmt.Errorf("rank %d died: %w", rank, err))
}

// Abort transitions the communicator to aborted status, closes the abort channel,
// marks all surviving ranks as aborted, and fences all outputs in this generation (#10443).
func (c *CollectiveMeshCommunicator) Abort(reason error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if atomic.LoadInt32(&c.aborted) != 0 {
		return
	}

	atomic.StoreInt32(&c.aborted, 1)
	c.abortReason = reason
	c.fabricStatus = FabricFenced

	// Close abortCh to notify all waiting peer goroutines
	c.abortOnce.Do(func() {
		close(c.abortCh)
	})

	// Mark remaining active ranks as ABORTED
	for r := 0; r < c.worldSize; r++ {
		if c.rankStatuses[r] == RankStatusActive {
			c.rankStatuses[r] = RankStatusAborted
		}
	}

	// Fence all outputs in this generation
	for _, out := range c.trackedOutputs {
		if out.Generation == c.generation {
			out.fence(reason)
		}
	}
}

// ExecuteCollective runs a rank's collective slice within the communicator epoch.
// If the communicator is aborted before or during execution, or if the context expires,
// it fails immediately and fences the output (#10443).
func (c *CollectiveMeshCommunicator) ExecuteCollective(
	ctx context.Context,
	rank int,
	input []float32,
	op func(ctx context.Context) ([]float32, error),
) (*CollectiveOutput, error) {
	c.mu.RLock()
	if rank < 0 || rank >= c.worldSize {
		c.mu.RUnlock()
		return nil, ErrRankNotFound
	}
	gen := c.generation
	aborted := atomic.LoadInt32(&c.aborted) != 0
	abortCh := c.abortCh
	abortErr := c.abortReason
	rankSt := c.rankStatuses[rank]
	c.mu.RUnlock()

	if aborted || rankSt != RankStatusActive {
		if abortErr == nil {
			abortErr = ErrCollectiveAborted
		}
		fencedOut := &CollectiveOutput{
			Generation:  gen,
			Rank:        rank,
			Data:        nil,
			fenced:      true,
			fenceReason: abortErr,
		}
		c.trackOutput(fencedOut)
		return fencedOut, fmt.Errorf("%w: %v", ErrCollectiveAborted, abortErr)
	}

	// Channel for operation completion
	type opResult struct {
		data []float32
		err  error
	}
	resCh := make(chan opResult, 1)

	go func() {
		data, err := op(ctx)
		resCh <- opResult{data: data, err: err}
	}()

	select {
	case <-abortCh:
		c.mu.RLock()
		reason := c.abortReason
		c.mu.RUnlock()
		if reason == nil {
			reason = ErrCollectiveAborted
		}

		fencedOut := &CollectiveOutput{
			Generation:  gen,
			Rank:        rank,
			fenced:      true,
			fenceReason: reason,
		}
		c.trackOutput(fencedOut)
		return fencedOut, fmt.Errorf("%w: peer aborted: %v", ErrCollectiveAborted, reason)

	case <-ctx.Done():
		c.Abort(ctx.Err())
		fencedOut := &CollectiveOutput{
			Generation:  gen,
			Rank:        rank,
			fenced:      true,
			fenceReason: ctx.Err(),
		}
		c.trackOutput(fencedOut)
		return fencedOut, ctx.Err()

	case res := <-resCh:
		if res.err != nil {
			c.ReportRankDeath(rank, res.err)
			fencedOut := &CollectiveOutput{
				Generation:  gen,
				Rank:        rank,
				fenced:      true,
				fenceReason: res.err,
			}
			c.trackOutput(fencedOut)
			return fencedOut, res.err
		}

		c.mu.Lock()
		if atomic.LoadInt32(&c.aborted) != 0 {
			reason := c.abortReason
			if reason == nil {
				reason = ErrCollectiveAborted
			}
			fencedOut := &CollectiveOutput{
				Generation:  gen,
				Rank:        rank,
				fenced:      true,
				fenceReason: reason,
			}
			c.trackedOutputs = append(c.trackedOutputs, fencedOut)
			c.mu.Unlock()
			return fencedOut, fmt.Errorf("%w: %v", ErrCollectiveAborted, reason)
		}

		out := &CollectiveOutput{
			Generation:  gen,
			Rank:        rank,
			Data:        res.data,
			fenced:      false,
			completedAt: time.Now(),
		}
		c.trackedOutputs = append(c.trackedOutputs, out)
		c.mu.Unlock()
		return out, nil
	}
}

func (c *CollectiveMeshCommunicator) trackOutput(out *CollectiveOutput) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.trackedOutputs = append(c.trackedOutputs, out)
}

// AdvanceEpoch advances the communicator to the next generation/epoch, resetting
// the abort channel and active rank statuses for surviving ranks (#10443).
//
// Invariant: Outputs produced under previous epochs REMAIN PERMANENTLY FENCED.
func (c *CollectiveMeshCommunicator) AdvanceEpoch(survivingRanks []int) uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.generation++
	atomic.StoreInt32(&c.aborted, 0)
	c.abortReason = nil
	c.abortOnce = sync.Once{}
	c.abortCh = make(chan struct{})
	c.fabricStatus = FabricHealthy

	// Reset rank statuses according to surviving ranks
	survMap := make(map[int]bool, len(survivingRanks))
	for _, r := range survivingRanks {
		survMap[r] = true
	}

	for r := 0; r < c.worldSize; r++ {
		if survMap[r] {
			c.rankStatuses[r] = RankStatusActive
			c.heartbeats[r] = time.Now()
		} else {
			c.rankStatuses[r] = RankStatusDead
		}
	}

	// Note: c.trackedOutputs is preserved; any outputs already fenced stay fenced!
	return c.generation
}
