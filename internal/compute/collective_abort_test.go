package compute

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// TestCollectiveAbortRankDeath verifies that when one rank dies, all peers are aborted and fenced (#10443).
func TestCollectiveAbortRankDeath(t *testing.T) {
	comm := NewCollectiveMeshCommunicator(CollectiveCommunicatorConfig{
		MeshID:           "test-mesh-4gpu",
		WorldSize:        4,
		HeartbeatTimeout: 1 * time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	errs := make([]error, 4)

	// Rank 0, 2, 3 wait in a long-running collective
	for r := 0; r < 4; r++ {
		if r == 1 {
			continue // Rank 1 will simulate death
		}
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			_, err := comm.ExecuteCollective(ctx, rank, []float32{1.0}, func(c context.Context) ([]float32, error) {
				// Simulate waiting for collective ring synchronization
				select {
				case <-c.Done():
					return nil, c.Err()
				case <-time.After(500 * time.Millisecond):
					return []float32{1.0}, nil
				}
			})
			errs[rank] = err
		}(r)
	}

	// Give peers a moment to start waiting
	time.Sleep(20 * time.Millisecond)

	// Rank 1 dies with an Xid error
	xidErr := errors.New("CUDA Xid 31 FAULT_PDE ACCESS_TYPE_VIRT_READ")
	comm.ReportRankDeath(1, xidErr)

	wg.Wait()

	// All peers (rank 0, 2, 3) must report abort
	for r := 0; r < 4; r++ {
		if r == 1 {
			continue
		}
		if errs[r] == nil || !errors.Is(errs[r], ErrCollectiveAborted) {
			t.Errorf("rank %d error = %v, want ErrCollectiveAborted", r, errs[r])
		}
	}

	// Verify rank statuses
	st1, _ := comm.RankStatus(1)
	if st1 != RankStatusDead {
		t.Fatalf("rank 1 status = %s, want DEAD", st1)
	}
	st0, _ := comm.RankStatus(0)
	if st0 != RankStatusAborted {
		t.Fatalf("rank 0 status = %s, want ABORTED", st0)
	}
}

// TestCollectiveAbortTimeout verifies heartbeat timeout detection and automatic abort (#10443).
func TestCollectiveAbortTimeout(t *testing.T) {
	comm := NewCollectiveMeshCommunicator(CollectiveCommunicatorConfig{
		MeshID:           "test-mesh-timeout",
		WorldSize:        2,
		HeartbeatTimeout: 50 * time.Millisecond,
	})

	now := time.Now()
	_ = comm.Heartbeat(0, now)
	_ = comm.Heartbeat(1, now)

	// Advance time past timeout for rank 1
	future := now.Add(100 * time.Millisecond)
	_ = comm.Heartbeat(0, future) // Rank 0 stays alive; Rank 1 does not heartbeat

	err := comm.CheckLiveness(future)
	if err == nil {
		t.Fatalf("expected timeout error from CheckLiveness")
	}

	if !comm.IsAborted() {
		t.Fatalf("communicator should be aborted after timeout")
	}

	st1, _ := comm.RankStatus(1)
	if st1 != RankStatusTimedOut {
		t.Fatalf("rank 1 status = %s, want TIMED_OUT", st1)
	}
}

// TestCollectiveAbortFencingStaleOutputs verifies that outputs from an aborted epoch are fenced (#10443).
func TestCollectiveAbortFencingStaleOutputs(t *testing.T) {
	comm := NewCollectiveMeshCommunicator(CollectiveCommunicatorConfig{
		MeshID:           "test-mesh-fence",
		WorldSize:        2,
		HeartbeatTimeout: 1 * time.Second,
	})

	ctx := context.Background()

	// Rank 0 completes a collective successfully
	out0, err := comm.ExecuteCollective(ctx, 0, []float32{1.0, 2.0}, func(c context.Context) ([]float32, error) {
		return []float32{1.0, 2.0}, nil
	})
	if err != nil {
		t.Fatalf("ExecuteCollective rank 0 failed: %v", err)
	}

	// Read output before abort: valid
	val, err := out0.Read()
	if err != nil || len(val) != 2 {
		t.Fatalf("Read before abort = (%v, %v), want [1, 2]", val, err)
	}

	// Mid-flight fabric failure occurs in this epoch
	comm.Abort(errors.New("GPU interconnect fabric cable severed"))

	// Verify output is now fenced
	if !out0.Fenced() {
		t.Fatalf("out0 should be marked Fenced after abort")
	}

	// Reading fenced output must fail closed
	_, err = out0.Read()
	if err == nil || !errors.Is(err, ErrOutputFenced) {
		t.Fatalf("Read on fenced output = %v, want ErrOutputFenced", err)
	}
}

// TestCollectiveAbortEpochAdvancement verifies that advancing epoch restores communicator health
// while previous outputs remain permanently fenced (#10443).
func TestCollectiveAbortEpochAdvancement(t *testing.T) {
	comm := NewCollectiveMeshCommunicator(CollectiveCommunicatorConfig{
		MeshID:           "test-mesh-epoch",
		WorldSize:        3,
		HeartbeatTimeout: 1 * time.Second,
	})

	ctx := context.Background()

	// Epoch 1 output
	outGen1, _ := comm.ExecuteCollective(ctx, 0, []float32{42.0}, func(c context.Context) ([]float32, error) {
		return []float32{42.0}, nil
	})

	// Rank 2 dies -> Epoch 1 aborted
	comm.ReportRankDeath(2, errors.New("rank 2 OOM"))
	if !comm.IsAborted() {
		t.Fatal("communicator should be aborted")
	}

	// Advance to Epoch 2 with surviving ranks [0, 1]
	newGen := comm.AdvanceEpoch([]int{0, 1})
	if newGen != 2 {
		t.Fatalf("new generation = %d, want 2", newGen)
	}
	if comm.IsAborted() {
		t.Fatal("communicator should NOT be aborted in new epoch")
	}

	// Invariant: Epoch 1 output must STILL be fenced!
	_, err := outGen1.Read()
	if err == nil || !errors.Is(err, ErrOutputFenced) {
		t.Fatalf("outGen1 read = %v, want ErrOutputFenced", err)
	}

	// Epoch 2 collective on surviving rank 0 succeeds and can be read
	outGen2, err := comm.ExecuteCollective(ctx, 0, []float32{99.0}, func(c context.Context) ([]float32, error) {
		return []float32{99.0}, nil
	})
	if err != nil {
		t.Fatalf("ExecuteCollective in Epoch 2 failed: %v", err)
	}

	data, err := outGen2.Read()
	if err != nil || len(data) != 1 || data[0] != 99.0 {
		t.Fatalf("Read Epoch 2 output = (%v, %v), want [99.0]", data, err)
	}
}

// TestCollectiveAbortInvalidRank verifies that invalid rank indices fail immediately (#10443).
func TestCollectiveAbortInvalidRank(t *testing.T) {
	comm := NewCollectiveMeshCommunicator(CollectiveCommunicatorConfig{
		MeshID:    "test-invalid-rank",
		WorldSize: 2,
	})

	if comm.MeshID() != "test-invalid-rank" {
		t.Fatalf("MeshID() = %q, want %q", comm.MeshID(), "test-invalid-rank")
	}

	ctx := context.Background()
	_, err := comm.ExecuteCollective(ctx, -1, []float32{1.0}, func(c context.Context) ([]float32, error) {
		return []float32{1.0}, nil
	})
	if !errors.Is(err, ErrRankNotFound) {
		t.Fatalf("ExecuteCollective rank -1 = %v, want ErrRankNotFound", err)
	}

	_, err = comm.ExecuteCollective(ctx, 2, []float32{1.0}, func(c context.Context) ([]float32, error) {
		return []float32{1.0}, nil
	})
	if !errors.Is(err, ErrRankNotFound) {
		t.Fatalf("ExecuteCollective rank 2 = %v, want ErrRankNotFound", err)
	}

	if err := comm.Heartbeat(-1, time.Now()); !errors.Is(err, ErrRankNotFound) {
		t.Fatalf("Heartbeat rank -1 = %v, want ErrRankNotFound", err)
	}
	if _, err := comm.RankStatus(-1); !errors.Is(err, ErrRankNotFound) {
		t.Fatalf("RankStatus rank -1 = %v, want ErrRankNotFound", err)
	}
}

// TestCollectiveAbortContextCancellation verifies that canceling context aborts the collective (#10443).
func TestCollectiveAbortContextCancellation(t *testing.T) {
	comm := NewCollectiveMeshCommunicator(CollectiveCommunicatorConfig{
		MeshID:    "test-cancel-ctx",
		WorldSize: 2,
	})

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	out, err := comm.ExecuteCollective(ctx, 0, []float32{1.0}, func(c context.Context) ([]float32, error) {
		<-c.Done()
		return nil, c.Err()
	})

	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("ExecuteCollective with canceled ctx = %v, want context.Canceled", err)
	}
	if out == nil || !out.Fenced() {
		t.Fatalf("output must be fenced on context cancel")
	}
	if !comm.IsAborted() {
		t.Fatalf("communicator should be aborted on context cancel")
	}
}

// TestCollectiveAbortHeartbeatAfterDeath verifies that dead ranks reject heartbeats (#10443).
func TestCollectiveAbortHeartbeatAfterDeath(t *testing.T) {
	comm := NewCollectiveMeshCommunicator(CollectiveCommunicatorConfig{
		MeshID:    "test-dead-hb",
		WorldSize: 2,
	})

	comm.ReportRankDeath(0, errors.New("fatal GPU PCIe drop"))
	err := comm.Heartbeat(0, time.Now())
	if !errors.Is(err, ErrRankDead) {
		t.Fatalf("Heartbeat on dead rank = %v, want ErrRankDead", err)
	}
}
