package compute

import (
	"sync"
	"testing"
)

func TestNUMADecodePool_RowBounds(t *testing.T) {
	for _, n := range []int{1, 2, 3, 7, 16, 64, 127, 256, 1000} {
		for _, w := range []int{1, 2, 3, 4, 8, 16, 32} {
			covered := make([]int, n)
			for i := 0; i < w; i++ {
				lo, hi := workerRowBounds(i, w, n)
				if lo < 0 || hi > n || lo > hi {
					t.Fatalf("n=%d w=%d worker=%d invalid range [%d, %d)", n, w, i, lo, hi)
				}
				for r := lo; r < hi; r++ {
					covered[r]++
				}
			}
			for r := 0; r < n; r++ {
				if covered[r] != 1 {
					t.Fatalf("n=%d w=%d row %d covered %d times (want 1)", n, w, r, covered[r])
				}
			}
		}
	}
}

func TestNUMADecodePool_BarrierFreeExecutionAndBitIdentity(t *testing.T) {
	topo := SynthesizeNUMATopology(4)
	sched := ScheduleDecodeNUMA(topo, 8)
	if !sched.Eligible {
		t.Fatalf("sched ineligible: %s", sched.Reason)
	}

	pool, err := NewNUMADecodePool(sched)
	if err != nil {
		t.Fatalf("NewNUMADecodePool failed: %v", err)
	}
	defer pool.Close()

	const out = 256
	const in = 256

	// Synthetic weights per node (byte-identical across replicas)
	src := make([]byte, out*in)
	for i := range src {
		src[i] = byte(i%251 + 1)
	}
	replicaSet, err := BuildNUMAReplicasForTopology(src, topo)
	if err != nil {
		t.Fatalf("BuildNUMAReplicasForTopology: %v", err)
	}
	defer replicaSet.Free()

	x := make([]float32, in)
	for i := range x {
		x[i] = float32(i)*0.01 + 0.5
	}

	// Reference output computed serially from src
	wantY := make([]float32, out)
	for o := 0; o < out; o++ {
		var sum float32
		row := src[o*in : (o+1)*in]
		for c := 0; c < in; c++ {
			sum += float32(row[c]) * x[c]
		}
		wantY[o] = sum
	}

	// Output computed via NUMADecodePool reading local replica per node
	gotY := make([]float32, out)
	nodeReads := make([]int, 4)
	var nodeMu sync.Mutex

	err = pool.Dispatch(out, func(nodeID, lo, hi int) {
		replica := replicaSet.For(nodeID)
		if replica == nil {
			t.Errorf("worker on node %d received nil replica", nodeID)
			return
		}
		nodeMu.Lock()
		nodeReads[nodeID]++
		nodeMu.Unlock()

		for o := lo; o < hi; o++ {
			var sum float32
			row := replica[o*in : (o+1)*in]
			for c := 0; c < in; c++ {
				sum += float32(row[c]) * x[c]
			}
			gotY[o] = sum
		}
	})
	if err != nil {
		t.Fatalf("Dispatch failed: %v", err)
	}

	// Verify every node participated
	for nodeID, count := range nodeReads {
		if count == 0 {
			t.Fatalf("node %d did not perform any replica reads", nodeID)
		}
	}

	// Verify bit-identical output
	for o := 0; o < out; o++ {
		if gotY[o] != wantY[o] {
			t.Fatalf("row %d: got %f, want %f (bit mismatch)", o, gotY[o], wantY[o])
		}
	}
}

func TestNUMADecodePool_ThreadSafety(t *testing.T) {
	topo := SynthesizeNUMATopology(2)
	sched := ScheduleDecodeNUMA(topo, 4)
	pool, err := NewNUMADecodePool(sched)
	if err != nil {
		t.Fatalf("NewNUMADecodePool: %v", err)
	}
	defer pool.Close()

	const out = 64
	const concurrency = 16
	var wg sync.WaitGroup
	wg.Add(concurrency)

	for g := 0; g < concurrency; g++ {
		go func(gid int) {
			defer wg.Done()
			y := make([]float32, out)
			err := pool.Dispatch(out, func(nodeID, lo, hi int) {
				for o := lo; o < hi; o++ {
					y[o] = float32(o + gid*1000)
				}
			})
			if err != nil {
				t.Errorf("goroutine %d dispatch failed: %v", gid, err)
				return
			}
			for o := 0; o < out; o++ {
				want := float32(o + gid*1000)
				if y[o] != want {
					t.Errorf("goroutine %d row %d = %f, want %f", gid, o, y[o], want)
					return
				}
			}
		}(g)
	}

	wg.Wait()
}
