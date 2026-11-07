//go:build cuda && cgo

package compute

import (
	"flag"
	"math"
	"runtime"
	"sort"
	"testing"
)

var cudaSamplingCandidatesRequired = flag.Bool("fak-cuda-sampling-candidates-required", false, "require physical raw candidate transport and retained-copy witness")

// This test qualifies the physical producer only. The served serial caller's
// full transfer/lifetime witness must also run before promotion.
// fak-test:runtime slow est=10s lane=optin
func TestCUDASamplingCandidatesDevice(t *testing.T) {
	var cb *cudaBackend
	if *cudaSamplingCandidatesRequired {
		raw, ok := Lookup("cuda")
		if !ok {
			t.Fatal("required CUDA backend absent")
		}
		cb, ok = raw.(*cudaBackend)
		if !ok {
			t.Fatalf("required native CUDA backend absent: %T", raw)
		}
	} else {
		cb = cudaOrSkip(t)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	deviceWork := func() DeviceWork {
		t.Helper()
		for _, work := range DeviceWorkSnapshot() {
			if work.Backend == "cuda" {
				return work
			}
		}
		t.Fatal("CUDA device-work reader absent")
		return DeviceWork{}
	}
	for _, n := range []int{67, 257, 65537} {
		row := make([]float32, n)
		for i := range row {
			row[i] = float32((i*7919)%1009-504) / 17
		}
		row[3] = math.Float32frombits(0x80000000)
		row[5] = 0
		row[n-1] = math.Float32frombits(1)
		for _, k := range []int{1, 64} {
			logits := mkResident(cb, []int{n}, row)
			var packet SamplingCandidatePacket
			before := cb.HostXferBytes()
			workBefore := deviceWork()
			if *cudaSamplingCandidatesRequired && !workBefore.SyncObserved {
				t.Fatal("required accounting witness needs an isolated process with observed CUDA work")
			}
			kept, available, err := cb.SamplingCandidates(logits, k, true, &packet)
			if err != nil || !available || !packet.Valid || packet.Count != k+1 {
				t.Fatalf("n=%d k=%d available=%t count=%d valid=%t err=%v", n, k, available, packet.Count, packet.Valid, err)
			}
			if got, want := cb.HostXferBytes()-before, uint64(4*(2+2*(k+1))); got != want {
				t.Fatalf("candidate transport=%d bytes, want %d", got, want)
			}
			workAfter := deviceWork()
			if workAfter.Dispatches-workBefore.Dispatches != 19 ||
				workAfter.ReadbackBytes-workBefore.ReadbackBytes != uint64(4*(2+2*(k+1))) ||
				workAfter.SyncObserved != workBefore.SyncObserved {
				t.Fatalf("candidate work accounting before=%+v after=%+v", workBefore, workAfter)
			}
			// The process-wide stage registry may legitimately fold later names
			// into "other" after unrelated tests fill its bounded label table.
			// Only the explicitly isolated required witness demands this label;
			// aggregate accounting above remains mandatory in ordinary runs.
			if *cudaSamplingCandidatesRequired {
				sites, ok := DeviceSyncSitesDelta(workBefore.SyncSites, workAfter.SyncSites)
				found := false
				for _, site := range sites {
					found = found || (site.Site == SyncSiteReadback && site.Stage == "sampling_candidates_retain" && site.D2HBytes == uint64(4*(2+2*(k+1))))
				}
				if !ok || !found {
					t.Fatalf("candidate readback stage absent: %+v", sites)
				}
			}
			ids := make([]int, n)
			for i := range ids {
				ids[i] = i
			}
			sort.Slice(ids, func(i, j int) bool {
				if row[ids[i]] != row[ids[j]] {
					return row[ids[i]] > row[ids[j]]
				}
				return ids[i] < ids[j]
			})
			gotIDs := append([]int(nil), packet.IDs[:packet.Count]...)
			sort.Ints(gotIDs)
			wantIDs := append([]int(nil), ids[:k+1]...)
			sort.Ints(wantIDs)
			for i := range gotIDs {
				if gotIDs[i] != wantIDs[i] {
					t.Fatalf("global candidate set differs at %d", i)
				}
			}
			for i, id := range packet.IDs[:packet.Count] {
				if math.Float32bits(packet.Raw[i]) != math.Float32bits(row[id]) {
					t.Fatal("raw candidate payload changed")
				}
			}
			copy := cb.Read(kept)
			for i := range row {
				if math.Float32bits(copy[i]) != math.Float32bits(row[i]) {
					t.Fatal("retained raw row changed")
				}
			}
			cb.Free(kept)
			cb.Free(logits)
		}
	}
	for _, invalid := range [][]float32{
		{1, 2, float32(math.NaN())}, {1, 2, float32(math.Inf(1))},
		{float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1))},
	} {
		logits := mkResident(cb, []int{len(invalid)}, invalid)
		var packet SamplingCandidatePacket
		_, available, err := cb.SamplingCandidates(logits, 1, false, &packet)
		if err != nil || !available || packet.Valid {
			t.Fatalf("invalid row: available=%t valid=%t err=%v", available, packet.Valid, err)
		}
		cb.Free(logits)
	}
}
