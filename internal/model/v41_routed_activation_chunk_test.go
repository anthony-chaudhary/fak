package model

import (
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"reflect"
	"testing"
)

// CPU orchestration fixture: physical chunk records are expanded only inside
// this fake transport. No native dispatch, status fill or GPU parity is claimed.
type v41ActivationChunkWitness struct {
	*v41ExpertPanelRowWitness
	chunks                  map[compute.Buffer][]error
	chunkCalls, chunkProbes []int
	enabled                 bool
	failAt                  int
	failure                 error
	partial                 string
	packedOutputs           map[compute.Buffer]bool
	packedCleanupPanic      error
}

func (b *v41ActivationChunkWitness) SupportsV41RoutedSwiGLUPanelChunk(g, u compute.Tensor, first, rows int) bool {
	b.chunkProbes = append(b.chunkProbes, rows)
	return b.enabled && (rows == 2 || rows == 4 || rows == 8) && len(g.Shape) == 2 && reflect.DeepEqual(g.Shape, u.Shape) && first >= 0 && first+rows <= g.Shape[0]
}
func (b *v41ActivationChunkWitness) V41RoutedSwiGLUPanelChunkDeferred(g, u compute.Tensor, first int, limit float32, weights []float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.chunkCalls = append(b.chunkCalls, len(weights))
	width := g.Shape[1]
	if !b.ownedTiles[g.Buf()] || !b.ownedTiles[u.Buf()] {
		return compute.Tensor{}, compute.V41DeferredStatus{}, errors.New("borrowed panel retired")
	}
	gv, uv := b.Backend.Read(g), b.Backend.Read(u)
	packed := make([]float32, len(weights)*width)
	verdicts := make([]error, len(weights))
	for i, w := range weights {
		row := first + i
		gt := b.Backend.Upload(compute.NewF32(compute.Default(), []int{width}, gv[row*width:(row+1)*width]), compute.F32)
		ut := b.Backend.Upload(compute.NewF32(compute.Default(), []int{width}, uv[row*width:(row+1)*width]), compute.F32)
		out, st, err := b.v41RoutedLayerTestBackend.V41RoutedSwiGLUDeferred(gt, ut, limit, w)
		if err != nil {
			panic(err)
		}
		copy(packed[i*width:], b.Backend.Read(out))
		verdicts[i] = b.pending[st.Record.Buf()]
		delete(b.pending, st.Record.Buf())
		b.Backend.Free(st.Record)
		b.Backend.Free(out)
		b.Backend.Free(gt)
		b.Backend.Free(ut)
	}
	out := b.tensor([]int{len(packed)}, packed)
	b.packedOutputs[out.Buf()] = true
	record := b.Backend.Upload(compute.NewF32(compute.Default(), []int{4 * len(weights)}, make([]float32, 4*len(weights))), compute.F32)
	b.chunks[record.Buf()] = verdicts
	st := compute.V41DeferredStatus{Record: record, Kind: compute.V41DeferredRoutedSwiGLU, Count: width, Records: len(weights)}
	if len(b.chunkCalls) == b.failAt {
		if b.partial == "output" || b.partial == "empty" {
			delete(b.chunks, record.Buf())
			b.Backend.Free(record)
			st = compute.V41DeferredStatus{}
		}
		if b.partial == "status" || b.partial == "empty" {
			b.v41ExpertTileWitness.Free(out)
			out = compute.Tensor{}
		}
		if b.partial == "alias" {
			b.v41ExpertTileWitness.Free(out)
			out = g
			return out, st, nil
		}
		return out, st, b.failure
	}
	return out, st, nil
}
func (b *v41ActivationChunkWitness) Free(x compute.Tensor) {
	delete(b.chunks, x.Buf())
	b.v41ExpertPanelRowWitness.Free(x)
	if b.packedOutputs[x.Buf()] && b.packedCleanupPanic != nil {
		panic(b.packedCleanupPanic)
	}
}
func (b *v41ActivationChunkWitness) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	var flat []compute.V41DeferredStatus
	for _, st := range statuses {
		if st.Records <= 1 {
			flat = append(flat, st)
			continue
		}
		verdicts, ok := b.chunks[st.Record.Buf()]
		if !ok || len(verdicts) != st.Records {
			return nil, nil, errors.New("chunk read twice or malformed")
		}
		for _, v := range verdicts {
			one := b.record(v)
			one.Kind, one.Count, one.Site = st.Kind, st.Count, st.Site
			flat = append(flat, one)
		}
		b.Free(st.Record)
	}
	return b.v41ExpertTileWitness.V41ReadDeferred(flat, rows)
}

// fak-test:runtime medium est=10s lane=default
func TestV41ActivationChunkConsumer(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41PinnedSlateFixture(t, false)
	inputs := make([][]float32, 32)
	weights := make([]float32, 32)
	for r := range inputs {
		inputs[r] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[r] {
			inputs[r][i] = float32((r*3+i)%13-6) / 16
		}
		weights[r] = float32(r+1) / 64
	}
	makeSession := func(enabled bool) (*Session, *v41ActivationChunkWitness) {
		row := &v41ExpertPanelRowWitness{v41ExpertGateUp32Witness: &v41ExpertGateUp32Witness{v41ExpertDownTileWitness: &v41ExpertDownTileWitness{v41ExpertTileWitness: &v41ExpertTileWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}, tiles: true, ownedTiles: map[compute.Buffer]bool{}}, downTiles: true}, grid: true}, enabledRows: true, refuseRow: -1, frees: map[compute.Buffer]int{}}
		b := &v41ActivationChunkWitness{v41ExpertPanelRowWitness: row, chunks: map[compute.Buffer][]error{}, packedOutputs: map[compute.Buffer]bool{}, enabled: enabled}
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s, b
	}
	check := func(b *v41ActivationChunkWitness) {
		t.Helper()
		if len(b.pending) != 0 || len(b.chunks) != 0 || len(b.ownedTiles) != 0 {
			t.Fatal("temporary owner leak")
		}
		for _, n := range b.frees {
			if n != 1 {
				t.Fatalf("owner released%d times", n)
			}
		}
	}
	for _, n := range []int{2, 4, 8, 15, 32} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			old, ob := makeSession(false)
			s, b := makeSession(true)
			want, wo, we := old.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:n], weights[:n])
			got, o, e := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:n], weights[:n])
			if we != nil || e != nil || wo != v41RoutedExpertHandled || o != wo || len(got) != n {
				t.Fatalf("%v/%v", we, e)
			}
			for r := range got {
				if !v41SameBits(got[r], want[r]) {
					t.Fatalf("row%d differs", r)
				}
			}
			if !reflect.DeepEqual(b.chunkCalls, b.downRows) || !reflect.DeepEqual(b.downRows, ob.downRows) || ob.gathers-b.gathers != len(b.chunkCalls) || b.fences != 1 || len(b.directRows) != n%2 {
				t.Fatalf("chunks%v down%v direct%v gathers%d/%d", b.chunkCalls, b.downRows, b.directRows, ob.gathers, b.gathers)
			}
			check(b)
			check(ob)
		})
	}
	for _, partial := range []string{"both", "output", "status", "empty", "alias"} {
		t.Run(partial, func(t *testing.T) {
			s, b := makeSession(true)
			b.failAt = 1
			b.partial = partial
			b.failure = &compute.BackendError{Backend: "fixture", Site: "chunk", Err: errors.New("selected")}
			out, o, e := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:8], weights[:8])
			if out != nil || o != v41RoutedExpertError || e == nil || len(b.chunkCalls) != 1 || len(b.downRows) != 0 || len(b.directRows) != 0 {
				t.Fatalf("selected failure replayed: %v", e)
			}
			if partial != "alias" && !errors.Is(e, b.failure) {
				t.Fatal("selected error replaced")
			}
			check(b)
		})
	}
	t.Run("earlier_down_verdict_before_later_chunk_error", func(t *testing.T) {
		s, b := makeSession(true)
		b.publishFaultAt = map[int]bool{1: true}
		b.failAt = 2
		b.failure = &compute.BackendError{Backend: "fixture", Site: "second chunk", Err: errors.New("later")}
		_, o, e := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:16], weights[:16])
		var numeric *V41ExpertOperationError
		if o != v41RoutedExpertError || !errors.As(e, &numeric) || numeric.Stage != "down" || errors.Is(e, b.failure) || len(b.downRows) != 1 {
			t.Fatalf("issued numeric order changed: %v", e)
		}
		check(b)
	})
	t.Run("down_recording_failure_prevents_later_chunk", func(t *testing.T) {
		s, b := makeSession(true)
		b.failDownAt = 1
		b.downFailure = &compute.BackendError{Backend: "fixture", Site: "down", Err: errors.New("recording")}
		_, o, e := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:16], weights[:16])
		if o != v41RoutedExpertError || !errors.Is(e, b.downFailure) || len(b.chunkCalls) != 1 {
			t.Fatalf("later chunk issued: %v", e)
		}
		check(b)
	})

	t.Run("earlier_down_before_later_row_chunk_numeric", func(t *testing.T) {
		s, b := makeSession(true)
		b.publishFaultAt = map[int]bool{1: true}
		b.activationFaults = map[float32]*compute.V41RoutedSwiGLUFault{weights[7]: {Stage: compute.V41RoutedSwiGLUStageActivation, Index: 3}}
		out, o, e := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:16], weights[:16])
		var numeric *V41ExpertOperationError
		if out != nil || o != v41RoutedExpertError || !errors.As(e, &numeric) || numeric.Stage != "down" || len(b.chunkCalls) != 2 || len(b.downRows) != 2 {
			t.Fatalf("logical row priority changed: %v", e)
		}
		check(b)
	})
	t.Run("chunk_start_cleanup_panic_is_selected", func(t *testing.T) {
		s, b := makeSession(true)
		b.publishFaultAt = map[int]bool{2: true}
		failure := &compute.BackendError{Backend: "fixture", Site: "packed cleanup", Err: errors.New("release")}
		b.packedCleanupPanic = failure
		out, o, e := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:8], weights[:8])
		// The single packed allocation belongs to chunk start, rather than
		// eight independent row owners. Its release panic precedes row 1.
		if out != nil || o != v41RoutedExpertError || !errors.Is(e, failure) || !s.BackendSessionClosed() {
			t.Fatalf("grouped cleanup boundary changed: %v", e)
		}
		check(b)
	})
	t.Run("read_failure_discards_all_numeric_verdicts", func(t *testing.T) {
		s, b := makeSession(true)
		b.publishFaultAt = map[int]bool{1: true}
		failure := &compute.BackendError{Backend: "fixture", Site: "readback", Err: errors.New("unknown completion")}
		b.v41ScalarCohortWitness.readFailure = failure
		out, o, e := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:16], weights[:16])
		if out != nil || o != v41RoutedExpertError || !errors.Is(e, failure) || !s.BackendSessionClosed() {
			t.Fatalf("read failure exposed verdict/output: %v", e)
		}
		check(b)
	})
}
