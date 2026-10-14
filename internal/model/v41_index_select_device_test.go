package model

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41IndexSelectCall struct {
	scores                            []float32
	rows, topK, topKBlocks, blockSize int
}

// This recorder selects through compute's portable rank-count mirror of the
// device kernel over cpu-ref storage. DeviceMemory tests routing, not silicon.
type v41IndexSelectTestBackend struct {
	*v41DenseTestBackend
	calls    []v41IndexSelectCall
	outputs  map[compute.Buffer]bool
	fault    string
	failCall int
	cause    error
}

func newV41IndexSelectTestBackend() *v41IndexSelectTestBackend {
	return &v41IndexSelectTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
}

func (b *v41IndexSelectTestBackend) V41IndexSelectAvailable() bool { return true }

func (b *v41IndexSelectTestBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 index-select ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if class != compute.MemoryActivation || dt != compute.F32 {
		panic("index-select upload is not a classed F32 activation")
	}
	return compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
}

func (b *v41IndexSelectTestBackend) V41IndexSelect(scores compute.Tensor, rows, topK, topKBlocks, blockSize int) (compute.Tensor, error) {
	if scores.Backend() != b || !reflect.DeepEqual(scores.Shape, []int{rows}) {
		return compute.Tensor{}, errors.New("invalid index-select contract")
	}
	call := v41IndexSelectCall{scores: append([]float32(nil), b.Backend.Read(scores)...), rows: rows, topK: topK, topKBlocks: topKBlocks, blockSize: blockSize}
	b.calls = append(b.calls, call)
	if len(b.calls) == b.failCall && b.fault == "dispatch" {
		return compute.Tensor{}, b.cause
	}
	rowsOut, err := v41IndexSelectRankCountRef(call.scores, topK, topKBlocks, blockSize)
	if err != nil {
		return compute.Tensor{}, err
	}
	values := make([]float32, len(rowsOut))
	for i, row := range rowsOut {
		values[i] = float32(row)
	}
	if len(b.calls) == b.failCall && b.fault == "unsorted" && len(values) > 1 {
		values[0], values[1] = values[1], values[0]
		if values[0] < 0 || values[1] < 0 {
			values[0] = float32(rows) // out of range when a sentinel blocks the swap witness
		}
	}
	out := compute.NewF32(b, []int{len(values)}, values)
	b.outputs[out.Buf()] = true
	return out, nil
}

func (b *v41IndexSelectTestBackend) Free(x compute.Tensor) {
	delete(b.outputs, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

func v41IndexSelectOnly(s *Session) {
	v41ScoreOnly(s)
	st := s.v41State()
	st.setIndexerScore(nil)
	st.indexScoreHealth = nil
}

// v41IndexSelectHost is the existing host selection pair the device must match.
func v41IndexSelectHost(scores []float32, topK, topKBlocks, blockSize int) ([]int32, error) {
	var mask []bool
	var err error
	if topKBlocks > 0 {
		if mask, err = V41SelectCandidateBlocks(scores, len(scores), topKBlocks, blockSize); err != nil {
			return nil, err
		}
	}
	return V41SelectIndexRows(scores, len(scores), topK, 0, mask)
}

// Refs #13668. The device rank-count selection is the host selection: the same
// rows, in the same order, with the same ties, pinning and -1 sentinels.
// fak-test:runtime fast est=500ms lane=default
func TestV41IndexSelectMatchesHostSelectors(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	inf, ninf := float32(math.Inf(1)), float32(math.Inf(-1))
	cases := [][]float32{
		{1, 1, 1, 1},
		{0, negZero, 0, negZero, 0},
		{3, ninf, 2, ninf, inf, 1},
		{ninf, ninf, ninf},
		{-1, -2, -3, -4, -5, -6, -7, -8, -9},
	}
	rng := rand.New(rand.NewSource(13668))
	for n := 1; n <= 70; n += 3 {
		row := make([]float32, n)
		for i := range row {
			// Few distinct values force ties across blocks.
			row[i] = float32(rng.Intn(5)-2) * 0.5
			if rng.Intn(11) == 0 {
				row[i] = ninf
			}
		}
		cases = append(cases, row)
	}
	checked := 0
	for ci, scores := range cases {
		for _, topK := range []int{1, 2, 3, 7, 64, 200} {
			for _, blocks := range [][2]int{{0, 0}, {1, 1}, {1, 4}, {2, 3}, {3, 8}, {100, 2}} {
				want, err := v41IndexSelectHost(scores, topK, blocks[0], blocks[1])
				if err != nil {
					t.Fatal(err)
				}
				got, err := v41IndexSelectRankCountRef(scores, topK, blocks[0], blocks[1])
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("case %d topK=%d blocks=%v: device=%v err=%v host=%v", ci, topK, blocks, got, err, want)
				}
				checked++
			}
		}
	}
	if checked < 500 {
		t.Fatalf("parity sweep too small: %d", checked)
	}
	if _, err := v41IndexSelectRankCountRef([]float32{1, float32(math.NaN())}, 1, 0, 0); err == nil {
		t.Fatal("NaN score was selected")
	}
}

// Refs #13668. The selected seam replaces only the selection: the forward is
// bit-identical to the host session, a failure fails closed without host
// re-selection, and a device-only session without the op refuses.
// fak-test:runtime medium est=5s lane=default
func TestV41IndexSelectDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("index-select software witness requires cpu-ref")
	}
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 1, 2}

	t.Run("seam-matches-host-selection", func(t *testing.T) {
		b := newV41IndexSelectTestBackend()
		s := &Session{M: &Model{}, Backend: b}
		t.Cleanup(s.Close)
		sel := s.v41DeviceGatherOpsFunc().indexSelectFunc()
		if sel == nil {
			t.Fatal("available index-select operation was not bound")
		}
		scores := []float32{0.5, 2, float32(math.Inf(-1)), 2, -1, 0.5, 3}
		for _, tc := range [][3]int{{3, 0, 0}, {4, 2, 2}, {10, 1, 3}} {
			want, err := v41IndexSelectHost(scores, tc[0], tc[1], tc[2])
			if err != nil {
				t.Fatal(err)
			}
			got, err := sel(0, scores, tc[0], tc[1], tc[2])
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("selection %v: got %v err %v want %v", tc, got, err, want)
			}
		}
		if empty, err := sel(0, nil, 4, 0, 0); err != nil || len(empty) != 0 || len(b.calls) != 3 {
			t.Fatalf("empty selection dispatched or failed: %v %v calls=%d", empty, err, len(b.calls))
		}
		if len(b.outputs) != 0 {
			t.Fatal("selected index output leaked")
		}
	})

	t.Run("forward-bit-identical", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		b := newV41IndexSelectTestBackend()
		s := v41DenseTestSession(t, m, b)
		v41IndexSelectOnly(s)
		host := m.NewSession()
		t.Cleanup(host.Close)
		v41ScoreTestBits(t, s.Prefill(tokens), host.Prefill(tokens))
		if len(b.calls) == 0 {
			t.Fatal("prefill never offered a selection to the device seam")
		}
		before := len(b.calls)
		v41ScoreTestBits(t, s.Step(3), host.Step(3))
		if len(b.calls) == before {
			t.Fatal("step never offered a selection to the device seam")
		}
		if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(host.v41Forward)) {
			t.Fatal("device selection changed retained index/top-k state")
		}
	})

	for _, fault := range []string{"dispatch", "unsorted"} {
		t.Run("selected-failure-"+fault, func(t *testing.T) {
			m := v41IndexerTestFixture(t)
			b := newV41IndexSelectTestBackend()
			s := v41DenseTestSession(t, m, b)
			v41IndexSelectOnly(s)
			s.Prefill(tokens)
			b.fault, b.failCall = fault, len(b.calls)+1
			b.cause = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			err := recoverError(func() { s.Step(3) })
			var closed *BackendForwardOperationError
			if !errors.As(err, &closed) || closed.Path != "v41-index-select" || !s.BackendSessionClosed() {
				t.Fatalf("selected index failure lost identity: %v", err)
			}
			if fault == "dispatch" && !errors.Is(err, b.cause) {
				t.Fatalf("selected index failure lost its cause: %v", err)
			}
			if len(b.calls) != b.failCall || len(b.outputs) != 0 {
				t.Fatal("selected index failure replayed or leaked")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || len(b.calls) != b.failCall {
				t.Fatalf("closed session retried the selection: %v", retry)
			}
		})
	}

	t.Run("device-only-without-op-refuses", func(t *testing.T) {
		s := &Session{M: &Model{}, Backend: newV41DenseTestBackend()}
		t.Cleanup(s.Close)
		ops := s.v41DeviceGatherOpsFunc()
		if ops.indexSelectFunc() != nil {
			t.Fatal("a backend without the operation selected a device index list")
		}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		_, err := v41IndexRowsSelected(0, []float32{1}, []float32{2}, []float32{1}, 1, 1, 1, 0, 0, 1, nil, nil, ops.indexSelectFunc())
		var closed *BackendForwardOperationError
		if !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &closed) || closed.Path != "device-only" || !s.BackendSessionClosed() {
			t.Fatalf("device-only missing index list did not fail closed: %v", err)
		}
	})
}

// v41IndexSelectRankCountRef mirrors the device kernel's integer-keyed rank
// count (shaders/v41_index_select.comp) so the parity test holds that
// formulation against the host sort-based selectors. It is test-only.
func v41IndexSelectRankCountRef(scores []float32, topK, topKBlocks, blockSize int) ([]int32, error) {
	rows := len(scores)
	if rows <= 0 || topK <= 0 || topKBlocks < 0 || (topKBlocks > 0 && blockSize <= 0) {
		return nil, fmt.Errorf("index select reference: invalid geometry rows=%d topK=%d blocks=%d/%d", rows, topK, topKBlocks, blockSize)
	}
	key := func(v float32) uint32 {
		bits := math.Float32bits(v)
		if bits == 0x80000000 {
			bits = 0
		}
		if bits&0x80000000 != 0 {
			return ^bits
		}
		return bits | 0x80000000
	}
	const keyNegInf, keyPosInf = uint32(0x007fffff), uint32(0xff800000)
	outranks := func(a uint32, ia int, b uint32, ib int) bool { return a > b || (a == b && ia < ib) }
	for i, v := range scores {
		if v != v {
			return nil, fmt.Errorf("index select reference: score %d is NaN", i)
		}
	}
	keep := func(int) bool { return true }
	if topKBlocks > 0 {
		blocks := (rows-1)/blockSize + 1
		best := make([]uint32, blocks)
		for b := range best {
			best[b] = keyNegInf
			for p := b * blockSize; p < min((b+1)*blockSize, rows); p++ {
				best[b] = max(best[b], key(scores[p]))
			}
		}
		best[(rows-1)/blockSize] = keyPosInf
		kept := make([]bool, blocks)
		for b, mine := range best {
			rank := 0
			for o, other := range best {
				if outranks(other, o, mine, b) {
					rank++
				}
			}
			kept[b] = rank < topKBlocks && mine != keyNegInf
		}
		keep = func(i int) bool { return kept[i/blockSize] }
	}
	masked := make([]uint32, rows)
	for i, v := range scores {
		masked[i] = keyNegInf
		if keep(i) {
			masked[i] = key(v)
		}
	}
	out := make([]int32, 0, min(topK, rows))
	for i, mine := range masked {
		rank := 0
		for j, other := range masked {
			if outranks(other, j, mine, i) {
				rank++
			}
		}
		if rank >= topK {
			continue
		}
		if mine == keyNegInf {
			out = append(out, -1)
		} else {
			out = append(out, int32(i))
		}
	}
	return out, nil
}
