package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The real preparation seam returns an alias, not a float payload. Allocation
// counts cover preparation alone, not dispatch, device staging, or elapsed time.
// fak-test:runtime fast est=20ms lane=default
func TestV41ColdPlainAttentionBorrow(t *testing.T) {
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = false
	defer func() { v41ResidentKVDisabled = prior }()
	q, sink := []float32{0.5, -0.25}, []float32{0.125}
	rows := [][]float32{{9, 8}, {0.25, 0.5}, {-0.5, 0.75}}
	keys, idx := []int32{1, 2}, []int32{0, 1, -1}
	opt := V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 2, N: 2, TopK: 3, Softmax: 0.5}
	view := v41ColdAttentionResidentView(&V41AttentionState{windowSize: 128}, 0, 2, 2, 0)
	calls := 0
	attend := func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
		calls++
		if layer != 0 || request.kv != nil || request.rowKV == nil || request.resident != view ||
			request.sinkKey != view.sinkKey || !reflect.DeepEqual(request.idx, idx) || !reflect.DeepEqual(request.plain, opt) {
			t.Fatal("borrowed dispatch changed request geometry, trailing padding, or owner")
		}
		for i := range keys {
			if &request.rowKV.window[i][0] != &rows[keys[i]][0] {
				t.Fatal("borrowed dispatch copied a float row")
			}
		}
		return append([]float32(nil), q...), nil
	}
	flat, window := v41ColdPlainAttentionInput(q, sink, rows, keys, idx, opt, view.owner, attend)
	if flat != nil || len(window) != 2 || cap(window) != 2 || calls != 0 {
		t.Fatal("preparation packed, escaped its window, or dispatched")
	}
	got, err := v41ColdPlainAttentionSink(0, q, flat, sink, window, idx, opt, view, attend)
	if err != nil || calls != 1 || !reflect.DeepEqual(got, q) {
		t.Fatalf("single dispatch failed: calls=%d err=%v", calls, err)
	}
	var heldFlat []float32
	var heldWindow [][]float32
	borrowAllocs := testing.AllocsPerRun(100, func() {
		heldFlat, heldWindow = v41ColdPlainAttentionInput(q, sink, rows, keys, idx, opt, view.owner, attend)
	})
	if heldFlat != nil || &heldWindow[0][0] != &rows[1][0] {
		t.Fatal("allocation witness lost the borrowed alias")
	}
	flatAllocs := testing.AllocsPerRun(100, func() {
		heldFlat, heldWindow = v41ColdPlainAttentionInput(q, sink, rows, keys, idx, opt, nil, attend)
	})
	if heldWindow != nil || len(heldFlat) != 4 || &heldFlat[0] == &rows[1][0] || borrowAllocs != 0 || flatAllocs < 1 {
		t.Fatalf("preparation allocation/copy witness: borrowed=%g flat=%g bytes=%d", borrowAllocs, flatAllocs, 4*len(heldFlat))
	}
	for _, decline := range []string{"nil-callback", "nil-owner", "disabled"} {
		v, fn := view.owner, v41SharedAttentionFunc(attend)
		switch decline {
		case "nil-callback":
			fn = nil
		case "nil-owner":
			v = nil
		case "disabled":
			v41ResidentKVDisabled = true
		}
		f, w := v41ColdPlainAttentionInput(q, sink, rows, keys, idx, opt, v, fn)
		if w != nil || !reflect.DeepEqual(f, heldFlat) {
			t.Fatalf("%s did not retain original flat input", decline)
		}
	}
	// The original wrapper rechecked disable/callback state after the timer.
	// A changed guard must still dispatch the historical flat/host path.
	_, err = v41ColdPlainAttentionSink(0, q, nil, sink, window, idx, opt, view, func(_ int, request v41SharedAttentionRequest) ([]float32, error) {
		if request.resident != nil || request.sinkKey != "" || request.rowKV != nil || !reflect.DeepEqual(request.kv, heldFlat) {
			t.Fatal("dispatch ignored changed resident-disable guard")
		}
		return append([]float32(nil), q...), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	v41ResidentKVDisabled = false
	got, err = v41ColdPlainAttentionSink(0, q, nil, sink, window, idx, opt, view, nil)
	want, hostErr := V41SparseAttentionSink(q, heldFlat, sink, idx, opt)
	if err != nil || hostErr != nil {
		t.Fatalf("changed callback guard failed: %v / %v", err, hostErr)
	}
	v41SharedAttentionSessionBits(t, "changed callback host fallback", got, want)
	t.Logf("preparation actual allocations borrowed/flat=%g/%g; packed host float bytes=0/%d", borrowAllocs, flatAllocs, 4*len(heldFlat))
}

// Competing malformed inputs must retain the flat validator's primary error,
// selected-operation wrapping and closure, before any backend allocation/upload.
// fak-test:runtime fast est=100ms lane=default
func TestV41ColdPlainAttentionInvalidPrecedence(t *testing.T) {
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = false
	defer func() { v41ResidentKVDisabled = prior }()
	for _, tc := range []struct{ name, first string }{
		{"query-row", "non-finite query"}, {"query-sink", "non-finite query"},
		{"row-sink", "non-finite KV"}, {"row-width-query", "plain KV length"},
		{"row-sink-shape", "sink length"}, {"row-index-shape", "index length"},
		{"unused-row", "non-finite KV"}, {"trailing-padding", "row index -2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var failures []string
			for _, candidate := range []bool{false, true} {
				q, sink := []float32{0.5, -0.25}, []float32{0.125}
				rows := [][]float32{{0.25, 0.5}, {-0.5, 0.75}}
				keys, idx := []int32{0, 1}, []int32{0, 1, -1}
				opt := V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 2, N: 2, TopK: 3, Softmax: 0.5}
				nan := float32(math.NaN())
				switch tc.name {
				case "query-row":
					q[0], rows[1][0] = nan, nan
				case "query-sink":
					q[0], sink[0] = nan, nan
				case "row-sink":
					rows[1][0], sink[0] = nan, nan
				case "row-width-query":
					rows[1], q[0] = rows[1][:1], nan
				case "row-sink-shape":
					rows[1][0], sink = nan, []float32{0, 0}
				case "row-index-shape":
					rows[1][0], idx = nan, idx[:2]
				case "unused-row":
					rows[1][0], idx[1] = nan, -1
				case "trailing-padding":
					idx[2] = -2
				}
				b := newV41ResidentKVTestBackend()
				s := v41DenseTestSession(t, v41IncrementalPlainModel(t, 1), b)
				selected := s.v41SharedAttentionFunc()
				view := v41ColdAttentionResidentView(&V41AttentionState{windowSize: 128}, 0, 1, 2, 0)
				var flat []float32
				var window [][]float32
				if candidate {
					flat, window = v41ColdPlainAttentionInput(q, sink, rows, keys, idx, opt, view.owner, selected)
				} else {
					for _, row := range rows {
						flat = append(flat, row...)
					}
				}
				if window != nil || b.uploads != 0 || len(b.allocations) != 0 || s.BackendSessionClosed() {
					t.Fatal("malformed preparation borrowed or produced effects")
				}
				var err error
				if candidate {
					_, err = v41ColdPlainAttentionSink(0, q, flat, sink, window, idx, opt, view, selected)
				} else {
					_, err = v41SparseAttentionSinkResident(0, q, flat, sink, idx, opt, view, selected)
				}
				var selectedErr *V41SharedAttentionOperationError
				var closed *BackendForwardOperationError
				if !errors.As(err, &selectedErr) || !errors.As(err, &closed) || !errors.Is(err, ErrV41ForwardStage) ||
					!s.BackendSessionClosed() || b.uploads != 0 || len(b.calls) != 0 || len(b.allocations) != 0 ||
					!strings.Contains(err.Error(), tc.first) {
					t.Fatalf("wrong primary failure/effects candidate=%t: %v", candidate, err)
				}
				failures = append(failures, err.Error())
				s.Close()
				if err := s.M.CloseWeights(); err != nil {
					t.Fatal(err)
				}
			}
			if failures[0] != failures[1] {
				t.Fatalf("primary failure changed: %q versus %q", failures[0], failures[1])
			}
		})
	}
	// Admission is not a validation bypass. The adapter must still reject any
	// original input changed after admission, before its first device effect.
	for _, input := range []string{"query", "row", "sink"} {
		t.Run("revalidate-"+input, func(t *testing.T) {
			b := newV41ResidentKVTestBackend()
			s := v41DenseTestSession(t, v41IncrementalPlainModel(t, 1), b)
			selected := s.v41SharedAttentionFunc()
			q, sink := []float32{0.5, -0.25}, []float32{0.125}
			rows := [][]float32{{0.25, 0.5}}
			keys, idx := []int32{0}, []int32{0, -1}
			opt := V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 2, N: 1, TopK: 2, Softmax: 0.5}
			view := v41ColdAttentionResidentView(&V41AttentionState{windowSize: 128}, 0, 0, 1, 0)
			flat, window := v41ColdPlainAttentionInput(q, sink, rows, keys, idx, opt, view.owner, selected)
			if flat != nil || window == nil {
				t.Fatal("valid input declined borrowing")
			}
			fault := q
			if input == "row" {
				fault = rows[0]
			} else if input == "sink" {
				fault = sink
			}
			fault[0] = float32(math.NaN())
			_, err := v41ColdPlainAttentionSink(0, q, flat, sink, window, idx, opt, view, selected)
			var selectedErr *V41SharedAttentionOperationError
			if !errors.As(err, &selectedErr) || !s.BackendSessionClosed() || b.uploads != 0 || len(b.allocations) != 0 {
				t.Fatalf("borrowed input bypassed full validation: %v", err)
			}
			s.Close()
			if err := s.M.CloseWeights(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// Observe the actual existing mirror staging payload. This probe is used only
// with panel(), whose tensor owner remains the embedded CPU-storage backend;
// it is not installed as a selected-attention operation backend.
type v41ColdPackingProbe struct {
	*v41ResidentKVTestBackend
	t       *testing.T
	current []float32
	bytes   int
	borrow  bool
}

func (b *v41ColdPackingProbe) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 resident KV rows" {
		values := b.Backend.Read(x)
		if !reflect.DeepEqual(values, b.current) || (b.borrow && &values[0] == &b.current[0]) {
			b.t.Fatal("mirror did not stage exactly a fresh copy of the missing row")
		}
		b.bytes += 4 * len(values)
	}
	return b.v41ResidentKVTestBackend.UploadClass(x, dt, class, site)
}

// Count copied float bytes from materialized flat inputs and actual missing-row
// staging buffers, including W=129 wrap. Common backend upload copies are not
// called host packing. No elapsed-time or physical bandwidth claim is made.
// fak-test:runtime fast est=100ms lane=default
func TestV41ColdPlainAttentionPackingBytes(t *testing.T) {
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = false
	defer func() { v41ResidentKVDisabled = prior }()
	const seq, width, dim = 130, 129, 32
	rows := make([][]float32, seq)
	for p := range rows {
		rows[p] = make([]float32, dim)
		for d := range rows[p] {
			rows[p][d] = float32(p*dim+d) / 128
		}
	}
	q := make([]float32, dim)
	attend := func(int, v41SharedAttentionRequest) ([]float32, error) { panic("packing probe dispatched") }
	var packed, uploads [2]int
	for arm, borrow := range []bool{false, true} {
		b := &v41ColdPackingProbe{v41ResidentKVTestBackend: newV41ResidentKVTestBackend(), t: t, borrow: borrow}
		s := v41DenseTestSession(t, v41IncrementalPlainModel(t, 1), b)
		mirror := s.newV41ResidentKV()
		owner := &V41AttentionState{windowSize: 128}
		for pos := range seq {
			keys := v41PlainWindowKeys(pos, width)
			idx := v41PlainWindowIndexList(keys)
			view := v41ColdAttentionResidentView(owner, 0, pos, len(keys), 0)
			opt := V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: dim, N: len(keys), TopK: len(idx), Softmax: 0.5}
			admit := owner
			if !borrow {
				admit = nil
			}
			flat, window := v41ColdPlainAttentionInput(q, nil, rows, keys, idx, opt, admit, attend)
			packed[arm] += 4 * len(flat)
			var rowKV *v41SharedAttentionRows
			if borrow {
				if flat != nil || window == nil || &window[0][0] != &rows[keys[0]][0] {
					t.Fatal("preparation did not borrow the exact original window")
				}
				rowKV = &v41SharedAttentionRows{window: window}
			}
			sourceRows := make([]int, len(keys))
			for i := range sourceRows {
				sourceRows[i] = i
			}
			b.current = rows[pos]
			panel, err := mirror.panel(0, view, flat, rowKV, dim, sourceRows)
			if err != nil {
				t.Fatal(err)
			}
			b.Free(panel)
		}
		if borrow {
			packed[arm] += b.bytes // write() packs the missing row only for rowKV.
		}
		uploads[arm] = b.bytes
		s.Close()
		if len(b.allocations) != 0 {
			t.Fatal("packing witness leaked a bank or panel")
		}
		if err := s.M.CloseWeights(); err != nil {
			t.Fatal(err)
		}
	}
	wantFlat := 4 * dim * (width*(width+1)/2 + width)
	wantBorrow := 4 * dim * seq
	if packed != [2]int{wantFlat, wantBorrow} || uploads != [2]int{wantBorrow, wantBorrow} || packed[1] >= packed[0] {
		t.Fatalf("actual packed bytes flat/borrow=%v uploads=%v; want=%d/%d", packed, uploads, wantFlat, wantBorrow)
	}
	t.Logf("actual packed host float bytes flat/borrow=%d/%d saved=%d; unchanged KV upload bytes=%v", packed[0], packed[1], packed[0]-packed[1], uploads)
}

// Residency stays enabled in both arms. The control rematerializes precisely
// the old flat callback payload, keeping owner, IDs and sink residency intact.
// fak-test:runtime medium est=10s lane=default
func TestV41ColdPlainAttentionSessionParity(t *testing.T) {
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = false
	defer func() { v41ResidentKVDisabled = prior }()
	for _, tc := range []struct {
		seq, window int
		rowLog      bool
	}{{4, 3, true}, {130, 129, true}, {4, 3, false}} {
		t.Run(fmt.Sprintf("seq%d_window%d_rowLog%t", tc.seq, tc.window, tc.rowLog), func(t *testing.T) {
			run := func(flatControl bool) (v41ColdAttentionRun, int, int) {
				m := v41IncrementalPlainModel(t, 1)
				m.Cfg.Window = []int{tc.window}
				recorder := newV41SharedAttentionTestBackend()
				var backend compute.Backend = recorder
				var bank *v41ResidentKVTestBackend
				if tc.rowLog {
					bank = newV41ResidentKVTestBackend()
					backend, recorder = bank, bank.v41SharedAttentionTestBackend
				}
				s := v41DenseTestSession(t, m, backend)
				v41SharedAttentionOnly(s)
				selected := s.v41State().sharedAttentionFunc()
				var result v41ColdAttentionRun
				var owner *V41AttentionState
				pos := 0
				s.v41State().setSharedAttention(func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
					if request.resident == nil || (owner != nil && request.resident.owner != owner) {
						t.Fatal("cold/Step owner missing or changed")
					}
					owner = request.resident.owner
					if pos < tc.seq {
						if request.rowKV == nil || request.kv != nil || request.idx[len(request.idx)-1] != -1 ||
							cap(request.rowKV.window) != len(request.rowKV.window) {
							t.Fatal("plain cold entry did not borrow its bounded window with trailing padding")
						}
						if flatControl {
							request.kv, request.rowKV = v41TestAttentionFlatKV(request), nil
						}
					}
					before := v41TestAttentionFlatKV(request)
					out, err := selected(layer, request)
					if !reflect.DeepEqual(before, v41TestAttentionFlatKV(request)) {
						t.Fatal("callback mutated input rows")
					}
					result.outputs = append(result.outputs, append([]float32(nil), out...))
					pos++
					return out, err
				})
				prefix, next := v41IncrementalPrefix(m.Cfg, tc.seq)
				result.logits = append(result.logits, append([]float32(nil), s.Prefill(prefix)...))
				result.states = append(result.states, captureV41ForwardSnapshot(s.v41State()))
				if owner == nil || owner != s.v41State().layerState(0) {
					t.Fatal("prefill did not commit exact seeded owner")
				}
				result.logits = append(result.logits, append([]float32(nil), s.Step(next)...))
				result.states = append(result.states, captureV41ForwardSnapshot(s.v41State()))
				result.calls = append(result.calls, recorder.calls...)
				bytes, gathers := recorder.uploadBytes, 0
				if bank != nil {
					gathers = bank.gathers
				}
				if pos != tc.seq+1 {
					t.Fatalf("actual calls=%d want=%d", pos, tc.seq+1)
				}
				v41ColdAttentionClose(t, s, recorder)
				return result, bytes, gathers
			}
			borrow, borrowBytes, borrowGathers := run(false)
			flat, flatBytes, flatGathers := run(true)
			if !reflect.DeepEqual(borrow.states, flat.states) || !reflect.DeepEqual(borrow.calls, flat.calls) ||
				borrowBytes != flatBytes || borrowGathers != flatGathers {
				t.Fatal("borrowing changed full continuation, attention operands, or transfer operations")
			}
			for i := range borrow.logits {
				v41SharedAttentionSessionBits(t, "cold/first-Step logits", borrow.logits[i], flat.logits[i])
			}
			for i := range borrow.outputs {
				v41SharedAttentionSessionBits(t, "every attention output", borrow.outputs[i], flat.outputs[i])
			}
			t.Logf("exact cold/Step output and continuation parity; H2D bytes borrowed/flat=%d/%d gathers=%d/%d", borrowBytes, flatBytes, borrowGathers, flatGathers)
		})
	}
}
