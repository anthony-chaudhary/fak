package model

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Admission uses only geometry, never ID filtering. This independent small
// integer oracle includes entirely unused compressed banks.
// fak-test:runtime fast est=10ms lane=default
func TestV41ColdAttentionResidentAdmission(t *testing.T) {
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = false
	defer func() { v41ResidentKVDisabled = prior }()
	owner := &V41AttentionState{windowSize: 128}
	attend := func(int, v41SharedAttentionRequest) ([]float32, error) { panic("admission dispatched") }
	for seq := 1; seq <= 16; seq++ {
		for _, window := range []int{-1, 1, 2, 3, 128, 130} {
			for groups := 0; groups <= 40; groups++ {
				rows := 0
				for pos := 0; pos < seq; pos++ {
					width := pos + 1
					if window > 0 && width > window {
						width = window
					}
					rows += width
				}
				got := v41ColdAttentionResidentOwner(owner, seq, window, groups, attend)
				want := seq > 1 && seq+groups <= rows
				if (got != nil) != want || (got != nil && got != owner) {
					t.Fatalf("seq=%d window=%d groups=%d: admission=%t want=%t", seq, window, groups, got != nil, want)
				}
			}
		}
	}
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		seq, window, groups int
		want                bool
	}{
		{129, 129, 0, true}, {130, 130, 0, false}, {130, -1, 0, false},
		{maxInt, 128, maxInt, true}, {maxInt, 2, maxInt, false},
		{maxInt, 2, maxInt - 1, true}, {maxInt, 1, 1, false},
		{4, 0, 0, false}, {4, -2, 0, false}, {4, 3, -1, false},
	} {
		if got := v41ColdAttentionResidentOwner(owner, tc.seq, tc.window, tc.groups, attend); (got != nil) != tc.want {
			t.Fatalf("boundary %+v: admitted=%t", tc, got != nil)
		}
	}
	// Overflow of the halved triangular product means savings exceed any int G.
	large := &V41AttentionState{windowSize: maxInt}
	if v41ColdAttentionResidentOwner(large, maxInt, maxInt, maxInt, attend) != large {
		t.Fatal("overflow-safe triangular admission declined")
	}
	if v41ColdAttentionResidentOwner(nil, 4, 3, 0, attend) != nil ||
		v41ColdAttentionResidentOwner(owner, 4, 3, 0, nil) != nil {
		t.Fatal("absent owner or callback admitted")
	}
	v41ResidentKVDisabled = true
	if v41ColdAttentionResidentOwner(owner, 4, 3, 0, attend) != nil {
		t.Fatal("disabled residency admitted")
	}
}

type v41ColdAttentionObserved struct {
	layer, pos, groups, selectedRows int
	bytes, kvRows, gathers           int
	resident                         bool
}

type v41ColdAttentionRun struct {
	logits, outputs [][]float32
	states          []*v41ForwardSnapshot
	calls           []v41SharedAttentionTestCall
	observed        []v41ColdAttentionObserved
}

// Named sinks belong to the model's immutable pool. Session.Close must retire
// every bank and transient while preserving exactly those borrowed handles;
// Model.CloseWeights must then release the sinks as well. Neither an arbitrary
// survivor count nor a borrowed flag alone establishes the correct lifetime.
func v41ColdAttentionClose(t *testing.T, s *Session, recorder *v41SharedAttentionTestBackend) {
	t.Helper()
	sinks := map[compute.Buffer]bool{}
	var keys []string
	for layer := 0; layer < s.M.Cfg.NumLayers; layer++ {
		key := "v41-attn-sink:" + layerName(layer, "attn.sink")
		if tensor, ok := s.halW[key]; ok {
			if _, borrowed := s.borrowedHALW[key]; !borrowed || !recorder.allocations[tensor.Buf()] {
				t.Fatalf("named sink %s lacks a live borrowed handle", key)
			}
			sinks[tensor.Buf()] = true
			keys = append(keys, key)
		}
	}
	modelOwned := map[compute.Buffer]bool{}
	if !s.M.VisitImmutableDeviceWeights(s.Backend, func(tensor compute.Tensor) bool {
		if recorder.allocations[tensor.Buf()] {
			modelOwned[tensor.Buf()] = true
		}
		return true
	}) || !reflect.DeepEqual(modelOwned, sinks) {
		t.Fatal("tracked model-owned buffers differ from the exact borrowed sink identities")
	}
	sessionOwned := len(recorder.allocations) - len(sinks)
	s.Close()
	if !reflect.DeepEqual(recorder.allocations, sinks) {
		t.Fatalf("Session.Close survivors differ from exact model-owned sinks: actual=%d want=%d", len(recorder.allocations), len(sinks))
	}
	if err := s.M.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if len(recorder.allocations) != 0 {
		t.Fatal("Model.CloseWeights retained tracked attention allocations")
	}
	sort.Strings(keys)
	t.Logf("attention lifetime: session-owned buffers released=%d; exact borrowed sinks retained=%v; after model close allocations=0", sessionOwned, keys)
}

// Observe real adapter uploads using the existing CPU-storage recorder. Query
// and sink bytes are removed only after checking the exact total byte count.
// This is a source/transfer contract, not a physical-device performance result.
func v41ColdAttentionSessionRun(t *testing.T, m *Model, seq int, step, rowLog, disabled, admitted bool) v41ColdAttentionRun {
	t.Helper()
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = disabled
	defer func() { v41ResidentKVDisabled = prior }()
	var backend compute.Backend
	var rows *v41ResidentKVTestBackend
	var recorder *v41SharedAttentionTestBackend
	if rowLog {
		rows = newV41ResidentKVTestBackend()
		backend, recorder = rows, rows.v41SharedAttentionTestBackend
	} else {
		recorder = newV41SharedAttentionTestBackend()
		backend = recorder
	}
	s := v41DenseTestSession(t, m, backend)
	v41SharedAttentionOnly(s)
	selected := s.v41State().sharedAttention
	if selected == nil {
		t.Fatal("selected attention missing")
	}
	var run v41ColdAttentionRun
	positions := make([]int, m.Cfg.NumLayers)
	owners := make([]*V41AttentionState, m.Cfg.NumLayers)
	previousGroups := make([]int, m.Cfg.NumLayers)
	gatherCount := func() int {
		if rows == nil {
			return 0
		}
		return rows.gathers
	}
	s.v41State().sharedAttention = func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
		pos := positions[layer]
		positions[layer]++
		width := pos + 1
		if w := m.Cfg.windowForLayer(layer); w > 0 && width > w {
			width = w
		}
		groups := request.plain.N - width
		selectedRows := 0
		for _, id := range request.idx {
			if id >= 0 {
				selectedRows++
			}
		}
		wantView := !disabled && (pos >= seq || admitted)
		if v := request.resident; v != nil {
			if v.pos != pos || v.firstKey != pos-width+1 || v.windowRows != width || v.compressedRows != groups ||
				v.sinkKey != layerName(layer, "attn.sink") || v.ringRows != v.owner.windowSize {
				t.Fatalf("layer=%d pos=%d wrong resident descriptor: %+v", layer, pos, v)
			}
			if owners[layer] == nil {
				owners[layer] = v.owner
			} else if owners[layer] != v.owner {
				t.Fatal("cold and Step descriptors changed owner identity")
			}
		}
		flat, ids := v41TestAttentionFlatKV(request), append([]int32(nil), request.idx...)
		before, gathers := recorder.uploadBytes, gatherCount()
		out, err := selected(layer, request)
		if err != nil {
			return nil, err
		}
		if !reflect.DeepEqual(flat, v41TestAttentionFlatKV(request)) || !reflect.DeepEqual(ids, request.idx) {
			t.Fatal("selected callback mutated original KV or IDs")
		}
		resident := rowLog && wantView
		kvRows, sinkValues, wantGathers := selectedRows, len(request.sink), 0
		if resident {
			wantGathers = 1
			kvRows = 1
			if pos == 0 {
				kvRows += groups
			} else {
				kvRows += groups - previousGroups[layer]
				sinkValues = 0
			}
		}
		wantBytes := 4 * (len(request.q) + kvRows*m.Cfg.HeadDim + sinkValues)
		delta := recorder.uploadBytes - before
		if delta != wantBytes || gatherCount()-gathers != wantGathers {
			t.Fatalf("layer=%d pos=%d actual bytes/gathers=%d/%d want=%d/%d", layer, pos, delta, gatherCount()-gathers, wantBytes, wantGathers)
		}
		if (request.resident != nil) != wantView {
			t.Fatalf("layer=%d pos=%d resident view=%t want=%t", layer, pos, request.resident != nil, wantView)
		}
		actualRows := (delta - 4*(len(request.q)+sinkValues)) / (4 * m.Cfg.HeadDim)
		run.observed = append(run.observed, v41ColdAttentionObserved{layer: layer, pos: pos, groups: groups, selectedRows: selectedRows,
			bytes: delta, kvRows: actualRows, gathers: gatherCount() - gathers, resident: resident})
		run.outputs = append(run.outputs, append([]float32(nil), out...))
		previousGroups[layer] = groups
		return out, nil
	}
	prefix, next := v41IncrementalPrefix(m.Cfg, seq)
	run.logits = append(run.logits, append([]float32(nil), s.Prefill(prefix)...))
	run.states = append(run.states, captureV41ForwardSnapshot(s.v41State()))
	if admitted && !disabled {
		for layer, owner := range owners {
			if owner == nil || owner != s.v41State().layerState(layer) {
				t.Fatal("cold mirror did not name the committed seeded owner")
			}
		}
	}
	if step {
		run.logits = append(run.logits, append([]float32(nil), s.Step(next)...))
		run.states = append(run.states, captureV41ForwardSnapshot(s.v41State()))
	}
	run.calls = append(run.calls, recorder.calls...)
	v41ColdAttentionClose(t, s, recorder)
	return run
}

// Actual cold Prefill and first Step preserve every attention operand/output,
// full-vocabulary logits, and the complete captured continuation state. Wrap
// crosses the device ring modulus; oversized cold windows remain per-call.
// Estimates are unmeasured; execution requires the separately reviewed grant.
// fak-test:runtime medium est=30s lane=default
func TestV41ColdAttentionResidentSession(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("software contract witness requires cpu-ref")
	}
	for _, tc := range []struct {
		name        string
		kind        string
		seq, window int
		step        bool
		rowLog      bool
		admitted    bool
	}{
		{"plain", "plain", 4, 3, true, true, true},
		{"compressed-source-reader", "ratio2", 5, 3, true, true, true},
		{"ratio-one-source-reader", "ratio1", 4, 3, true, true, true},
		{"ring-wrap", "plain", 130, 3, true, true, true},
		{"maximum-window-wrap", "plain", 130, 129, true, true, true},
		{"oversized-window", "plain", 130, 130, false, true, false},
		{"oversized-global", "plain", 130, -1, false, true, false},
		{"missing-row-log", "plain", 4, 3, true, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			build := func() *Model {
				var m *Model
				switch tc.kind {
				case "ratio1":
					m = v41RatioOneForwardFixture(t, false)
				case "ratio2":
					m = v41ReaderWidthFixture(t)
				default:
					m = v41IncrementalPlainModel(t, 2)
				}
				m.Cfg.Window = []int{tc.window, tc.window}
				return m
			}
			on := v41ColdAttentionSessionRun(t, build(), tc.seq, tc.step, tc.rowLog, false, tc.admitted)
			off := v41ColdAttentionSessionRun(t, build(), tc.seq, tc.step, tc.rowLog, true, tc.admitted)
			if !reflect.DeepEqual(on.states, off.states) || !reflect.DeepEqual(on.calls, off.calls) {
				t.Fatal("resident path changed attention operands or complete continuation state")
			}
			if len(on.logits) != len(off.logits) || len(on.outputs) != len(off.outputs) {
				t.Fatal("resident path changed output count")
			}
			for i := range on.logits {
				v41SharedAttentionSessionBits(t, "cold/first-Step logits", on.logits[i], off.logits[i])
			}
			for i := range on.outputs {
				v41SharedAttentionSessionBits(t, "every attention output", on.outputs[i], off.outputs[i])
			}
			var onRows, offRows, onBytes, offBytes, gathers int
			for _, r := range on.observed {
				if r.pos < tc.seq {
					onRows += r.kvRows
					onBytes += r.bytes
					gathers += r.gathers
				}
			}
			for _, r := range off.observed {
				if r.pos < tc.seq {
					offRows += r.kvRows
					offBytes += r.bytes
				}
			}
			if onRows > offRows || onBytes > offBytes {
				t.Fatal("cold admission increased measured KV rows or attention upload bytes")
			}
			if tc.admitted && tc.rowLog && (onBytes == offBytes || gathers != 2*tc.seq) {
				t.Fatal("cold witness did not exercise residency and sink reuse")
			}
			t.Logf("cold fixture=%s actual KV rows=%d/%d delta=%d attention H2D bytes=%d/%d gathers=%d (CPU storage only)", tc.name, onRows, offRows, onRows-offRows, onBytes, offBytes, gathers)
		})
	}
}

// Pinned projected inputs expose duplicate selections, future masks, unused
// bank rows, and an unfavorable tiny bank that must decline before allocation.
// fak-test:runtime fast est=100ms lane=default
func TestV41ColdAttentionResidentProjectedRows(t *testing.T) {
	prior := v41ResidentKVDisabled
	defer func() { v41ResidentKVDisabled = prior }()
	for _, tc := range []struct {
		name                    string
		seq, window, ratio      int
		ids                     []int32
		wantOnRows, wantOffRows int
		admitted                bool
	}{
		{"ratio1-duplicates", 4, 3, 1, []int32{0, 0, -1, 3}, 8, 18, true},
		{"ratio2-duplicates-future", 4, 3, 2, []int32{0, 0, -1, 3}, 8, 15, true},
		{"entire-bank-unused", 4, 3, 1, []int32{-1}, 8, 9, true},
		{"tiny-unused-bank-declines", 2, 1, 1, []int32{-1}, 2, 2, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			type result struct {
				out                  [][]float32
				calls                []v41SharedAttentionTestCall
				rows, bytes, gathers int
			}
			run := func(disabled bool) result {
				v41ResidentKVDisabled = disabled
				b := newV41ResidentKVTestBackend()
				s := v41DenseTestSession(t, v41IncrementalPlainModel(t, 1), b)
				selected := s.v41SharedAttentionFunc()
				owner, err := NewV41AttentionState(2, 8)
				if err != nil {
					t.Fatal(err)
				}
				windows := [][]float32{{0.25, -0.5}, {0.75, 0.125}, {-0.5, 0.25}, {0.125, -0.75}}
				bank := [][]float32{{-0.25, 0.5}, {3, -2}, {-1, 4}, {0.125, 0.25}}
				if err := owner.seedTemporal(windows[:tc.seq], 0, tc.window); err != nil {
					t.Fatal(err)
				}
				admitted := v41ColdAttentionResidentOwner(owner, tc.seq, tc.window, len(bank), selected)
				if (admitted != nil) != (tc.admitted && !disabled) {
					t.Fatal("unexpected projected admission")
				}
				var got result
				for pos := 0; pos < tc.seq; pos++ {
					width := min(pos+1, tc.window)
					window := windows[pos-width+1 : pos+1]
					before := b.uploadBytes
					out, err := v41CombinedAttentionResident(0, pos, tc.ratio, 1, 2, []float32{0.5, -0.25}, []float32{0.125}, window, bank, tc.ids, 0.5,
						v41ColdAttentionResidentView(admitted, 0, pos, width, len(bank)), selected)
					if err != nil {
						t.Fatal(err)
					}
					var wantKV []float32
					for _, row := range window {
						wantKV = append(wantKV, row...)
					}
					for _, id := range tc.ids {
						if id >= 0 && (int(id)+1)*tc.ratio <= pos+1 {
							wantKV = append(wantKV, bank[id]...)
						}
					}
					if !reflect.DeepEqual(b.calls[pos].kv, wantKV) {
						t.Fatal("causal masking, duplicate order, or selected row values changed")
					}
					sinkValues := 1
					if admitted != nil && pos > 0 {
						sinkValues = 0
					}
					delta := b.uploadBytes - before
					kvBytes := delta - 4*(2+sinkValues)
					if kvBytes < 0 || kvBytes%8 != 0 {
						t.Fatal("invalid observed KV byte accounting")
					}
					got.rows += kvBytes / 8
					got.out = append(got.out, out)
				}
				got.calls, got.bytes, got.gathers = b.calls, b.uploadBytes, b.gathers
				v41ColdAttentionClose(t, s, b.v41SharedAttentionTestBackend)
				return got
			}
			on, off := run(false), run(true)
			if !reflect.DeepEqual(on.calls, off.calls) || !reflect.DeepEqual(on.out, off.out) ||
				on.rows != tc.wantOnRows || off.rows != tc.wantOffRows || on.bytes > off.bytes {
				t.Fatalf("projected parity/accounting: actual rows=%d/%d want=%d/%d bytes=%d/%d", on.rows, off.rows, tc.wantOnRows, tc.wantOffRows, on.bytes, off.bytes)
			}
			wantGathers := 0
			if tc.admitted {
				wantGathers = tc.seq
			}
			if on.gathers != wantGathers || off.gathers != 0 {
				t.Fatalf("gathers=%d/%d want=%d/0", on.gathers, off.gathers, wantGathers)
			}
			t.Logf("projected fixture=%s actual KV rows=%d/%d delta=%d attention H2D bytes=%d/%d gathers=%d (CPU storage only)", tc.name, on.rows, off.rows, on.rows-off.rows, on.bytes, off.bytes, on.gathers)
		})
	}
	// Malformed future/unselected rows and invalid IDs cannot be hidden by the
	// residency admission or masks; reject before the selected callback runs.
	v41ResidentKVDisabled = false
	for _, invalid := range []string{"future-row", "unused-row", "padding-id"} {
		t.Run(invalid, func(t *testing.T) {
			called := false
			attend := func(int, v41SharedAttentionRequest) ([]float32, error) {
				called = true
				return nil, fmt.Errorf("unexpected dispatch")
			}
			bank := [][]float32{{0, 1}, {2, 3}, {4, 5}, {6, 7}}
			ids := []int32{0, -1}
			switch invalid {
			case "future-row":
				bank[3][1] = float32(math.NaN())
			case "unused-row":
				bank[1][0] = float32(math.Inf(1))
			case "padding-id":
				ids[1] = -2
			}
			owner := &V41AttentionState{windowSize: 128}
			admitted := v41ColdAttentionResidentOwner(owner, 4, 3, 4, attend)
			_, err := v41CombinedAttentionResident(0, 0, 2, 1, 2, []float32{1, 0}, []float32{0}, [][]float32{{0, 1}}, bank, ids, 1,
				v41ColdAttentionResidentView(admitted, 0, 0, 1, len(bank)), attend)
			if err == nil || called {
				t.Fatalf("invalid input reached selected callback: called=%t err=%v", called, err)
			}
		})
	}
}
