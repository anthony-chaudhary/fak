package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Isolate shared attention from the independently witnessed device seams.
func v41SharedAttentionOnly(s *Session) {
	st := s.v41State()
	st.expertGateUp, st.expertDown = nil, nil
	st.denseProjection, st.groupedOutput, st.engramProjection = nil, nil, nil
	st.mhcProjection, st.finalNorm, st.queryNorm, st.kvNorm = nil, nil, nil, nil
	st.ffnNorm, st.sharedActivation, st.tailRoPE = nil, nil, nil
}

func v41SharedAttentionSessionNear(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) == 0 || len(got) != len(want) {
		t.Fatalf("logit widths=%d/%d", len(got), len(want))
	}
	var maxAbs, maxRel float64
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("non-finite logit comparison at %d: %g/%g", i, g, w)
		}
		delta := math.Abs(g - w)
		maxAbs = math.Max(maxAbs, delta)
		maxRel = math.Max(maxRel, delta/math.Max(math.Abs(w), 1e-30))
		// Frozen software continuation tolerance, not a hardware qualification.
		if delta > 1e-4+1e-4*math.Abs(w) {
			t.Fatalf("logit %d=%g want=%g abs=%g", i, g, w, delta)
		}
	}
	t.Logf("continuation maximum absolute=%g relative=%g", maxAbs, maxRel)
}

func v41SharedAttentionSessionBits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s widths=%d/%d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] bits=%08x want=%08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

// One bounded software witness follows the actual public entry points. The
// recorder performs independently rounded F32 math over CPU-owned storage; its
// DeviceMemory capability tests routing and never proves physical dispatch.
// Runtime estimate is unmeasured until the separately authorized witness runs.
// No parallel execution: existing session route probes are package-global.
// fak-test:runtime medium est=10s lane=default
func TestV41SharedAttentionSession(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("shared-attention software witness requires cpu-ref")
	}
	newSession := func(t *testing.T, m *Model) (*Session, *v41SharedAttentionTestBackend) {
		t.Helper()
		b := newV41SharedAttentionTestBackend()
		s := v41DenseTestSession(t, m, b)
		v41SharedAttentionOnly(s)
		if s.v41State().sharedAttention == nil {
			t.Fatal("available shared attention was not bound")
		}
		return s, b
	}

	for _, role := range []bool{false, true} {
		name := "plain-window-boundary"
		if role {
			name = "source-reader-group-boundary"
		}
		t.Run(name, func(t *testing.T) {
			var m *Model
			if role {
				m = v41ReaderWidthFixture(t) // source and reader both ratio 2
			} else {
				m = v41IncrementalPlainModel(t, 2)
				m.Cfg.Window = []int{2, 2}
			}
			s, b := newSession(t, m)
			host := m.NewSession()
			t.Cleanup(host.Close)
			selected := s.v41State().sharedAttention
			requests := 0
			s.v41State().sharedAttention = func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
				// Cold Prefill is layer-major; Step and suffix are position-major.
				wantLayer, pos := requests/2, requests%2
				if requests >= 4 {
					wantLayer, pos = requests%2, requests/2
				}
				requests++
				if layer != wantLayer {
					t.Fatalf("request layer=%d want=%d", layer, wantLayer)
				}
				q, kv, sink := append([]float32(nil), request.q...), append([]float32(nil), request.kv...), append([]float32(nil), request.sink...)
				idx := append([]int32(nil), request.idx...)
				values := make([][]float32, len(request.values))
				for i := range values {
					values[i] = append([]float32(nil), request.values[i]...)
				}
				rows := min(pos+1, 2)
				if role {
					window := pos + 1
					if configured := m.Cfg.windowForLayer(layer); configured > 0 {
						window = min(window, configured)
					}
					groups := max(2, pos+1) / 2 // cold payload also carries the future first group
					rows = window + (pos+1)/2
					o := request.plain
					if request.mode != compute.V41SharedAttentionPlain || o.B != 1 || o.M != 1 || o.N != window+groups || o.TopK != window+groups || o.HeadDim != m.Cfg.HeadDim || o.Heads != m.Cfg.NumHeads || o.Softmax != m.Cfg.attnScale() || o.Inverse != nil || o.RopeDim != 0 {
						t.Fatalf("combined source/window options changed: layer=%d pos=%d opt=%+v", layer, pos, o)
					}
					wantIdx := make([]int32, window+groups)
					for i := range wantIdx {
						wantIdx[i] = int32(i)
						if i >= window && (i-window+1)*2 > pos+1 {
							wantIdx[i] = -1
						}
					}
					if !reflect.DeepEqual(request.idx, wantIdx) {
						t.Fatalf("combined positional list=%v want=%v", request.idx, wantIdx)
					}
					for _, value := range request.kv[window*m.Cfg.HeadDim:] {
						if math.Float32bits(value)&0xffff != 0 {
							t.Fatal("published compressed row lost its BF16 boundary")
						}
					}
				} else {
					o := request.plain
					if request.mode != compute.V41SharedAttentionPlain || o.B != 1 || o.M != 1 || o.Heads != m.Cfg.NumHeads || o.HeadDim != m.Cfg.HeadDim || o.N != rows || o.TopK != rows+1 || o.Softmax != m.Cfg.attnScale() || o.TopKLength != nil || o.Inverse != nil || o.RopeDim != 0 {
						t.Fatalf("plain configured-window options changed at pos=%d: %+v", pos, o)
					}
					wantIdx := make([]int32, rows+1)
					for i := 0; i < rows; i++ {
						wantIdx[i] = int32(i)
					}
					wantIdx[rows] = -1
					if !reflect.DeepEqual(request.idx, wantIdx) {
						t.Fatalf("plain positional list=%v want=%v", request.idx, wantIdx)
					}
				}
				before := len(b.calls)
				out, err := selected(layer, request)
				if err != nil {
					return nil, err
				}
				v41SharedAttentionSessionBits(t, "Q input", request.q, q)
				v41SharedAttentionSessionBits(t, "KV input", request.kv, kv)
				v41SharedAttentionSessionBits(t, "sink input", request.sink, sink)
				if !reflect.DeepEqual(request.idx, idx) {
					t.Fatal("selected contraction mutated caller indices")
				}
				for i := range values {
					v41SharedAttentionSessionBits(t, "source input", request.values[i], values[i])
				}
				if len(b.calls) != before+1 {
					t.Fatal("nonempty attention skipped or repeated device selection")
				}
				call := b.calls[before]
				if call.rows != rows || call.heads != m.Cfg.NumHeads || call.dim != m.Cfg.HeadDim || call.scale != m.Cfg.attnScale() || call.mode != request.mode || !call.hasSink {
					t.Fatalf("gathered call geometry differs at pos=%d: %+v", pos, call)
				}
				wantKV := kv
				if role {
					wantKV = nil
					for _, id := range request.idx {
						if id >= 0 {
							wantKV = append(wantKV, kv[int(id)*m.Cfg.HeadDim:(int(id)+1)*m.Cfg.HeadDim]...)
						}
					}
				}
				v41SharedAttentionSessionBits(t, "ordered gathered KV", call.kv, wantKV)
				return out, nil
			}

			got := s.Prefill([]int{1, 2})
			if calls := m.V41ExpertFaultAttribution().Prefill.AttentionContractionCalls; calls != 4 {
				t.Fatalf("cold prefill attempted contractions=%d want=4, including windows before completed groups", calls)
			}
			v41SharedAttentionSessionNear(t, got, host.Prefill([]int{1, 2}))
			if !s.v41IncrementalEligible() {
				t.Fatal("cold prefill did not seed incremental state")
			}
			readStep, restoreStep := v41SessionIncrementalCalls(t)
			defer restoreStep()
			beforeAttempts := m.V41ExpertFaultAttribution().Decode.AttentionContractionCalls
			got = s.Step(3)
			if readStep() != 1 {
				t.Fatal("Step did not take the single-token route")
			}
			if delta := m.V41ExpertFaultAttribution().Decode.AttentionContractionCalls - beforeAttempts; delta != 2 {
				t.Fatalf("Step attempted contractions=%d want=2", delta)
			}
			v41SharedAttentionSessionNear(t, got, host.Step(3))
			readSuffix, restoreSuffix := v41SuffixPrefillCalls(t)
			defer restoreSuffix()
			beforeAttempts = m.V41ExpertFaultAttribution().Prefill.AttentionContractionCalls
			got = s.Prefill([]int{4, 5})
			if readSuffix() != 2 {
				t.Fatal("continued Prefill replayed the prefix")
			}
			if delta := m.V41ExpertFaultAttribution().Prefill.AttentionContractionCalls - beforeAttempts; delta != 4 {
				t.Fatalf("suffix attempted contractions=%d want=4", delta)
			}
			v41SharedAttentionSessionNear(t, got, host.Prefill([]int{4, 5}))
			cold := m.NewSession()
			t.Cleanup(cold.Close)
			v41SharedAttentionSessionNear(t, got, cold.Prefill([]int{1, 2, 3, 4, 5}))
			if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(host.v41Forward)) {
				t.Fatal("selected continuation changed ring/compressor/publication state")
			}
			wantCalls := 10
			if role {
				for pos, pair := range [][2]int{{0, 2}, {1, 3}, {4, 5}, {6, 7}, {8, 9}} {
					owner, reader := b.calls[pair[0]], b.calls[pair[1]]
					compressed := (pos + 1) / 2 * m.Cfg.HeadDim
					v41SharedAttentionSessionBits(t, "reader uses source rows", reader.kv[len(reader.kv)-compressed:], owner.kv[len(owner.kv)-compressed:])
				}
			}
			if requests != 10 || len(b.calls) != wantCalls || b.uploads != 3*wantCalls || b.reads != wantCalls || len(b.allocations) != 0 {
				t.Fatalf("request/dispatch/transfer counts=%d/%d/%d/%d live=%d", requests, len(b.calls), b.uploads, b.reads, len(b.allocations))
			}
		})
	}

	t.Run("incomplete-first-group-step", func(t *testing.T) {
		m := v41IncrementalExpertFixture(t, true, false)
		s, b := newSession(t, m)
		state, err := NewV41AttentionState(m.Cfg.HeadDim, 8)
		if err != nil {
			t.Fatal(err)
		}
		// Empty retained state still contracts its own first window row before
		// any compressed group exists.
		s.v41State().setLayerState(0, 1, state)
		scratch := &v41ProjScratch{sharedAttention: s.v41State().sharedAttention}
		got, stats, err := m.forwardV41Step(1, s.v41State(), scratch)
		if err != nil || !stats.Committed || stats.LayerCalls != 1 || len(b.calls) != 1 || b.calls[0].rows != 1 || b.uploads != 3 || scratch.sharedAttention != nil {
			t.Fatalf("incomplete group lost its window or scratch semantics: stats=%+v err=%v", stats, err)
		}
		v41SharedAttentionSessionNear(t, got, lastLogits(m.Forward([]int{1})))
		v41SharedAttentionSessionNear(t, s.Step(2), lastLogits(m.Forward([]int{1, 2})))
		if len(b.calls) != 2 || b.calls[1].rows != 3 || len(state.partialInputs) != 0 {
			t.Fatal("first completed group did not join the two window rows exactly once")
		}
	})

	t.Run("clone-restore-rebind", func(t *testing.T) {
		m := v41ReaderWidthFixture(t)
		source, b := newSession(t, m)
		source.Prefill([]int{1, 2, 3}) // snapshot includes an incomplete source group
		want := captureV41ForwardSnapshot(source.v41Forward)
		unbound := want.clone().restore()
		if unbound.sharedAttention != nil || unbound.callbackOwner != nil {
			t.Fatal("continuation snapshot captured the source callback")
		}
		snap, err := source.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer snap.Close()
		clone, err := snap.Clone()
		if err != nil {
			t.Fatal(err)
		}
		defer clone.Close()
		target, branch := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
		for i, item := range []struct {
			snapshot *PrefixSnapshot
			session  *Session
		}{{snap, target}, {clone, branch}} {
			if err := item.snapshot.Restore(item.session); err != nil {
				t.Fatal(err)
			}
			st := item.session.v41Forward
			if st == nil || st.callbackOwner != item.session || st.sharedAttention == nil || !reflect.DeepEqual(captureV41ForwardSnapshot(st), want) {
				t.Fatalf("restore %d failed eager owner binding or changed continuation", i)
			}
			v41SharedAttentionOnly(item.session)
		}
		source.Close()
		before := len(b.calls)
		got := target.Step(4)
		if !reflect.DeepEqual(captureV41ForwardSnapshot(branch.v41Forward), want) {
			t.Fatal("one restored branch mutated the other snapshot's rows")
		}
		v41SharedAttentionSessionNear(t, got, branch.Prefill([]int{4}))
		cold := m.NewSession()
		t.Cleanup(cold.Close)
		v41SharedAttentionSessionNear(t, got, cold.Prefill([]int{1, 2, 3, 4}))
		if len(b.calls) != before+4 || len(b.allocations) != 0 {
			t.Fatal("restored callbacks used closed source owner, replayed, or leaked")
		}
		before = len(b.calls)
		target.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		var refused *BackendForwardOperationError
		if err := recoverError(func() { target.Step(5) }); !errors.As(err, &refused) || refused.Stage != "decode: architecture uses host model compute" || len(b.calls) != before {
			t.Fatalf("whole-architecture DeviceOnly admission changed: %v", err)
		}
	})

	t.Run("decline-before-selection", func(t *testing.T) {
		for _, missing := range []bool{false, true} {
			m := v41IncrementalPlainModel(t, 1)
			b := newV41SharedAttentionTestBackend()
			b.supported = false
			var backend compute.Backend = b
			if missing {
				backend = b.v41DenseTestBackend
			}
			s := v41DenseTestSession(t, m, backend)
			v41SharedAttentionOnly(s)
			if s.v41State().sharedAttention != nil {
				t.Fatal("declined capability bound a selected callback")
			}
			host := m.NewSession()
			t.Cleanup(host.Close)
			v41SharedAttentionSessionBits(t, "declined prefill", s.Prefill([]int{1, 2}), host.Prefill([]int{1, 2}))
			v41SharedAttentionSessionBits(t, "declined Step", s.Step(3), host.Step(3))
			if len(b.calls) != 0 || b.uploads != 0 || b.reads != 0 {
				t.Fatal("declined attention dispatched or transferred")
			}
		}
	})

	for _, tc := range []struct {
		fault, entry, stage string
		role                bool
	}{
		{"dispatch", "cold", "shared attention", false},
		{"dispatch", "step", "shared attention", false},
		{"dispatch", "suffix", "shared attention", true},
		{"KV upload", "step", "KV upload", false},
		{"read", "suffix", "readback", true},
		{"shape", "step", "output validation", false},
		{"nonfinite", "suffix", "readback", true},
	} {
		t.Run(tc.fault+"-"+tc.entry, func(t *testing.T) {
			var m *Model
			if tc.role {
				m = v41ReaderWidthFixture(t)
			} else {
				m = v41IncrementalPlainModel(t, 2)
			}
			s, b := newSession(t, m)
			if tc.entry != "cold" {
				s.Prefill([]int{1, 2, 3})
			}
			before := captureV41ForwardSnapshot(s.v41Forward)
			layers := append([]*V41AttentionState(nil), s.v41Forward.layers...)
			b.fault, b.failCall = tc.fault, len(b.calls)+m.Cfg.NumLayers
			// A selected failure never grants permission to replay on the host,
			// even when its nested cause carries the structural refusal sentinel.
			b.cause = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: fmt.Errorf("selected cause: %w", ErrV41ForwardStage)}
			invoke := func() {
				if tc.entry == "step" {
					s.Step(4)
				} else {
					s.Prefill([]int{4})
				}
			}
			err := recoverError(invoke)
			var selected *V41SharedAttentionOperationError
			var closed *BackendForwardOperationError
			if !errors.As(err, &selected) || selected.Layer != m.Cfg.NumLayers-1 || !errors.As(err, &closed) || closed.Path != "v41-shared-attention" || closed.Stage != tc.stage {
				t.Fatalf("failure lost selected operation/layer/stage: %v", err)
			}
			if tc.fault == "dispatch" || tc.fault == "KV upload" || tc.fault == "read" {
				if !errors.Is(err, b.cause) || !errors.Is(err, ErrV41ForwardStage) {
					t.Fatalf("selected failure lost nested cause: %v", err)
				}
			}
			wantCalls := b.failCall
			if tc.fault == "KV upload" {
				wantCalls--
			}
			if !s.BackendSessionClosed() || len(b.calls) != wantCalls || len(b.allocations) != 0 || b.output != nil || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
				t.Fatal("selected failure replayed, leaked, or advanced committed state")
			}
			for i, state := range layers {
				if s.v41Forward.layerState(i) != state {
					t.Fatal("rollback replaced retained layer ownership")
				}
			}
			if retry := recoverError(invoke); retry != s.halFailure || len(b.calls) != wantCalls {
				t.Fatalf("closed session retried selected attention: %v", retry)
			}
		})
	}

	t.Run("later-suffix-failure-keeps-successful-token", func(t *testing.T) {
		m := v41ReaderWidthFixture(t)
		s, b := newSession(t, m)
		s.Prefill([]int{1, 2})
		// Advance a detached host control through the first suffix token. The
		// second token completes a source group, then fails at the reader; only
		// that token's ring/group/publication changes must roll back.
		expected := captureV41ForwardSnapshot(s.v41Forward).restore()
		scratch := &v41ProjScratch{sharedAttention: s.v41State().sharedAttention}
		before := len(b.calls)
		if _, _, err := m.forwardV41Step(3, expected, scratch); err != nil {
			t.Fatal(err)
		}
		if scratch.sharedAttention != nil || len(b.calls) != before {
			t.Fatal("detached host state reused or retained the prior owner's callback")
		}
		b.fault, b.failCall = "dispatch", before+2*m.Cfg.NumLayers
		b.cause = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
		read, restore := v41SuffixPrefillCalls(t)
		defer restore()
		err := recoverError(func() { s.Prefill([]int{3, 4}) })
		var selected *V41SharedAttentionOperationError
		if !errors.As(err, &selected) || !errors.Is(err, b.cause) || !errors.Is(err, ErrV41ForwardStage) || !s.BackendSessionClosed() || len(b.calls) != b.failCall || len(b.allocations) != 0 || read() != 1 {
			t.Fatalf("later suffix failure replayed or lost its committed boundary: %v", err)
		}
		if !reflect.DeepEqual(s.v41Forward.history, []int{1, 2, 3}) || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(expected)) {
			t.Fatal("later suffix failure failed to retain exactly the earlier successful token")
		}
		if retry := recoverError(func() { s.Prefill([]int{4}) }); retry != s.halFailure || len(b.calls) != b.failCall {
			t.Fatalf("closed suffix session retried: %v", retry)
		}
	})
}
