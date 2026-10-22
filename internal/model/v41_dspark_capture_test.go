package model

import (
	"context"
	"errors"
	"math"
	"os"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This is a synthetic full-four-stream target, never checkpoint/hardware proof.
// Alias immutable layer-0 weights into three layers; each layer has real Engram.
func v41DSparkCaptureFixture(t *testing.T) (*Model, V41EngramLayout) {
	t.Helper()
	m, layout := v41FullEngramModel(t)
	t.Cleanup(func() { _ = m.CloseWeights() })
	m.Cfg.NumLayers = 3
	m.Cfg.Window = []int{-1, -1, -1}
	meta := *m.Cfg.DeepSeekV41
	draft := *meta.DSpark
	draft.TargetLayerIDs = []int{0, 1, 2}
	meta.DSpark = &draft
	meta.CompressRatios = make([]int, 6)
	// This synthetic three-layer target uses window-only attention.
	meta.KVSourceLayerIDs = nil
	meta.IndexSourceLayerIDs = nil
	meta.CandidateSourceLayerID = -1
	meta.EngramLayerIDs = []int{0, 1, 2}
	meta.EngramNumEmbeddings = []int{48, 48, 48}
	m.Cfg.DeepSeekV41 = &meta
	first := make(map[string]tensorMeta)
	for name, tm := range m.manifest {
		if strings.HasPrefix(name, "model.layers.0.") {
			first[name] = tm
		}
	}
	for layer := 1; layer < 3; layer++ {
		for name, tm := range first {
			m.manifest[strings.Replace(name, "model.layers.0.", "model.layers."+itoa(layer)+".", 1)] = tm
		}
	}
	layout.Rows = []uint32{48, 48, 48}
	multipliers, primes := layout.Multipliers[0], layout.Primes[0]
	layout.Multipliers, layout.Primes = make([][]uint64, 3), make([][]uint32, 3)
	sources := make([]V41EngramRowSource, 3)
	for layer := range sources {
		layout.Multipliers[layer] = append([]uint64(nil), multipliers...)
		layout.Primes[layer] = append([]uint32(nil), primes...)
		sources[layer] = &v41EngramMemorySource{packed: v41EngramTestPackedRows(48), rows: 48}
	}
	if err := m.wireV41Engram(layout, sources, int64(V41EngramPackedRowBytes)*4); err != nil {
		t.Fatal(err)
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("window-only target fixture admission: %v", err)
	}
	return m, layout
}

func v41DSparkCaptureSession(t *testing.T, m *Model) *Session {
	t.Helper()
	s := m.NewSession()
	t.Cleanup(s.Close)
	s.v41State()
	return s
}

// Independent ordered-F32 mean and BF16 RNE; never calls the production mean.
func v41DSparkCaptureMeanOracle(streams [][]float32) []float32 {
	out := make([]float32, len(streams[0]))
	for d := range out {
		var sum float32
		for stream := 0; stream < 4; stream++ {
			sum = float32(sum + streams[stream][d])
		}
		u := math.Float32bits(float32(sum * .25))
		u += 0x7fff + ((u >> 16) & 1)
		out[d] = math.Float32frombits(u & 0xffff0000)
	}
	return out
}

func v41DSparkCaptureRowsClose(t *testing.T, got, want []v41DSparkContextRow) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("row count %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i].Position != want[i].Position || got[i].Token != want[i].Token || !slices.Equal(got[i].Layers, want[i].Layers) {
			t.Fatalf("row %d lineage differs", i)
		}
		for tap := 0; tap < 3; tap++ {
			if len(got[i].Hidden[tap]) != len(want[i].Hidden[tap]) {
				t.Fatal("tap width differs")
			}
			for d, value := range got[i].Hidden[tap] {
				w := want[i].Hidden[tap][d]
				if math.Abs(float64(value-w)) > 1e-4*(1+math.Abs(float64(w))) {
					t.Fatalf("row %d tap %d dim %d: %g, want %g", i, tap, d, value, w)
				}
			}
		}
	}
}

func v41DSparkCapturedCold(t *testing.T, s *Session, ids []int) ([]v41DSparkContextRow, *Activations) {
	t.Helper()
	c, err := newV41DSparkCapture(context.Background(), s, ids, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.close()
	act, err := s.M.forwardV41WithHeadSelection(ids, s.v41Forward, true)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := c.committed()
	if err != nil {
		t.Fatal(err)
	}
	return rows, act
}

// fak-test:runtime medium est=8s lane=default
func TestV41DSparkCaptureNativePaths(t *testing.T) {
	m, layout := v41DSparkCaptureFixture(t)
	t.Run("window-only-source-declaration-refuses", func(t *testing.T) {
		bad := &Model{Cfg: m.Cfg}
		meta := *m.Cfg.DeepSeekV41
		meta.KVSourceLayerIDs = []int{2}
		bad.Cfg.DeepSeekV41 = &meta
		err := bad.v41KVSourceForwardAdmitted()
		var stage *V41ForwardError
		if !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &stage) || stage.Stage != v41StageAttention || stage.Layer != 2 {
			t.Fatalf("ratio-zero source admission = %v, want typed attention refusal at layer 2", err)
		}
		if err := m.v41ForwardAdmitted(); err != nil {
			t.Fatalf("negative control changed the original fixture: %v", err)
		}
	})
	t.Run("cold-incremental-and-independent-first-tap", func(t *testing.T) {
		s := v41DSparkCaptureSession(t, m)
		ids := []int{1, 3, 5}
		rows, _ := v41DSparkCapturedCold(t, s, ids)
		embedded, err := m.v41EmbeddingPanelOn(ids, nil)
		if err != nil {
			t.Fatal(err)
		}
		streams := make([][][]float32, len(ids))
		for i := range ids {
			streams[i] = make([][]float32, 4)
			for stream := 0; stream < 4; stream++ {
				streams[i][stream] = make([]float32, m.Cfg.HiddenSize)
				for d := range streams[i][stream] {
					u := math.Float32bits(embedded[i*m.Cfg.HiddenSize+d])
					u += 0x7fff + ((u >> 16) & 1)
					streams[i][stream][d] = math.Float32frombits(u & 0xffff0000)
				}
			}
		}
		injected := v41OracleFullEngramUpdates(t, m, layout, 0, streams, ids, float32(m.Cfg.RMSNormEps))
		moved, differsFromZero := false, false
		for i := range rows {
			want := v41DSparkCaptureMeanOracle(injected[i])
			for d, value := range rows[i].Hidden[0] {
				if math.Abs(float64(value-want[d])) > 1e-4*(1+math.Abs(float64(want[d]))) {
					t.Fatalf("cold pre-attention tap differs from independent Engram/mean at %d/%d", i, d)
				}
				moved = moved || value != streams[i][0][d]
				differsFromZero = differsFromZero || value != injected[i][0][d]
			}
		}
		if !moved || !differsFromZero {
			t.Fatal("first-tap witness cannot distinguish pre-Engram or stream zero")
		}
		c, err := newV41DSparkCapture(context.Background(), s, []int{7}, false, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer c.close()
		if _, _, err := m.forwardV41Step(7, s.v41Forward, nil); err != nil {
			t.Fatal(err)
		}
		last, err := c.committed()
		if err != nil {
			t.Fatal(err)
		}
		ref := v41DSparkCaptureSession(t, m)
		all, act := v41DSparkCapturedCold(t, ref, []int{1, 3, 5, 7})
		v41DSparkCaptureRowsClose(t, append(rows, last...), all)
		if reflect.DeepEqual(all[3].Hidden[2], act.Hidden[0][3*m.Cfg.HiddenSize:]) {
			t.Fatal("last pre-attention tap equals post-layer output; weak placement witness")
		}
		owned := all[0].Hidden[0][0]
		last[0].Hidden[0][0] = 12345
		if all[0].Hidden[0][0] != owned || s.v41Forward.dsparkCapture != nil {
			t.Fatal("capture storage or binding was retained/shared")
		}
	})
	t.Run("retained-six-all-prefixes", func(t *testing.T) {
		for keep := 0; keep <= 6; keep++ {
			s, ref := v41DSparkCaptureSession(t, m), v41DSparkCaptureSession(t, m)
			v41DSparkCapturedCold(t, s, []int{1, 2})
			v41DSparkCapturedCold(t, ref, []int{1, 2})
			ids := []int{3, 4, 5, 6, 7, 0}
			r, err := m.beginV41StepRound(s.v41Forward, 6)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = r.abort() })
			c, err := newV41DSparkCapture(context.Background(), s, ids, false, r)
			if err != nil {
				_ = r.abort()
				t.Fatal(err)
			}
			t.Cleanup(c.close)
			for _, id := range ids {
				tx, err := r.beginStep(id, nil)
				if err != nil {
					t.Fatal(err)
				}
				for tx.nextLayer < tx.layers {
					if err := tx.advanceLayer(); err != nil {
						t.Fatal(err)
					}
				}
				if _, err := tx.finalize(); err != nil {
					t.Fatal(err)
				}
				if tx.captureValidate != nil {
					t.Fatal("closed transaction retained capture callback")
				}
			}
			if _, err := c.committed(); !errors.Is(err, errV41DSparkCapture) {
				t.Fatal("speculative rows escaped as committed")
			}
			if keep > 0 {
				staged, err := c.staged(r, keep)
				if err != nil || len(staged) != keep || staged[0].Position != 2 || staged[keep-1].Token != ids[keep-1] {
					t.Fatalf("staging retained prefix %d: %v", keep, err)
				}
			}
			if err := r.finish(keep); err != nil {
				t.Fatal(err)
			}
			if _, err := c.staged(r, max(1, keep)); !errors.Is(err, errV41DSparkCapture) {
				t.Fatal("closed round replay admitted")
			}
			c.close()
			for _, id := range ids[:keep] {
				if _, _, err := m.forwardV41Step(id, ref.v41Forward, nil); err != nil {
					t.Fatal(err)
				}
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(ref.v41Forward)) {
				t.Fatalf("six-step retained prefix %d differs from serial target", keep)
			}
			// Both branches must remain usable after the rejected suffix is drained.
			got, _, err := m.forwardV41Step(1, s.v41Forward, nil)
			want, _, refErr := m.forwardV41Step(1, ref.v41Forward, nil)
			if err != nil || refErr != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("continuation after keep %d: %v/%v", keep, err, refErr)
			}
		}
	})
}

// Tail capture is the input to future BootstrapCommittedTail, not that missing
// draft operation itself. This verifies absolute lineage and bounded ownership.
// fak-test:runtime medium est=12s lane=default
func TestV41DSparkCaptureTailAndFailure(t *testing.T) {
	m, _ := v41DSparkCaptureFixture(t)
	for _, count := range []int{127, 128, 129} {
		s := v41DSparkCaptureSession(t, m)
		ids := make([]int, count)
		for i := range ids {
			ids[i] = i % m.Cfg.VocabSize
		}
		rows, _ := v41DSparkCapturedCold(t, s, ids)
		if len(rows) != min(count, 128) || rows[0].Position != max(0, count-128) || rows[len(rows)-1].Position != count-1 {
			t.Fatal("tail capture lost absolute positions or 128-row bound")
		}
		for _, row := range rows {
			if row.Token != ids[row.Position] {
				t.Fatal("tail token lineage differs")
			}
		}
	}
	t.Run("failed-zero-append-recapture", func(t *testing.T) {
		s := v41DSparkCaptureSession(t, m)
		v41DSparkCapturedCold(t, s, []int{1, 2})
		before := captureV41ForwardSnapshot(s.v41Forward)
		c, err := newV41DSparkCapture(context.Background(), s, nil, true, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer c.close()
		fault := errors.New("late cold head fault")
		s.v41Forward.setFinalNorm(func([]float32) ([]float32, error) { return nil, fault })
		_, err = m.forwardV41(nil, s.v41Forward)
		if !errors.Is(err, fault) {
			t.Fatalf("head cause: %v", err)
		}
		if _, err := c.committed(); !errors.Is(err, errV41DSparkCapture) || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
			t.Fatal("failed recapture published taps or changed target temporal state")
		}
	})
	t.Run("cancel-during-final-normalization", func(t *testing.T) {
		for _, noLogits := range []bool{false, true} {
			s := v41DSparkCaptureSession(t, m)
			v41DSparkCapturedCold(t, s, []int{1, 2})
			before := captureV41ForwardSnapshot(s.v41Forward)
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)
			c, err := newV41DSparkCapture(ctx, s, []int{3}, false, nil)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(c.close)
			s.v41Forward.setFinalNorm(func(x []float32) ([]float32, error) { cancel(); return x, nil })
			_, _, err = m.forwardV41StepCore(3, s.v41Forward, nil, false, !noLogits)
			if !errors.Is(err, context.Canceled) || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatalf("late cancellation changed target state (noLogits=%v): %v", noLogits, err)
			}
			c.close()
			cancel()
		}
	})
	t.Run("late-cold-unwind", func(t *testing.T) {
		for _, mode := range []string{"panic", "panic-nil", "goexit"} {
			s := v41DSparkCaptureSession(t, m)
			c, err := newV41DSparkCapture(context.Background(), s, []int{1}, true, nil)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(c.close)
			fault := errors.New("late cold cleanup")
			done := make(chan struct{})
			returned := false
			var caught any
			go func() {
				defer close(done)
				defer func() { caught = recover() }()
				_, _ = c.runCold(func() (*Activations, error) {
					defer func() {
						switch mode {
						case "panic":
							panic(fault)
						case "panic-nil":
							panic(nil)
						default:
							runtime.Goexit()
						}
					}()
					return m.forwardV41WithHeadModeCore([]int{1}, s.v41Forward, true, true)
				})
				returned = true
			}()
			<-done
			if returned || !c.failed || c.coldSucceeded || c.run != nil || (mode == "panic" && caught != fault) {
				t.Fatalf("late %s became successful capture: returned=%v caught=%v", mode, returned, caught)
			}
			if len(s.v41Forward.history) != 1 {
				t.Fatal("late-fault witness did not cross actual target publication")
			}
			if _, err := c.committed(); !errors.Is(err, errV41DSparkCapture) {
				t.Fatal("late target cleanup failure published capture")
			}
		}
	})
	t.Run("late-trace-publication-error", func(t *testing.T) {
		s := v41DSparkCaptureSession(t, m)
		c, err := newV41DSparkCapture(context.Background(), s, []int{1}, true, nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(c.close)
		previous := v41ActivationTraceEnvPath
		path := t.TempDir() + "/missing/trace.json"
		v41ActivationTraceEnvPath = func() string { return path }
		t.Cleanup(func() { v41ActivationTraceEnvPath = previous })
		_, err = m.forwardV41([]int{1}, s.v41Forward)
		var pathError *os.PathError
		if err == nil || !errors.As(err, &pathError) || !strings.Contains(err.Error(), path) ||
			!slices.Equal(s.v41Forward.history, []int{1}) || !c.failed || c.coldSucceeded || c.run != nil {
			t.Fatalf("late trace destination failure did not cross target publication: history=%v path=%q err=%v", s.v41Forward.history, path, err)
		}
		if _, err := c.committed(); !errors.Is(err, errV41DSparkCapture) {
			t.Fatal("failed trace publication supplied committed taps")
		}
	})
	t.Run("pretrained-capture-limits", func(t *testing.T) {
		s := v41DSparkCaptureSession(t, m)
		width, layers := m.Cfg.HiddenSize, m.Cfg.NumLayers
		defer func() { m.Cfg.HiddenSize, m.Cfg.NumLayers = width, layers }()
		m.Cfg.HiddenSize = 5121
		if _, err := newV41DSparkCapture(context.Background(), s, []int{1}, true, nil); !errors.Is(err, errV41DSparkCapture) {
			t.Fatal("oversized capture width admitted")
		}
		m.Cfg.HiddenSize, m.Cfg.NumLayers = width, 41
		if _, err := newV41DSparkCapture(context.Background(), s, []int{1}, true, nil); !errors.Is(err, errV41DSparkCapture) {
			t.Fatal("oversized target depth admitted")
		}
	})
	t.Run("wrong-token-stale-state-and-duplicate", func(t *testing.T) {
		s := v41DSparkCaptureSession(t, m)
		v41DSparkCapturedCold(t, s, []int{1})
		c, err := newV41DSparkCapture(context.Background(), s, []int{2}, false, nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(c.close)
		_, streams := v41FullEngramAsymmetricStreams([]int{2}, m.Cfg.HiddenSize)
		if err := c.captureStep(s.v41Forward, 0, 2, 1, streams[0], nil, true); err != nil {
			t.Fatal(err)
		}
		want := v41DSparkCaptureMeanOracle(streams[0])
		if !reflect.DeepEqual(c.rows[0].Hidden[0], want) {
			t.Fatal("ordered mean differs")
		}
		streams[0][0][0] += 1
		if !reflect.DeepEqual(c.rows[0].Hidden[0], want) {
			t.Fatal("capture aliases source streams")
		}
		if err := c.captureStep(s.v41Forward, 0, 2, 1, streams[0], nil, true); !errors.Is(err, errV41DSparkCapture) {
			t.Fatal("duplicate tap admitted")
		}
		c.close()
		c, err = newV41DSparkCapture(context.Background(), s, []int{2}, false, nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(c.close)
		if err := c.captureStep(s.v41Forward, 0, 3, 1, streams[0], nil, true); !errors.Is(err, errV41DSparkCapture) {
			t.Fatal("wrong token admitted")
		}
		c.close()
		c, err = newV41DSparkCapture(context.Background(), s, []int{2}, false, nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(c.close)
		s.v41Forward = captureV41ForwardSnapshot(s.v41Forward).restore()
		if err := c.check(); !errors.Is(err, errV41DSparkCapture) {
			t.Fatal("restored target silently rebound old capture")
		}
		c.close()
	})
}

// A cpu-ref STORAGE recorder for the optional contract, not a device kernel.
type v41DSparkMeanRangeRecorder struct {
	compute.Backend
	allow              bool
	mode               string
	source             compute.Buffer
	calls, sourceReads int
	reads              []int
	fresh              compute.Buffer
	frees              map[compute.Buffer]int
	cause              error
}

func (b *v41DSparkMeanRangeRecorder) SupportsV41DSparkMeanRange(_ compute.Tensor, _, count int) bool {
	return b.allow && count > 0 && count <= 128
}

func (b *v41DSparkMeanRangeRecorder) V41DSparkMeanRange(input compute.Tensor, first, count int) (compute.Tensor, error) {
	b.calls++
	if b.mode == "alias" {
		out := input
		out.Shape = []int{count, input.Shape[2]}
		return out, nil
	}
	// Direct storage access simulates a device op; never calls the model's Read.
	in := b.Backend.Read(input)
	H := input.Shape[2]
	values := make([]float32, count*H)
	for row := 0; row < count; row++ {
		streams := make([][]float32, 4)
		for stream := 0; stream < 4; stream++ {
			start := ((first+row)*4 + stream) * H
			streams[stream] = in[start : start+H]
		}
		copy(values[row*H:], v41DSparkCaptureMeanOracle(streams))
	}
	if b.mode == "non-bf16" {
		values[0] = 1.001
	}
	out := compute.NewF32(b, []int{count, H}, values)
	b.fresh = out.Buf()
	if b.mode == "partial-error" || b.mode == "partial-error-free-fault" {
		return out, b.cause
	}
	return out, nil
}

func (b *v41DSparkMeanRangeRecorder) Read(t compute.Tensor) []float32 {
	if b.mode == "read-free-fault" {
		panic(&compute.BackendError{Backend: "mean-recorder", Site: "read", Err: b.cause})
	}
	if t.Buf() == b.source {
		b.sourceReads++
	}
	b.reads = append(b.reads, t.Numel())
	return b.Backend.Read(t)
}

func (b *v41DSparkMeanRangeRecorder) Free(t compute.Tensor) {
	b.frees[t.Buf()]++
	if t.Buf() == b.fresh && strings.HasSuffix(b.mode, "free-fault") {
		panic(errors.New("secondary output Free fault"))
	}
	b.Backend.Free(t)
}

// fak-test:runtime fast est=100ms lane=default
func TestV41DSparkCaptureResidentContract(t *testing.T) {
	const (
		first       = 1
		count       = v41WindowSize
		sourceRows  = first + count
		streamCount = compute.V41MHCStreams
		width       = 4
	)
	for _, mode := range []string{"ok", "declined", "alias", "non-bf16", "partial-error", "partial-error-free-fault", "read-free-fault"} {
		t.Run(mode, func(t *testing.T) {
			be := &v41DSparkMeanRangeRecorder{Backend: compute.Default(), allow: mode != "declined", mode: mode,
				frees: make(map[compute.Buffer]int), cause: errors.New("selected mean fault")}
			values := make([]float32, sourceRows*streamCount*width)
			for i := range values {
				values[i] = float32(i%31-15) / 16
			}
			input := compute.NewF32(be, []int{sourceRows, streamCount, width}, values)
			be.source = input.Buf()
			streams, err := newV41DeviceStreams(be, input)
			if err != nil {
				t.Fatal(err)
			}
			defer streams.Free()
			s := &Session{Backend: be, M: &Model{}}
			c := &v41DSparkCapture{owner: s, width: width}
			r := &v41MHCResident{d: &v41MHCDevice{s: s}, streams: streams, rows: sourceRows, hidden: width}
			rows, err := c.residentMeans(37, r, first, count, sourceRows)
			if be.sourceReads != 0 || be.frees[input.Buf()] != 0 {
				t.Fatal("capture read or freed borrowed full source panel")
			}
			if mode == "declined" {
				if !errors.Is(err, errV41DSparkMeanRangeUnavailable) || be.calls != 0 || len(be.reads) != 0 || s.halClosed {
					t.Fatal("missing capability selected work or read/fell back")
				}
				return
			}
			if mode != "ok" {
				var selected *V41MHCOperationError
				if !errors.As(err, &selected) || !s.halClosed || ((strings.HasPrefix(mode, "partial-error") || mode == "read-free-fault") && !errors.Is(err, be.cause)) {
					t.Fatalf("selected fault lost retirement/cause: %v", err)
				}
			} else {
				if err != nil || len(rows) != count || !reflect.DeepEqual(be.reads, []int{count * width}) {
					t.Fatalf("bounded output read: %v/%v", be.reads, err)
				}
				oracleStreams := make([][]float32, streamCount)
				for i := range oracleStreams {
					start := (first*streamCount + i) * width
					oracleStreams[i] = values[start : start+width]
				}
				want := v41DSparkCaptureMeanOracle(oracleStreams)
				if !reflect.DeepEqual(rows[0], want) {
					t.Fatal("range offset or mean order differs")
				}
			}
			if be.fresh != nil && be.frees[be.fresh] != 1 {
				t.Fatal("fresh/partial output ownership not drained exactly once")
			}
		})
	}
	// A quarter-first collapse can stay finite here. The required ordered
	// unscaled F32 sum overflows and must refuse rather than change arithmetic.
	huge := math.Float32frombits(0x7f7f0000)
	if _, err := v41DSparkMeanStreams([][]float32{{huge}, {huge}, {huge}, {huge}}, 1); err == nil {
		t.Fatal("overflowing mean silently changed to quarter-first arithmetic")
	}
}
