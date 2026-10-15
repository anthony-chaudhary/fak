package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This CPU-backed recorder observes selected operation shapes and logical
// completions. It measures neither physical submissions/fences nor GPU time.
// The independent host router evaluates the actual uploaded logits in each row.
type v41ColdRouterRecorder struct {
	*v41RouterBiasRecorder
	owner     sync.Mutex
	session   *Session
	owned     bool
	rows      []int
	picks     []routePick
	seen      int
	completed int
	numeric   map[int]compute.V41RouterTopKFault
	malformed int
	after     func()
}

type v41ColdRouterOwnedRecorder struct{ *v41ColdRouterRecorder }

func (b *v41ColdRouterOwnedRecorder) ForwardExecutionOwner() *sync.Mutex { return &b.owner }

// Mirror v41RouterResidentRecorder's complete F32 ownership boundary. Merely
// embedding a CPU backend does not make its buffers belong to this recorder.
func (b *v41ColdRouterOwnedRecorder) own(shape []int, data []float32, site string) compute.Tensor {
	out := compute.NewF32(b, append([]int(nil), shape...), append([]float32(nil), data...))
	b.roles[out.Buf()] = site
	return out
}

func (b *v41ColdRouterOwnedRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	return b.UploadClass(x, dt, compute.MemoryActivation, "unclassified")
}

func (b *v41ColdRouterOwnedRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, site string) compute.Tensor {
	if dt != compute.F32 {
		panic("cold router fixture only supports owned F32 uploads")
	}
	return b.own(x.Shape, b.Backend.Read(x), site)
}

func (b *v41ColdRouterOwnedRecorder) projectionOutput(tmp compute.Tensor) compute.Tensor {
	defer b.Backend.Free(tmp)
	return b.own(tmp.Shape, b.Backend.Read(tmp), "projection output")
}

func (b *v41ColdRouterOwnedRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	return b.projectionOutput(b.Backend.MatMul(w, x))
}

func (b *v41ColdRouterOwnedRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	return b.projectionOutput(b.Backend.BatchedMatMul(w, x, rows))
}

func (b *v41ColdRouterOwnedRecorder) requireLive(x compute.Tensor) {
	if _, owned := b.roles[x.Buf()]; owned {
		if x.Backend() != b || b.frees[x.Buf()] != 0 {
			panic("cold router fixture rejected a foreign or retired buffer")
		}
	} else if x.Backend() != b.Backend {
		panic("cold router fixture rejected an unknown buffer owner")
	}
}

func (b *v41ColdRouterOwnedRecorder) Read(x compute.Tensor) []float32 {
	b.requireLive(x)
	return b.Backend.Read(x)
}

func (b *v41ColdRouterOwnedRecorder) Free(x compute.Tensor) {
	b.requireLive(x)
	b.v41RouterBiasRecorder.Free(x)
}

func (b *v41ColdRouterRecorder) V41RouterTopK(logits, bias compute.Tensor, rows, experts, topK int, scale float32, full, hasBias bool) (compute.V41RouterPicks, error) {
	// Match the production recording-domain provider. Without it, the public
	// Session entrypoint correctly declines canonical native composition.
	if b.owned {
		if b.session == nil || b.session.backendExecutionScope == nil || !b.session.backendExecutionScope.live || b.session.backendExecutionScope.owner != &b.owner {
			return compute.V41RouterPicks{}, errors.New("router witness has no live owned Session scope")
		}
		if b.owner.TryLock() {
			b.owner.Unlock()
			return compute.V41RouterPicks{}, errors.New("router witness entered without its execution owner held")
		}
		if logits.Backend() != b.session.Backend || b.roles[logits.Buf()] == "" || b.frees[logits.Buf()] != 0 ||
			hasBias && (bias.Backend() != b.session.Backend || b.roles[bias.Buf()] == "" || b.frees[bias.Buf()] != 0) {
			return compute.V41RouterPicks{}, errors.New("router witness received foreign or retired selected operands")
		}
	} else if b.session == nil || b.session.backendExecutionScope != nil {
		return compute.V41RouterPicks{}, errors.New("unowned router control unexpectedly acquired a native scope")
	}
	if !reflect.DeepEqual(logits.Shape, []int{rows, experts}) || rows <= 0 {
		return compute.V41RouterPicks{}, errors.New("router received an incorrect row extent")
	}
	start := b.seen
	b.seen += rows
	b.rows = append(b.rows, rows)
	out, err := b.v41RouterBiasRecorder.V41RouterTopK(logits, bias, rows, experts, topK, scale, full, hasBias)
	if err != nil {
		return out, err
	}
	b.completed++
	if b.malformed >= start && b.malformed < start+rows {
		out.Experts[(b.malformed-start)*topK] = int32(experts)
	}
	// Model the existing primitive's status-before-picks contract, including
	// mixed corrupt publications. Every numeric status is ordered by row.
	for row := 0; row < rows; row++ {
		if fault, ok := b.numeric[start+row]; ok {
			fault.Row = row
			return out, &fault
		}
	}
	for i, expert := range out.Experts {
		b.picks = append(b.picks, routePick{expert: int(expert), weight: out.Weights[i]})
	}
	if b.after != nil {
		b.after()
	}
	return out, nil
}

func v41ColdRouterSession(t *testing.T, scalar bool, unowned ...bool) (*Model, *Session, *v41ColdRouterRecorder) {
	t.Helper()
	m := v41FullStepPatchedNorms(t)
	if m.Cfg.NumLayers != 1 {
		t.Fatal("row-count witness requires the one-layer full-geometry fixture")
	}
	b := &v41ColdRouterRecorder{v41RouterBiasRecorder: newV41RouterBiasRecorder(), malformed: -1}
	b.owned = len(unowned) == 0 || !unowned[0]
	t.Cleanup(func() {
		for buffer, role := range b.roles {
			if b.frees[buffer] != 1 {
				t.Errorf("%s allocation freed %d times, want1", role, b.frees[buffer])
			}
		}
	})
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("CloseWeights: %v", err)
		}
	})
	var backend compute.Backend = b
	if b.owned {
		backend = &v41ColdRouterOwnedRecorder{b}
	}
	s := v41DenseTestSession(t, m, backend)
	b.session = s
	if scalar {
		// Replacing the ordinary callback must remove canonical composition
		// capability, while preserving the identical selected row operation.
		state := s.v41State()
		state.setRouterTopK(state.routerTopK())
	}
	return m, s, b
}

func v41ColdRouterFailure(t *testing.T, s *Session, ids []int) error {
	t.Helper()
	var caught any
	func() {
		defer func() { caught = recover() }()
		_ = s.Prefill(ids)
	}()
	err, ok := caught.(error)
	var closed *BackendForwardOperationError
	if !ok || !errors.As(err, &closed) || closed.Path != "v41-router-topk" || !s.BackendSessionClosed() || len(s.v41State().history) != 0 {
		t.Fatalf("failure did not close and roll back: panic=%v closed=%t history=%d", caught, s.BackendSessionClosed(), len(s.v41State().history))
	}
	return err
}

func v41ColdRouterSamePicks(a, b []routePick) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].expert != b[i].expert || math.Float32bits(a[i].weight) != math.Float32bits(b[i].weight) {
			return false
		}
	}
	return true
}

func v41ColdRouterTraceChild(t *testing.T, mode string) {
	t.Helper()
	path := os.Getenv(ExpertRouteTraceEnv)
	if path == "" {
		t.Fatal("child has no isolated trace destination")
	}
	if mode == "after-first-failure" {
		t.Setenv(ExpertRouteTraceEnv, "")
		m, s, b := v41ColdRouterSession(t, false)
		b.numeric = map[int]compute.V41RouterTopKFault{0: {Stage: compute.V41RouterTopKStageLogit, Expert: 7}}
		ids, _ := v41IncrementalPrefix(m.Cfg, 3)
		_ = v41ColdRouterFailure(t, s, ids)
		if !reflect.DeepEqual(b.rows, []int{1}) {
			t.Fatal("failed first token dispatched later rows")
		}
		// The failed request must not have consumed the trace's lazy init.
		t.Setenv(ExpertRouteTraceEnv, path)
	} else {
		if processExpertRouteSink() == nil {
			t.Fatal("child did not open its isolated process trace sink")
		}
		// An already-open sink stays live after configuration removal. This
		// exercises the existing-sink guard rather than the environment guard.
		t.Setenv(ExpertRouteTraceEnv, "")
	}
	m, s, b := v41ColdRouterSession(t, false)
	b.numeric = map[int]compute.V41RouterTopKFault{1: {Stage: compute.V41RouterTopKStageLogit, Expert: 7}}
	ids, _ := v41IncrementalPrefix(m.Cfg, 3)
	_ = v41ColdRouterFailure(t, s, ids)
	if sink := processExpertRouteSink(); sink != nil {
		t.Cleanup(func() { _ = sink.f.Close() })
	} else {
		t.Fatal("successful token did not initialize the trace")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 || !reflect.DeepEqual(b.rows, []int{1, 1}) {
		t.Fatalf("prior successful trace lost: rows=%v JSONL=%q", b.rows, data)
	}
	var header ExpertRouteTraceHeader
	var event ExpertRouteEvent
	if json.Unmarshal([]byte(lines[0]), &header) != nil || header.Schema != ExpertRouteTraceSchema ||
		json.Unmarshal([]byte(lines[1]), &event) != nil || event.Token != 0 || event.Layer != 0 || len(event.Experts) != m.Cfg.NumExpertsPerTok || len(event.Weights) != len(event.Experts) {
		t.Fatalf("invalid retained first-token trace: %q", data)
	}
}

// Existing Session APIs make the test-only parent fail on actual scalar call
// counts, not on undefined production helpers. The shared primitive's numeric
// ordering is retained; synthetic mixed corrupt-output diagnostics are not
// claimed identical to scalar calls (the mixed-publication case records this).
// fak-test:runtime medium est=20s lane=default
func TestV41ColdRouterPanel(t *testing.T) {
	const traceChildEnv = "FAK_TEST_V41_COLD_ROUTER_TRACE_CHILD"
	if mode := os.Getenv(traceChildEnv); mode != "" {
		v41ColdRouterTraceChild(t, mode)
		return
	}
	for _, rows := range []int{1, 2, 63, 64, 65, 66, 129} {
		t.Run(fmt.Sprintf("rows-%d", rows), func(t *testing.T) {
			m, s, b := v41ColdRouterSession(t, false)
			_, control, serial := v41ColdRouterSession(t, true)
			ids, next := v41IncrementalPrefix(m.Cfg, rows)
			got, want := s.Prefill(ids), control.Prefill(ids)
			if len(got) == 0 || !v41SameBits(got, want) || !v41ColdRouterSamePicks(b.picks, serial.picks) || !reflect.DeepEqual(s.v41State().history, control.v41State().history) {
				t.Fatal("cold panel changed picks, exact weights, last logits or committed history")
			}
			expected := []int{1}
			for left := rows - 1; left > 0; {
				n := min(64, left)
				expected = append(expected, n)
				left -= n
			}
			if !reflect.DeepEqual(b.rows, expected) || len(serial.rows) != rows || b.completed != len(expected) || serial.completed != rows {
				t.Fatalf("selected rows=%v logical completions=%d; want %v; scalar rows=%v completions=%d", b.rows, b.completed, expected, serial.rows, serial.completed)
			}
			for _, count := range serial.rows {
				if count != 1 {
					t.Fatal("ordinary-callback control unexpectedly batched")
				}
			}
			t.Logf("cold_router_tokens=%d selected_rows=%v logical_calls=%d logical_completions=%d scalar_calls=%d", rows, b.rows, len(b.rows), b.completed, len(serial.rows))
			// The retained window is128 rows. Stay inside it for this independent
			// cold→incremental comparison; the129 case witnesses chunk tails only.
			if rows < 128 {
				before := len(b.rows)
				if !v41SameBits(s.Step(next), control.Step(next)) || len(b.rows) != before+1 || b.rows[before] != 1 {
					t.Fatal("subsequent single decode changed output or row count")
				}
			}
		})
	}
	for _, tc := range []struct {
		name  string
		rows  int
		fault map[int]compute.V41RouterTopKFault
		want  []int
		field string
	}{
		{"first-row", 3, map[int]compute.V41RouterTopKFault{0: {Stage: compute.V41RouterTopKStageLogit, Expert: 4}, 1: {Stage: compute.V41RouterTopKStageBias, Expert: 3}}, []int{1}, "logits"},
		{"first-panel", 3, map[int]compute.V41RouterTopKFault{1: {Stage: compute.V41RouterTopKStageBias, Expert: 3}, 2: {Stage: compute.V41RouterTopKStageLogit, Expert: 4}}, []int{1, 2}, "correction_bias"},
		{"later-chunk", 130, map[int]compute.V41RouterTopKFault{66: {Stage: compute.V41RouterTopKStageBias, Expert: 3}, 70: {Stage: compute.V41RouterTopKStageLogit, Expert: 4}}, []int{1, 64, 64}, "correction_bias"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, s, b := v41ColdRouterSession(t, false)
			b.numeric = tc.fault
			ids, _ := v41IncrementalPrefix(m.Cfg, tc.rows)
			err := v41ColdRouterFailure(t, s, ids)
			var fault *v4RouteError
			if !errors.As(err, &fault) || fault.Field != tc.field || !reflect.DeepEqual(b.rows, tc.want) || b.completed != len(tc.want) {
				t.Fatalf("fault=%v selected rows=%v completions=%d", err, b.rows, b.completed)
			}
		})
	}
	t.Run("malformed-output", func(t *testing.T) {
		m, s, b := v41ColdRouterSession(t, false)
		b.malformed = 1
		ids, _ := v41IncrementalPrefix(m.Cfg, 3)
		err := v41ColdRouterFailure(t, s, ids)
		var fault *v4RouteError
		if !errors.As(err, &fault) || fault.Field != "device_route" || !reflect.DeepEqual(b.rows, []int{1, 2}) {
			t.Fatalf("malformed output escaped or replayed: %v rows=%v", err, b.rows)
		}
	})
	t.Run("mixed-publication-precedence", func(t *testing.T) {
		for _, scalar := range []bool{false, true} {
			m, s, b := v41ColdRouterSession(t, scalar)
			b.malformed = 1
			b.numeric = map[int]compute.V41RouterTopKFault{2: {Stage: compute.V41RouterTopKStageLogit, Expert: 7}}
			ids, _ := v41IncrementalPrefix(m.Cfg, 3)
			err := v41ColdRouterFailure(t, s, ids)
			field, rows := "logits", []int{1, 2}
			if scalar {
				field, rows = "device_route", []int{1, 1}
			}
			var fault *v4RouteError
			if !errors.As(err, &fault) || fault.Field != field || !reflect.DeepEqual(b.rows, rows) {
				t.Fatalf("documented mixed diagnostic scalar=%t: %v rows=%v", scalar, err, b.rows)
			}
		}
	})
	for _, mode := range []string{"observer", "event-observer", "trace-env", "custom-callback", "unowned-backend"} {
		t.Run(mode, func(t *testing.T) {
			m, s, b := v41ColdRouterSession(t, false, mode == "unowned-backend")
			events, callbacks := 0, 0
			switch mode {
			case "observer":
				m.SetExpertRouteObserver(func(int, []int) { events++ })
			case "event-observer":
				m.SetExpertRouteEventObserver(func(ExpertRouteEvent) { events++ })
			case "trace-env":
				t.Setenv(ExpertRouteTraceEnv, t.TempDir()+"/routes.jsonl")
			case "custom-callback":
				state := s.v41State()
				ordinary := state.routerTopK()
				state.setRouterTopK(func(layer int, logits, bias []float32, cfg v41RouterConfig, full bool) ([]routePick, error) {
					callbacks++
					return ordinary(layer, logits, bias, cfg, full)
				})
			}
			b.numeric = map[int]compute.V41RouterTopKFault{1: {Stage: compute.V41RouterTopKStageLogit, Expert: 7}}
			ids, _ := v41IncrementalPrefix(m.Cfg, 3)
			_ = v41ColdRouterFailure(t, s, ids)
			if !reflect.DeepEqual(b.rows, []int{1, 1}) || (mode == "observer" || mode == "event-observer") && events != 1 || mode == "custom-callback" && callbacks != 2 {
				t.Fatalf("fallback lost scalar effects: mode=%s rows=%v events=%d callbacks=%d", mode, b.rows, events, callbacks)
			}
		})
	}
	t.Run("scope-changed-after-first-chunk", func(t *testing.T) {
		m, s, b := v41ColdRouterSession(t, false)
		b.after = func() {
			if len(b.rows) == 2 {
				s.cacheGeometryEpoch++
			}
		}
		ids, _ := v41IncrementalPrefix(m.Cfg, 66)
		err := v41ColdRouterFailure(t, s, ids)
		if !errors.Is(err, errV41RouterProjectionScope) || !reflect.DeepEqual(b.rows, []int{1, 64}) {
			t.Fatalf("changed scope continued or replayed: %v rows=%v", err, b.rows)
		}
	})
	for _, mode := range []string{"already-open-trace-sink", "after-first-failure"} {
		t.Run(mode, func(t *testing.T) {
			path := t.TempDir() + "/routes.jsonl"
			cmd := exec.Command(os.Args[0], "-test.run=^TestV41ColdRouterPanel$", "-test.timeout=30s")
			cmd.Env = append(os.Environ(), traceChildEnv+"="+mode, ExpertRouteTraceEnv+"="+path)
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("isolated trace control: %v\n%s", err, output)
			}
		})
	}
}
