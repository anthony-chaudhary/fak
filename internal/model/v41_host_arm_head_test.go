package model

import (
	"errors"
	"math"
	"path/filepath"
	"reflect"
	"testing"
)

func v41HostHeadBitsEqual(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d want %d", len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("element %d bits %08x want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func v41HostHeadLogitCounts(stages []string) (counts [3]int) {
	for _, stage := range stages {
		switch stage {
		case "logit-scale":
			counts[0]++
		case "logit-softcap":
			counts[1]++
		case "logit-token-suppression":
			counts[2]++
		}
	}
	return
}

// fak-test:runtime fast est=10ms lane=default
func TestV41HostArmHeadPostprocess(t *testing.T) {
	for _, tc := range []struct {
		name  string
		cfg   Config
		input []float32
		want  []string
	}{
		{"empty", Config{LogitScale: 2, LogitSoftcap: 1, SuppressTokens: []int{0}}, nil, nil},
		{"identity", Config{}, []float32{1, -2}, nil},
		{"near-one", Config{LogitScale: math.Nextafter(1, 2)}, []float32{1, -2}, nil},
		{"scale", Config{LogitScale: -math.SmallestNonzeroFloat64}, []float32{1, -1}, []string{"logit-scale"}},
		{"invalid-suppression", Config{SuppressTokens: []int{-1, 2, 90}}, []float32{1, -2}, nil},
		{"all", Config{LogitScale: 0.5, LogitSoftcap: 2, SuppressTokens: []int{-1, 1, 1, 99}}, []float32{1, -2}, []string{"logit-scale", "logit-softcap", "logit-token-suppression"}},
		{"nan-softcap", Config{LogitSoftcap: math.NaN()}, []float32{1, -2}, []string{"logit-softcap"}},
		{"nonfinite", Config{LogitScale: 2, LogitSoftcap: 3}, []float32{float32(math.Inf(1)), float32(math.NaN())}, []string{"logit-scale", "logit-softcap"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stages []string
			input := append([]float32(nil), tc.input...)
			want := append([]float32(nil), input...)
			logitScaleInPlace(want, tc.cfg)
			got, err := v41ScaleHeadLogitsObserved(input, tc.cfg, nil, nil, func(stage string) { stages = append(stages, stage) })
			if err != nil {
				t.Fatal(err)
			}
			v41HostHeadBitsEqual(t, got, want)
			if !reflect.DeepEqual(stages, tc.want) {
				t.Fatalf("stages=%v want=%v", stages, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41HostArmHeadSelectedScale(t *testing.T) {
	fault := errors.New("selected scale fault")
	for _, fail := range []bool{false, true} {
		var stages []string
		calls := 0
		cfg := Config{LogitScale: 0.5, LogitSoftcap: 2, SuppressTokens: []int{1}}
		input := []float32{4, 3}
		got, err := v41ScaleHeadLogitsObserved(input, cfg, func(row []float32, factor float32) ([]float32, error) {
			calls++
			if fail {
				return nil, fault
			}
			out := append([]float32(nil), row...)
			for i := range out {
				out[i] *= factor
			}
			return out, nil
		}, nil, func(stage string) { stages = append(stages, stage) })
		if calls != 1 {
			t.Fatalf("selected calls=%d", calls)
		}
		if fail {
			if !errors.Is(err, fault) || !errors.Is(err, ErrV41LogitScaleOperation) || got != nil || len(stages) != 0 {
				t.Fatalf("selected failure replayed: %v %v", err, stages)
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}
			want := []float32{4, 3}
			logitScaleInPlace(want, cfg)
			v41HostHeadBitsEqual(t, got, want)
			if !reflect.DeepEqual(stages, []string{"logit-softcap", "logit-token-suppression"}) {
				t.Fatalf("selected stages=%v", stages)
			}
		}
		v41HostHeadBitsEqual(t, input, []float32{4, 3})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmHeadNormalization(t *testing.T) {
	for _, full := range []bool{false, true} {
		var m *Model
		if full {
			m = v41FullStepPatchedNorms(t)
		} else {
			m = v41IncrementalPlainModel(t, 1)
		}
		input := make([]float32, m.Cfg.HiddenSize)
		for i := range input {
			input[i] = float32(i%7-3) / 11
		}
		var stages []string
		observe := func(stage string) { stages = append(stages, stage) }
		got, err := m.v41PrepareHeadInputBoundObserved(input, nil, nil, observe)
		if err != nil {
			t.Fatal(err)
		}
		want, err := m.v41PrepareHeadInputBound(input, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		v41HostHeadBitsEqual(t, got, want)
		expected := []string{"rms-norm"}
		if full {
			expected = []string{"bf16-boundary", "rms-norm", "bf16-boundary"}
		}
		if !reflect.DeepEqual(stages, expected) {
			t.Fatalf("full=%v stages=%v want=%v", full, stages, expected)
		}
		stages = nil
		project := func(_ int, _ string, row []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
			return make([]float32, out*rows), v41ProjectionHandled, nil
		}
		if _, err := m.v41HeadWithLogitScaleBoundObserved(input, project, nil, nil, nil, nil, observe); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(stages[:len(expected)], expected) {
			t.Fatalf("cold helper stages=%v", stages)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmHeadNormFailures(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	fault := errors.New("selected head norm fault")
	for _, mode := range []string{"norm", "round", "invalid-first", "invalid-later", "selected-success"} {
		t.Run(mode, func(t *testing.T) {
			input := make([]float32, m.Cfg.HiddenSize)
			for i := range input {
				input[i] = 0.25
			}
			var normalize v41FinalNormFunc
			var round v41BF16BoundaryFunc
			var expected []string
			switch mode {
			case "norm":
				normalize = func([]float32) ([]float32, error) { return nil, fault }
				expected = []string{"bf16-boundary"}
			case "round":
				round = func(int, string, []float32) ([]float32, error) { return nil, fault }
			case "invalid-first":
				input[0] = float32(math.NaN())
			case "invalid-later":
				input[1] = float32(math.Inf(1))
				expected = []string{"bf16-boundary"}
			case "selected-success":
				normalize = func(row []float32) ([]float32, error) { return append([]float32(nil), row...), nil }
				round = func(_ int, _ string, row []float32) ([]float32, error) { return append([]float32(nil), row...), nil }
			}
			var stages []string
			_, err := m.v41FullFinalNormWithBF16BoundaryObserved(input, normalize, round, func(stage string) { stages = append(stages, stage) })
			if mode == "selected-success" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil {
				t.Fatal("expected failure")
			}
			if (mode == "norm" || mode == "round") && !errors.Is(err, fault) {
				t.Fatalf("lost selected cause: %v", err)
			}
			if !reflect.DeepEqual(stages, expected) {
				t.Fatalf("stages=%v want=%v", stages, expected)
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41HostArmHeadStepAndTrace(t *testing.T) {
	for _, trace := range []bool{false, true} {
		t.Run(map[bool]string{false: "ordinary", true: "trace"}[trace], func(t *testing.T) {
			withV41ActivationTraceEnv(t, "")
			m := v41FullStepPatchedNorms(t)
			m.Cfg.LogitScale, m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = 0.5, 2, []int{0}
			s := v41DenseTestSession(t, m, newV41DenseTestBackend())
			control := v41DenseTestSession(t, m, newV41DenseTestBackend())
			s.v41State().setFinalNorm(nil)
			control.v41State().setFinalNorm(nil)
			v41HostHeadBitsEqual(t, s.Prefill([]int{1}), control.Prefill([]int{1}))
			var stages []string
			s.hostStageObserver = func(stage string) { stages = append(stages, stage) }
			if trace {
				withV41ActivationTraceEnv(t, filepath.Join(t.TempDir(), "head-trace.json"))
			}
			got := s.Step(2)
			withV41ActivationTraceEnv(t, "")
			v41HostHeadBitsEqual(t, got, control.Step(2))
			wantStages := []string{"bf16-boundary", "rms-norm", "bf16-boundary", "logit-scale", "logit-softcap", "logit-token-suppression"}
			if len(stages) < len(wantStages) || !reflect.DeepEqual(stages[len(stages)-len(wantStages):], wantStages) {
				t.Fatalf("head stages=%v", stages)
			}
			if got := v41HostHeadLogitCounts(stages); got != [3]int{1, 1, 1} {
				t.Fatalf("Step head-exclusive counts=%v", got)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmHeadScaleLoan(t *testing.T) {
	withV41ActivationTraceEnv(t, "")
	m := v41IncrementalPlainModel(t, 1)
	m.Cfg.SetV41HeadScaleLoan(true)
	m.Cfg.LogitScale, m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = 0.5, 2, []int{0}
	b := newV41HeadScaleLoanRecorder()
	s := v41DenseTestSession(t, m, b)
	s.v41State().setFinalNorm(nil)
	b.headWeight = s.weightHAL(m.headName()).Buf()
	s.Prefill([]int{1})
	s.Step(2)
	var stages []string
	s.hostStageObserver = func(stage string) { stages = append(stages, stage) }
	before := b.retainedScaleCalls
	s.Step(3)
	if b.retainedScaleCalls != before+1 {
		t.Fatal("retained scale not executed")
	}
	wantStages := []string{"rms-norm", "logit-softcap", "logit-token-suppression"}
	if len(stages) < len(wantStages) || !reflect.DeepEqual(stages[len(stages)-len(wantStages):], wantStages) {
		t.Fatalf("loan stages=%v", stages)
	}
	if got := v41HostHeadLogitCounts(stages); got != [3]int{0, 1, 1} {
		t.Fatalf("loan head-exclusive counts=%v", got)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmHeadSkippedProjection(t *testing.T) {
	withV41ActivationTraceEnv(t, "")
	m, st, _ := v41RoundModeStates(t)
	m.Cfg.LogitScale, m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = 0.5, 2, []int{0}
	var stages []string
	owner := &Session{hostStageObserver: func(stage string) { stages = append(stages, stage) }}
	st.callbackOwner = owner
	round, err := m.beginV41StepRound(st, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer round.abort()
	tx := v41RoundPreparedTestStep(t, round, 4)
	if err := tx.finalizeNoLogits(); err != nil {
		t.Fatal(err)
	}
	for _, stage := range stages {
		if stage == "logit-scale" || stage == "logit-softcap" || stage == "logit-token-suppression" {
			t.Fatalf("skipped projection observed %v", stages)
		}
	}
}
