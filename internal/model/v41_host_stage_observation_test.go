package model

import (
	"errors"
	"reflect"
	"sync"
	"testing"
)

func TestV41HostStageSuccessfulDenseOnly(t *testing.T) {
	m := v41DenseTestFixture(t, false, false)
	s := &Session{M: m, hostPathBypass: "device-only-session"}
	shape := v41ProjectionShape(t, m, "attn.wq_a.weight")
	panel := make([]float32, shape[1])
	for i := range panel {
		panel[i] = float32(i+1) / 16
	}
	want, err := m.v41ProjectionRows(0, "attn.wq_a.weight", panel, shape[0], shape[1], 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	f := s.v41ObserveDenseHost(nil)
	for i := 0; i < 2; i++ {
		got, outcome, err := f(0, "attn.wq_a.weight", panel, shape[0], shape[1], 1)
		if err != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v/%v/%v want %v", got, outcome, err, want)
		}
	}
	got := s.HostStages()
	if !reflect.DeepEqual(got, []string{v41DenseHostStage}) {
		t.Fatal(got)
	}
	got[0] = "mutated"
	if s.HostStages()[0] != v41DenseHostStage {
		t.Fatal("observation aliases caller")
	}
	if len((&Session{}).HostStages()) != 0 || s.HostFallbackObserved() {
		t.Fatal("observation leaked or became refusal")
	}
}

func TestV41HostStageNoSelectedReplay(t *testing.T) {
	for _, outcome := range []v41DenseProjectionOutcome{v41ProjectionHandled, v41ProjectionError} {
		s := &Session{hostPathBypass: "device-only-session"} // nil model proves no host dereference.
		cause := errors.New("selected failure")
		calls := 0
		f := s.v41ObserveDenseHost(func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			calls++
			return []float32{7}, outcome, cause
		})
		got, actual, err := f(0, "x", nil, 1, 1, 1)
		if calls != 1 || actual != outcome || err != cause || len(got) != 1 || len(s.HostStages()) != 0 {
			t.Fatal("selected outcome changed or host counted")
		}
	}
	s := &Session{}
	_, outcome, err := s.v41ObserveDenseHost(nil)(0, "x", nil, 1, 1, 1)
	if outcome != v41ProjectionDeclined || err != nil || len(s.HostStages()) != 0 {
		t.Fatal("unadmitted session observed")
	}
}

func TestV41HostStageFailedAdmissionNotObserved(t *testing.T) {
	s := &Session{M: &Model{}, hostPathBypass: "device-only-session"}
	_, outcome, err := s.v41ObserveDenseHost(nil)(0, "missing", nil, 0, 0, 0)
	if outcome != v41ProjectionError || err == nil || len(s.HostStages()) != 0 {
		t.Fatal("failed admission counted as execution")
	}
}

func TestV41HostStageConcurrentSnapshot(t *testing.T) {
	s := &Session{}
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); s.noteV41DenseHostSuccess(); _ = s.HostStages() }()
	}
	wg.Wait()
	if !reflect.DeepEqual(s.HostStages(), []string{v41DenseHostStage}) {
		t.Fatal(s.HostStages())
	}
}
