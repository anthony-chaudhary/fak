package model

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

func TestIssue8833SelectorDefaultAndParser(t *testing.T) {
	t.Setenv(mixedQKVEnv, "")
	if got := newMixedQKVSessionFromEnv().mode; got != mixedQKVOff {
		t.Fatalf("default=%v", got)
	}
	for value, want := range map[string]mixedQKVMode{"control": mixedQKVControl, " candidate ": mixedQKVCandidate} {
		got, err := parseMixedQKVMode(value)
		if err != nil || got != want {
			t.Fatalf("parse %q=(%v,%v), want %v", value, got, err, want)
		}
	}
	if _, err := parseMixedQKVMode("on"); err == nil {
		t.Fatal("invalid selector accepted")
	}
}

func issue8833Model() *Model {
	cfg := Config{HiddenSize: 4096, NumHeads: 16, NumKVHeads: 4, HeadDim: 256, AttnOutputGate: true,
		LayerTypes: []string{"linear_attention", "full_attention"}}
	return &Model{Cfg: cfg, q8w: map[string]*q8Tensor{"q": {out: 8192, in: 4096}, "k": {out: 1024, in: 4096}},
		q4kw: map[string]*q4kTensor{"v": {out: 1024, in: 4096}}}
}

func TestIssue8833GeometryFamilyAndDeclineGating(t *testing.T) {
	m := issue8833Model()
	var calls atomic.Int32
	s := &Session{M: m, mixedQKV: mixedQKVSession{mode: mixedQKVCandidate, exec: func(mixedQKVRequest) mixedQKVResult {
		calls.Add(1)
		return mixedQKVResult{Err: errors.New("declined")}
	}}}
	if out, handled, err := s.tryMixedQKV(nil, "q", "k", "v", nil, make([]float32, 4096), 8192, 1024, 1024); err != nil || handled || out != nil {
		t.Fatalf("decline=(%v,%v,%v)", out, handled, err)
	}
	if calls.Load() != 1 {
		t.Fatalf("calls=%d", calls.Load())
	}
	m.Cfg.HiddenSize--
	_, _, _ = s.tryMixedQKV(nil, "q", "k", "v", nil, nil, 8192, 1024, 1024)
	if calls.Load() != 1 {
		t.Fatal("wrong geometry dispatched")
	}
	m.Cfg.HiddenSize++
	delete(m.q4kw, "v")
	_, _, _ = s.tryMixedQKV(nil, "q", "k", "v", nil, nil, 8192, 1024, 1024)
	if calls.Load() != 1 {
		t.Fatal("wrong quant family dispatched")
	}
}

func TestIssue8833PostSubmitNeverFallsBack(t *testing.T) {
	s := &Session{M: issue8833Model(), mixedQKV: mixedQKVSession{mode: mixedQKVCandidate, exec: func(mixedQKVRequest) mixedQKVResult {
		return mixedQKVResult{Submitted: true, Err: errMixedQKVPostSubmit}
	}}}
	_, handled, err := s.tryMixedQKV(nil, "q", "k", "v", nil, nil, 8192, 1024, 1024)
	if !handled || !errors.Is(err, errMixedQKVPostSubmit) {
		t.Fatalf("handled=%v err=%v", handled, err)
	}
}

func TestIssue8833ConcurrentStateAndCloseZeroOwnership(t *testing.T) {
	sessions := make([]*Session, 32)
	var wg sync.WaitGroup
	for i := range sessions {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sessions[i] = &Session{mixedQKV: mixedQKVSession{mode: mixedQKVCandidate, exec: func(mixedQKVRequest) mixedQKVResult { return mixedQKVResult{} }}}
		}(i)
	}
	wg.Wait()
	for _, s := range sessions {
		s.Close()
		if s.mixedQKV.mode != mixedQKVOff || s.mixedQKV.exec != nil {
			t.Fatal("Close retained mixed-QKV ownership")
		}
	}
}
