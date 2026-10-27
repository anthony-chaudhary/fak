package model

import (
	"errors"
	"os"
	"reflect"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Forward math is production V4.1 math with CPU-reference tensors/KV. Only
// submission and one secondary KV cleanup failure are injected.
type v41EndFaultBackend struct {
	compute.Backend
	beginOK bool
	begins  int
	ends    int
	end     func() error
	clones  int
	onFree  func()
}

type v41EndFaultKV struct {
	compute.KVStore
	backend *v41EndFaultBackend
}

func (b *v41EndFaultBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	return &v41EndFaultKV{KVStore: b.Backend.NewKV(cfg), backend: b}
}

func (kv *v41EndFaultKV) Clone() compute.KVStore {
	kv.backend.clones++
	return kv.KVStore.Clone()
}

func (kv *v41EndFaultKV) Free() {
	if kv.backend.onFree != nil {
		kv.backend.onFree()
	}
	kv.KVStore.Free()
}

func (b *v41EndFaultBackend) BeginV41StepBatch() bool {
	b.begins++
	return b.beginOK
}

func (b *v41EndFaultBackend) EndV41StepBatch() error {
	b.ends++
	if b.end != nil {
		return b.end()
	}
	return nil
}

// fak-test:runtime fast est=1s lane=default
func TestV41PrefillEndFaultRetiresSession(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	t.Setenv(v41SuffixUbatchEnv, "")
	t.Setenv(v41SuffixRefoldMaxHistoryEnv, "")
	t.Setenv(v41ActivationTraceEnvVar, "")
	backend, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("production cpu-ref backend is not registered")
	}
	newSession := func(t *testing.T) (*Session, *v41EndFaultBackend) {
		t.Helper()
		b := &v41EndFaultBackend{Backend: backend, beginOK: true}
		s, err := v41IncrementalPlainModel(t, 1).NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			b.onFree = nil
			s.Close()
		})
		return s, b
	}
	type outcome struct {
		returned bool
		unwound  bool
		payload  any
	}
	invoke := func(f func()) outcome {
		var out outcome
		done := make(chan struct{})
		go func() {
			defer close(done)
			func() {
				defer func() { out.payload = recover() }()
				f()
				out.returned = true
			}()
			// A panic (including legacy nil) returns through the recovery
			// boundary. Goexit does not, and must never become a panic/result.
			out.unwound = true
		}()
		<-done
		return out
	}
	assertRetired := func(t *testing.T, s *Session, b *v41EndFaultBackend, cause error) {
		t.Helper()
		if !s.BackendSessionClosed() || s.halFailure == nil {
			t.Fatalf("failed submission left reusable session: closed=%v failure=%v", s.BackendSessionClosed(), s.halFailure)
		}
		if cause != nil && !errors.Is(s.halFailure, cause) {
			t.Fatalf("sticky failure lost End cause: %v", s.halFailure)
		}
		begins, ends, clones := b.begins, b.ends, b.clones
		for _, retry := range []func(){func() { s.Step(4) }, func() { s.Prefill([]int{4}) }, func() { s.PrefillNoLogits([]int{4}) }} {
			got := invoke(retry)
			if got.returned || !got.unwound || got.payload != s.halFailure {
				t.Fatalf("retry did not return sticky refusal: %+v", got)
			}
		}
		for _, snapshot := range []func() (*PrefixSnapshot, error){s.PrefixSnapshot, s.PrefixSnapshotHostOnly} {
			p, err := snapshot()
			if p != nil {
				p.Close()
			}
			if p != nil || err != s.halFailure {
				t.Fatalf("failed session published prefix or lost sticky cause: prefix=%v err=%v", p != nil, err)
			}
		}
		if b.begins != begins || b.ends != ends || b.clones != clones {
			t.Fatalf("retired session resubmitted/cloned: begin/end/clone %d/%d/%d -> %d/%d/%d", begins, ends, clones, b.begins, b.ends, b.clones)
		}
	}
	for _, mode := range []string{"Prefill", "PrefillNoLogits", "panic", "panic_nil", "goexit", "cleanup_panic"} {
		t.Run(mode, func(t *testing.T) {
			if mode == "panic_nil" {
				t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil=1")
			}
			s, b := newSession(t)
			sentinel := errors.New("End submission sentinel")
			closedBeforeCleanup := false
			if mode == "cleanup_panic" {
				b.onFree = func() {
					closedBeforeCleanup = s.BackendSessionClosed()
					b.onFree = nil
					panic("secondary KV cleanup panic")
				}
			}
			committed := false
			b.end = func() error {
				st := s.v41Forward
				committed = st != nil && reflect.DeepEqual(st.history, []int{1, 2, 3}) && len(st.layers) == 1 && st.layers[0] != nil && st.layers[0].nextWindowPos == 3
				if mode != "PrefillNoLogits" {
					committed = committed && st.forwardCoverage != nil && st.forwardCoverage.Complete
				}
				switch mode {
				case "panic":
					panic(sentinel)
				case "panic_nil":
					panic(nil)
				case "goexit":
					runtime.Goexit()
				}
				return sentinel
			}
			got := invoke(func() {
				if mode == "PrefillNoLogits" {
					s.PrefillNoLogits([]int{1, 2, 3})
				} else {
					s.Prefill([]int{1, 2, 3})
				}
			})
			if !committed || b.begins != 1 || b.ends != 1 || got.returned {
				t.Fatalf("not an End fault after real forward commit: committed=%v begin/end=%d/%d result=%+v", committed, b.begins, b.ends, got)
			}
			if got.unwound != (mode != "goexit") {
				t.Fatalf("Goexit/panic identity changed: %+v", got)
			}
			var cause error
			switch mode {
			case "Prefill", "PrefillNoLogits", "cleanup_panic":
				cause = sentinel
				err, ok := got.payload.(error)
				if !ok || !errors.Is(err, sentinel) {
					t.Fatalf("End error identity changed: %v", got.payload)
				}
			case "panic":
				if got.payload != sentinel {
					t.Fatalf("End panic identity changed: %v", got.payload)
				}
			default:
				if got.payload != nil {
					t.Fatalf("nil panic/Goexit converted: %v", got.payload)
				}
			}
			if mode == "cleanup_panic" && !closedBeforeCleanup {
				t.Fatal("terminal latch was not visible before interrupted KV cleanup")
			}
			assertRetired(t, s, b, cause)
		})
	}
	t.Run("warm second token", func(t *testing.T) {
		s, b := newSession(t)
		s.Prefill([]int{1, 2, 3})
		prefix, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		defer prefix.Close()
		b.begins, b.ends = 0, 0
		var histories [][]int
		sentinel := errors.New("second suffix End sentinel")
		b.end = func() error {
			histories = append(histories, append([]int(nil), s.v41Forward.history...))
			if b.ends == 2 {
				return sentinel
			}
			return nil
		}
		got := invoke(func() { s.Prefill([]int{4, 5, 6}) })
		err, ok := got.payload.(error)
		if got.returned || !ok || !errors.Is(err, sentinel) || !reflect.DeepEqual(histories, [][]int{{1, 2, 3, 4}, {1, 2, 3, 4, 5}}) {
			t.Fatalf("warm per-token commits/rejection changed: histories=%v result=%+v", histories, got)
		}
		if !reflect.DeepEqual(prefix.v41.history, []int{1, 2, 3}) {
			t.Fatal("failed suffix mutated independently owned prefix")
		}
		assertRetired(t, s, b, sentinel)
	})
	t.Run("warm ubatch", func(t *testing.T) {
		s, b := newSession(t)
		s.Prefill([]int{1, 2, 3})
		t.Setenv(v41SuffixUbatchEnv, "2")
		b.begins, b.ends = 0, 0
		sentinel := errors.New("suffix ubatch End sentinel")
		committed := false
		b.end = func() error {
			committed = reflect.DeepEqual(s.v41Forward.history, []int{1, 2, 3, 4, 5, 6})
			return sentinel
		}
		got := invoke(func() { s.Prefill([]int{4, 5, 6}) })
		err, ok := got.payload.(error)
		if got.returned || !ok || !errors.Is(err, sentinel) || !committed || b.begins != 1 || b.ends != 1 {
			t.Fatalf("not a committed ubatch End fault: committed=%v begin/end=%d/%d result=%+v", committed, b.begins, b.ends, got)
		}
		assertRetired(t, s, b, sentinel)
	})
	for _, mode := range []string{"success", "declined", "disabled"} {
		t.Run(mode, func(t *testing.T) {
			s, b := newSession(t)
			b.beginOK = mode != "declined"
			if mode == "disabled" {
				t.Setenv(v41StepBatchEnv, "0")
			}
			s.PrefillNoLogits([]int{1, 2, 3})
			if len(s.Step(4)) != s.M.Cfg.VocabSize || s.BackendSessionClosed() || !reflect.DeepEqual(s.v41Forward.history, []int{1, 2, 3, 4}) {
				t.Fatal("healthy session did not remain usable")
			}
			wantBegin, wantEnd := 2, 2
			if mode == "declined" {
				wantEnd = 0
			} else if mode == "disabled" {
				wantBegin, wantEnd = 0, 0
			}
			if b.begins != wantBegin || b.ends != wantEnd {
				t.Fatalf("begin/end=%d/%d want=%d/%d", b.begins, b.ends, wantBegin, wantEnd)
			}
			p, err := s.PrefixSnapshot()
			if err != nil || p == nil || p.Tokens != 4 {
				t.Fatalf("healthy prefix: %v %v", p, err)
			}
			p.Close()
			s.Close()
			for _, snapshot := range []func() (*PrefixSnapshot, error){s.PrefixSnapshot, s.PrefixSnapshotHostOnly} {
				p, err := snapshot()
				if p != nil || err == nil {
					t.Fatalf("ordinary closed session snapshot: %v %v", p, err)
				}
			}
		})
	}
	t.Run("structural refusal still refolds", func(t *testing.T) {
		s, b := newSession(t)
		s.Prefill([]int{1, 2, 3})
		s.v41Forward.layerState(0).nextWindowPos += 5
		b.begins, b.ends = 0, 0
		if len(s.Prefill([]int{4, 5})) != s.M.Cfg.VocabSize || s.BackendSessionClosed() {
			t.Fatal("successful End poisoned a recoverable structural refusal")
		}
		if !reflect.DeepEqual(s.v41Forward.history, []int{1, 2, 3, 4, 5}) || b.begins != 2 || b.ends != 2 {
			t.Fatalf("structural refold changed: history=%v begin/end=%d/%d", s.v41Forward.history, b.begins, b.ends)
		}
	})
}
