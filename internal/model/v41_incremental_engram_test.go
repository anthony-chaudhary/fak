package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=15s lane=default
func TestV41IncrementalEngramSessionParity(t *testing.T) {
	for _, compressed := range []bool{false, true} {
		for _, n := range []int{1, 2, 3, 4, 7, 32, 120} {
			t.Run(fmt.Sprintf("compressed=%t/prefix=%d", compressed, n), func(t *testing.T) {
				t.Parallel()
				m, _ := v41IncrementalEngramFixture(t, compressed)
				oracle, _ := v41IncrementalEngramFixture(t, compressed)
				prefix, next := v41IncrementalPrefix(m.Cfg, n)
				s := m.NewSession()
				assertV41RowsClose(t, "cold prefill", s.Prefill(prefix), v41IncrementalEngramCold(t, oracle, prefix), 1e-6)
				if !s.v41IncrementalEligible() {
					t.Fatal("prepared Engram session is not incrementally eligible")
				}
				history := append([]int(nil), prefix...)
				for _, tokens := range [][]int{{next}, {2, 5, 1}, {6}} {
					before := v41IncrementalEngramRequests(m)
					var got []float32
					if len(tokens) == 1 {
						got = s.Step(tokens[0])
					} else {
						got = s.Prefill(tokens)
					}
					history = append(history, tokens...)
					assertV41RowsClose(t, "incremental Engram", got, v41IncrementalEngramCold(t, oracle, history), 1e-6)
					if delta := v41IncrementalEngramRequests(m) - before; delta != int64(48*len(tokens)) {
						t.Fatalf("Engram requests=%d want=%d current columns only", delta, 48*len(tokens))
					}
					if !reflect.DeepEqual(s.v41Forward.history, history) {
						t.Fatal("session history differs from committed tokens")
					}
					for l, pos := range v41AllNextWindowPos(s.v41Forward) {
						if pos != len(history) {
							t.Fatalf("layer %d cursor=%d history=%d", l, pos, len(history))
						}
					}
				}
			})
		}
	}
}

// fak-test:justify why=invariant when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41IncrementalEngramFixtureIsNonzero(t *testing.T) {
	t.Parallel()
	m, _ := v41IncrementalEngramFixture(t, false)
	plain, _ := v41IncrementalEngramFixture(t, false)
	plain.Cfg.DeepSeekV41.EngramLayerIDs = nil
	ids := []int{1, 3, 2, 5, 7}
	with := v41IncrementalEngramCold(t, m, ids)
	without := v41IncrementalEngramCold(t, plain, ids)
	if v41RowsAgree(with, without, 1e-6) {
		t.Fatal("nonzero Engram fixture does not change decoded logits")
	}
	stage := m.v41EngramStageFor()
	a, err := NewV41EngramHashState(stage.layout)
	if err != nil {
		t.Fatal(err)
	}
	first, err := a.Hash([]int{1, 2, 3, 4}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewV41EngramHashState(stage.layout)
	if err != nil {
		t.Fatal(err)
	}
	second, err := b.Hash([]int{7, 6, 5, 4}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(first[len(first)-48:], second[len(second)-48:]) {
		t.Fatal("fixture rows do not depend on preceding token IDs")
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=8s lane=default
func TestV41IncrementalEngramRollbackRetry(t *testing.T) {
	for _, fault := range []string{"gather", "malformed", "later-layer", "later-expert", "final-norm"} {
		t.Run(fault, func(t *testing.T) {
			t.Parallel()
			m, sources := v41IncrementalEngramFixture(t, true)
			oracle, _ := v41IncrementalEngramFixture(t, true)
			prefix := []int{1, 3, 2}
			s := m.NewSession()
			s.Prefill(prefix)
			st := s.v41Forward
			before := captureV41ForwardSnapshot(st)
			beforeRequests := v41IncrementalEngramRequests(m)
			injectedAfterAdmission := false
			var undo func()
			switch fault {
			case "gather":
				sources[1].fail = true
				undo = func() { sources[1].fail = false }
			case "malformed":
				sources[1].malformed = true
				undo = func() { sources[1].malformed = false }
			case "later-layer":
				st.layerState(2).nextWindowPos += 5
				undo = func() { st.layerState(2).nextWindowPos = len(prefix) }
			case "later-expert":
				name := layerName(2, "ffn.shared_experts.w2.weight")
				meta := m.manifest[name]
				sources[1].onRead = func() {
					delete(m.manifest, name)
					injectedAfterAdmission = true
				}
				undo = func() { sources[1].onRead = nil; m.manifest[name] = meta }
			case "final-norm":
				meta := m.manifest["model.norm.weight"]
				sources[1].onRead = func() {
					delete(m.manifest, "model.norm.weight")
					injectedAfterAdmission = true
				}
				undo = func() { sources[1].onRead = nil; m.manifest["model.norm.weight"] = meta }
			}
			_, stats, err := m.forwardV41Step(5, st, &v41ProjScratch{})
			undo()
			if err == nil {
				t.Fatal("injected incremental fault returned success")
			}
			if v41IncrementalEngramRequests(m) <= beforeRequests {
				t.Fatal("fault did not reach the incremental Engram gather")
			}
			if (fault == "later-expert" || fault == "final-norm") && !injectedAfterAdmission {
				t.Fatal("consumer failure was not injected by a post-admission Engram row read")
			}
			if fault == "gather" || fault == "malformed" {
				var typed *V41ForwardError
				if !errors.As(err, &typed) || !errors.Is(err, ErrV41NativeUnsupported) || typed.Stage != v41StageEngram || typed.Layer != 2 {
					t.Fatalf("Engram fault is not a typed native refusal: %v", err)
				}
			} else if !errors.Is(err, ErrV41ForwardStage) {
				t.Fatalf("fault lacks forward-stage identity: %v", err)
			}
			if stats.Committed || !reflect.DeepEqual(before, captureV41ForwardSnapshot(st)) {
				t.Fatal("failed token changed the complete committed continuation snapshot")
			}
			got, stats, err := m.forwardV41Step(5, st, &v41ProjScratch{})
			if err != nil || !stats.Committed {
				t.Fatalf("retry err=%v committed=%t", err, stats.Committed)
			}
			assertV41RowsClose(t, "retry", got, v41IncrementalEngramCold(t, oracle, append(prefix, 5)), 1e-6)
			got, _, err = m.forwardV41Step(6, st, &v41ProjScratch{})
			if err != nil {
				t.Fatal(err)
			}
			assertV41RowsClose(t, "after retry", got, v41IncrementalEngramCold(t, oracle, append(prefix, 5, 6)), 1e-6)
		})
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=4s lane=default
func TestV41IncrementalEngramForkIsolation(t *testing.T) {
	t.Parallel()
	m, _ := v41IncrementalEngramFixture(t, true)
	oracle, _ := v41IncrementalEngramFixture(t, true)
	s := m.NewSession()
	prefix := []int{1, 3, 2}
	s.Prefill(prefix)
	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	clone, err := snap.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	a, b := m.NewSession(), m.NewSession()
	if err := snap.Restore(a); err != nil {
		t.Fatal(err)
	}
	if err := clone.Restore(b); err != nil {
		t.Fatal(err)
	}
	for _, turn := range []struct {
		s    *Session
		tail []int
	}{{a, []int{5}}, {b, []int{7}}, {s, []int{4}}, {a, []int{5, 6}}, {b, []int{7, 1}}, {s, []int{4, 2}}} {
		got := turn.s.Step(turn.tail[len(turn.tail)-1])
		history := append(append([]int(nil), prefix...), turn.tail...)
		assertV41RowsClose(t, "independent branch", got, v41IncrementalEngramCold(t, oracle, history), 1e-6)
		if !reflect.DeepEqual(turn.s.v41Forward.history, history) {
			t.Fatal("sibling branch modified history")
		}
	}
}

func v41IncrementalEngramPhaseJSON(t *testing.T, m *Model) map[string]map[string]json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func v41IncrementalEngramJSONInt(t *testing.T, fields map[string]json.RawMessage, key string) int64 {
	t.Helper()
	var n int64
	if raw, ok := fields[key]; !ok {
		t.Fatalf("default phase JSON omits %s", key)
	} else if err := json.Unmarshal(raw, &n); err != nil {
		t.Fatal(err)
	}
	return n
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=4s lane=default
func TestV41IncrementalEngramDefaultPhaseAttribution(t *testing.T) {
	t.Parallel()
	m, _ := v41IncrementalEngramFixture(t, false)
	s := m.NewSession()
	s.Prefill([]int{1, 3, 2})
	s.Step(5)
	s.Prefill([]int{6, 1, 4})
	s.Step(7)
	fields := v41IncrementalEngramPhaseJSON(t, m)
	for phase, tokens := range map[string]int64{"prefill": 3, "decode": 2} {
		if got := v41IncrementalEngramJSONInt(t, fields[phase], "incremental_engram_injections"); got != tokens*2 {
			t.Errorf("%s injections=%d want=%d", phase, got, tokens*2)
		}
		if got := v41IncrementalEngramJSONInt(t, fields[phase], "incremental_engram_rows"); got != tokens*48 {
			t.Errorf("%s rows=%d want=%d", phase, got, tokens*48)
		}
		if got := v41IncrementalEngramJSONInt(t, fields[phase], "incremental_engram_hash_tokens"); got <= 0 || got > tokens*4 {
			t.Errorf("%s hash processed tokens=%d bound=(0,%d]", phase, got, tokens*4)
		}
		if got := v41IncrementalEngramJSONInt(t, fields[phase], "incremental_engram_nanos"); got <= 0 {
			t.Errorf("%s duration absent", phase)
		}
	}
	plain, _ := v41IncrementalEngramFixture(t, false)
	plain.Cfg.DeepSeekV41.EngramLayerIDs = nil
	p := plain.NewSession()
	p.Prefill([]int{1, 3, 2})
	p.Step(5)
	for _, phase := range v41IncrementalEngramPhaseJSON(t, plain) {
		for _, key := range []string{"incremental_engram_injections", "incremental_engram_rows", "incremental_engram_hash_tokens", "incremental_engram_nanos"} {
			if got := v41IncrementalEngramJSONInt(t, phase, key); got != 0 {
				t.Errorf("no-Engram %s=%d", key, got)
			}
		}
	}
}
