package model

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
)

// This counts the allocations of the actual full-model validator, then proves
// that the admitted transaction avoids exactly one such scan. The paired paths
// retain embedding and journal setup; no global admission probe is installed.
// fak-test:runtime fast est=1s lane=default
func TestV41CohortSingleAdmission(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	t.Run("one scan removed from transaction", func(t *testing.T) {
		st := v41IncrementalState(t, m, []int{1, 2})
		before := captureV41ForwardSnapshot(st)
		var scratch v41ProjScratch
		scan := testing.AllocsPerRun(5, func() {
			if err := m.v41ForwardAdmitted(); err != nil {
				panic(err)
			}
		})
		run := func(admitted bool) {
			var tx *v41StepTransaction
			var err error
			if admitted {
				tx, err = m.beginV41StepAdmitted(3, st, &scratch)
			} else {
				tx, err = m.beginV41Step(3, st, &scratch)
			}
			if err != nil {
				panic(err)
			}
			tx.abort()
		}
		checked := testing.AllocsPerRun(5, func() { run(false) })
		admitted := testing.AllocsPerRun(5, func() { run(true) })
		if scan <= 0 || checked-admitted != scan {
			t.Fatalf("removed validation allocations=%g, want one scan=%g (checked=%g admitted=%g)", checked-admitted, scan, checked, admitted)
		}
		if !reflect.DeepEqual(before, captureV41ForwardSnapshot(st)) {
			t.Fatal("paired transaction setup/abort changed continuation")
		}
		t.Logf("one validation scan=%g allocations; checked=%g admitted=%g", scan, checked, admitted)
	})

	t.Run("real cohort parity and per-dispatch validation", func(t *testing.T) {
		var lanes []v41StepCohortLane
		var refs []*Session
		for i, prefix := range [][]int{{1, 2}, {2, 3}} {
			s := m.NewSession()
			t.Cleanup(s.Close)
			if len(s.Prefill(prefix)) == 0 {
				t.Fatal("prefill returned no logits")
			}
			ref := m.NewSession()
			t.Cleanup(ref.Close)
			ref.v41Forward = captureV41ForwardSnapshot(s.v41State()).clone().restore()
			refs = append(refs, ref)
			lanes = append(lanes, v41StepCohortLane{session: s, id: 4 + i})
		}
		var owner sync.Mutex
		got, err := m.forwardV41StepCohort(&owner, lanes)
		if err != nil {
			t.Fatal(err)
		}
		for i, lane := range lanes {
			want, stats, err := m.forwardV41Step(lane.id, refs[i].v41State(), nil)
			if err != nil || !stats.Committed || !got[i].stats.Committed || !reflect.DeepEqual(got[i].logits, want) || !reflect.DeepEqual(captureV41ForwardSnapshot(lane.session.v41Forward), captureV41ForwardSnapshot(refs[i].v41Forward)) {
				t.Fatalf("cohort lane %d differs from checked serial step: %v", i, err)
			}
		}
		before := captureV41ForwardSnapshot(lanes[0].session.v41Forward)
		name := "model.norm.weight"
		meta := m.manifest[name]
		delete(m.manifest, name)
		defer func() { m.manifest[name] = meta }()
		wantErr := m.v41ForwardAdmitted()
		_, err = m.forwardV41StepCohort(&owner, lanes)
		if wantErr == nil || fmt.Sprint(err) != fmt.Sprint(wantErr) || !reflect.DeepEqual(before, captureV41ForwardSnapshot(lanes[0].session.v41Forward)) {
			t.Fatalf("next dispatch reused stale admission or mutated a lane: %v / %v", err, wantErr)
		}
		if !owner.TryLock() {
			t.Fatal("admission failure retained the device owner")
		}
		owner.Unlock()
	})

	t.Run("unchecked entry keeps failure precedence", func(t *testing.T) {
		st := v41IncrementalState(t, m, []int{1, 2})
		name := "model.norm.weight"
		meta := m.manifest[name]
		delete(m.manifest, name)
		defer func() { m.manifest[name] = meta }()
		wantErr := m.v41ForwardAdmitted()
		tx, err := m.beginV41Step(m.Cfg.VocabSize, st, nil)
		if wantErr == nil || fmt.Sprint(err) != fmt.Sprint(wantErr) || tx == nil || !tx.closed {
			t.Fatalf("unchecked entry lost model-before-token validation: %v / %v", err, wantErr)
		}
		_, nilErr := m.beginV41Step(m.Cfg.VocabSize, nil, nil)
		if !errors.Is(nilErr, ErrV41ForwardStage) || fmt.Sprint(nilErr) == fmt.Sprint(wantErr) {
			t.Fatalf("nil-state failure no longer precedes model validation: %v", nilErr)
		}
	})
}
