package model

import (
	"fmt"
	"reflect"
	"testing"
)

// The actual checked and admitted full-step drivers consume identical restored
// continuations. Their allocation delta must equal one real validation scan.
// fak-test:runtime fast est=1s lane=default
func TestV41StepSingleAdmission(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	t.Run("one scan removed from full step", func(t *testing.T) {
		seed := captureV41ForwardSnapshot(v41IncrementalState(t, m, []int{1, 2}))
		scan := testing.AllocsPerRun(5, func() {
			if err := m.v41ForwardAdmitted(); err != nil {
				panic(err)
			}
		})
		run := func(admitted bool) ([]float32, *v41ForwardSnapshot) {
			st := seed.clone().restore()
			var logits []float32
			var stats v41StepStats
			var err error
			if admitted {
				logits, stats, err = m.forwardV41StepAdmitted(3, st, nil)
			} else {
				logits, stats, err = m.forwardV41Step(3, st, nil)
			}
			if err != nil || !stats.Committed || len(logits) == 0 {
				panic(fmt.Sprintf("full-step fixture did not commit: %v", err))
			}
			return logits, captureV41ForwardSnapshot(st)
		}
		checked := testing.AllocsPerRun(5, func() { _, _ = run(false) })
		admitted := testing.AllocsPerRun(5, func() { _, _ = run(true) })
		if scan <= 0 || checked-admitted != scan {
			t.Fatalf("removed validation allocations=%g, want one scan=%g (checked=%g admitted=%g)", checked-admitted, scan, checked, admitted)
		}
		want, wantState := run(false)
		got, gotState := run(true)
		if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(gotState, wantState) {
			t.Fatal("admitted full step differs from checked continuation")
		}
		t.Logf("one validation scan=%g allocations; checked=%g admitted=%g", scan, checked, admitted)
	})

	t.Run("public step still validates every dispatch", func(t *testing.T) {
		s := m.NewSession()
		t.Cleanup(s.Close)
		if len(s.Prefill([]int{1, 2})) == 0 {
			t.Fatal("prefill returned no logits")
		}
		ref := captureV41ForwardSnapshot(s.v41State()).clone().restore()
		want, stats, err := m.forwardV41Step(3, ref, nil)
		if err != nil || !stats.Committed {
			t.Fatal(err)
		}
		got := s.Step(3)
		if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(ref)) {
			t.Fatal("public Step differs from checked serial continuation")
		}
		before := captureV41ForwardSnapshot(s.v41Forward)
		name := "model.norm.weight"
		meta := m.manifest[name]
		delete(m.manifest, name)
		defer func() { m.manifest[name] = meta }()
		wantErr := m.v41ForwardAdmitted()
		var primary any
		func() {
			defer func() { primary = recover() }()
			s.Step(4)
		}()
		if wantErr == nil || fmt.Sprint(primary) != fmt.Sprint(wantErr) || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
			t.Fatalf("public Step reused stale admission or mutated continuation: %v / %v", primary, wantErr)
		}
		_, _, err = m.forwardV41Step(m.Cfg.VocabSize, ref, nil)
		if fmt.Sprint(err) != fmt.Sprint(wantErr) {
			t.Fatalf("standalone step no longer checks model before token: %v / %v", err, wantErr)
		}
	})
}
