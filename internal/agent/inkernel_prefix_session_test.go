package agent

import (
	"slices"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestOwnedPrefixSessionRestoreParityAndConsumption(t *testing.T) {
	p := reusePlanner(false, false, tinyCfg())
	source := p.m.NewSession()
	defer source.Close()
	source.Prefill([]int{3, 7, 11})
	snapshot, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	restored, err := p.restoreOwnedPrefixSnapshot(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	if snapshot.Cache != nil {
		t.Fatal("snapshot retained transferred cache")
	}
	want := append([]float32(nil), source.Step(17)...)
	got := restored.Step(17)
	if !slices.Equal(got, want) {
		t.Fatal("restored continuation differs from source")
	}
	duplicate, err := p.restoreOwnedPrefixSnapshot(snapshot)
	if err == nil || duplicate != nil {
		if duplicate != nil {
			duplicate.Close()
		}
		t.Fatalf("consumed snapshot returned session=%v error=%v", duplicate, err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestOwnedPrefixSessionClosedSnapshotRefuses(t *testing.T) {
	p := reusePlanner(false, false, tinyCfg())
	source := p.m.NewSession()
	defer source.Close()
	source.Prefill([]int{3, 7})
	snapshot, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	snapshot.Close()
	restored, err := p.restoreOwnedPrefixSnapshot(snapshot)
	if err == nil || restored != nil {
		if restored != nil {
			restored.Close()
		}
		t.Fatalf("closed snapshot returned session=%v error=%v", restored, err)
	}
}
