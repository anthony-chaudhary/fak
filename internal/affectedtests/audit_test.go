package affectedtests

import (
	"reflect"
	"testing"
)

func TestAuditSelectionReportsUnselectedFailure(t *testing.T) {
	selected := TestObservation{
		Complete: true,
		Packages: []PackageObservation{{Package: "internal/selected"}},
	}
	full := TestObservation{
		Complete: true,
		Packages: []PackageObservation{
			{Package: "internal/selected"},
			{Package: "internal/unselected", Failed: true},
		},
	}

	got := AuditSelection(selected, full)
	if got.Sound {
		t.Fatal("AuditSelection() Sound = true, want false")
	}
	if !got.Complete {
		t.Fatal("AuditSelection() Complete = false, want true")
	}
	if want := []string{"internal/unselected"}; !reflect.DeepEqual(got.SelectorMisses, want) {
		t.Fatalf("AuditSelection() SelectorMisses = %v, want %v", got.SelectorMisses, want)
	}
}

func TestAuditSelectionAcceptsSameObservedFailures(t *testing.T) {
	selected := TestObservation{
		Complete: true,
		Packages: []PackageObservation{{Package: "internal/failing", Failed: true}},
	}
	full := TestObservation{
		Complete: true,
		Packages: []PackageObservation{{Package: "internal/failing", Failed: true}},
	}

	got := AuditSelection(selected, full)
	if !got.Sound {
		t.Fatalf("AuditSelection() Sound = false, misses %v", got.SelectorMisses)
	}
	if len(got.SelectorMisses) != 0 {
		t.Fatalf("AuditSelection() SelectorMisses = %v, want empty", got.SelectorMisses)
	}
}

func TestAuditSelectionFailsClosedOnIncompleteObservation(t *testing.T) {
	tests := []struct {
		name     string
		selected TestObservation
		full     TestObservation
	}{
		{
			name:     "selected incomplete",
			selected: TestObservation{Complete: false},
			full:     TestObservation{Complete: true},
		},
		{
			name:     "full incomplete",
			selected: TestObservation{Complete: true},
			full:     TestObservation{Complete: false},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AuditSelection(tt.selected, tt.full)
			if got.Complete {
				t.Fatal("AuditSelection() Complete = true, want false")
			}
			if got.Sound {
				t.Fatal("AuditSelection() Sound = true, want false")
			}
		})
	}
}

func TestAuditSelectionSortsAndDeduplicatesFailures(t *testing.T) {
	selected := TestObservation{
		Complete: true,
		Packages: []PackageObservation{
			{Package: "z/pkg", Failed: true},
			{Package: "a/pkg", Failed: true},
			{Package: "z/pkg", Failed: true},
		},
	}
	full := TestObservation{
		Complete: true,
		Packages: []PackageObservation{
			{Package: "m/pkg", Failed: true},
			{Package: "a/pkg", Failed: true},
			{Package: "b/pkg", Failed: true},
			{Package: "m/pkg", Failed: true},
		},
	}

	got := AuditSelection(selected, full)
	if want := []string{"a/pkg", "z/pkg"}; !reflect.DeepEqual(got.SelectedFailures, want) {
		t.Fatalf("AuditSelection() SelectedFailures = %v, want %v", got.SelectedFailures, want)
	}
	if want := []string{"a/pkg", "b/pkg", "m/pkg"}; !reflect.DeepEqual(got.FullFailures, want) {
		t.Fatalf("AuditSelection() FullFailures = %v, want %v", got.FullFailures, want)
	}
	if want := []string{"b/pkg", "m/pkg"}; !reflect.DeepEqual(got.SelectorMisses, want) {
		t.Fatalf("AuditSelection() SelectorMisses = %v, want %v", got.SelectorMisses, want)
	}
}
