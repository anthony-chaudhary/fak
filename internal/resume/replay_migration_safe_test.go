package resume

import (
	"reflect"
	"testing"
)

// TestReplayMigrationSafePlainAllowed: a fully-read plain single-sequence free-form
// turn is the ONLY replay-migratable shape — a token replay reconstructs its state.
func TestReplayMigrationSafePlainAllowed(t *testing.T) {
	v := ReplayMigrationSafe(ReplayShape{ShapeKnown: true, Sequences: 1})
	if !v.ReplaySafe {
		t.Fatalf("plain single-sequence request: want ReplaySafe, got refused with %v", v.Reasons)
	}
	if len(v.Reasons) != 0 {
		t.Fatalf("allowed verdict must carry no reasons, got %v", v.Reasons)
	}
}

// TestReplayMigrationSafeStructuredRefused: a constrained/guided decode request is
// refused — the grammar FSM would restart from its schema root on a fresh worker.
func TestReplayMigrationSafeStructuredRefused(t *testing.T) {
	v := ReplayMigrationSafe(ReplayShape{ShapeKnown: true, Sequences: 1, Structured: true})
	if v.ReplaySafe {
		t.Fatal("structured-decode request must be refused")
	}
	if !hasReason(v.Reasons, ReplayReasonStructuredDecode) {
		t.Fatalf("want reason %q, got %v", ReplayReasonStructuredDecode, v.Reasons)
	}
}

// TestReplayMigrationSafeMultiSequenceRefused: n>1 is refused — per-sequence sampler
// state is not reconstructable from the delivered token stream alone.
func TestReplayMigrationSafeMultiSequenceRefused(t *testing.T) {
	v := ReplayMigrationSafe(ReplayShape{ShapeKnown: true, Sequences: 3})
	if v.ReplaySafe {
		t.Fatal("n>1 request must be refused")
	}
	if !hasReason(v.Reasons, ReplayReasonMultiSequence) {
		t.Fatalf("want reason %q, got %v", ReplayReasonMultiSequence, v.Reasons)
	}
}

// TestReplayMigrationSafeCombinedAllReasons: a request that trips several traits at
// once carries every reason, in the fixed sorted order.
func TestReplayMigrationSafeCombinedAllReasons(t *testing.T) {
	v := ReplayMigrationSafe(ReplayShape{ShapeKnown: true, Sequences: 4, Structured: true})
	if v.ReplaySafe {
		t.Fatal("structured + n>1 request must be refused")
	}
	want := []string{ReplayReasonMultiSequence, ReplayReasonStructuredDecode} // sorted
	if !reflect.DeepEqual(v.Reasons, want) {
		t.Fatalf("want all reasons sorted %v, got %v", want, v.Reasons)
	}
}

// TestReplayMigrationSafeUnknownShapeFailsClosed: the zero value (shape unread) and a
// degenerate non-positive sequence count both refuse fail-closed.
func TestReplayMigrationSafeUnknownShapeFailsClosed(t *testing.T) {
	cases := []struct {
		name  string
		shape ReplayShape
	}{
		{"zero value / shape unread", ReplayShape{}},
		{"shape unread but sequences set", ReplayShape{Sequences: 1}},
		{"degenerate zero sequence count", ReplayShape{ShapeKnown: true, Sequences: 0}},
		{"degenerate negative sequence count", ReplayShape{ShapeKnown: true, Sequences: -2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := ReplayMigrationSafe(tc.shape)
			if v.ReplaySafe {
				t.Fatalf("%s must refuse fail-closed", tc.name)
			}
			if !hasReason(v.Reasons, ReplayReasonUnknownShape) {
				t.Fatalf("want reason %q, got %v", ReplayReasonUnknownShape, v.Reasons)
			}
		})
	}
}

// TestReplayMigrationSafeDeterministic: same shape yields the same verdict every call.
func TestReplayMigrationSafeDeterministic(t *testing.T) {
	shape := ReplayShape{ShapeKnown: true, Sequences: 5, Structured: true}
	first := ReplayMigrationSafe(shape)
	for i := 0; i < 8; i++ {
		if got := ReplayMigrationSafe(shape); !reflect.DeepEqual(got, first) {
			t.Fatalf("call %d diverged: %v vs %v", i, got, first)
		}
	}
}

func hasReason(reasons []string, want string) bool {
	for _, r := range reasons {
		if r == want {
			return true
		}
	}
	return false
}
