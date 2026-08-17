package deliverystages

import (
	"reflect"
	"strings"
	"testing"
)

func TestDefaultRegistryIsCompleteAndValid(t *testing.T) {
	r := Default()
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	if len(r.Stages) != 21 {
		t.Fatalf("stages = %d, want 21", len(r.Stages))
	}
	if len(r.Bottlenecks) != 14 {
		t.Fatalf("bottlenecks = %d, want 14", len(r.Bottlenecks))
	}
	for _, id := range []StageID{"intent", "issue-contract", "value-centrality", "scope", "dependency-readiness", "capacity-admission", "lane-admission", "context-acquisition", "authoring", "recording", "compile-admission", "build", "static-analysis", "affected-tests", "full-tests", "evidence", "integration", "release-admission", "release-publication", "runtime-observation", "closure"} {
		if _, ok := r.Stage(id); !ok {
			t.Errorf("missing stage %q", id)
		}
	}
}

func TestRecordingDoesNotMeanCompileAdmission(t *testing.T) {
	r := Default()
	recording, _ := r.Stage("recording")
	if reflect.DeepEqual(recording, mustStage(t, r, "compile-admission")) {
		t.Fatal("recording and compile admission collapsed")
	}
	if !contains(recording.Invalidates, "compile-admission") {
		t.Fatalf("recording must invalidate downstream compile admission: %v", recording.Invalidates)
	}
	if contains(recording.Prerequisites, "compile-admission") {
		t.Fatal("recording depends on compile admission")
	}
}

func TestDependencyAwareInvalidationIsExactSuffix(t *testing.T) {
	r := Default()
	got, err := r.InvalidatedAfter("build")
	if err != nil {
		t.Fatal(err)
	}
	want := []StageID{"static-analysis", "affected-tests", "full-tests", "evidence", "integration", "release-admission", "release-publication", "runtime-observation", "closure"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("invalidated = %v, want %v", got, want)
	}
}

func TestCrosswalkAvoidsAggregateCIRed(t *testing.T) {
	r := Default()
	cases := map[string]struct {
		stage      StageID
		bottleneck BottleneckID
	}{
		"go build": {"build", "compile"}, "dos arbitrate": {"lane-admission", "collision-lease"},
		"REFUSE_AT_CAP": {"capacity-admission", "capacity-seat"}, "pre-push": {"integration", "integration-drift"},
		"shipgate": {"release-admission", "release-policy"}, "CI red": {"full-tests", "unknown-irreducible"},
	}
	for local, want := range cases {
		got, ok := r.ResolveLocal(local)
		if !ok || got.Stage != want.stage || got.Bottleneck != want.bottleneck {
			t.Errorf("%q = %#v, %v", local, got, ok)
		}
	}
}

func TestRegistryRejectsMalformedGraphs(t *testing.T) {
	t.Run("duplicate", func(t *testing.T) {
		r := Default()
		r.Stages = append(r.Stages, r.Stages[0])
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "duplicate") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("dangling", func(t *testing.T) {
		r := Default()
		r.Stages[0].Prerequisites = []StageID{"missing"}
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "unknown") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("cycle", func(t *testing.T) {
		r := Default()
		r.Stages[0].Prerequisites = []StageID{"closure"}
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "cycle") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("no split", func(t *testing.T) {
		r := Default()
		r.Stages[0].SplitDimensions = nil
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "retry/split") {
			t.Fatalf("err=%v", err)
		}
	})
}

func mustStage(t *testing.T, r Registry, id StageID) Stage {
	t.Helper()
	stage, ok := r.Stage(id)
	if !ok {
		t.Fatal(id)
	}
	return stage
}
func contains(values []StageID, want StageID) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
