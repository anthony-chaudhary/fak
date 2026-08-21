package laneadmit

import (
	"fmt"
	"testing"
)

func TestCompareLocalKeepsLockAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{"fak native lane and tree admission": {"native", true}, "geometry-only tree overlap": {"baseline", true}, "DOS arbitrate": {"integration", false}, "GitHub Actions concurrency groups": {"external", false}, "Kubernetes Lease coordination": {"external", false}, "etcd concurrency mutex": {"external", false}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d want %d: %#v", len(got.Arms), len(want), got.Arms)
	}
	for _, arm := range got.Arms {
		expected, ok := want[arm.Name]
		if !ok {
			t.Fatalf("unexpected arm %q", arm.Name)
		}
		if arm.Kind != expected.kind || arm.Available != expected.available {
			t.Errorf("arm %q=%q available=%v want %q/%v", arm.Name, arm.Kind, arm.Available, expected.kind, expected.available)
		}
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Decisions != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Decisions != 5 {
		t.Fatalf("native=%#v", got.Arms[0])
	}
}

func BenchmarkDecideLaneTreeAdmission(b *testing.B) {
	tax, live, requests := comparisonFixture()
	var got Verdict
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, req := range requests {
			got = Decide(req, live, tax)
		}
	}
	if !got.Admit {
		b.Fatal(got)
	}
}

var scalingLeaseCounts = []int{1, 8, 64, 512, 4096}

var scalingScenarios = []struct {
	name      string
	wantAdmit bool
}{
	{name: "disjoint", wantAdmit: true},
	{name: "first_conflict", wantAdmit: false},
	{name: "last_conflict", wantAdmit: false},
	{name: "all_conflict", wantAdmit: false},
	{name: "self_lease", wantAdmit: true},
	{name: "read_only", wantAdmit: true},
}

func TestDecideScalingFixtures(t *testing.T) {
	for _, scenario := range scalingScenarios {
		for _, n := range scalingLeaseCounts {
			t.Run(fmt.Sprintf("%s/N=%d", scenario.name, n), func(t *testing.T) {
				req, live, tax := decideScalingFixture(scenario.name, n)
				got := Decide(req, live, tax)
				if got.Admit != scenario.wantAdmit {
					t.Fatalf("admit=%v want %v: %+v", got.Admit, scenario.wantAdmit, got)
				}
				switch scenario.name {
				case "first_conflict", "last_conflict":
					if len(got.Conflicts) != 1 {
						t.Fatalf("conflicts=%d want 1", len(got.Conflicts))
					}
				case "all_conflict":
					if len(got.Conflicts) != n {
						t.Fatalf("conflicts=%d want %d", len(got.Conflicts), n)
					}
				default:
					if len(got.Conflicts) != 0 {
						t.Fatalf("conflicts=%d want 0", len(got.Conflicts))
					}
				}
			})
		}
	}
}

func BenchmarkDecideScaling(b *testing.B) {
	for _, scenario := range scalingScenarios {
		b.Run(scenario.name, func(b *testing.B) {
			for _, n := range scalingLeaseCounts {
				b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
					req, live, tax := decideScalingFixture(scenario.name, n)
					var got Verdict
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						got = Decide(req, live, tax)
					}
					if got.Admit != scenario.wantAdmit {
						b.Fatalf("admit=%v want %v: %+v", got.Admit, scenario.wantAdmit, got)
					}
				})
			}
		})
	}
}

func decideScalingFixture(scenario string, n int) (Request, []Lease, Taxonomy) {
	req := Request{
		Surface: SurfaceDispatch,
		Lane:    "target",
		Tree:    []string{"internal/target/**"},
		Holder:  "benchmark",
	}
	live := make([]Lease, n)
	for i := range live {
		live[i] = Lease{
			ID:     fmt.Sprintf("lease-%04d", i),
			Lane:   fmt.Sprintf("peer-%04d", i),
			Tree:   []string{fmt.Sprintf("internal/peer/%04d/**", i)},
			Holder: "peer",
		}
	}

	switch scenario {
	case "disjoint":
	case "first_conflict":
		live[0].Tree = append([]string(nil), req.Tree...)
	case "last_conflict":
		live[len(live)-1].Tree = append([]string(nil), req.Tree...)
	case "all_conflict":
		for i := range live {
			live[i].ID = fmt.Sprintf("lease-%04d", len(live)-i)
			live[i].Lane = req.Lane
		}
	case "self_lease":
		req.LeaseID = live[0].ID
		live[0].Lane = req.Lane
		live[0].Tree = append([]string(nil), req.Tree...)
	case "read_only":
		req.ReadOnly = true
		for i := range live {
			live[i].Lane = req.Lane
			live[i].Tree = append([]string(nil), req.Tree...)
		}
	default:
		panic("unknown scaling scenario: " + scenario)
	}

	return req, live, Taxonomy{Loaded: true}
}
