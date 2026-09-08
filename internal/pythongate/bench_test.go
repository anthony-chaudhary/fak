package pythongate

import (
	"fmt"
	"testing"
)

// BenchmarkBaselineSet measures the overhead of materializing the grandfathered
// Python tools slice into an allowlist lookup map.
func BenchmarkBaselineSet(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		set := baselineSet()
		if len(set) == 0 {
			b.Fatal("empty baseline set")
		}
	}
}

// BenchmarkOffensesAgainst_Clean measures gate evaluation latency on a clean repository
// where all tracked Python tools match the grandfathered allowlist (no offenses).
func BenchmarkOffensesAgainst_Clean(b *testing.B) {
	allowed := baselineSet()
	tracked := make([]string, len(grandfathered))
	copy(tracked, grandfathered)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offenses := offensesAgainst(tracked, allowed)
		if len(offenses) != 0 {
			b.Fatalf("expected 0 offenses, got %d", len(offenses))
		}
	}
}

// BenchmarkOffensesAgainst_WithOffenses measures gate evaluation and offense collection
// when unauthorized (new) Python tools are discovered in the tree.
func BenchmarkOffensesAgainst_WithOffenses(b *testing.B) {
	allowed := baselineSet()
	tracked := make([]string, 0, len(grandfathered)+30)
	tracked = append(tracked, grandfathered...)
	for i := 0; i < 30; i++ {
		tracked = append(tracked, fmt.Sprintf("tools/unauthorized_script_%03d.py", i))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offenses := offensesAgainst(tracked, allowed)
		if len(offenses) != 30 {
			b.Fatalf("expected 30 offenses, got %d", len(offenses))
		}
	}
}

// BenchmarkOffensesAgainst_AllOffenses measures sorting and allocation pressure when
// every tracked path is un-grandfathered.
func BenchmarkOffensesAgainst_AllOffenses(b *testing.B) {
	allowed := make(map[string]bool)
	tracked := make([]string, 100)
	for i := 0; i < 100; i++ {
		tracked[i] = fmt.Sprintf("tools/new_tool_%03d.py", 100-i)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offenses := offensesAgainst(tracked, allowed)
		if len(offenses) != 100 {
			b.Fatalf("expected 100 offenses, got %d", len(offenses))
		}
	}
}

// BenchmarkOffenseString measures rendering an Offense struct into its structured
// refusal diagnostic message carrying ReasonNewPythonTool.
func BenchmarkOffenseString(b *testing.B) {
	offense := Offense{Path: "tools/new_experimental_worker.py"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		msg := offense.String()
		if len(msg) == 0 {
			b.Fatal("empty offense string")
		}
	}
}

// BenchmarkAdmittedTestCompanions_FastFilter measures the non-spawning fast filter
// in test companion admission when tracked tools do not contain companion paths.
func BenchmarkAdmittedTestCompanions_FastFilter(b *testing.B) {
	tracked := make(map[string]bool, len(grandfathered))
	for _, p := range grandfathered {
		tracked[p] = true
	}
	baseline := baselineSet()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		admitted := admittedTestCompanions(".", tracked, baseline)
		if len(admitted) != 0 {
			b.Fatalf("expected 0 admitted companions, got %d", len(admitted))
		}
	}
}

// BenchmarkGateEvaluation_Synthetic measures end-to-end gate evaluation flow on
// synthetic inputs (baseline set construction, tracked map conversion, and ratchet evaluation).
func BenchmarkGateEvaluation_Synthetic(b *testing.B) {
	tracked := make([]string, len(grandfathered))
	copy(tracked, grandfathered)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		allowed := baselineSet()
		trackedSet := make(map[string]bool, len(tracked))
		for _, path := range tracked {
			trackedSet[path] = true
		}
		for path := range admittedTestCompanions(".", trackedSet, allowed) {
			allowed[path] = true
		}
		offenses := offensesAgainst(tracked, allowed)
		if len(offenses) != 0 {
			b.Fatalf("expected 0 offenses, got %d", len(offenses))
		}
	}
}
