package debtlane

import (
	"fmt"
	"math"
)

// EvaluateMaturityCurve maps evidence to a continuous 0.0 - 10.0 maturity score
// and assigns the corresponding lifecycle rung.
func EvaluateMaturityCurve(e Evidence) (score float64, rung string) {
	if !e.HasCode {
		return 0.0, "proposed"
	}

	// 1/10 maturity curve: code exists, but is a bare stub or skeleton without tests or substance.
	if e.CodeLines < 25 && !e.HasTests {
		return 1.0, "stub"
	}

	// 2/10: functional prototype (substantial code, but untested).
	if !e.HasTests {
		score = 2.0
		if e.CodeLines > 100 {
			score += 0.5
		}
		if e.ExcessComments {
			score -= 0.5
		}
		if score < 0 {
			score = 0
		}
		return score, "prototyped"
	}

	// Tested: base 4.0.
	score = 4.0
	if e.TestFilesCount > 1 {
		score += 0.5
	}

	// Integration: reachable from production commands.
	if e.Integrated {
		score += 1.5
	}

	// Dogfooded: verified in runtime proof / active harness.
	if e.Dogfooded {
		score += 1.0
	}

	// Benchmarked: measured performance or authority entry.
	if e.Benchmarked {
		score += 1.0
	}

	// Engineering rigor: clean self-documenting structure and bounded blast radius constraints.
	if !e.ExcessComments && e.CodeLines > 0 {
		score += 0.5
	}
	if e.DependentsCount <= 25 {
		score += 0.5
	}

	// Production grade bonus: clean comprehensive surface.
	if e.Integrated && e.Dogfooded && e.Benchmarked && e.HasTests && !e.ExcessComments {
		score += 1.0
	}

	// Excess comments penalty: formulaic noise or comment bloat is bad debt.
	if e.ExcessComments {
		score -= 0.5
	}

	// An unintegrated unit of work cannot advance past the tested ceiling (5.0).
	if !e.Integrated && score > 5.0 {
		score = 5.0
	}

	// Clamp to [0.0, 10.0].
	if score < 0 {
		score = 0
	}
	if score > 10.0 {
		score = 10.0
	}
	score = math.Round(score*10) / 10

	switch {
	case score >= 9.5 && e.Integrated && e.Dogfooded && e.Benchmarked && e.HasTests && !e.ExcessComments:
		rung = "production_grade"
	case score >= 8.5 && e.Integrated && e.Dogfooded && e.Benchmarked && !e.ExcessComments:
		rung = "hardened"
	case score >= 7.5 && e.Integrated && e.Dogfooded && e.Benchmarked:
		rung = "benchmarked"
	case score >= 6.5 && e.Integrated && e.Dogfooded:
		rung = "dogfooded"
	case score >= 5.0 && e.Integrated:
		rung = "integrated"
	case score >= 3.5 && e.HasTests:
		rung = "tested"
	case score >= 1.5:
		rung = "prototyped"
	default:
		rung = "stub"
	}
	return score, rung
}

// NextActionForGap renders the next actionable step to advance on the maturity curve.
func NextActionForGap(lane, unit string, score, target float64, e Evidence) string {
	if score >= target {
		return fmt.Sprintf("%s is at target maturity (%.1f/%.1f); keep tests and benchmarks green", lane, score, target)
	}

	switch {
	case !e.HasCode:
		return fmt.Sprintf("implement %s: add initial implementation in %s", lane, unit)
	case !e.HasTests:
		return fmt.Sprintf("test %s: add unit tests (*_test.go) covering %s to climb past 2.0", lane, unit)
	case !e.Integrated:
		return fmt.Sprintf("integrate %s: connect into production command or registration graph", lane)
	case !e.Dogfooded:
		return fmt.Sprintf("dogfood %s: execute real runtime path and capture passing proof", lane)
	case !e.Benchmarked:
		return fmt.Sprintf("benchmark %s: add substantive Benchmark* functions measuring production operations with b.N loops, or register authoritative benchmark", lane)
	case e.ExcessComments:
		return fmt.Sprintf("clean %s: prune excess comment bloat and formulaic noise (%.1f%% comment ratio); make code self-documenting and verify invariants in tests", lane, e.CommentRatio*100)
	default:
		return fmt.Sprintf("advance %s: complete production contracts and verified defaults to reach %.1f", lane, target)
	}
}
