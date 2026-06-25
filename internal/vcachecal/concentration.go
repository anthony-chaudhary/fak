package vcachecal

import "math"

// concentration.go is the vCache workload-concentration measurement — issue #716 scope
// 3 and §5.2. vCache only pays off when popularity CONCENTRATES onto a small anchor set
// (a steep Zipf workload): "7 anchors → 85%" implies a Zipf exponent s ≈ 1.74, and the
// coverage is roughly V-independent over 10³–5×10³. At the opposite extreme — a flat
// workload (s ≤ 1) — the head is huge and vCache barely helps: at s=1 the top-7 of 1000
// covers only 34.6%, and covering 90% needs thousands of anchors. The actionable gate
// (scope 3): MEASURE s BEFORE trusting vCache, and at s ≤ 1 do not warm the tail —
// manufacture skew (canonicalize/aggregate so popularity concentrates) or abstain.

// RankedVBlock is one vBlock's workload weight, ranked DESCENDING by Weight. The scope-3
// ranking key is frequency × size × reuse-density; the frequency is the cachemeta
// Lifecycle.AccessRatePerSec-driven reuse intensity (issue: "Lifecycle.AccessRatePerSec-
// driven"), size is the prefix length in tokens, and reuse-density is the expected
// reuses before the TTL.
type RankedVBlock struct {
	Key          string
	Frequency    float64
	Size         float64
	ReuseDensity float64
}

// Weight is the scope-3 ranking weight: frequency × size × reuse-density.
func (v RankedVBlock) Weight() float64 { return v.Frequency * v.Size * v.ReuseDensity }

// Concentration is the fitted workload concentration: the Zipf exponent s, the top-N
// coverage curve, and the defeated flag. Issue #716 acceptance: "Workload concentration
// s is measured and surfaced; a flat workload (s ≤ 1) is flagged as 'vCache will not
// help — manufacture skew or abstain.'"
type Concentration struct {
	ZipfS          float64 // the fitted Zipf exponent s
	Measured       bool
	Defeated       bool            // s ≤ 1 → structurally defeated (§5.2 corollary)
	TopNCoverage   map[int]float64 // rank N → cumulative weight share of the top-N anchors
	Recommendation string
}

// FitConcentration fits the Zipf exponent s from the ranked vBlocks (which MUST be
// sorted descending by Weight) via least-squares on log(weight) vs log(rank): a Zipf law
// has weight(rank) ∝ 1/rank^s, so log(weight) = C − s·log(rank) and s = −slope. It then
// computes the top-N coverage curve and flags a flat workload (s ≤ 1) as structurally
// defeated — the §5.2 gate. Fewer than two positive weights cannot fit s; the result is
// conservatively defeated (Measured = false, s = 0).
func FitConcentration(ranked []RankedVBlock) Concentration {
	c := Concentration{TopNCoverage: map[int]float64{}}
	if len(ranked) == 0 {
		c.Defeated = true
		c.Recommendation = "no workload ranked — measure concentration before trusting vCache"
		return c
	}
	total := 0.0
	for _, v := range ranked {
		total += v.Weight()
	}
	var sumX, sumY, sumXY, sumX2 float64
	n := 0
	for i, v := range ranked {
		w := v.Weight()
		if w <= 0 {
			continue
		}
		x := math.Log(float64(i + 1)) // rank is 1-based
		y := math.Log(w)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
		n++
	}
	if n >= 2 {
		denom := float64(n)*sumX2 - sumX*sumX
		if denom != 0 {
			slope := (float64(n)*sumXY - sumX*sumY) / denom
			c.ZipfS = -slope
			c.Measured = true
		}
	}
	if total > 0 {
		cum := 0.0
		for i, v := range ranked {
			cum += v.Weight()
			c.TopNCoverage[i+1] = cum / total
		}
	}
	if c.ZipfS <= 1.0 {
		c.Defeated = true
		c.Recommendation = "workload is flat (s<=1): vCache will not help — manufacture skew (canonicalize/aggregate) or abstain"
	} else {
		c.Recommendation = "workload is skewed (s>1): a small anchor set captures most of the volume — vCache can exploit it"
	}
	return c
}
