package vcachecal

import "time"

type ComparisonArm struct {
	Name           string
	Kind           string
	Available      bool
	Correct        bool
	Latency        time.Duration
	Buckets        int
	AllocatedBytes int64
	CapturedValue  float64
	CostUSD        float64
	Note           string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func allocationFixture() ([]BudgetBucket, int64, int) {
	return []BudgetBucket{
		{Key: "concentrated", Ranked: []RankedVBlock{{Key: "a", Frequency: 9, Size: 1, ReuseDensity: 1}, {Key: "b", Frequency: 1, Size: 1, ReuseDensity: 1}}},
		{Key: "flat", Ranked: []RankedVBlock{{Key: "a", Frequency: 1, Size: 1, ReuseDensity: 1}, {Key: "b", Frequency: 1, Size: 1, ReuseDensity: 1}}},
		{Key: "unmeasured"},
	}, 1200, 1
}
func sumBudgets(s []BucketShare) int64 {
	var n int64
	for _, x := range s {
		n += x.Budget
	}
	return n
}
func allocationValue(s []BucketShare) float64 {
	weights := map[string]float64{"concentrated": .9, "flat": .5, "unmeasured": 0}
	var v float64
	for _, x := range s {
		v += float64(x.Budget) * weights[x.Key]
	}
	return v
}
func equalShare(b []BudgetBucket, total int64) []BucketShare {
	return AllocateByConcentration(b, total, 0)
}
func volumeShare(b []BudgetBucket, total int64) []BucketShare {
	out := make([]BucketShare, len(b))
	var sum float64
	vol := make([]float64, len(b))
	for i, x := range b {
		for _, r := range x.Ranked {
			vol[i] += r.Frequency
		}
		sum += vol[i]
	}
	for i, x := range b {
		out[i] = BucketShare{Key: x.Key}
		if sum > 0 {
			out[i].Share = vol[i] / sum
		}
		out[i].Budget = int64(float64(total) * out[i].Share)
	}
	var used int64
	for _, x := range out {
		used += x.Budget
	}
	if len(out) > 0 {
		out[0].Budget += total - used
	}
	return out
}
func CompareLocal() ComparisonResult {
	buckets, total, topK := allocationFixture()
	start := time.Now()
	native := AllocateByConcentration(buckets, total, topK)
	nl := time.Since(start)
	start = time.Now()
	equal := equalShare(buckets, total)
	el := time.Since(start)
	start = time.Now()
	volume := volumeShare(buckets, total)
	vl := time.Since(start)
	correct := len(native) == 3 && sumBudgets(native) == total && native[0].Budget > native[1].Budget && native[2].Budget == 400
	return ComparisonResult{Workload: "allocate 1200 cache bytes across concentrated, flat, and unmeasured buckets at top-K one", Arms: []ComparisonArm{
		{Name: "fak native concentration-weighted allocation", Kind: "native", Available: true, Correct: correct, Latency: nl, Buckets: 3, AllocatedBytes: sumBudgets(native), CapturedValue: allocationValue(native)},
		{Name: "equal-share cache allocation", Kind: "baseline", Available: true, Correct: false, Latency: el, Buckets: 3, AllocatedBytes: sumBudgets(equal), CapturedValue: allocationValue(equal), Note: "conserves budget but ignores measured concentration"},
		{Name: "request-volume proportional allocation", Kind: "baseline", Available: true, Correct: false, Latency: vl, Buckets: 3, AllocatedBytes: sumBudgets(volume), CapturedValue: allocationValue(volume), Note: "uses demand volume but can starve an unmeasured tenant"},
		{Name: "fak + LMCache", Kind: "integration", Note: "requires the real first-class LMCache runtime and shared pool"},
		{Name: "fak + Mooncake", Kind: "integration", Note: "requires the real first-class Mooncake runtime and shared pool"},
		{Name: "vLLM cache-aware routing", Kind: "external", Note: "requires a real vLLM deployment and equivalent shared budget"},
		{Name: "SGLang HiCache and cache-aware scheduling", Kind: "external", Note: "requires a real SGLang deployment and equivalent shared budget"},
	}}
}
