package benchscore

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestGoodputCalculator(t *testing.T) {
	c := GoodputConstraints{
		MaxTTFTMS:              100,
		MaxTTSTMS:              30,
		MaxITLMS:               25,
		MinOutputTokPerSecUser: 20,
	}
	samples := []GoodputSample{
		{TTFTMS: 80, TTSTMS: 20, ITLMS: 20, OutputTokPerSecPerUser: 30},  // good
		{TTFTMS: 120, TTSTMS: 20, ITLMS: 20, OutputTokPerSecPerUser: 30}, // TTFT fail
		{TTFTMS: 80, TTSTMS: 20, ITLMS: 20, OutputTokPerSecPerUser: 10},  // tok/s fail
		{TTFTMS: 90, TTSTMS: 25, ITLMS: 22, OutputTokPerSecPerUser: 25},  // good
	}
	res := ComputeGoodput(samples, c, 2.0)
	if res.GoodRequests != 2 || res.TotalRequests != 4 || math.Abs(res.GoodputRPS-1.0) > 1e-12 {
		t.Fatalf("unexpected goodput result: %+v", res)
	}

	// Invalid duration or empty samples return -1 sentinel.
	if got := ComputeGoodput(samples, c, 0).GoodputRPS; got != InvalidGoodputSentinel {
		t.Fatalf("zero duration goodput = %v, want -1", got)
	}
	if got := ComputeGoodput(nil, c, 1.0).GoodputRPS; got != InvalidGoodputSentinel {
		t.Fatalf("empty samples goodput = %v, want -1", got)
	}
}
