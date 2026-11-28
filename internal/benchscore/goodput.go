// Structural adaptation of triton-inference-server/perf_analyzer GenAI-Perf
// goodput calculator (llm_goodput_calculator.py:70 @ 24f6b9af4db5, BSD-3-Clause).
package benchscore

// InvalidGoodputSentinel is returned when goodput cannot be computed (e.g. non-positive duration).
const InvalidGoodputSentinel = -1.0

// GoodputConstraints defines per-request SLO ceilings (0 disables a constraint).
type GoodputConstraints struct {
	MaxTTFTMS              float64 `json:"max_ttft_ms,omitempty"`
	MaxTTSTMS              float64 `json:"max_ttst_ms,omitempty"`
	MaxITLMS               float64 `json:"max_itl_ms,omitempty"`
	MinOutputTokPerSecUser float64 `json:"min_output_tok_per_sec_user,omitempty"`
}

// GoodputSample holds one request's observed metrics for SLO filtering.
type GoodputSample struct {
	TTFTMS                 float64 `json:"ttft_ms"`
	TTSTMS                 float64 `json:"ttst_ms"`
	ITLMS                  float64 `json:"itl_ms"`
	OutputTokPerSecPerUser float64 `json:"output_tok_per_sec_user"`
}

// Satisfies reports whether s meets all active constraints in c.
func (c GoodputConstraints) Satisfies(s GoodputSample) bool {
	if c.MaxTTFTMS > 0 && s.TTFTMS > c.MaxTTFTMS {
		return false
	}
	if c.MaxTTSTMS > 0 && s.TTSTMS > c.MaxTTSTMS {
		return false
	}
	if c.MaxITLMS > 0 && s.ITLMS > c.MaxITLMS {
		return false
	}
	if c.MinOutputTokPerSecUser > 0 && s.OutputTokPerSecPerUser < c.MinOutputTokPerSecUser {
		return false
	}
	return true
}

// GoodputResult holds goodput (good requests / duration_sec) or InvalidGoodputSentinel (-1).
type GoodputResult struct {
	TotalRequests int     `json:"total_requests"`
	GoodRequests  int     `json:"good_requests"`
	DurationSec   float64 `json:"duration_sec"`
	GoodputRPS    float64 `json:"goodput_rps"`
}

// ComputeGoodput filters samples against c and returns good / durationSec,
// or InvalidGoodputSentinel (-1) when durationSec <= 0 or samples is empty.
func ComputeGoodput(samples []GoodputSample, c GoodputConstraints, durationSec float64) GoodputResult {
	if len(samples) == 0 || durationSec <= 0 {
		return GoodputResult{
			TotalRequests: len(samples),
			DurationSec:   durationSec,
			GoodputRPS:    InvalidGoodputSentinel,
		}
	}
	good := 0
	for _, s := range samples {
		if c.Satisfies(s) {
			good++
		}
	}
	return GoodputResult{
		TotalRequests: len(samples),
		GoodRequests:  good,
		DurationSec:   durationSec,
		GoodputRPS:    float64(good) / durationSec,
	}
}
