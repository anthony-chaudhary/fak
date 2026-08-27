package model

import "fmt"

type CPUStreamingSample struct {
	Threads          int     `json:"threads"`
	LocalBytes       int64   `json:"local_bytes"`
	RemoteBytes      int64   `json:"remote_bytes"`
	LLCMisses        int64   `json:"llc_misses"`
	HugePages        bool    `json:"huge_pages"`
	SIMDWidth        int     `json:"simd_width"`
	PrefetchDistance int     `json:"prefetch_distance"`
	Nanoseconds      int64   `json:"nanoseconds"`
	Joules           float64 `json:"joules"`
	AcceptedTokens   int     `json:"accepted_tokens"`
}
type CPUStreamingReceipt struct {
	Schema             string  `json:"schema"`
	Engine             string  `json:"engine"`
	Samples            int     `json:"samples"`
	BestThreads        int     `json:"best_threads"`
	SustainableGBps    float64 `json:"sustainable_gbps"`
	RemoteTrafficRatio float64 `json:"remote_traffic_ratio"`
	LLCMisses          int64   `json:"llc_misses"`
	JoulesPerAccepted  float64 `json:"joules_per_accepted_token"`
	Saturated          bool    `json:"saturated"`
	Regression         bool    `json:"regression"`
	StopRule           string  `json:"stop_rule"`
	Rollback           string  `json:"rollback"`
}

func MeasureCPUStreaming(samples []CPUStreamingSample) (CPUStreamingReceipt, error) {
	if len(samples) < 2 {
		return CPUStreamingReceipt{}, fmt.Errorf("model: need scaling samples")
	}
	r := CPUStreamingReceipt{Schema: "fak-cpu-streaming/1", Engine: "fak-native", Samples: len(samples), StopRule: "stop when added threads improve bandwidth <2% or increase remote ratio", Rollback: "restore prior thread/page binding"}
	best := -1.0
	var prev float64
	for i, s := range samples {
		if s.Threads <= 0 || s.LocalBytes < 0 || s.RemoteBytes < 0 || s.LLCMisses < 0 || s.Nanoseconds <= 0 || s.Joules < 0 || s.AcceptedTokens < 0 {
			return CPUStreamingReceipt{}, fmt.Errorf("model: invalid CPU streaming sample")
		}
		gbps := float64(s.LocalBytes+s.RemoteBytes) / float64(s.Nanoseconds)
		if gbps > best {
			best = gbps
			r.BestThreads = s.Threads
			r.SustainableGBps = gbps
			total := s.LocalBytes + s.RemoteBytes
			if total > 0 {
				r.RemoteTrafficRatio = float64(s.RemoteBytes) / float64(total)
			}
			r.LLCMisses = s.LLCMisses
			if s.AcceptedTokens > 0 {
				r.JoulesPerAccepted = s.Joules / float64(s.AcceptedTokens)
			}
		}
		if i > 0 && gbps <= prev*1.02 {
			r.Saturated = true
		}
		if i > 0 && gbps < prev {
			r.Regression = true
		}
		prev = gbps
	}
	return r, nil
}
