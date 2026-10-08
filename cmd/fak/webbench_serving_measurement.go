package main

import (
	"fmt"
	"io"
	"math"

	"github.com/anthony-chaudhary/fak/internal/webbench"
)

func writeServingMeasurementEvidence(w io.Writer, stats *webbench.ServingStats) {
	wall, output, goodput, slo := "unmeasured", "unmeasured/unmeasured", "unmeasured", "unmeasured"
	observedMax, basis := "unmeasured", "unmeasured"
	if stats != nil {
		if stats.WallSeconds != nil && finitePositive(*stats.WallSeconds) {
			wall = fmt.Sprintf("%.6g", *stats.WallSeconds)
		}
		if stats.SuccessfulOutputTokensExact != nil && *stats.SuccessfulOutputTokensExact >= 0 {
			qualified := "unmeasured"
			if stats.SLOSuccessfulOutputTokensExact != nil && *stats.SLOSuccessfulOutputTokensExact >= 0 &&
				*stats.SLOSuccessfulOutputTokensExact <= *stats.SuccessfulOutputTokensExact {
				qualified = fmt.Sprintf("%d", *stats.SLOSuccessfulOutputTokensExact)
			}
			output = fmt.Sprintf("%d/%s", *stats.SuccessfulOutputTokensExact, qualified)
		}
		if stats.GoodputTokensS.Status == "measured" && stats.GoodputTokensS.Value != nil && finiteNonnegative(*stats.GoodputTokensS.Value) {
			goodput = fmt.Sprintf("%.6g", *stats.GoodputTokensS.Value)
		}
		if stats.GoodputSLOSeconds != nil && finitePositive(*stats.GoodputSLOSeconds) {
			slo = fmt.Sprintf("%.6g", *stats.GoodputSLOSeconds)
		}
		if stats.ObservedInFlightBasis == "client_http" && stats.ObservedMaxInFlight != nil &&
			*stats.ObservedMaxInFlight >= 0 && *stats.ObservedMaxInFlight <= stats.Requests {
			observedMax = fmt.Sprintf("%d", *stats.ObservedMaxInFlight)
			basis = "client_http"
		}
	}
	fmt.Fprintf(w, "throughput=%s wall_s=%s output_exact=%s goodput_tok_s=%s slo_s=%s observed_http_max=%s basis=%s", formatServingThroughput(stats), wall, output, goodput, slo, observedMax, basis)
}

func formatServingThroughput(stats *webbench.ServingStats) string {
	if stats == nil || stats.ThroughputTokensS.Status != "measured" || stats.ThroughputTokensS.Value == nil ||
		!finiteNonnegative(*stats.ThroughputTokensS.Value) || !servingThroughputUnit(stats.ThroughputTokensS.Unit) {
		return "unmeasured"
	}
	return fmt.Sprintf("%.6g %s", *stats.ThroughputTokensS.Value, stats.ThroughputTokensS.Unit)
}

func servingThroughputUnit(unit string) bool {
	switch unit {
	case "usage.completion_tokens/s", "stream_content_events/s", "estimated_content_tokens/s", "output_token_estimate/s":
		return true
	default:
		return false
	}
}

func servingSweepSelectionStats(rep *webbench.ServingSweepReport, track webbench.ServingTrack, concurrency int) *webbench.ServingStats {
	if rep == nil {
		return nil
	}
	var found *webbench.ServingStats
	for pointIndex := range rep.Points {
		point := &rep.Points[pointIndex]
		if point.Concurrency != concurrency {
			continue
		}
		for trackIndex := range point.Tracks {
			trackPoint := &point.Tracks[trackIndex]
			if trackPoint.Track != track {
				continue
			}
			if found != nil {
				return nil
			}
			if trackPoint.Status != "valid" && trackPoint.Status != "measured" {
				return nil
			}
			if trackPoint.MeasurementStatus != "" && trackPoint.MeasurementStatus != "measured" {
				return nil
			}
			found = &trackPoint.Stats
		}
	}
	return found
}

func finitePositive(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func finiteNonnegative(value float64) bool {
	return value >= 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}
