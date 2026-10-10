package webbench

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
)

type ServingSweepClaimKind string

const (
	ServingSweepClaimNone         ServingSweepClaimKind = "none"
	ServingSweepClaimPeak         ServingSweepClaimKind = "peak"
	ServingSweepClaimCapacityPeak ServingSweepClaimKind = "capacity_peak"
	ServingSweepClaimSLAKnee      ServingSweepClaimKind = "sla_knee"
)

type ServingSweepClaim struct {
	Kind  ServingSweepClaimKind
	Track ServingTrack
}

var servingSweepTokenRatePattern = regexp.MustCompile(`(?i)\b(?:tok(?:en)?s?\s*/\s*s(?:ec(?:ond)?s?)?\b|tokens?\s+per\s+second\b|token\s+throughput\b)`)

var servingSweepTrackPattern = regexp.MustCompile(`\b(ours|vllm|sglang|fak-fronts-fleet)\b`)

func ParseServingSweepClaim(claim string) ServingSweepClaim {
	text := strings.ToLower(claim)
	kind := ServingSweepClaimNone
	switch {
	case strings.Contains(text, "sla knee") || strings.Contains(text, "sla-knee") || strings.Contains(text, "p99 knee"):
		kind = ServingSweepClaimSLAKnee
	case strings.Contains(text, "capacity-valid peak") || strings.Contains(text, "capacity valid peak"):
		kind = ServingSweepClaimCapacityPeak
	case strings.Contains(text, "serving peak") || strings.Contains(text, "peak serving") || strings.Contains(text, "peak throughput"):
		kind = ServingSweepClaimPeak
	default:
		return ServingSweepClaim{Kind: ServingSweepClaimNone}
	}
	track := TrackOurs
	if match := servingSweepTrackPattern.FindString(text); match != "" {
		track = ServingTrack(match)
	}
	return ServingSweepClaim{Kind: kind, Track: track}
}

func ValidateServingSweepClaim(claim string, report *ServingSweepReport) error {
	parsed := ParseServingSweepClaim(claim)
	if parsed.Kind == ServingSweepClaimNone {
		return nil
	}
	if report == nil {
		return errors.New("serving peak/knee claim requires a fak.serving-sweep.v1 artifact")
	}
	if report.Schema != ServingSweepSchema {
		return fmt.Errorf("serving sweep artifact schema = %q, want %q", report.Schema, ServingSweepSchema)
	}
	if err := EvaluateServingSweep(report); err != nil {
		return fmt.Errorf("serving sweep artifact is invalid: %w", err)
	}
	summary := servingSweepSummary(report, parsed.Track)
	if summary == nil {
		return fmt.Errorf("serving sweep claim requires measured %s track; rerun the sweep with --tracks %s", parsed.Track, parsed.Track)
	}
	if summary.Status != "measured" || summary.ValidPoints < 2 || summary.Peak == nil || summary.PeakStatus != "measured" {
		reason := summary.Reason
		if reason == "" {
			reason = summary.PeakReason
		}
		if reason == "" {
			reason = "fewer than two comparable capacity-valid points or no measured peak"
		}
		return fmt.Errorf("serving peak claim for %s refused: %s; rerun with at least two in-capacity identity-stable points", parsed.Track, reason)
	}
	for _, unit := range []string{"usage.completion_tokens/s", "stream_content_events/s", "estimated_content_tokens/s", "output_token_estimate/s"} {
		if strings.Contains(strings.ToLower(claim), unit) && unit != summary.Peak.ThroughputUnit {
			return fmt.Errorf("serving throughput claim unit %s differs from measured unit %s", unit, summary.Peak.ThroughputUnit)
		}
	}
	// Non-token observations can support their own explicitly named rate, but
	// must never certify prose that presents events or estimates as exact tokens.
	if summary.Peak.TokenCountBasis != "usage.completion_tokens" {
		if servingSweepTokenRatePattern.MatchString(claim) {
			return fmt.Errorf("serving token throughput claim refused: measured unit is %s with token count basis %s", summary.Peak.ThroughputUnit, summary.Peak.TokenCountBasis)
		}
		if parsed.Kind != ServingSweepClaimSLAKnee && !strings.Contains(strings.ToLower(claim), summary.Peak.ThroughputUnit) {
			return fmt.Errorf("serving peak claim must explicitly identify measured throughput unit %s", summary.Peak.ThroughputUnit)
		}
	}
	if parsed.Kind != ServingSweepClaimSLAKnee {
		return nil
	}
	if !servingSweepSLAConfigured(report) {
		return errors.New("serving SLA-knee claim requires a configured p99 TTFT or ITL budget")
	}
	if summary.SLAStatus != "measured" || summary.SLAKnee == nil {
		reason := summary.SLAReason
		if reason == "" {
			reason = "no measured point satisfies every configured p99 budget"
		}
		return fmt.Errorf("serving SLA-knee claim for %s refused: %s", parsed.Track, reason)
	}
	return nil
}

func servingSweepSummary(report *ServingSweepReport, track ServingTrack) *ServingSweepTrackSummary {
	for i := range report.Tracks {
		if report.Tracks[i].Track == track {
			return &report.Tracks[i]
		}
	}
	return nil
}
