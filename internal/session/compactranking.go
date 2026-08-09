package session

import (
	"fmt"
	"io"
	"sort"
)

const (
	CompactRankFires           = "fires"
	CompactRankPeakResident    = "peak-resident"
	CompactRankCumulativeInput = "cumulative-input"
)

// writeCompactTrajectoryRanking renders a token-oriented top-session view. The
// default compact-audit report ranks compaction activity; this view answers the
// distinct operator question "which trajectories consumed or residently held the
// most tokens?" without conflating cumulative provider work with live context.

// WriteCompactTrajectoryRanking writes the selected token-trajectory view.
func WriteCompactTrajectoryRanking(w io.Writer, reports []CompactSessionReport, topN int, rank string) error {
	return writeCompactTrajectoryRanking(w, reports, topN, rank)
}
func writeCompactTrajectoryRanking(w io.Writer, reports []CompactSessionReport, topN int, rank string) error {
	if rank != CompactRankPeakResident && rank != CompactRankCumulativeInput {
		return fmt.Errorf("unknown compact trajectory rank %q (want %s or %s)", rank, CompactRankPeakResident, CompactRankCumulativeInput)
	}
	if topN <= 0 {
		return nil
	}
	ranked := append([]CompactSessionReport(nil), reports...)
	sort.SliceStable(ranked, func(i, j int) bool {
		left, right := compactRankValue(ranked[i], rank), compactRankValue(ranked[j], rank)
		if left != right {
			return left > right
		}
		return ranked[i].SessionID < ranked[j].SessionID
	})
	if len(ranked) > topN {
		ranked = ranked[:topN]
	}
	fmt.Fprintf(w, "  top %d sessions by %s tokens:\n", len(ranked), rank)
	for _, s := range ranked {
		id := s.SessionID
		if id == "" {
			id = "(unknown)"
		}
		peakRatio := 0.0
		if s.ContextWindow > 0 {
			peakRatio = float64(s.PeakResidentTokens) / float64(s.ContextWindow)
		}
		fmt.Fprintf(w, "    %s  %s\n", id, s.Verdict)
		fmt.Fprintf(w, "      peak RESIDENT %d/%d (%.1f%%) · final RESIDENT %d · cumulative input %d · %d fires\n",
			s.PeakResidentTokens, s.ContextWindow, 100*peakRatio, s.FinalResidentTokens,
			s.CumulativeInputTokens, len(s.Fires))
	}
	return nil
}

func compactRankValue(s CompactSessionReport, rank string) int64 {
	if rank == CompactRankPeakResident {
		return int64(s.PeakResidentTokens)
	}
	return int64(s.CumulativeInputTokens)
}
