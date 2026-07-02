package main

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

type vcacheSessionSummaryOptions struct {
	SinceDays       float64
	Max             int
	NamespacePrefix string
	AllNamespaces   bool
}

func applyRecentSessionSummary(rep *vcacheStatusReport, opts vcacheSessionSummaryOptions) {
	var since *float64
	if opts.SinceDays >= 0 {
		v := opts.SinceDays
		since = &v
	}
	nsPrefix := strings.TrimSpace(opts.NamespacePrefix)
	if opts.AllNamespaces {
		nsPrefix = ""
	} else if nsPrefix == "" {
		cwd, err := os.Getwd()
		if err != nil {
			rep.RecentSessionsError = fmt.Sprintf("current workspace namespace: %v", err)
			return
		}
		nsPrefix = sessionaudit.ProjectNamespace(cwd)
	}
	discover := sessionaudit.DiscoverOptions{
		SinceDays:       since,
		NamespacePrefix: nsPrefix,
	}
	recs, err := sessionaudit.Discover(discover)
	if err != nil {
		rep.RecentSessionsError = err.Error()
		return
	}
	totalDiscovered := len(recs)
	if opts.Max > 0 && len(recs) > opts.Max {
		recs = recs[:opts.Max]
	}
	sessions := make([]sessionaudit.Session, 0, len(recs))
	for _, rec := range recs {
		if rec.Kind == "subagent" {
			continue
		}
		s := sessionaudit.Analyze(rec.Path)
		s.Kind = rec.Kind
		sessions = append(sessions, s)
	}
	agg := sessionaudit.AggregateSessions(sessions)
	summary := sessionaudit.BuildCompactReport(sessions, agg, nsPrefix, since, false, opts.Max, totalDiscovered, nil, time.Now())
	rep.RecentSessions = &summary
}

func printVCacheSessionSummary(w io.Writer, summary sessionaudit.CompactReport) {
	fmt.Fprintf(w, "recent sessions: %d/%d sessions, scope %s, context %d tok, cache-read %.1f%%, I:O %.1f, cost $%.2f",
		summary.Scope.Audited,
		summary.Scope.Discovered,
		summary.Scope.NamespaceFilter,
		summary.Totals.TotalContextTokens,
		100*summary.Totals.CacheReadShare,
		summary.Totals.IORatio,
		summary.Totals.EstimatedCostUSD)
	if summary.Scope.Clipped {
		fmt.Fprint(w, " (clipped)")
	}
	fmt.Fprintln(w)
	for _, tier := range summary.Tiers {
		if tier.Tier == "fable" || tier.Tier == "opus" {
			fmt.Fprintf(w, "  %s: output %d (%.1f%%), cost $%.2f (%.1f%%)\n",
				tier.Tier,
				tier.OutputTokens,
				100*tier.OutputShare,
				tier.EstimatedCostUSD,
				100*tier.CostShare)
		}
	}
	if len(summary.TopLongContext) > 0 {
		top := summary.TopLongContext[0]
		fmt.Fprintf(w, "  top long-context: %s context %d tok, cache-read %.1f%%, model %s\n",
			top.Session,
			top.TotalContextTokens,
			100*top.CacheReadShare,
			top.TopModel)
	}
	for _, rec := range summary.Recommendations {
		fmt.Fprintf(w, "  recommendation: %s [%s] %s (%s)\n",
			rec.Kind,
			rec.Severity,
			rec.Action,
			rec.Evidence,
		)
	}
}

func dominantVCacheGovernorDecision(families []vcacheobserve.Family) string {
	if len(families) == 0 {
		return ""
	}
	counts := make(map[string]int, len(families))
	best := ""
	for _, family := range families {
		decision := string(family.GovernorDecision)
		counts[decision]++
		if best == "" || counts[decision] > counts[best] || counts[decision] == counts[best] && decision < best {
			best = decision
		}
	}
	return best
}
