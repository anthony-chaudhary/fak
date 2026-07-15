package gateway

import (
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// journalReframePass is the thin gateway sink for #4420. Empty path keeps the
// journal disabled; write failure is telemetry-only and never changes the emitted
// note. Control records classify the source without applying a rewrite.
func journalReframePass(path, traceID, arm, text string, now time.Time) string {
	if path == "" {
		return text
	}
	result := negframe.ReframeResult{Text: text}
	if arm == "treatment" {
		result = negframe.ReframePass(text)
	} else {
		result.ResidualNegatives = len(negframe.Classify("gateway-runtime", text))
	}
	_ = negframe.AppendReframeJournal(path, negframe.NewReframeJournalRow(traceID, arm, result, now), negframe.DefaultJournalMaxRows)
	return result.Text
}

func journalReframeFragments(path, traceID, site, arm string, fragments []negframe.Fragment, now time.Time) string {
	var out strings.Builder
	result := negframe.ReframeResult{}
	for _, fragment := range fragments {
		if !fragment.FakAuthored || arm != "treatment" {
			out.WriteString(fragment.Text)
			if fragment.FakAuthored && arm != "treatment" {
				result.ResidualNegatives += len(negframe.Classify("gateway-runtime", fragment.Text))
			}
			continue
		}
		r := negframe.ReframePass(fragment.Text)
		out.WriteString(r.Text)
		result.Applied += r.Applied
		result.VerbatimFallback += r.VerbatimFallback
		result.ResidualNegatives += r.ResidualNegatives
	}
	result.Text = out.String()
	if path != "" {
		_ = negframe.AppendReframeJournal(path, negframe.NewReframeJournalSiteRow(traceID, site, arm, result, now), negframe.DefaultJournalMaxRows)
	}
	return result.Text
}

func reframeJournalPath() string { return os.Getenv("FAK_NEGFRAME_JOURNAL") }
func reframeJournalArm() string {
	if os.Getenv("FAK_NEGFRAME_ARM") == "control" {
		return "control"
	}
	return "treatment"
}
