package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/auditusage"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// sharePtr mirrors the pointer-share the fold hands back; the package already has an
// f64 helper of its own, so this one stays named for what it carries.
func sharePtr(v float64) *float64 { return &v }

// The renderer's one job that matters: an unmeasured corpus and an all-vendor
// corpus must not read the same. Both are "we self-hosted nothing" to a careless
// formatter, and only one of them is a finding.
func TestSelfHostedLineNeverPrintsAnUnmeasuredZero(t *testing.T) {
	unmeasured := selfHostedLine(&auditusage.SelfHostedRollup{
		Reason:       string(gatewayusageledger.ShareNotInstrumented),
		OutputTokens: 12000,
		Rows:         41,
	})
	if strings.Contains(unmeasured, "0.0%") || strings.Contains(unmeasured, "0%") {
		t.Errorf("an unmeasured corpus rendered a percentage: %q", unmeasured)
	}
	for _, want := range []string{"NOT INSTRUMENTED", "41", "12000", "not a measured zero"} {
		if !strings.Contains(unmeasured, want) {
			t.Errorf("line %q missing %q", unmeasured, want)
		}
	}

	earned := selfHostedLine(&auditusage.SelfHostedRollup{
		OutputShare:              sharePtr(0),
		ClassifiedOutputFraction: 1,
		VendorTurns:              4,
		VendorOutputTokens:       500,
		OutputTokens:             500,
		Rows:                     1,
	})
	if !strings.Contains(earned, "0.0%") {
		t.Errorf("an earned zero must render as a percentage: %q", earned)
	}
	if strings.Contains(earned, "NOT INSTRUMENTED") {
		t.Errorf("a measured corpus rendered as uninstrumented: %q", earned)
	}
}

// A share is only meaningful against the volume it covers, so the renderer is not
// allowed to print one without the other.
func TestSelfHostedLineQualifiesTheShareWithItsCoverage(t *testing.T) {
	line := selfHostedLine(&auditusage.SelfHostedRollup{
		OutputShare:              sharePtr(0.75),
		ClassifiedOutputFraction: 0.4,
		SelfHostedTurns:          3,
		SelfHostedOutputTokens:   600,
		VendorTurns:              1,
		VendorOutputTokens:       200,
		OutputTokens:             2000,
		Rows:                     2,
	})
	if !strings.Contains(line, "75.0%") {
		t.Errorf("missing the share: %q", line)
	}
	if !strings.Contains(line, "40.0%") {
		t.Errorf("missing the coverage qualifier — an unqualified share gets quoted as a fleet number: %q", line)
	}
	if !strings.Contains(line, "600/800") {
		t.Errorf("missing the auditable numerator/denominator: %q", line)
	}
}

func TestSelfHostedLineHandlesTheRemainingCases(t *testing.T) {
	if got := selfHostedLine(nil); got != "" {
		t.Errorf("no rollup must render no line, got %q", got)
	}
	noOutput := selfHostedLine(&auditusage.SelfHostedRollup{
		Reason:          string(gatewayusageledger.ShareNoClassifiedOutput),
		SelfHostedTurns: 2,
		VendorTurns:     1,
		Rows:            3,
	})
	if !strings.Contains(noOutput, "undefined") || !strings.Contains(noOutput, "3 turns classified") {
		t.Errorf("no-output case rendered as %q", noOutput)
	}
	// A reason the fold grows later must surface as itself rather than as a blank
	// line or an invented number.
	future := selfHostedLine(&auditusage.SelfHostedRollup{Reason: "some_future_reason", Rows: 7})
	if !strings.Contains(future, "some_future_reason") {
		t.Errorf("an unknown reason must still be shown: %q", future)
	}
}
