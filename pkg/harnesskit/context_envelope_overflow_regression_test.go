package harnesskit

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestDeriveContextEnvelopeOverrideBounds(t *testing.T) {
	t.Parallel()
	maxInt := int(^uint(0) >> 1)
	minInt := -maxInt - 1
	wantSmallDefault := ContextEnvelope{
		ServedWindow: 8192, ContextWindow: 8192, OutputTokens: 4096,
		ReserveTokens: 7168, CompactTrigger: 1024, KeepRecentTokens: 4096,
		SummaryTokens: 4096, PostCompactTokens: 24576, ReclaimTokens: -23552,
		Reason: ReasonWindowTooSmallForHarness, Provenance: ContextEnvelopeProvenance,
	}
	cases := []struct {
		name             string
		served           int
		fixed            int
		output           int
		source           WindowSource
		refusal          string
		wantErr          error
		wantSmallDefault bool
	}{
		{name: "small default", served: 8192, wantSmallDefault: true},
		{name: "negative fallback", served: 8192, fixed: -1, output: -1, wantSmallDefault: true},
		{name: "min int fallback", served: 8192, fixed: minInt, output: minInt, wantSmallDefault: true},
		{name: "max int fixed", served: 8192, fixed: maxInt, refusal: "fixed"},
		{name: "max int output", served: 8192, output: maxInt, refusal: "output"},
		{name: "both oversized fixed first", served: 8192, fixed: maxInt, output: maxInt, refusal: "fixed"},
		{name: "small fixed boundary", served: 65536, fixed: 65536},
		{name: "small fixed above", served: 65536, fixed: 65537, refusal: "fixed"},
		{name: "small output boundary", served: 65536, output: 65536},
		{name: "small output above", served: 65536, output: 65537, refusal: "output"},
		{name: "cap fixed boundary", served: QualityCapTokens * 2, fixed: QualityCapTokens},
		{name: "cap fixed above", served: QualityCapTokens * 2, fixed: QualityCapTokens + 1, refusal: "fixed"},
		{name: "cap output boundary", served: QualityCapTokens * 2, output: QualityCapTokens},
		{name: "cap output above", served: QualityCapTokens * 2, output: QualityCapTokens + 1, refusal: "output"},
		{name: "max int served bounded overrides", served: maxInt, fixed: QualityCapTokens, output: QualityCapTokens},
		{name: "source refusal first", served: 8192, source: WindowDerived, fixed: maxInt, output: maxInt, wantErr: ErrDerivedWindow},
		{name: "window refusal first", served: 0, fixed: maxInt, output: maxInt, wantErr: ErrNoServedWindow},
	}
	// Observe stable identities through the existing API so this regression also
	// compiles against the parent, before the new refusal sentinels exist.
	refusals := make(map[string]error)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			source := tc.source
			if source == "" {
				source = WindowServed
			}
			got, err := DeriveContextEnvelope(ContextEnvelopeInput{
				ServedWindow: tc.served, Source: source,
				FixedPromptTokens: tc.fixed, MaxOutputTokens: tc.output,
			})
			if tc.refusal != "" {
				if err == nil {
					t.Fatalf("oversized %s override accepted: %+v", tc.refusal, got)
				}
				if got != (ContextEnvelope{}) {
					t.Fatalf("refusal returned a nonzero envelope: %+v", got)
				}
				if first, ok := refusals[tc.refusal]; ok && err != first {
					t.Fatalf("%s refusal changed sentinel identity: %v / %v", tc.refusal, first, err)
				}
				refusals[tc.refusal] = err
				return
			}
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) || got != (ContextEnvelope{}) {
					t.Fatalf("existing refusal changed: envelope=%+v error=%v, want %v", got, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("valid override rejected: %v", err)
			}
			if tc.wantSmallDefault && got != wantSmallDefault {
				t.Fatalf("small-window fallback changed:\n got %+v\nwant %+v", got, wantSmallDefault)
			}
			if tc.output > 0 && got.OutputTokens != tc.output {
				t.Fatalf("output override changed: got %d, want %d", got.OutputTokens, tc.output)
			}
			if tc.fixed > 0 && got.PostCompactTokens != tc.fixed+got.SummaryTokens+got.KeepRecentTokens {
				t.Fatalf("fixed-prefix override changed: %+v", got)
			}
		})
	}
	if refusals["fixed"] != nil && refusals["fixed"] == refusals["output"] {
		t.Fatal("fixed-prefix and output refusals share a sentinel")
	}
}
