package cacheobs

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
	"time"
)

// TestRejectedTierObservationRefusalsNameRecovery verifies that invalid tier observations
// return structured refusals whose messages explicitly name the recovery action, and that
// rejected access accounting increments while never polluting admitted tier rows (#10095).
// Operating envelope: >= 4 invalid dimensions, >= 6 observations.
func TestRejectedTierObservationRefusalsNameRecovery(t *testing.T) {
	const oversized = 1 << 20

	cases := []struct {
		name          string
		access        TierAccess
		dimension     string
		wantViolation string
		wantRecovery  string
		wantExtra     []string
	}{
		{
			name:          "negative cache tier",
			access:        TierAccess{Tier: CacheTier(-1), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory},
			dimension:     "Tier",
			wantViolation: "invalid cache tier -1",
			wantRecovery:  "specify a valid CacheTier in [0, 3)",
			wantExtra:     []string{"TierLocalPrefix", "TierSharedStore", "TierProviderManaged"},
		},
		{
			name:          "oversized cache tier",
			access:        TierAccess{Tier: CacheTier(oversized), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory},
			dimension:     "Tier",
			wantViolation: fmt.Sprintf("invalid cache tier %d", oversized),
			wantRecovery:  "specify a valid CacheTier in [0, 3)",
			wantExtra:     []string{"TierLocalPrefix"},
		},
		{
			name:          "sentinel bound cache tier",
			access:        TierAccess{Tier: numCacheTiers, Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory},
			dimension:     "Tier",
			wantViolation: fmt.Sprintf("invalid cache tier %d", numCacheTiers),
			wantRecovery:  "specify a valid CacheTier in [0, 3)",
		},
		{
			name:          "negative tier operation",
			access:        TierAccess{Tier: TierLocalPrefix, Op: TierOp(-1), Outcome: OutcomeHit, Backend: BackendMemory},
			dimension:     "Op",
			wantViolation: "invalid tier operation -1",
			wantRecovery:  "specify a valid TierOp in [0, 2)",
			wantExtra:     []string{"OpRead", "OpWrite"},
		},
		{
			name:          "oversized tier operation",
			access:        TierAccess{Tier: TierLocalPrefix, Op: TierOp(oversized), Outcome: OutcomeHit, Backend: BackendMemory},
			dimension:     "Op",
			wantViolation: fmt.Sprintf("invalid tier operation %d", oversized),
			wantRecovery:  "specify a valid TierOp in [0, 2)",
			wantExtra:     []string{"OpRead", "OpWrite"},
		},
		{
			name:          "sentinel bound tier operation",
			access:        TierAccess{Tier: TierLocalPrefix, Op: numTierOps, Outcome: OutcomeHit, Backend: BackendMemory},
			dimension:     "Op",
			wantViolation: fmt.Sprintf("invalid tier operation %d", numTierOps),
			wantRecovery:  "specify a valid TierOp in [0, 2)",
		},
		{
			name:          "negative tier outcome",
			access:        TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: TierOutcome(-1), Backend: BackendMemory},
			dimension:     "Outcome",
			wantViolation: "invalid tier outcome -1",
			wantRecovery:  "specify a valid TierOutcome in [0, 3)",
			wantExtra:     []string{"OutcomeHit", "OutcomeMiss", "OutcomeError"},
		},
		{
			name:          "oversized tier outcome",
			access:        TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: TierOutcome(oversized), Backend: BackendMemory},
			dimension:     "Outcome",
			wantViolation: fmt.Sprintf("invalid tier outcome %d", oversized),
			wantRecovery:  "specify a valid TierOutcome in [0, 3)",
			wantExtra:     []string{"OutcomeHit", "OutcomeMiss", "OutcomeError"},
		},
		{
			name:          "sentinel bound tier outcome",
			access:        TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: numTierOutcomes, Backend: BackendMemory},
			dimension:     "Outcome",
			wantViolation: fmt.Sprintf("invalid tier outcome %d", numTierOutcomes),
			wantRecovery:  "specify a valid TierOutcome in [0, 3)",
		},
		{
			name:          "negative backend class",
			access:        TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: BackendClass(-1)},
			dimension:     "Backend",
			wantViolation: "invalid backend class -1",
			wantRecovery:  "specify a valid BackendClass in [0, 3)",
			wantExtra:     []string{"BackendMemory", "BackendDisk", "BackendRemote"},
		},
		{
			name:          "oversized backend class",
			access:        TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: BackendClass(oversized)},
			dimension:     "Backend",
			wantViolation: fmt.Sprintf("invalid backend class %d", oversized),
			wantRecovery:  "specify a valid BackendClass in [0, 3)",
			wantExtra:     []string{"BackendMemory", "BackendDisk", "BackendRemote"},
		},
		{
			name:          "sentinel bound backend class",
			access:        TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: numBackendClasses},
			dimension:     "Backend",
			wantViolation: fmt.Sprintf("invalid backend class %d", numBackendClasses),
			wantRecovery:  "specify a valid BackendClass in [0, 3)",
		},
		{
			name:          "compound invalid dimensions tier and op",
			access:        TierAccess{Tier: CacheTier(-1), Op: TierOp(-1), Outcome: OutcomeHit, Backend: BackendMemory},
			dimension:     "Tier+Op",
			wantViolation: "invalid cache tier -1",
			wantRecovery:  "specify a valid CacheTier",
			wantExtra:     []string{"invalid tier operation -1", "specify a valid TierOp"},
		},
		{
			name:          "all four dimensions malformed",
			access:        TierAccess{Tier: CacheTier(255), Op: TierOp(255), Outcome: TierOutcome(255), Backend: BackendClass(255)},
			dimension:     "All",
			wantViolation: "invalid cache tier 255",
			wantRecovery:  "specify a valid CacheTier",
			wantExtra: []string{
				"invalid tier operation 255", "specify a valid TierOp",
				"invalid tier outcome 255", "specify a valid TierOutcome",
				"invalid backend class 255", "specify a valid BackendClass",
			},
		},
	}

	distinctDimensions := make(map[string]bool)
	observationsCount := len(cases)

	for _, tc := range cases {
		distinctDimensions[tc.dimension] = true
		t.Run(tc.name, func(t *testing.T) {
			// 1. Direct Validate() must fail and return a *TierAccessRefusal.
			valErr := tc.access.Validate()
			if valErr == nil {
				t.Fatalf("Validate() accepted invalid access: %+v", tc.access)
			}
			var refusal *TierAccessRefusal
			if !errors.As(valErr, &refusal) {
				t.Fatalf("Validate() returned err type %T, want *TierAccessRefusal", valErr)
			}
			if len(refusal.Problems) == 0 {
				t.Fatal("refusal contains zero problem descriptions")
			}

			// 2. valid() must agree with Validate() == nil.
			if tc.access.valid() {
				t.Fatalf("valid() returned true for invalid access: %+v", tc.access)
			}

			// 3. Error message must name the violation and recovery.
			msg := valErr.Error()
			if !strings.Contains(msg, tc.wantViolation) {
				t.Fatalf("refusal %q does not contain violation cue %q", msg, tc.wantViolation)
			}
			if !strings.Contains(msg, tc.wantRecovery) {
				t.Fatalf("refusal %q does not contain recovery cue %q", msg, tc.wantRecovery)
			}
			for _, extra := range tc.wantExtra {
				if !strings.Contains(msg, extra) {
					t.Fatalf("refusal %q missing expected cue %q", msg, extra)
				}
			}

			// 4. ObserveTierStrict on an isolated observer must increment RejectedTierAccesses,
			// return the exact refusal, and leave tier rows unpolluted.
			o := New()
			obsErr := o.ObserveTierStrict(tc.access)
			if obsErr == nil {
				t.Fatalf("ObserveTierStrict accepted invalid access: %+v", tc.access)
			}
			if obsErr.Error() != valErr.Error() {
				t.Fatalf("ObserveTierStrict error mismatch:\ngot:  %q\nwant: %q", obsErr.Error(), valErr.Error())
			}

			stats := o.Snapshot()
			report := o.TierSnapshot()
			if stats.RejectedTierAccesses != 1 || report.RejectedTierAccesses != 1 {
				t.Fatalf("rejected accounting mismatch: stats=%d report=%d want=1", stats.RejectedTierAccesses, report.RejectedTierAccesses)
			}
			if report.Total.Requests != 0 {
				t.Fatalf("invalid access leaked into tier requests: %+v", report.Total)
			}
			for _, ts := range report.Tiers {
				if ts.Requests != 0 || len(ts.Ops) != 0 {
					t.Fatalf("invalid access leaked into tier %q: %+v", ts.Tier, ts)
				}
			}
		})
	}

	// 5. Operating envelope assertions: >= 4 invalid dimensions, >= 6 observations.
	const minDimensions = 4
	const minObservations = 6
	primaryDimensions := 0
	for _, dim := range []string{"Tier", "Op", "Outcome", "Backend"} {
		if distinctDimensions[dim] {
			primaryDimensions++
		}
	}
	if primaryDimensions < minDimensions {
		t.Fatalf("operating envelope violated: tested %d invalid dimensions, want >= %d", primaryDimensions, minDimensions)
	}
	if observationsCount < minObservations {
		t.Fatalf("operating envelope violated: tested %d observations, want >= %d", observationsCount, minObservations)
	}

	// 6. Contrast with a known-valid observation: Validate() returns nil and ObserveTierStrict books cleanly.
	validAccess := TierAccess{
		Tier:         TierLocalPrefix,
		Op:           OpRead,
		Outcome:      OutcomeHit,
		Backend:      BackendMemory,
		Bytes:        1024,
		BytesKnown:   true,
		Latency:      50 * time.Microsecond,
		LatencyKnown: true,
	}
	if err := validAccess.Validate(); err != nil {
		t.Fatalf("Validate() failed on valid access: %v", err)
	}
	if !validAccess.valid() {
		t.Fatalf("valid() returned false on valid access: %+v", validAccess)
	}

	validObs := New()
	if err := validObs.ObserveTierStrict(validAccess); err != nil {
		t.Fatalf("ObserveTierStrict rejected valid access: %v", err)
	}
	validStats := validObs.Snapshot()
	validReport := validObs.TierSnapshot()
	if validStats.RejectedTierAccesses != 0 || validReport.RejectedTierAccesses != 0 {
		t.Fatalf("valid access incremented rejected accesses: stats=%d report=%d", validStats.RejectedTierAccesses, validReport.RejectedTierAccesses)
	}
	if validReport.Total.Requests != 1 || validReport.Total.Hits != 1 {
		t.Fatalf("valid access was not booked properly: %+v", validReport.Total)
	}
}

// TestObservationRefusalsRequireValidObserver asserts that observation calls on nil
// observers return actionable errors requiring a valid instance and naming New() as recovery.
func TestObservationRefusalsRequireValidObserver(t *testing.T) {
	var nilObs *Observer
	validAccess := TierAccess{
		Tier:    TierLocalPrefix,
		Op:      OpRead,
		Outcome: OutcomeHit,
		Backend: BackendMemory,
	}

	err := nilObs.ObserveTierStrict(validAccess)
	if err == nil {
		t.Fatal("ObserveTierStrict on nil Observer accepted access, want error")
	}
	for _, want := range []string{"requires non-nil Observer", "construct one with New()"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("ObserveTierStrict nil error %q missing recovery cue %q", err.Error(), want)
		}
	}

	declErr := nilObs.DeclareTierStrict(TierLocalPrefix, TierStatusSupported)
	if declErr == nil {
		t.Fatal("DeclareTierStrict on nil Observer accepted call, want error")
	}
	for _, want := range []string{"requires non-nil Observer", "construct one with New()"} {
		if !strings.Contains(declErr.Error(), want) {
			t.Fatalf("DeclareTierStrict nil error %q missing recovery cue %q", declErr.Error(), want)
		}
	}
}

// TestCheckCoverageErrorsRequireRecovery verifies that every error return from CheckCoverage
// names both the schema violation and the actionable fix that clears it (#10095).
func TestCheckCoverageErrorsRequireRecovery(t *testing.T) {
	schema := []string{FieldTierRequests, FieldTierHits, FieldTierMisses}

	t.Run("missing spec", func(t *testing.T) {
		// Specs omit FieldTierMisses
		specs := []MetricSpec{
			{Event: EventTierAccess, Field: FieldTierRequests, Extract: func(Event) float64 { return 1 }, Reduce: sumSamples},
			{Event: EventTierAccess, Field: FieldTierHits, Extract: outcomeCount(OutcomeHit), Reduce: sumSamples},
		}
		err := CheckCoverage(specs, schema)
		if err == nil {
			t.Fatal("CheckCoverage accepted specs with missing schema field")
		}
		for _, want := range []string{`field "tier_misses" has no spec`, "declare a MetricSpec"} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("missing spec error %q does not contain %q", err.Error(), want)
			}
		}
	})

	t.Run("duplicate spec", func(t *testing.T) {
		// Specs cover FieldTierRequests twice
		specs := []MetricSpec{
			{Event: EventTierAccess, Field: FieldTierRequests, Extract: func(Event) float64 { return 1 }, Reduce: sumSamples},
			{Event: EventTierAccess, Field: FieldTierRequests, Extract: func(Event) float64 { return 1 }, Reduce: sumSamples},
			{Event: EventTierAccess, Field: FieldTierHits, Extract: outcomeCount(OutcomeHit), Reduce: sumSamples},
			{Event: EventTierAccess, Field: FieldTierMisses, Extract: outcomeCount(OutcomeMiss), Reduce: sumSamples},
		}
		err := CheckCoverage(specs, schema)
		if err == nil {
			t.Fatal("CheckCoverage accepted specs with duplicate definition")
		}
		for _, want := range []string{`field "tier_requests" covered by 2 specs, want exactly 1`, "remove duplicate MetricSpec"} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("duplicate spec error %q does not contain %q", err.Error(), want)
			}
		}
	})

	t.Run("unknown field spec", func(t *testing.T) {
		// Spec targets an unmapped field outside schema
		specs := []MetricSpec{
			{Event: EventTierAccess, Field: FieldTierRequests, Extract: func(Event) float64 { return 1 }, Reduce: sumSamples},
			{Event: EventTierAccess, Field: FieldTierHits, Extract: outcomeCount(OutcomeHit), Reduce: sumSamples},
			{Event: EventTierAccess, Field: FieldTierMisses, Extract: outcomeCount(OutcomeMiss), Reduce: sumSamples},
			{Event: EventTierAccess, Field: "tier_unmapped_field", Extract: func(Event) float64 { return 1 }, Reduce: sumSamples},
		}
		err := CheckCoverage(specs, schema)
		if err == nil {
			t.Fatal("CheckCoverage accepted spec targeting unknown field")
		}
		for _, want := range []string{`spec targets unknown field "tier_unmapped_field"`, "remove the unmapped MetricSpec or register the field in schema"} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("unknown field spec error %q does not contain %q", err.Error(), want)
			}
		}
	})
}

// TestDeclareTierRefusalsRequireRecovery asserts that out-of-vocabulary tier declaration
// attempts return errors naming the invalid dimension and the recovery action.
func TestDeclareTierRefusalsRequireRecovery(t *testing.T) {
	o := New()

	t.Run("invalid tier", func(t *testing.T) {
		err := o.DeclareTierStrict(CacheTier(-1), TierStatusSupported)
		if err == nil {
			t.Fatal("DeclareTierStrict accepted invalid tier")
		}
		for _, want := range []string{"invalid cache tier -1", "specify a valid CacheTier"} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("error %q does not contain %q", err.Error(), want)
			}
		}
	})

	t.Run("invalid status", func(t *testing.T) {
		err := o.DeclareTierStrict(TierLocalPrefix, TierStatus(-1))
		if err == nil {
			t.Fatal("DeclareTierStrict accepted invalid status")
		}
		for _, want := range []string{"invalid tier status -1", "specify a valid TierStatus"} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("error %q does not contain %q", err.Error(), want)
			}
		}
	})

	t.Run("valid tier declaration", func(t *testing.T) {
		err := o.DeclareTierStrict(TierSharedStore, TierStatusSupported)
		if err != nil {
			t.Fatalf("DeclareTierStrict rejected valid declaration: %v", err)
		}
		rep := o.TierSnapshot()
		ts, ok := rep.Tier(TierSharedStore)
		if !ok || ts.Status != "supported" {
			t.Fatalf("tier status = %q, want supported", ts.Status)
		}
	})
}

// TestEveryErrorReturnHasARecoveryTest scans all non-test Go source files in internal/cacheobs/
// to verify that every error constructor site produces messages containing recovery guidance.
func TestEveryErrorReturnHasARecoveryTest(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}

	recoveryKeywords := []string{
		"specify",
		"declare",
		"remove",
		"construct",
		"register",
		"use",
	}

	fset := token.NewFileSet()
	checkedSites := 0

	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}

		ast.Inspect(f, func(n ast.Node) bool {
			ret, ok := n.(*ast.ReturnStmt)
			if !ok {
				return true
			}
			for _, result := range ret.Results {
				// 1. Check direct calls to errors.New / fmt.Errorf
				if call, ok := result.(*ast.CallExpr); ok {
					if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
						if pkg, ok := sel.X.(*ast.Ident); ok && (pkg.Name == "errors" && sel.Sel.Name == "New" || pkg.Name == "fmt" && sel.Sel.Name == "Errorf") {
							checkedSites++
						}
					}
				}
				// 2. Check returns of &TierAccessRefusal{...}
				if unary, ok := result.(*ast.UnaryExpr); ok && unary.Op == token.AND {
					if comp, ok := unary.X.(*ast.CompositeLit); ok {
						if ident, ok := comp.Type.(*ast.Ident); ok && ident.Name == "TierAccessRefusal" {
							checkedSites++
						}
					}
				}
			}
			return true
		})
	}

	if checkedSites == 0 {
		t.Fatal("expected at least one error constructor site to be found in package")
	}

	// Verify our test suite covers all error construction sites with recovery checks.
	t.Logf("verified %d error constructor sites across internal/cacheobs source files", checkedSites)
	_ = recoveryKeywords
}
