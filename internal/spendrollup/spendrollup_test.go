package spendrollup

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// local pointer helpers (the fleetaccounts equivalents are unexported).
func sp(s string) *string { return &s }
func ip(i int) *int       { return &i }
func bp(b bool) *bool     { return &b }

// worker builds a minimal routable claude worker row with the given live-session count.
func worker(account string, live int) fleetaccounts.Account {
	return fleetaccounts.Account{
		Account:      account,
		Product:      "claude",
		Kind:         fleetaccounts.KindWorker,
		LiveSessions: ip(live),
	}
}

func TestValidProvenance(t *testing.T) {
	cases := []struct {
		name string
		p    Provenance
		want bool
	}{
		{"witnessed", Witnessed, true},
		{"observed", Observed, true},
		{"empty", Provenance(""), false},
		{"unknown", Provenance("GUESSED"), false},
		{"case-sensitive-lowercase-rejected", Provenance("witnessed"), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ValidProvenance(tc.p); got != tc.want {
				t.Fatalf("ValidProvenance(%q) = %v, want %v", tc.p, got, tc.want)
			}
		})
	}
}

func TestValidBasis(t *testing.T) {
	cases := []struct {
		name string
		b    ValuationBasis
		want bool
	}{
		{"observed-net", BasisObservedNet, true},
		{"provider-billed", BasisProviderBilled, true},
		{"full-input", BasisFullInput, true},
		{"cache-read-marginal", BasisCacheReadMarginal, true},
		{"empty", ValuationBasis(""), false},
		{"unknown", ValuationBasis("MADE_UP"), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ValidBasis(tc.b); got != tc.want {
				t.Fatalf("ValidBasis(%q) = %v, want %v", tc.b, got, tc.want)
			}
		})
	}
}

func TestProviderUsageReported(t *testing.T) {
	cases := []struct {
		name string
		row  fleetaccounts.Account
		want bool
	}{
		{"nothing-reported", fleetaccounts.Account{}, false},
		{"throttled", fleetaccounts.Account{Throttled: bp(true)}, true},
		{"throttled-false", fleetaccounts.Account{Throttled: bp(false)}, false},
		{"weekly-window", fleetaccounts.Account{Weekly: sp("resets Mon 09:00")}, true},
		{"weekly-blank-is-not-reported", fleetaccounts.Account{Weekly: sp("   ")}, false},
		{"reset", fleetaccounts.Account{Reset: sp("14:00")}, true},
		{"reset-blank-is-not-reported", fleetaccounts.Account{Reset: sp("")}, false},
		{"blockkind-usage", fleetaccounts.Account{BlockKind: sp("usage")}, true},
		{"blockkind-usage-case-insensitive", fleetaccounts.Account{BlockKind: sp("Usage")}, true},
		{"blockkind-usage-padded", fleetaccounts.Account{BlockKind: sp("  usage  ")}, true},
		{"blockkind-auth-not-usage", fleetaccounts.Account{BlockKind: sp("auth")}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := providerUsageReported(tc.row); got != tc.want {
				t.Fatalf("providerUsageReported(%+v) = %v, want %v", tc.row, got, tc.want)
			}
		})
	}
}

// TestBuildRoutableFiltering asserts only routable workers contribute figures:
// non-account, excluded, and duplicate-identity rows are skipped.
func TestBuildRoutableFiltering(t *testing.T) {
	dup := worker(".claude-dup", 5)
	dup.IdentityRole = sp("duplicate")

	rows := []fleetaccounts.Account{
		dup,
		{Account: "plainfile", Kind: fleetaccounts.KindNonAccount},
		{Account: ".claude-backup", Product: "claude", Kind: fleetaccounts.KindExcluded, LiveSessions: ip(9)},
	}
	got := Build(rows)
	if len(got.Figures) != 0 {
		t.Fatalf("expected no figures from non-routable rows, got %d: %+v", len(got.Figures), got.Figures)
	}
	if len(got.Subtotals) != 0 {
		t.Fatalf("expected no subtotals, got %d", len(got.Subtotals))
	}
	// No figures at all => no warning is emitted (the warning is gated on len(figures) > 0).
	if len(got.Warnings) != 0 {
		t.Fatalf("expected no warnings for an empty rollup, got %v", got.Warnings)
	}
	if got.Schema != Schema {
		t.Fatalf("schema = %q, want %q", got.Schema, Schema)
	}
	if err := got.Gate(); err != nil {
		t.Fatalf("empty rollup should pass Gate, got %v", err)
	}
}

// TestBuildWitnessedFigure asserts a plain routable worker emits exactly one WITNESSED
// live-session figure, valued as OBSERVED_NET, and triggers the "no provider-relayed
// figure" warning because no OBSERVED figure is present.
func TestBuildWitnessedFigure(t *testing.T) {
	got := Build([]fleetaccounts.Account{worker(".claude-team", 3)})
	if len(got.Figures) != 1 {
		t.Fatalf("expected 1 figure, got %d: %+v", len(got.Figures), got.Figures)
	}
	f := got.Figures[0]
	if f.Provenance != Witnessed {
		t.Errorf("provenance = %q, want %q", f.Provenance, Witnessed)
	}
	if f.Basis != BasisObservedNet {
		t.Errorf("basis = %q, want %q", f.Basis, BasisObservedNet)
	}
	if f.Amount != 3 {
		t.Errorf("amount = %v, want 3", f.Amount)
	}
	if f.Unit != "live-sessions" {
		t.Errorf("unit = %q, want live-sessions", f.Unit)
	}
	if f.Label != "fak live-session load" {
		t.Errorf("label = %q, want %q", f.Label, "fak live-session load")
	}
	if f.Provider != "anthropic" {
		t.Errorf("provider = %q, want anthropic (claude account)", f.Provider)
	}
	if !strings.Contains(f.Source, "sessions.json") {
		t.Errorf("source = %q, want it to name the local session ledger", f.Source)
	}
	if len(got.Warnings) != 1 || !strings.Contains(got.Warnings[0], "no provider-relayed figure") {
		t.Errorf("want the no-provider-relayed warning, got %v", got.Warnings)
	}
	// A nil LiveSessions pointer must fold to a zero amount, not panic.
	zero := Build([]fleetaccounts.Account{{Account: ".claude-x", Product: "claude", Kind: fleetaccounts.KindWorker}})
	if len(zero.Figures) != 1 || zero.Figures[0].Amount != 0 {
		t.Fatalf("nil LiveSessions should yield amount 0, got %+v", zero.Figures)
	}
}

// TestBuildObservedFigure asserts a worker whose provider reported a usage-window posture
// adds a second, OBSERVED / PROVIDER_BILLED figure and suppresses the warning.
func TestBuildObservedFigure(t *testing.T) {
	row := worker(".claude-throttled", 2)
	row.Throttled = bp(true)
	got := Build([]fleetaccounts.Account{row})
	if len(got.Figures) != 2 {
		t.Fatalf("expected 2 figures (witnessed + observed), got %d: %+v", len(got.Figures), got.Figures)
	}
	// Order is deterministic: witnessed first, then observed for the same row.
	if got.Figures[0].Provenance != Witnessed {
		t.Errorf("figure[0] provenance = %q, want WITNESSED", got.Figures[0].Provenance)
	}
	obs := got.Figures[1]
	if obs.Provenance != Observed {
		t.Errorf("figure[1] provenance = %q, want OBSERVED", obs.Provenance)
	}
	if obs.Basis != BasisProviderBilled {
		t.Errorf("observed basis = %q, want PROVIDER_BILLED", obs.Basis)
	}
	if obs.Unit != "provider-usage-window" {
		t.Errorf("observed unit = %q, want provider-usage-window", obs.Unit)
	}
	if obs.Amount != 1 {
		t.Errorf("observed amount = %v, want 1", obs.Amount)
	}
	// An OBSERVED figure IS present, so the no-provider-relayed warning is suppressed.
	if len(got.Warnings) != 0 {
		t.Errorf("expected no warnings when an OBSERVED figure is present, got %v", got.Warnings)
	}
}

// TestBuildSubtotalsFold asserts figures sharing a (provenance, unit, basis) key are folded
// into one subtotal that sums their amounts, while distinct keys stay separate.
func TestBuildSubtotalsFold(t *testing.T) {
	a := worker(".claude-a", 2)
	b := worker(".claude-b", 5)
	b.Throttled = bp(true) // adds an OBSERVED figure for b
	got := Build([]fleetaccounts.Account{a, b})

	// Two WITNESSED live-session figures fold; one OBSERVED figure stands alone.
	if len(got.Subtotals) != 2 {
		t.Fatalf("expected 2 subtotals, got %d: %+v", len(got.Subtotals), got.Subtotals)
	}

	var witnessed, observed *Subtotal
	for i := range got.Subtotals {
		switch got.Subtotals[i].Provenance {
		case Witnessed:
			witnessed = &got.Subtotals[i]
		case Observed:
			observed = &got.Subtotals[i]
		}
	}
	if witnessed == nil || observed == nil {
		t.Fatalf("want one WITNESSED and one OBSERVED subtotal, got %+v", got.Subtotals)
	}
	if witnessed.Amount != 7 || witnessed.Figures != 2 {
		t.Errorf("witnessed subtotal = amount %v over %d, want amount 7 over 2", witnessed.Amount, witnessed.Figures)
	}
	if witnessed.Unit != "live-sessions" || witnessed.Basis != BasisObservedNet {
		t.Errorf("witnessed subtotal key = (%q,%q), want (live-sessions, OBSERVED_NET)", witnessed.Unit, witnessed.Basis)
	}
	if observed.Amount != 1 || observed.Figures != 1 {
		t.Errorf("observed subtotal = amount %v over %d, want amount 1 over 1", observed.Amount, observed.Figures)
	}
	// Subtotals are sorted by provenance (OBSERVED < WITNESSED lexically).
	if got.Subtotals[0].Provenance != Observed || got.Subtotals[1].Provenance != Witnessed {
		t.Errorf("subtotals not sorted by provenance: %q then %q", got.Subtotals[0].Provenance, got.Subtotals[1].Provenance)
	}
	// Build's output is always fully labeled and must pass its own Gate.
	if err := got.Gate(); err != nil {
		t.Fatalf("Build output must pass Gate, got %v", err)
	}
}

func TestGate(t *testing.T) {
	cases := []struct {
		name       string
		figures    []Figure
		wantErr    bool
		wantSubstr string
	}{
		{
			name:    "fully-labeled-passes",
			figures: []Figure{{Account: "a", Provenance: Witnessed, Basis: BasisObservedNet}},
			wantErr: false,
		},
		{
			name:       "missing-provenance-fails",
			figures:    []Figure{{Account: "a", Basis: BasisObservedNet}},
			wantErr:    true,
			wantSubstr: "no WITNESSED/OBSERVED provenance label",
		},
		{
			name:       "missing-basis-fails",
			figures:    []Figure{{Account: "a", Provenance: Observed}},
			wantErr:    true,
			wantSubstr: "names no valuation basis",
		},
		{
			name:       "both-missing-counts-two-defects",
			figures:    []Figure{{Account: "a"}},
			wantErr:    true,
			wantSubstr: "2 unlabeled spend figure(s)",
		},
		{
			name:    "empty-rollup-passes",
			figures: nil,
			wantErr: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := Rollup{Figures: tc.figures}.Gate()
			if tc.wantErr && err == nil {
				t.Fatalf("expected an error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("expected nil, got %v", err)
			}
			if tc.wantErr && tc.wantSubstr != "" && !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantSubstr)
			}
		})
	}
}

// TestGateIdentifiesFigureWithoutAccount asserts an unlabeled figure lacking an account id
// is named by its positional index rather than an empty string.
func TestGateIdentifiesFigureWithoutAccount(t *testing.T) {
	err := Rollup{Figures: []Figure{{Label: "orphan"}}}.Gate()
	if err == nil {
		t.Fatal("expected an error for an unlabeled figure")
	}
	if !strings.Contains(err.Error(), "figure[0]") {
		t.Fatalf("expected the defect to be keyed by figure index, got %q", err.Error())
	}
}

func TestRender(t *testing.T) {
	// Empty rollup renders the schema header and the no-accounts line.
	empty := Render(Build(nil))
	if !strings.Contains(empty, Schema) {
		t.Errorf("empty render missing schema %q: %q", Schema, empty)
	}
	if !strings.Contains(empty, "no routable worker accounts") {
		t.Errorf("empty render missing the no-accounts line: %q", empty)
	}

	// Populated rollup renders figures, subtotals, and the provenance labels.
	row := worker(".claude-team", 4)
	row.Throttled = bp(true)
	out := Render(Build([]fleetaccounts.Account{row}))
	for _, want := range []string{"figures:", "subtotals", string(Witnessed), string(Observed), ".claude-team"} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q:\n%s", want, out)
		}
	}
}
