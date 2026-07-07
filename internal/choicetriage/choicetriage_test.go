package choicetriage

import "testing"

func TestTriageDispositions(t *testing.T) {
	cases := []struct {
		name string
		in   Signal
		want Disposition
		hum  bool
	}{
		{
			name: "release decision is the irreducible human residual",
			in:   Signal{Severity: "decision", Source: "release", Question: "cut the release?", OptionCount: 2},
			want: HumanResidual,
			hum:  true,
		},
		{
			name: "auth/login page is a human residual even phrased as a page",
			in:   Signal{Severity: "page", Source: "gateway", Detail: "seat needs re-auth / login", OptionCount: 3},
			want: HumanResidual,
			hum:  true,
		},
		{
			name: "priority call is a human residual",
			in:   Signal{Severity: "decision", Question: "which priority wins?", OptionCount: 2},
			want: HumanResidual,
			hum:  true,
		},
		{
			name: "missing-report page with a runnable fix is obvious, not a page",
			in: Signal{
				Severity: "page", Source: "cadence", Question: "cadence report missing",
				Action: "generate `fak cadence --json` and pass it with --cadence", OptionCount: 3,
			},
			want: TakeObvious,
		},
		{
			name: "action that opens with a command literal is obvious",
			in:   Signal{Source: "program", Action: "`fak program report --json`", OptionCount: 2},
			want: TakeObvious,
		},
		{
			name: "a single surfaced option is a fake choice — take it",
			in:   Signal{Source: "agent", Question: "let agents continue?", OptionCount: 1},
			want: TakeObvious,
		},
		{
			name: "roadmap-scale work with no single command files a ticket",
			in:   Signal{Source: "milestone", Question: "advance the roadmap epic?", Detail: "multiple frontiers open", OptionCount: 2},
			want: FileTicket,
		},
		{
			name: "explicit oversized scope files a ticket",
			in:   Signal{Source: "watch", Question: "reduce measured friction?", ScopeLarge: true, OptionCount: 2},
			want: FileTicket,
		},
		{
			name: "unclear-but-knowable defaults to a fresh context window, not a human",
			in:   Signal{Source: "watch", Question: "investigate the friction spike?", Detail: "throughput dipped", OptionCount: 3},
			want: FreshContext,
		},
		{
			name: "the zero signal is a fresh-context evaluation, never a page",
			in:   Signal{},
			want: FreshContext,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Triage(tc.in)
			if got.Disposition != tc.want {
				t.Fatalf("disposition = %q, want %q (reason: %s)", got.Disposition, tc.want, got.Reason)
			}
			if !got.Disposition.Valid() {
				t.Fatalf("disposition %q is not a closed member", got.Disposition)
			}
			if got.NeedsHuman != tc.hum {
				t.Fatalf("NeedsHuman = %v, want %v", got.NeedsHuman, tc.hum)
			}
			if got.NeedsHuman != got.Disposition.NeedsHuman() {
				t.Fatalf("verdict.NeedsHuman %v disagrees with disposition.NeedsHuman() %v", got.NeedsHuman, got.Disposition.NeedsHuman())
			}
			if got.Resolve == "" || got.Reason == "" {
				t.Fatalf("verdict must carry a reason and a resolve, got %+v", got)
			}
		})
	}
}

// TestHumanResidualIsEarnedNotDefault pins the doctrine: the default for an
// ambiguous choice is FreshContext, and HumanResidual only fires on a real
// authority signal. This is the inversion the package exists to enforce.
func TestHumanResidualIsEarnedNotDefault(t *testing.T) {
	ambiguous := Signal{Source: "program", Question: "what next here?", Detail: "unclear", OptionCount: 3}
	if v := Triage(ambiguous); v.Disposition != FreshContext {
		t.Fatalf("ambiguous choice should default to FRESH_CONTEXT, got %q", v.Disposition)
	}
	if v := Triage(ambiguous); v.NeedsHuman {
		t.Fatal("ambiguous choice must not be routed to a human by default")
	}
	earned := Signal{Source: "release", Question: "approve the publish?", OptionCount: 2}
	if v := Triage(earned); !v.NeedsHuman {
		t.Fatal("a real approve/publish authority signal must earn HUMAN_RESIDUAL")
	}
}
