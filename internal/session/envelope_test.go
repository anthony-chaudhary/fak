package session

import (
	"reflect"
	"testing"
	"time"
)

// TestBudgetEnvelopeParseFlags pins ParseEnvelopeFlags's string syntax at every axis and its
// boundary tokens (empty = not-stated, "unbounded" = explicit sentinel, a real value = a
// ceiling), so the CLI's flag strings map onto Envelope deterministically.
func TestBudgetEnvelopeParseFlags(t *testing.T) {
	cases := []struct {
		name                                        string
		tokens, wallClock, turns, spend, throughput string
		want                                        Envelope
		wantErr                                     bool
	}{
		{
			name: "all empty is the zero envelope",
			want: Envelope{},
		},
		{
			name:   "tokens only",
			tokens: "50000",
			want:   Envelope{Tokens: 50000},
		},
		{
			name:      "wall clock duration",
			wallClock: "10m",
			want:      Envelope{WallClock: 10 * time.Minute},
		},
		{
			name:  "turns only",
			turns: "25",
			want:  Envelope{Turns: 25},
		},
		{
			name:  "spend dollar sign",
			spend: "$5.25",
			want:  Envelope{SpendCapCents: 525},
		},
		{
			name:  "spend bare dollars",
			spend: "5",
			want:  Envelope{SpendCapCents: 500},
		},
		{
			name:       "throughput floor",
			throughput: "42.5",
			want:       Envelope{ThroughputFloor: 42.5},
		},
		{
			name:   "explicit unbounded tokens",
			tokens: "unbounded",
			want:   Envelope{Tokens: Unbounded},
		},
		{
			name:  "explicit unbounded turns case-insensitive",
			turns: "UNBOUNDED",
			want:  Envelope{Turns: Unbounded},
		},
		{
			name:       "every axis together",
			tokens:     "100000",
			wallClock:  "1h30m",
			turns:      "40",
			spend:      "$12.00",
			throughput: "10",
			want: Envelope{
				Tokens:          100000,
				WallClock:       90 * time.Minute,
				Turns:           40,
				SpendCapCents:   1200,
				ThroughputFloor: 10,
			},
		},
		{name: "malformed tokens", tokens: "abc", wantErr: true},
		{name: "zero tokens rejected (ambiguous)", tokens: "0", wantErr: true},
		{name: "negative tokens rejected", tokens: "-5", wantErr: true},
		{name: "malformed wall clock", wallClock: "not-a-duration", wantErr: true},
		{name: "negative wall clock rejected", wallClock: "-5m", wantErr: true},
		{name: "malformed turns", turns: "xyz", wantErr: true},
		{name: "malformed spend", spend: "free", wantErr: true},
		{name: "negative spend rejected", spend: "-5", wantErr: true},
		{name: "malformed throughput", throughput: "fast", wantErr: true},
		{name: "negative throughput rejected", throughput: "-1", wantErr: true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := ParseEnvelopeFlags(c.tokens, c.wallClock, c.turns, c.spend, c.throughput)
			if c.wantErr {
				if err == nil {
					t.Fatalf("ParseEnvelopeFlags(%q,%q,%q,%q,%q) = %+v, want error", c.tokens, c.wallClock, c.turns, c.spend, c.throughput, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseEnvelopeFlags(%q,%q,%q,%q,%q): unexpected error: %v", c.tokens, c.wallClock, c.turns, c.spend, c.throughput, err)
			}
			if got != c.want {
				t.Fatalf("ParseEnvelopeFlags(%q,%q,%q,%q,%q) = %+v, want %+v", c.tokens, c.wallClock, c.turns, c.spend, c.throughput, got, c.want)
			}
		})
	}
}

// TestBudgetEnvelopeParseIsDeterministic is the issue's Done condition made literal: the same
// Envelope always parses to a byte-identical ParsedEnvelope (no clock read, no randomness), so
// "inspect the parsed deterministic budget" is a real equality assertion.
func TestBudgetEnvelopeParseIsDeterministic(t *testing.T) {
	e := Envelope{Tokens: 50000, WallClock: 10 * time.Minute, Turns: 25, SpendCapCents: 500, ThroughputFloor: 20}
	a := e.Parse()
	b := e.Parse()
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("Parse is not deterministic: %+v != %+v", a, b)
	}
	want := ParsedEnvelope{
		Envelope:   e,
		Budget:     Budget{TurnsLeft: 25, TokensLeft: 50000},
		TimeBudget: TimeBudget{LimitNanos: int64(10 * time.Minute)},
		Pace:       Pace{},
	}
	if !reflect.DeepEqual(a, want) {
		t.Fatalf("Parse() = %+v, want %+v", a, want)
	}
}

// TestBudgetEnvelopeZeroIsUnbounded pins that an envelope stating nothing produces a Budget
// and TimeBudget byte-identical to the runtime's own pre-envelope defaults (DefaultState's
// Budget and NewTimeBudget()) — an unused envelope must never introduce a hidden cap.
func TestBudgetEnvelopeZeroIsUnbounded(t *testing.T) {
	var e Envelope
	if !e.IsZero() {
		t.Fatalf("zero Envelope.IsZero() = false, want true")
	}
	got := e.Parse()
	wantBudget := DefaultState("t").Budget
	if got.Budget != wantBudget {
		t.Fatalf("zero envelope Budget = %+v, want %+v (DefaultState's budget)", got.Budget, wantBudget)
	}
	if got.TimeBudget != NewTimeBudget() {
		t.Fatalf("zero envelope TimeBudget = %+v, want the unbounded zero value", got.TimeBudget)
	}
	if got.TimeBudget.Bounded() {
		t.Fatalf("zero envelope TimeBudget reports Bounded() = true")
	}
}

// TestBudgetEnvelopeExplicitUnbounded pins that the "unbounded" string sentinel round-trips
// through ToBudget as the same Unbounded (-1) constant Budget itself uses, distinguishing it
// from "not stated" (which also lands on Unbounded via the default, but through a different
// path — both must agree on the wire value).
func TestBudgetEnvelopeExplicitUnbounded(t *testing.T) {
	e, err := ParseEnvelopeFlags("unbounded", "", "unbounded", "", "")
	if err != nil {
		t.Fatalf("ParseEnvelopeFlags: %v", err)
	}
	b := e.ToBudget()
	if b.TokensLeft != Unbounded || b.TurnsLeft != Unbounded {
		t.Fatalf("ToBudget() = %+v, want both axes Unbounded", b)
	}
}

// TestBudgetEnvelopeToThroughput pins the throughput floor's projection: it seeds only the
// expected rate, leaving the observed rate at zero (no observation yet) so ThroughputRatio
// reads it as "no constraint" until a runtime observation lands.
func TestBudgetEnvelopeToThroughput(t *testing.T) {
	e := Envelope{ThroughputFloor: 30}
	th := e.ToThroughput()
	if th.ExpectedTokensPerSec != 30 || th.ObservedTokensPerSec != 0 {
		t.Fatalf("ToThroughput() = %+v, want {Observed:0 Expected:30}", th)
	}
	if r := th.ThroughputRatio(); r != 1.0 {
		t.Fatalf("ThroughputRatio() with no observation = %v, want 1.0 (no constraint)", r)
	}
}

// TestBudgetEnvelopeSpendCapRoundTrips pins that a stated spend cap survives Parse
// unmodified (advisory today — see envelope.go's file header — but never silently dropped).
func TestBudgetEnvelopeSpendCapRoundTrips(t *testing.T) {
	e := Envelope{SpendCapCents: 999}
	got := e.Parse()
	if got.Envelope.SpendCapCents != 999 {
		t.Fatalf("Parse().Envelope.SpendCapCents = %d, want 999", got.Envelope.SpendCapCents)
	}
}
