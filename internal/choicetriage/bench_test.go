package choicetriage

import "testing"

// TestBenchmarkTriageSanity verifies that the workloads exercised in benchmarks
// yield valid verdicts with expected dispositions.
func TestBenchmarkTriageSanity(t *testing.T) {
	signals := sampleSignals()
	for _, sig := range signals {
		v := Triage(sig)
		if !v.Disposition.Valid() {
			t.Fatalf("invalid disposition %q for signal %+v", v.Disposition, sig)
		}
		if v.Reason == "" || v.Resolve == "" {
			t.Fatalf("empty reason or resolve in verdict %+v", v)
		}
	}
}

func sampleSignals() []Signal {
	return []Signal{
		{
			Severity:    "decision",
			Source:      "release",
			Question:    "cut the release now?",
			Detail:      "all tests passed",
			OptionCount: 2,
		},
		{
			Severity:    "page",
			Source:      "gateway",
			Question:    "re-auth needed",
			Detail:      "seat needs credential login",
			OptionCount: 2,
		},
		{
			Severity:    "page",
			Source:      "cadence",
			Question:    "cadence report missing",
			Action:      "generate `fak cadence --json` and pass it with --cadence",
			OptionCount: 3,
		},
		{
			Source:      "program",
			Action:      "`fak program report --json`",
			OptionCount: 2,
		},
		{
			Source:      "agent",
			Question:    "let agents continue?",
			OptionCount: 1,
		},
		{
			Source:      "milestone",
			Question:    "advance the roadmap epic?",
			Detail:      "multiple frontiers open",
			OptionCount: 2,
		},
		{
			Source:      "watch",
			Question:    "reduce measured friction?",
			ScopeLarge:  true,
			OptionCount: 2,
		},
		{
			Source:      "watch",
			Question:    "investigate the friction spike?",
			Detail:      "throughput dipped",
			OptionCount: 3,
		},
		{},
	}
}

// BenchmarkTriage measures throughput of evaluating surfaced choice signals
// across the four closed dispositions.
func BenchmarkTriage(b *testing.B) {
	cases := []struct {
		name string
		sig  Signal
	}{
		{
			name: "HumanResidual_Authority",
			sig: Signal{
				Severity:    "decision",
				Source:      "release",
				Question:    "approve and cut the release now?",
				Detail:      "policy sign-off required",
				OptionCount: 2,
			},
		},
		{
			name: "TakeObvious_CommandLiteral",
			sig: Signal{
				Source:      "program",
				Action:      "`fak program report --json`",
				OptionCount: 2,
			},
		},
		{
			name: "TakeObvious_ActionHint",
			sig: Signal{
				Severity:    "page",
				Source:      "cadence",
				Question:    "cadence report missing",
				Action:      "generate report with fak cadence --json",
				OptionCount: 3,
			},
		},
		{
			name: "TakeObvious_SingleOption",
			sig: Signal{
				Source:      "agent",
				Question:    "let agents continue?",
				OptionCount: 1,
			},
		},
		{
			name: "FileTicket_ScopeHints",
			sig: Signal{
				Source:      "milestone",
				Question:    "advance the roadmap epic?",
				Detail:      "multiple frontiers open across backlog",
				OptionCount: 2,
			},
		},
		{
			name: "FileTicket_ScopeLarge",
			sig: Signal{
				Source:      "watch",
				Question:    "reduce friction spikes?",
				ScopeLarge:  true,
				OptionCount: 2,
			},
		},
		{
			name: "FreshContext_Default",
			sig: Signal{
				Source:      "watch",
				Question:    "investigate the friction spike?",
				Detail:      "throughput dipped on queue",
				OptionCount: 3,
			},
		},
		{
			name: "FreshContext_ZeroSignal",
			sig:  Signal{},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				v := Triage(tc.sig)
				if !v.Disposition.Valid() {
					b.Fatalf("invalid disposition: %s", v.Disposition)
				}
			}
		})
	}
}

// BenchmarkTriageBatch measures the throughput of triaging a production batch
// of mixed surfaced choices, such as from an operator brief or stop hook scan.
func BenchmarkTriageBatch(b *testing.B) {
	batch := sampleSignals()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range batch {
			v := Triage(batch[j])
			if !v.Disposition.Valid() {
				b.Fatalf("invalid disposition: %s", v.Disposition)
			}
		}
	}
}
