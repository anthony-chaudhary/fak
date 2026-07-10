package loopindex

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// doneConditionInput is the fixture the issue names: missing receipts, a same-
// family refusal, an unavailable auditor, a refute, and a pass. It must yield
// transparent denominators and a non-green dark-loop/coverage verdict.
func doneConditionInput() CrossAuditInput {
	const now = 1_000_000_000_000
	return CrossAuditInput{
		NowUnixNano:     now,
		StaleAfterNanos: 3_600_000_000_000, // 1h
		EligibleIssues:  5,                 // issues 101..105; 105 has no record (missing receipt)
		Records: []AuditRecord{
			{ // pass
				IssueNumber: 101, Class: "infra", Outcome: OutcomePass, Independence: IndependenceAdmitted,
				AuthorModel: "claude-opus", AuditorModel: "gpt-audit", CalibrationVersion: "cal/v1",
				RecordedAtUnixNano: now - 60_000_000_000, DurationNanos: 2_000_000_000,
				CostMeasured: true, InputTokens: 1000, OutputTokens: 200, TotalTokens: 1200,
				CostMicrosUSD: 3400, CostBasis: "provider-reported",
			},
			{ // refute with finding
				IssueNumber: 102, Class: "infra", Outcome: OutcomeRefute, Severity: "HIGH", Independence: IndependenceAdmitted,
				AuthorModel: "claude-opus", AuditorModel: "gpt-audit", CalibrationVersion: "cal/v1",
				RecordedAtUnixNano: now - 120_000_000_000, DurationNanos: 4_000_000_000,
				CostMeasured: true, InputTokens: 1500, OutputTokens: 400, TotalTokens: 1900,
				CostMicrosUSD: 5600, CostBasis: "provider-reported",
			},
			{ // unavailable auditor — a dark-loop signal; independence was admitted before the provider dropped
				IssueNumber: 103, Class: "bug", Outcome: OutcomeUnavailable, Independence: IndependenceAdmitted,
				AuthorModel: "claude-opus", AuditorModel: "gpt-audit", CalibrationVersion: "cal/v1",
				RecordedAtUnixNano: now - 30_000_000_000,
			},
			{ // same-family refusal — no audit ran; independence refused
				IssueNumber: 104, Class: "bug", Outcome: OutcomeRefused, Independence: IndependenceRefused,
				AuthorModel: "claude-opus", AuditorModel: "claude-opus", CalibrationVersion: "cal/v1",
				RecordedAtUnixNano: now - 10_000_000_000,
			},
		},
		Loop: LoopHealth{
			Present: true, Running: true, LastTickUnixNano: now - 5_000_000_000,
			PendingIssues: 1, InflightIssues: 0, Retries: 2, DeadLetters: 0,
			Providers: []ProviderHealth{{Name: "gpt-audit", Available: false}},
		},
	}
}

func TestCrossAuditDoneConditionIsNonGreenWithTransparentDenominators(t *testing.T) {
	sc := ScoreCrossAudit(doneConditionInput())

	if sc.OK {
		t.Fatal("done-condition fixture must not be OK")
	}
	if sc.Verdict != "ACTION" {
		t.Fatalf("verdict = %q, want ACTION", sc.Verdict)
	}
	if !sc.Health.DarkLoop {
		t.Fatal("done-condition fixture must raise a dark-loop verdict (unavailable auditor + provider down)")
	}

	// Transparent coverage denominators: unavailable/refused do NOT count as audited.
	c := sc.Coverage
	if c.Eligible != 5 || c.Audited != 2 || c.Attempted != 4 || c.Pending != 3 || c.Missing != 1 || c.UnavailableOnly != 2 {
		t.Fatalf("coverage denominators not transparent: %+v", c)
	}
	if c.AuditedRate != 0.4 {
		t.Fatalf("audited_rate = %v, want 0.4 (2/5)", c.AuditedRate)
	}

	// Independence shows refused separately from admitted.
	i := sc.Independence
	if i.Admitted != 3 || i.Refused != 1 || i.Unknown != 0 || i.Total != 4 {
		t.Fatalf("independence not transparent: %+v", i)
	}

	// Pass rate is over completed audits and labelled not-correctness.
	q := sc.Quality
	if q.Completed != 2 || q.Passes != 1 || q.Findings != 1 || q.PassRate != 0.5 {
		t.Fatalf("quality sample/rate wrong: %+v", q)
	}
	if !strings.Contains(q.PassRateBasis, "NOT a correctness rate") {
		t.Fatalf("pass_rate_basis must disclaim correctness, got %q", q.PassRateBasis)
	}

	// Economics fields carry provenance/sample counts; only measured, reported cost is summed.
	e := sc.Economics
	if e.TokenSampleCount != 2 || e.TotalTokens != 3100 {
		t.Fatalf("token provenance wrong: %+v", e)
	}
	if e.CostSampleCount != 2 || e.CostMicrosUSD != 9000 {
		t.Fatalf("cost provenance wrong: %+v", e)
	}
	if e.LatencySampleCount != 2 || e.AvgDurationNanos != 3_000_000_000 {
		t.Fatalf("latency provenance wrong: %+v", e)
	}

	// Dark-loop debt and coverage debt are both named.
	if !contains(sc.Debts, "dark-loop") || !contains(sc.Debts, "coverage-incomplete") ||
		!contains(sc.Debts, "open-findings") || !contains(sc.Debts, "independence-unproven") {
		t.Fatalf("debts missing an expected entry: %v", sc.Debts)
	}
	if sc.Grade != "F" {
		t.Fatalf("dark loop must grade F, got %q", sc.Grade)
	}
}

func TestCrossAuditZeroAuditedOfNonzeroEligibleIsDark(t *testing.T) {
	// The acceptance-gate rule: zero audited of nonzero eligible exits non-green,
	// even with a live loop and no other alarm.
	sc := ScoreCrossAudit(CrossAuditInput{
		EligibleIssues: 4,
		Loop:           LoopHealth{Present: true, Running: true},
	})
	if sc.OK {
		t.Fatal("zero audited of nonzero eligible must not be OK")
	}
	if !sc.Health.DarkLoop {
		t.Fatal("zero coverage must be a dark loop")
	}
	if !contains(sc.Debts, "coverage-dark") {
		t.Fatalf("expected coverage-dark debt, got %v", sc.Debts)
	}
	if sc.Coverage.Pending != 4 || sc.Coverage.Missing != 4 {
		t.Fatalf("coverage denominators wrong: %+v", sc.Coverage)
	}
}

func TestCrossAuditGreenWhenCompleteAndLive(t *testing.T) {
	const now = 2_000_000_000_000
	sc := ScoreCrossAudit(CrossAuditInput{
		NowUnixNano: now, StaleAfterNanos: 3_600_000_000_000, EligibleIssues: 2,
		Records: []AuditRecord{
			{IssueNumber: 1, Class: "infra", Outcome: OutcomePass, Independence: IndependenceAdmitted,
				AuthorModel: "a", AuditorModel: "b", RecordedAtUnixNano: now - 1_000, DurationNanos: 1_000_000_000,
				CostMeasured: true, TotalTokens: 100, CostMicrosUSD: 10, CostBasis: "provider-reported"},
			{IssueNumber: 2, Class: "infra", Outcome: OutcomePass, Independence: IndependenceAdmitted,
				AuthorModel: "a", AuditorModel: "b", RecordedAtUnixNano: now - 2_000, DurationNanos: 1_000_000_000,
				CostMeasured: true, TotalTokens: 100, CostMicrosUSD: 10, CostBasis: "provider-reported"},
		},
		Loop: LoopHealth{Present: true, Running: true, LastTickUnixNano: now - 1_000,
			Providers: []ProviderHealth{{Name: "b", Available: true}}},
	})
	if !sc.OK || sc.Verdict != "OK" {
		t.Fatalf("complete, live, admitted loop must be OK; got verdict=%q debts=%v", sc.Verdict, sc.Debts)
	}
	if sc.Health.DarkLoop {
		t.Fatal("healthy loop must not be dark")
	}
	if sc.Grade != "A" {
		t.Fatalf("full coverage must grade A, got %q", sc.Grade)
	}
}

func TestCrossAuditAbsentLoopIsDark(t *testing.T) {
	// An absent loop state (Present=false) is treated as not-running: a dark loop,
	// even if the receipt side looks fully covered. This is the "silently stopped"
	// alarm the issue names.
	sc := ScoreCrossAudit(CrossAuditInput{
		EligibleIssues: 1,
		Records: []AuditRecord{
			{IssueNumber: 1, Class: "infra", Outcome: OutcomePass, Independence: IndependenceAdmitted},
		},
		Loop: LoopHealth{Present: false},
	})
	if sc.OK {
		t.Fatal("absent loop must not be OK")
	}
	if !sc.Health.DarkLoop {
		t.Fatal("absent loop must be dark")
	}
}

func TestScoreCrossAuditDeterministic(t *testing.T) {
	in := doneConditionInput()
	a, err := json.Marshal(ScoreCrossAudit(in))
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(ScoreCrossAudit(in))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) {
		t.Fatalf("ScoreCrossAudit not deterministic:\n%s\n%s", a, b)
	}
}

// TestCrossAuditSchemaPinned pins the JSON contract. A change to any field name
// or block shape is a schema break and must update this golden deliberately.
func TestCrossAuditSchemaPinned(t *testing.T) {
	sc := ScoreCrossAudit(doneConditionInput())
	got, err := json.MarshalIndent(sc, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != crossAuditGolden {
		t.Fatalf("cross-audit scorecard JSON drifted from the pinned schema.\n--- got ---\n%s\n--- want ---\n%s", got, crossAuditGolden)
	}
}

func TestRenderCrossAuditHeadline(t *testing.T) {
	var b bytes.Buffer
	RenderCrossAudit(&b, ScoreCrossAudit(doneConditionInput()))
	out := b.String()
	for _, want := range []string{
		"cross-audit scorecard", "dark_loop=true", "coverage:", "independence:",
		"NOT a correctness rate", "sample counts are provenance", "ALARM:", "next:",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("Render output missing %q:\n%s", want, out)
		}
	}
}

const crossAuditGolden = `{
  "schema": "fak.crossaudit.scorecard.v1",
  "ok": false,
  "verdict": "ACTION",
  "grade": "F",
  "finding": "audit loop NOT green: dark-loop,coverage-incomplete,open-findings,independence-unproven (audited 2/5, dark_loop=true)",
  "reason": "1 auditor provider(s) unavailable; 1 audit(s) recorded UNAVAILABLE",
  "next_action": "restore the audit loop: bring the auditor provider back, restart the loop, or clear dead letters — an unobserved loop is a security-theater claim",
  "debts": [
    "dark-loop",
    "coverage-incomplete",
    "open-findings",
    "independence-unproven"
  ],
  "alarms": [
    "1 auditor provider(s) unavailable",
    "1 audit(s) recorded UNAVAILABLE"
  ],
  "coverage": {
    "eligible": 5,
    "records": 4,
    "attempted": 4,
    "audited": 2,
    "pending": 3,
    "missing": 1,
    "unavailable_only": 2,
    "audited_rate": 0.4,
    "attempted_rate": 0.8,
    "rate_basis": "audited (completed) / eligible; UNAVAILABLE and REFUSED are not audited"
  },
  "independence": {
    "total": 4,
    "admitted": 3,
    "refused": 1,
    "unknown": 0,
    "admitted_rate": 0.75,
    "rate_basis": "admitted / all records; refused and unknown shown separately"
  },
  "quality": {
    "verdicts": [
      {
        "name": "PASS",
        "count": 1
      },
      {
        "name": "REFUSED",
        "count": 1
      },
      {
        "name": "REFUTE",
        "count": 1
      },
      {
        "name": "UNAVAILABLE",
        "count": 1
      }
    ],
    "severities": [
      {
        "name": "HIGH",
        "count": 1
      }
    ],
    "completed": 2,
    "passes": 1,
    "findings": 1,
    "pass_rate": 0.5,
    "pass_rate_basis": "PASS / completed audits — an independence-checked no-refutation rate, NOT a correctness rate",
    "per_class_yield": [
      {
        "class": "infra",
        "completed": 2,
        "refutes": 1,
        "yield_rate": 0.5
      }
    ]
  },
  "model_mix": {
    "authors": [
      {
        "name": "claude-opus",
        "count": 4
      }
    ],
    "auditors": [
      {
        "name": "gpt-audit",
        "count": 3
      },
      {
        "name": "claude-opus",
        "count": 1
      }
    ],
    "calibrations": [
      {
        "name": "cal/v1",
        "count": 4
      }
    ]
  },
  "economics": {
    "records": 4,
    "token_sample_count": 2,
    "input_tokens": 2500,
    "output_tokens": 600,
    "total_tokens": 3100,
    "cost_sample_count": 2,
    "cost_micros_usd": 9000,
    "cost_basis_counts": [
      {
        "name": "provider-reported",
        "count": 2
      }
    ],
    "latency_sample_count": 2,
    "total_duration_nanos": 6000000000,
    "avg_duration_nanos": 3000000000
  },
  "health": {
    "loop_present": true,
    "loop_running": true,
    "last_tick_age_nanos": 5000000000,
    "stale": false,
    "pending_issues": 1,
    "inflight_issues": 0,
    "retries": 2,
    "dead_letters": 0,
    "providers": [
      {
        "name": "gpt-audit",
        "available": false
      }
    ],
    "unavailable_providers": 1,
    "unavailable_audits": 1,
    "newest_record_age_nanos": 10000000000,
    "oldest_record_age_nanos": 120000000000,
    "freshness_sample_count": 4,
    "dark_loop": true
  }
}`
