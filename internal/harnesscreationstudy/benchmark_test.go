package harnesscreationstudy

import (
	"encoding/json"
	"testing"
)

var (
	benchStudySink  Study
	benchResultSink Result
)

func benchmarkStudy() Study {
	return Study{
		Schema: Schema,
		ID:     "study-benchmark",
		Protocol: Protocol{
			Frozen:                true,
			TenMinuteLimitSeconds: 600,
			AssistancePolicy:      "task-card-and-help-only",
			FailuresInDenominator: true,
			TaskDigest:            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			Parity: MatchedStudySpec{
				Frozen:                true,
				MinimumPairs:          2,
				MaxMedianElapsedRatio: 1.25,
				CounterbalancedOrder:  true,
			},
		},
		Baseline: Baseline{
			ID:       "baseline-alt",
			Runnable: true,
			Tuned:    true,
			Frozen:   true,
			Evidence: "evidence/baseline.json",
		},
		Runs: []Run{
			{
				ID:               "pair-1-fak",
				ParticipantID:    "builder-1",
				Track:            "ten-minute",
				Arm:              "fak",
				PairID:           "pair-1",
				TaskDigest:       "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				MachineID:        "node-1",
				PairOrder:        "fak-first",
				ArmPosition:      1,
				ParticipantClass: "unfamiliar-builder",
				Independent:      true,
				OS:               "linux",
				CPU:              "amd64",
				NetworkState:     "online",
				CacheState:       "empty",
				Outcome:          "success",
				ElapsedSeconds:   320,
				Receipt:          "receipts/1-fak.json",
			},
			{
				ID:               "pair-1-base",
				ParticipantID:    "builder-1",
				Track:            "ten-minute",
				Arm:              "baseline",
				PairID:           "pair-1",
				TaskDigest:       "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				MachineID:        "node-1",
				PairOrder:        "fak-first",
				ArmPosition:      2,
				ParticipantClass: "unfamiliar-builder",
				Independent:      true,
				OS:               "linux",
				CPU:              "amd64",
				NetworkState:     "online",
				CacheState:       "empty",
				Outcome:          "success",
				ElapsedSeconds:   350,
				Receipt:          "receipts/1-base.json",
			},
			{
				ID:               "pair-2-base",
				ParticipantID:    "builder-2",
				Track:            "ten-minute",
				Arm:              "baseline",
				PairID:           "pair-2",
				TaskDigest:       "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				MachineID:        "node-1",
				PairOrder:        "baseline-first",
				ArmPosition:      1,
				ParticipantClass: "unfamiliar-builder",
				Independent:      true,
				OS:               "linux",
				CPU:              "amd64",
				NetworkState:     "online",
				CacheState:       "empty",
				Outcome:          "success",
				ElapsedSeconds:   340,
				Receipt:          "receipts/2-base.json",
			},
			{
				ID:               "pair-2-fak",
				ParticipantID:    "builder-2",
				Track:            "ten-minute",
				Arm:              "fak",
				PairID:           "pair-2",
				TaskDigest:       "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				MachineID:        "node-1",
				PairOrder:        "baseline-first",
				ArmPosition:      2,
				ParticipantClass: "unfamiliar-builder",
				Independent:      true,
				OS:               "linux",
				CPU:              "amd64",
				NetworkState:     "online",
				CacheState:       "empty",
				Outcome:          "success",
				ElapsedSeconds:   310,
				Receipt:          "receipts/2-fak.json",
			},
			{
				ID:                    "weekend-run",
				ParticipantID:         "builder-3",
				Track:                 "weekend",
				ParticipantClass:      "unfamiliar-builder",
				Independent:           true,
				Outcome:               "success",
				ElapsedSeconds:        3600,
				IndependentlyAuthored: true,
				ConformancePassed:     true,
				Receipt:               "receipts/weekend.json",
			},
			{
				ID:               "calibration-run",
				ParticipantID:    "maintainer-1",
				Track:            "ten-minute",
				ParticipantClass: "maintainer-calibration",
				Independent:      false,
				Outcome:          "success",
				ElapsedSeconds:   120,
				Receipt:          "receipts/calib.json",
			},
		},
	}
}

// BenchmarkHarnessCreationStudy benchmarks parsing and evaluation of a complete study dataset.
func BenchmarkHarnessCreationStudy(b *testing.B) {
	study := benchmarkStudy()
	raw, err := json.Marshal(study)
	if err != nil {
		b.Fatalf("failed to marshal benchmark study: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parsed, err := Parse(raw)
		if err != nil {
			b.Fatalf("parse failed: %v", err)
		}
		res := Evaluate(parsed)
		if res.TenMinute.ClaimStatus == "" {
			b.Fatal("unexpected empty claim status")
		}
		benchResultSink = res
	}
}

// BenchmarkParse benchmarks schema decoding and invariant validation of study JSON payloads.
func BenchmarkParse(b *testing.B) {
	study := benchmarkStudy()
	raw, err := json.Marshal(study)
	if err != nil {
		b.Fatalf("failed to marshal benchmark study: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parsed, err := Parse(raw)
		if err != nil {
			b.Fatalf("parse failed: %v", err)
		}
		benchStudySink = parsed
	}
}

// BenchmarkEvaluate benchmarks aggregation and parity evaluation over a pre-parsed study.
func BenchmarkEvaluate(b *testing.B) {
	study := benchmarkStudy()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Evaluate(study)
		if res.Parity.ClaimStatus == "" {
			b.Fatal("unexpected empty parity claim status")
		}
		benchResultSink = res
	}
}
