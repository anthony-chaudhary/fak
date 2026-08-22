package main

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestDevelopmentScalingReportReconcilesCanonicalReceipts(t *testing.T) {
	report, err := analyzeDevelopmentFixture(canonicalDevelopmentFixture)
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != developmentReportSchema || report.Workload.ID == "" || report.Workload.ReceiptDigest == "" {
		t.Fatalf("report identity is incomplete: %+v", report)
	}
	gotSeats := make([]int, 0, len(report.Arms))
	for _, arm := range report.Arms {
		gotSeats = append(gotSeats, arm.Seats)
		if arm.AcceptedClosures != report.Workload.Items {
			t.Fatalf("%d-seat useful closures=%d, want %d", arm.Seats, arm.AcceptedClosures, report.Workload.Items)
		}
		if arm.Losses.sum() != arm.Losses.TotalWorkerMS || arm.WIPAreaWorkerMS != arm.Losses.TotalWorkerMS {
			t.Fatalf("%d-seat accounting does not reconcile: %+v", arm.Seats, arm)
		}
		if arm.CriticalPathMS > arm.MakespanMS || arm.Speedup > float64(arm.Seats) || arm.ParallelEfficiency > 1 {
			t.Fatalf("%d-seat timing is impossible: %+v", arm.Seats, arm)
		}
	}
	if !reflect.DeepEqual(gotSeats, developmentSeatGrid) {
		t.Fatalf("seat grid=%v, want %v", gotSeats, developmentSeatGrid)
	}
	thirty := report.Arms[len(report.Arms)-1]
	if thirty.Seats != 30 || thirty.DuplicateAttempts == 0 || thirty.UnverifiableAttempts == 0 || thirty.CollisionEvents == 0 || thirty.RetryAttempts == 0 {
		t.Fatalf("30-seat arm does not retain duplicate/unverifiable/retry/collision cost: %+v", thirty)
	}
	if report.DominantLimiter == "" || report.NextExperiment == "" {
		t.Fatalf("operator decision is incomplete: %+v", report)
	}
}

func TestDevelopmentScalingNegativeFixturesFailClosed(t *testing.T) {
	for _, tc := range developmentNegativeFixtures() {
		t.Run(tc.name, func(t *testing.T) {
			body, err := json.Marshal(tc.fixture)
			if err != nil {
				t.Fatal(err)
			}
			_, err = analyzeDevelopmentFixture(body)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("analyzeDevelopmentFixture() error=%v, want substring %q", err, tc.wantErr)
			}
		})
	}
}

func TestDevelopmentSelfcheckMatchesCapturedReceipt(t *testing.T) {
	var jsonOut, operatorOut bytes.Buffer
	if err := runDevelopmentScaling(true, &jsonOut, &operatorOut); err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("testdata/issue-8477-selfcheck.json")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(jsonOut.Bytes(), want) {
		t.Fatalf("selfcheck JSON drifted from captured receipt\n--- got ---\n%s\n--- want ---\n%s", jsonOut.Bytes(), want)
	}
	operator := strings.TrimSpace(operatorOut.String())
	if lines := strings.Count(operator, "\n") + 1; lines > 10 {
		t.Fatalf("operator report has %d lines, want one screen (<=10)\n%s", lines, operator)
	}
	for _, required := range []string{"SEATS ADMIT CLOSE", "   30", "dominant limiter:", "next experiment:", "selfcheck: PASS"} {
		if !strings.Contains(operator, required) {
			t.Fatalf("operator report missing %q\n%s", required, operator)
		}
	}
}
