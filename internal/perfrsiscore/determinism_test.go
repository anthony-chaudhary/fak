package perfrsiscore

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// TestDeterminism proves that scoring the performance RSI loop across repeated
// sequential runs produces bitwise-identical and deep-equal results for Score,
// ScoreLoopTurn, ComposeV1, Compare, RenderHuman, RenderMarkdown, and
// FormatLoopTurnReceipt without state leakage or map iteration drift.
func TestDeterminism(t *testing.T) {
	e := fixture(t)
	refReport := Score(e)
	refReportJSON, err := MarshalJSON(refReport)
	if err != nil {
		t.Fatalf("marshal reference report: %v", err)
	}
	refHuman := RenderHuman(refReport)
	refMarkdown := RenderMarkdown(refReport)

	const inputPath = "testdata/complete.json"
	refReceipt := ScoreLoopTurn(inputPath)
	refReceiptFormatted := FormatLoopTurnReceipt(refReceipt)

	compositionInputs := committedCompositionInputs(t)
	refComposed, err := ComposeV1("issue-9767-determinism", compositionInputs)
	if err != nil {
		t.Fatalf("reference ComposeV1: %v", err)
	}
	refComposedJSON, err := json.Marshal(refComposed)
	if err != nil {
		t.Fatalf("marshal reference composed: %v", err)
	}

	priorReport := refReport
	priorReport.Snapshot = "prior-snapshot"
	refComparisonReport := refReport
	if err := Compare(&refComparisonReport, priorReport); err != nil {
		t.Fatalf("reference Compare: %v", err)
	}

	const iterations = 100
	for i := 0; i < iterations; i++ {
		gotReport := Score(e)
		if !reflect.DeepEqual(gotReport, refReport) {
			t.Fatalf("iteration %d: Score report diverged from reference", i)
		}
		gotReportJSON, err := MarshalJSON(gotReport)
		if err != nil {
			t.Fatalf("iteration %d: marshal report: %v", i, err)
		}
		if !bytes.Equal(gotReportJSON, refReportJSON) {
			t.Fatalf("iteration %d: report JSON diverged from reference", i)
		}

		if gotHuman := RenderHuman(gotReport); gotHuman != refHuman {
			t.Fatalf("iteration %d: RenderHuman diverged from reference", i)
		}
		if gotMarkdown := RenderMarkdown(gotReport); gotMarkdown != refMarkdown {
			t.Fatalf("iteration %d: RenderMarkdown diverged from reference", i)
		}

		gotReceipt := ScoreLoopTurn(inputPath)
		if !reflect.DeepEqual(gotReceipt, refReceipt) {
			t.Fatalf("iteration %d: ScoreLoopTurn receipt diverged from reference", i)
		}
		if gotFormatted := FormatLoopTurnReceipt(gotReceipt); gotFormatted != refReceiptFormatted {
			t.Fatalf("iteration %d: FormatLoopTurnReceipt diverged from reference", i)
		}

		gotComposed, err := ComposeV1("issue-9767-determinism", compositionInputs)
		if err != nil {
			t.Fatalf("iteration %d: ComposeV1 failed: %v", i, err)
		}
		if !reflect.DeepEqual(gotComposed, refComposed) {
			t.Fatalf("iteration %d: ComposeV1 composed evidence diverged from reference", i)
		}
		gotComposedJSON, err := json.Marshal(gotComposed)
		if err != nil {
			t.Fatalf("iteration %d: marshal composed: %v", i, err)
		}
		if !bytes.Equal(gotComposedJSON, refComposedJSON) {
			t.Fatalf("iteration %d: composed JSON diverged from reference", i)
		}

		gotComparisonReport := gotReport
		if err := Compare(&gotComparisonReport, priorReport); err != nil {
			t.Fatalf("iteration %d: Compare failed: %v", i, err)
		}
		if !reflect.DeepEqual(gotComparisonReport, refComparisonReport) {
			t.Fatalf("iteration %d: Compare report diverged from reference", i)
		}
	}
}

// TestPerfRSIScore_Determinism validates determinism across distinct evidence topologies:
// saturated 100x clean, degraded with missing/unknown debt, and unavailable inputs.
func TestPerfRSIScore_Determinism(t *testing.T) {
	t.Run("CleanSaturated", func(t *testing.T) {
		e := fixture(t)
		for i := range e.Dimensions {
			target := *e.Dimensions[i].Target
			e.Dimensions[i].Current = &target
		}
		ref := Score(e)
		refJSON, err := json.Marshal(ref)
		if err != nil {
			t.Fatalf("marshal clean ref: %v", err)
		}
		for i := 0; i < 50; i++ {
			got := Score(e)
			if !reflect.DeepEqual(got, ref) {
				t.Fatalf("clean iteration %d diverged", i)
			}
			gotJSON, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal clean got: %v", err)
			}
			if !bytes.Equal(gotJSON, refJSON) {
				t.Fatalf("clean iteration %d JSON diverged", i)
			}
		}
	})

	t.Run("DegradedUnknownDebt", func(t *testing.T) {
		e := fixture(t)
		e.Dimensions[0].Current = nil
		e.Dimensions[3].Current = nil
		ref := Score(e)
		refJSON, err := json.Marshal(ref)
		if err != nil {
			t.Fatalf("marshal degraded ref: %v", err)
		}
		for i := 0; i < 50; i++ {
			got := Score(e)
			if !reflect.DeepEqual(got, ref) {
				t.Fatalf("degraded iteration %d diverged", i)
			}
			gotJSON, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal degraded got: %v", err)
			}
			if !bytes.Equal(gotJSON, refJSON) {
				t.Fatalf("degraded iteration %d JSON diverged", i)
			}
		}
	})

	t.Run("UnavailableLoopTurn", func(t *testing.T) {
		ref := ScoreLoopTurn("")
		for i := 0; i < 50; i++ {
			got := ScoreLoopTurn("")
			if !reflect.DeepEqual(got, ref) {
				t.Fatalf("unavailable iteration %d diverged", i)
			}
		}
	})
}

// TestPerfRSIScore_RaceWitness serves as a concurrent execution and race witness:
// multiple goroutines simultaneously evaluate, compose, format, and render the
// performance RSI scorecard to witness that all operations are race-free and deterministic.
func TestPerfRSIScore_RaceWitness(t *testing.T) {
	e := fixture(t)
	refReport := Score(e)
	refReportJSON, err := MarshalJSON(refReport)
	if err != nil {
		t.Fatalf("marshal reference report: %v", err)
	}
	refHuman := RenderHuman(refReport)
	refMarkdown := RenderMarkdown(refReport)

	const inputPath = "testdata/complete.json"
	refReceipt := ScoreLoopTurn(inputPath)
	refReceiptFormatted := FormatLoopTurnReceipt(refReceipt)

	compositionInputs := committedCompositionInputs(t)
	refComposed, err := ComposeV1("issue-9767-concurrent-witness", compositionInputs)
	if err != nil {
		t.Fatalf("reference ComposeV1: %v", err)
	}

	priorReport := refReport
	priorReport.Snapshot = "prior-snapshot"
	refComparisonReport := refReport
	if err := Compare(&refComparisonReport, priorReport); err != nil {
		t.Fatalf("reference Compare: %v", err)
	}

	const sampleUsageLines = `{"schema":"fak-performance-rsi-usage/1","at":"2026-08-28T12:00:00Z","status":"scored","reason":"SCORE_COMPLETE","snapshot":"test-1","invocation_outcomes":{"success":1,"refusal":0,"error":0}}
{"schema":"fak-performance-rsi-usage/1","at":"2026-08-28T12:05:00Z","status":"unavailable","reason":"SCORE_INPUT_UNAVAILABLE","invocation_outcomes":{"success":0,"refusal":1,"error":0}}`
	refFold, err := FoldUsageReader(strings.NewReader(sampleUsageLines))
	if err != nil {
		t.Fatalf("reference FoldUsageReader: %v", err)
	}
	refFormattedFold := FormatUsageFold(refFold)

	const goroutines = 32
	var wg sync.WaitGroup
	errCh := make(chan error, goroutines*8)
	start := make(chan struct{})

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			<-start

			gotReport := Score(e)
			if !reflect.DeepEqual(gotReport, refReport) {
				errCh <- fmt.Errorf("worker %d: Score diverged from reference", workerID)
				return
			}
			gotReportJSON, err := MarshalJSON(gotReport)
			if err != nil {
				errCh <- fmt.Errorf("worker %d: marshal report: %w", workerID, err)
				return
			}
			if !bytes.Equal(gotReportJSON, refReportJSON) {
				errCh <- fmt.Errorf("worker %d: report JSON diverged from reference", workerID)
				return
			}

			if gotHuman := RenderHuman(gotReport); gotHuman != refHuman {
				errCh <- fmt.Errorf("worker %d: RenderHuman diverged from reference", workerID)
				return
			}
			if gotMarkdown := RenderMarkdown(gotReport); gotMarkdown != refMarkdown {
				errCh <- fmt.Errorf("worker %d: RenderMarkdown diverged from reference", workerID)
				return
			}

			gotReceipt := ScoreLoopTurn(inputPath)
			if !reflect.DeepEqual(gotReceipt, refReceipt) {
				errCh <- fmt.Errorf("worker %d: ScoreLoopTurn diverged from reference", workerID)
				return
			}
			if gotFormatted := FormatLoopTurnReceipt(gotReceipt); gotFormatted != refReceiptFormatted {
				errCh <- fmt.Errorf("worker %d: FormatLoopTurnReceipt diverged from reference", workerID)
				return
			}

			gotComposed, err := ComposeV1("issue-9767-concurrent-witness", compositionInputs)
			if err != nil {
				errCh <- fmt.Errorf("worker %d: ComposeV1 failed: %w", workerID, err)
				return
			}
			if !reflect.DeepEqual(gotComposed, refComposed) {
				errCh <- fmt.Errorf("worker %d: ComposeV1 diverged from reference", workerID)
				return
			}

			gotComparisonReport := gotReport
			if err := Compare(&gotComparisonReport, priorReport); err != nil {
				errCh <- fmt.Errorf("worker %d: Compare failed: %w", workerID, err)
				return
			}
			if !reflect.DeepEqual(gotComparisonReport, refComparisonReport) {
				errCh <- fmt.Errorf("worker %d: Compare report diverged from reference", workerID)
				return
			}

			gotFold, err := FoldUsageReader(strings.NewReader(sampleUsageLines))
			if err != nil {
				errCh <- fmt.Errorf("worker %d: FoldUsageReader failed: %w", workerID, err)
				return
			}
			if !reflect.DeepEqual(gotFold, refFold) {
				errCh <- fmt.Errorf("worker %d: FoldUsageReader diverged from reference", workerID)
				return
			}
			if gotFormattedFold := FormatUsageFold(gotFold); gotFormattedFold != refFormattedFold {
				errCh <- fmt.Errorf("worker %d: FormatUsageFold diverged from reference", workerID)
				return
			}
		}(g)
	}

	close(start)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Error(err)
	}
}
