package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

func TestDisambiguationCommittedFreshnessClean(t *testing.T) {
	original := committedDisambiguationProbe
	t.Cleanup(func() { committedDisambiguationProbe = original })
	generated, err := disambiguation.GeneratePublicIndex()
	if err != nil {
		t.Fatal(err)
	}
	committedDisambiguationProbe = func() ([]byte, error) { return generated, nil }
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"committed-freshness", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.CommittedFreshnessReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Verdict != disambiguation.CommittedFreshnessClean {
		t.Fatalf("report=%+v", report)
	}
}

func TestDisambiguationCommittedFreshnessOverlayDrift(t *testing.T) {
	original := committedDisambiguationProbe
	t.Cleanup(func() { committedDisambiguationProbe = original })
	committedDisambiguationProbe = func() ([]byte, error) { return []byte("different"), nil }
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"committed-freshness", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.CommittedFreshnessReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Verdict != disambiguation.CommittedFreshnessOverlayDrift {
		t.Fatalf("report=%+v", report)
	}
}

func TestDisambiguationCommittedFreshnessUnavailable(t *testing.T) {
	original := committedDisambiguationProbe
	t.Cleanup(func() { committedDisambiguationProbe = original })
	committedDisambiguationProbe = func() ([]byte, error) { return nil, errors.New("git unavailable") }
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"committed-freshness", "--json"}); code != 1 {
		t.Fatalf("code=%d want 1 stderr=%s", code, stderr.String())
	}
	var report disambiguation.CommittedFreshnessReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Verdict != disambiguation.CommittedFreshnessUnavailable {
		t.Fatalf("report=%+v", report)
	}
}
