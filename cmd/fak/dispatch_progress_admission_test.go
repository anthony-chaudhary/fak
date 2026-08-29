package main

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestDispatchFinishFirstAdmissionMatrix(t *testing.T) {
	tests := []struct {
		name         string
		input        dispatchFinishFirstAdmissionInput
		state        string
		allowedFresh int
	}{
		{
			name: "divergence", state: dispatchFinishFirstStateDiverging, allowedFresh: 0,
			input: dispatchFinishFirstAdmissionInput{EvidenceAvailable: true, GitHubAvailable: true, WIPFilesDelta: 3, WIPLinesDelta: 40, RequestedFreshStarts: 4, Finishers: 3},
		},
		{
			name: "stale oldest", state: dispatchFinishFirstStateStaleOldest, allowedFresh: 0,
			input: dispatchFinishFirstAdmissionInput{EvidenceAvailable: true, GitHubAvailable: true, OldestWIPMinutes: 1441, CloseRate: 0, RequestedFreshStarts: 2, Finishers: 2},
		},
		{
			name: "github unavailable", state: dispatchFinishFirstStateGitHubUnavailable, allowedFresh: 0,
			input: dispatchFinishFirstAdmissionInput{EvidenceAvailable: true, GitHubAvailable: false, RequestedFreshStarts: 3, Finishers: 4},
		},
		{
			name: "bounded convergence", state: dispatchFinishFirstStateRecovering, allowedFresh: 1,
			input: dispatchFinishFirstAdmissionInput{EvidenceAvailable: true, GitHubAvailable: true, WIPFilesDelta: -2, WIPLinesDelta: -20, CloseRate: 1, ConsecutiveConvergingWindows: 2, RecoveringFromDivergence: true, RequestedFreshStarts: 4, Finishers: 2},
		},
		{
			name: "converged", state: dispatchFinishFirstStateConverged, allowedFresh: 4,
			input: dispatchFinishFirstAdmissionInput{EvidenceAvailable: true, GitHubAvailable: true, WIPFilesDelta: -1, CloseRate: 1, ConsecutiveConvergingWindows: 3, RecoveringFromDivergence: true, RequestedFreshStarts: 4, Finishers: 2},
		},
		{
			name: "explicit override", state: dispatchFinishFirstStateOverride, allowedFresh: 4,
			input: dispatchFinishFirstAdmissionInput{EvidenceAvailable: true, GitHubAvailable: true, WIPFilesDelta: 86, OldestWIPMinutes: 5210, RequestedFreshStarts: 4, Finishers: 5, Override: true},
		},
		{
			name: "observed 86 file 5210 minute class", state: dispatchFinishFirstStateDiverging, allowedFresh: 0,
			input: dispatchFinishFirstAdmissionInput{EvidenceAvailable: true, GitHubAvailable: true, WIPFilesDelta: 86, WIPLinesDelta: 1200, OldestWIPMinutes: 5210, CloseRate: 0, ConsecutiveDivergingWindows: 4, RequestedFreshStarts: 6, Finishers: 5},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := evaluateDispatchFinishFirstAdmission(tt.input)
			if got.State != tt.state || got.AllowedFreshStarts != tt.allowedFresh || got.DeniedFreshStarts != tt.input.RequestedFreshStarts-tt.allowedFresh {
				t.Fatalf("admission = %+v", got)
			}
			if got.AllowedFinishers != tt.input.Finishers {
				t.Fatalf("finishers = %d, want preserved %d", got.AllowedFinishers, tt.input.Finishers)
			}
		})
	}
}

func TestDispatchFinishFirstSnapshotHysteresisAndCloseRate(t *testing.T) {
	now := time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)
	snapshots := []progressInventorySnapshot{
		{ObservedAt: now.Add(-40 * time.Minute), WIPFiles: 10, WIPLines: 100, OldestWIPMinutes: 500, GitHubAvailable: true},
		{ObservedAt: now.Add(-30 * time.Minute), WIPFiles: 12, WIPLines: 140, OldestWIPMinutes: 510, GitHubAvailable: true},
		{ObservedAt: now.Add(-20 * time.Minute), WIPFiles: 11, WIPLines: 130, OldestWIPMinutes: 520, GitHubAvailable: true},
		{ObservedAt: now.Add(-10 * time.Minute), WIPFiles: 9, WIPLines: 90, OldestWIPMinutes: 530, GitHubAvailable: true},
	}
	rows := []map[string]any{{"closed_now": 0}, {"closed_now": 2}, {"closed_now": 1}}
	in := dispatchFinishFirstInputsFromSnapshots(snapshots, rows, 5, 3, false)
	if in.WIPFilesDelta != -2 || in.WIPLinesDelta != -40 || in.ConsecutiveConvergingWindows != 2 || !in.RecoveringFromDivergence || in.CloseRate != 1 {
		t.Fatalf("snapshot input = %+v", in)
	}
	got := evaluateDispatchFinishFirstAdmission(in)
	if got.State != dispatchFinishFirstStateRecovering || got.AllowedFreshStarts != 1 || got.AllowedFinishers != 3 {
		t.Fatalf("admission = %+v", got)
	}
}

func TestDispatchFinishFirstUnobservedPreservesLegacyCap(t *testing.T) {
	got := evaluateDispatchFinishFirstAdmission(dispatchFinishFirstAdmissionInput{RequestedFreshStarts: 2, Finishers: 1})
	if got.State != dispatchFinishFirstStateUnobserved || got.AllowedFreshStarts != 2 || got.DeniedFreshStarts != 0 || got.AllowedFinishers != 1 {
		t.Fatalf("unobserved admission = %+v", got)
	}
}

func TestDispatchFinishFirstAdmissionJSONCarriesAuditablePreflightFields(t *testing.T) {
	admission := evaluateDispatchFinishFirstAdmission(dispatchFinishFirstAdmissionInput{
		EvidenceAvailable: true, GitHubAvailable: true, WIPFilesDelta: 2,
		RequestedFreshStarts: 3, Finishers: 4,
	})
	data, err := json.Marshal(admission)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, key := range []string{`"state"`, `"inputs"`, `"denied_fresh_starts"`, `"allowed_finishers"`, `"override"`, `"recovery"`} {
		if !strings.Contains(text, key) {
			t.Fatalf("admission JSON missing %s: %s", key, text)
		}
	}
}
