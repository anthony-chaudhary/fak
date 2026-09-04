package studydrift

import (
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func TestRefreshSourceOfflineStatesAndImmutableEvidence(t *testing.T) {
	oldBytes := []byte("old source")
	pin := PinnedSource{Repository: "owner/repo", URL: "https://old", Revision: "r1", Bytes: oldBytes, Digest: DigestSource(oldBytes), ObservationID: "obs-1", DecisionID: "decision-1"}
	at := "2026-08-26T12:00:00Z"
	tests := []struct {
		name        string
		observation SourceObservation
		want        RefreshStatus
	}{
		{"unchanged", SourceObservation{ObservedAt: at, URL: pin.URL, Revision: pin.Revision, Bytes: oldBytes, Digest: pin.Digest, ObservationID: "obs-2", DecisionID: "decision-2"}, RefreshUnchanged},
		{"moved", SourceObservation{ObservedAt: at, URL: "https://new", Revision: "r2", Bytes: oldBytes, Digest: pin.Digest, ObservationID: "obs-2", DecisionID: "decision-2"}, RefreshMoved},
		{"changed", SourceObservation{ObservedAt: at, URL: pin.URL, Revision: "r2", Bytes: []byte("new source"), Digest: DigestSource([]byte("new source")), ObservationID: "obs-2", DecisionID: "decision-2"}, RefreshChanged},
		{"unavailable", SourceObservation{ObservedAt: at, Unavailable: true}, RefreshUnavailable},
		{"unverifiable", SourceObservation{ObservedAt: at, Bytes: []byte("tampered"), Digest: pin.Digest}, RefreshUnverifiable},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			before := clonePinnedSource(pin)
			receipt := RefreshSource(pin, test.observation)
			if receipt.Status != test.want || receipt.ObservedAt != at {
				t.Fatalf("receipt = %+v", receipt)
			}
			if !reflect.DeepEqual(pin, before) {
				t.Fatalf("canonical input mutated: got %+v want %+v", pin, before)
			}
			if test.want == RefreshUnavailable || test.want == RefreshUnverifiable {
				if receipt.Canonical != nil {
					t.Fatalf("failure state mutated canonical: %+v", receipt.Canonical)
				}
			}
			if test.want == RefreshChanged {
				if string(receipt.Prior.Bytes) != "old source" || receipt.Prior.Digest != pin.Digest {
					t.Fatalf("prior evidence not preserved: %+v", receipt.Prior)
				}
				if receipt.Supersedes == nil || receipt.Supersedes.Observation != "obs-1" || receipt.Supersedes.Decision != "decision-1" {
					t.Fatalf("supersession = %+v", receipt.Supersedes)
				}
			}
		})
	}
}

func TestRefreshSourceCopiesCallerBytes(t *testing.T) {
	oldBytes := []byte("old")
	newBytes := []byte("new")
	pin := PinnedSource{Bytes: oldBytes, Digest: DigestSource(oldBytes)}
	observation := SourceObservation{ObservedAt: "2026-08-26T12:00:00Z", Bytes: newBytes, Digest: DigestSource(newBytes), ObservationID: "obs-2", DecisionID: "decision-2"}
	receipt := RefreshSource(pin, observation)
	oldBytes[0], newBytes[0] = 'X', 'X'
	if string(receipt.Prior.Bytes) != "old" || string(receipt.Observation.Bytes) != "new" || string(receipt.Canonical.Bytes) != "new" {
		t.Fatalf("receipt aliases caller bytes: %+v", receipt)
	}
}

func TestSelectDueIsBoundedAndPriorityAware(t *testing.T) {
	registry := studymonitor.Registry{Repositories: []studymonitor.Repository{
		{Repository: "low", Status: "active", Priority: 3, LastChecked: "2026-01-01"},
		{Repository: "fresh", Status: "active", Priority: 1, LastChecked: "2026-08-25"},
		{Repository: "high", Status: "active", Priority: 1, LastChecked: "2026-02-01"},
		{Repository: "paused", Status: "paused", Priority: 0, LastChecked: "2026-01-01"},
		{Repository: "middle", Status: "active", Priority: 2, LastChecked: "2026-03-01"},
	}}
	due := SelectDue(registry, time.Date(2026, 8, 26, 0, 0, 0, 0, time.UTC), 30, 2)
	if got := []string{due[0].Repository, due[1].Repository}; !reflect.DeepEqual(got, []string{"high", "middle"}) {
		t.Fatalf("due = %v", got)
	}
	if got := SelectDue(registry, time.Now(), 30, 0); len(got) != 0 {
		t.Fatalf("zero limit selected %d", len(got))
	}
}

func BenchmarkStudyDrift(b *testing.B) {
	oldBytes := []byte("module github.com/example/lib\n\ngo 1.26\n")
	oldDigest := DigestSource(oldBytes)
	pin := PinnedSource{
		Repository:    "example/lib",
		URL:           "https://github.com/example/lib",
		Revision:      "v1.0.0",
		Bytes:         oldBytes,
		Digest:        oldDigest,
		ObservationID: "obs-100",
		DecisionID:    "dec-100",
	}

	newBytes := []byte("module github.com/example/lib\n\ngo 1.26\n\nrequire github.com/pkg/errors v0.9.1\n")
	newDigest := DigestSource(newBytes)
	obs := SourceObservation{
		ObservedAt:    "2026-09-04T12:00:00Z",
		URL:           "https://github.com/example/lib",
		Revision:      "v1.1.0",
		Bytes:         newBytes,
		Digest:        newDigest,
		ObservationID: "obs-101",
		DecisionID:    "dec-101",
	}

	b.Run("RefreshSourceChanged", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			receipt := RefreshSource(pin, obs)
			if receipt.Status != RefreshChanged {
				b.Fatalf("unexpected status: %s", receipt.Status)
			}
		}
	})

	b.Run("DigestSource", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = DigestSource(newBytes)
		}
	})

	registry := studymonitor.Registry{
		Repositories: []studymonitor.Repository{
			{Repository: "repo-a", Status: "active", Priority: 2, LastChecked: "2026-08-01"},
			{Repository: "repo-b", Status: "active", Priority: 1, LastChecked: "2026-07-15"},
			{Repository: "repo-c", Status: "active", Priority: 3, LastChecked: "2026-06-01"},
			{Repository: "repo-d", Status: "paused", Priority: 1, LastChecked: "2026-05-01"},
		},
	}
	now := time.Date(2026, 9, 4, 0, 0, 0, 0, time.UTC)
	b.Run("SelectDue", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			due := SelectDue(registry, now, 14, 2)
			if len(due) != 2 {
				b.Fatalf("unexpected count: %d", len(due))
			}
		}
	})
}
