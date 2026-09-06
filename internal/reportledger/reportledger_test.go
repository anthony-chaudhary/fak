package reportledger

import (
	"fmt"
	"reflect"
	"testing"
)

type ledgerRow struct {
	Date        string
	GeneratedAt string
	Payload     string
}

func (r ledgerRow) LedgerDate() string        { return r.Date }
func (r ledgerRow) LedgerGeneratedAt() string { return r.GeneratedAt }

func TestLatestBeforeOrdersByDateThenGeneratedAt(t *testing.T) {
	prior := []ledgerRow{
		{Date: "2026-08-30", GeneratedAt: "2026-08-30T09:00:00Z", Payload: "earlier-day"},
		{Date: "2026-08-31", GeneratedAt: "2026-08-31T08:00:00Z", Payload: "same-day-earlier"},
		{Date: "2026-08-31", GeneratedAt: "2026-08-31T17:00:00Z", Payload: "same-day-later"},
	}
	got, ok := LatestBefore(ledgerRow{Date: "2026-08-31", GeneratedAt: "2026-08-31T20:00:00Z"}, prior)
	if !ok || got.Payload != "same-day-later" {
		t.Fatalf("LatestBefore = (%#v, %v), want the same-day later tick", got, ok)
	}
}

// A row whose generated-at equals the reference row's is its own prior
// generation; re-appending must not trend a row against itself.
func TestLatestBeforeExcludesOwnGeneration(t *testing.T) {
	self := ledgerRow{Date: "2026-08-31", GeneratedAt: "2026-08-31T17:00:00Z"}
	prior := []ledgerRow{
		{Date: "2026-08-31", GeneratedAt: "2026-08-31T17:00:00Z", Payload: "self"},
		{Date: "2026-08-31", GeneratedAt: "2026-08-31T08:00:00Z", Payload: "same-day-earlier"},
	}
	got, ok := LatestBefore(self, prior)
	if !ok || got.Payload != "same-day-earlier" {
		t.Fatalf("LatestBefore = (%#v, %v), want the earlier same-day tick", got, ok)
	}
}

// An empty generated-at never matches the self stamp (jsonlledger skips the
// comparison on empty tiebreaks), so a row without a stamp stays a candidate.
func TestLatestBeforeKeepsRowsWithoutGeneratedAt(t *testing.T) {
	prior := []ledgerRow{{Date: "2026-08-31", Payload: "no-stamp"}}
	got, ok := LatestBefore(ledgerRow{Date: "2026-08-31", GeneratedAt: "2026-08-31T17:00:00Z"}, prior)
	if !ok || got.Payload != "no-stamp" {
		t.Fatalf("LatestBefore = (%#v, %v), want the unstamped candidate", got, ok)
	}
}

func TestLatestBeforeWithoutCandidates(t *testing.T) {
	got, ok := LatestBefore(ledgerRow{Date: "2026-09-01"}, nil)
	if ok || !reflect.DeepEqual(got, ledgerRow{}) {
		t.Fatalf("LatestBefore = (%#v, %v), want (zero, false)", got, ok)
	}
}

func BenchmarkLatestBefore(b *testing.B) {
	prior := make([]ledgerRow, 100)
	for i := 0; i < len(prior); i++ {
		prior[i] = ledgerRow{
			Date:        fmt.Sprintf("2026-08-%02d", (i%30)+1),
			GeneratedAt: fmt.Sprintf("2026-08-%02dT%02d:00:00Z", (i%30)+1, i%24),
			Payload:     fmt.Sprintf("payload-%d", i),
		}
	}
	target := ledgerRow{
		Date:        "2026-09-01",
		GeneratedAt: "2026-09-01T12:00:00Z",
		Payload:     "target",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, ok := LatestBefore(target, prior)
		if !ok {
			b.Fatal("LatestBefore failed to find matching row")
		}
		_ = got
	}
}
