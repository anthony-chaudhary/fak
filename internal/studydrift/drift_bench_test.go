package studydrift

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

var (
	benchReceiptSink RefreshReceipt
	benchDigestSink  string
	benchReposSink   []studymonitor.Repository
)

func BenchmarkDigestSource_64B(b *testing.B) {
	data := bytes.Repeat([]byte("a"), 64)
	b.ReportAllocs()
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDigestSink = DigestSource(data)
	}
}

func BenchmarkDigestSource_4KB(b *testing.B) {
	data := bytes.Repeat([]byte("a"), 4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDigestSink = DigestSource(data)
	}
}

func BenchmarkDigestSource_64KB(b *testing.B) {
	data := bytes.Repeat([]byte("a"), 65536)
	b.ReportAllocs()
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDigestSink = DigestSource(data)
	}
}

func BenchmarkRefreshSource_Unchanged(b *testing.B) {
	payload := []byte("package example\n\nconst Version = \"1.0.0\"\n")
	digest := DigestSource(payload)
	pin := PinnedSource{
		Repository:    "org/repo",
		URL:           "https://example.com/org/repo",
		Revision:      "v1.0.0",
		Bytes:         payload,
		Digest:        digest,
		ObservationID: "obs-001",
		DecisionID:    "dec-001",
	}
	obs := SourceObservation{
		ObservedAt:    "2026-09-05T12:00:00Z",
		URL:           pin.URL,
		Revision:      pin.Revision,
		Bytes:         payload,
		Digest:        digest,
		ObservationID: "obs-002",
		DecisionID:    "dec-002",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReceiptSink = RefreshSource(pin, obs)
	}
	if benchReceiptSink.Status != RefreshUnchanged {
		b.Fatalf("unexpected status: %s", benchReceiptSink.Status)
	}
}

func BenchmarkRefreshSource_Moved(b *testing.B) {
	payload := []byte("package example\n\nconst Version = \"1.0.0\"\n")
	digest := DigestSource(payload)
	pin := PinnedSource{
		Repository:    "org/repo",
		URL:           "https://example.com/org/repo-old",
		Revision:      "v1.0.0",
		Bytes:         payload,
		Digest:        digest,
		ObservationID: "obs-001",
		DecisionID:    "dec-001",
	}
	obs := SourceObservation{
		ObservedAt:    "2026-09-05T12:00:00Z",
		URL:           "https://example.com/org/repo-new",
		Revision:      "v1.0.1",
		Bytes:         payload,
		Digest:        digest,
		ObservationID: "obs-002",
		DecisionID:    "dec-002",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReceiptSink = RefreshSource(pin, obs)
	}
	if benchReceiptSink.Status != RefreshMoved {
		b.Fatalf("unexpected status: %s", benchReceiptSink.Status)
	}
}

func BenchmarkRefreshSource_Changed(b *testing.B) {
	oldPayload := []byte("package example\n\nconst Version = \"1.0.0\"\n")
	newPayload := []byte("package example\n\nconst Version = \"1.1.0\"\n")
	pin := PinnedSource{
		Repository:    "org/repo",
		URL:           "https://example.com/org/repo",
		Revision:      "v1.0.0",
		Bytes:         oldPayload,
		Digest:        DigestSource(oldPayload),
		ObservationID: "obs-001",
		DecisionID:    "dec-001",
	}
	obs := SourceObservation{
		ObservedAt:    "2026-09-05T12:00:00Z",
		URL:           pin.URL,
		Revision:      "v1.1.0",
		Bytes:         newPayload,
		Digest:        DigestSource(newPayload),
		ObservationID: "obs-002",
		DecisionID:    "dec-002",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReceiptSink = RefreshSource(pin, obs)
	}
	if benchReceiptSink.Status != RefreshChanged {
		b.Fatalf("unexpected status: %s", benchReceiptSink.Status)
	}
}

func BenchmarkRefreshSource_Unavailable(b *testing.B) {
	payload := []byte("package example\n")
	pin := PinnedSource{
		Repository: "org/repo",
		Bytes:      payload,
		Digest:     DigestSource(payload),
	}
	obs := SourceObservation{
		ObservedAt:  "2026-09-05T12:00:00Z",
		Unavailable: true,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReceiptSink = RefreshSource(pin, obs)
	}
	if benchReceiptSink.Status != RefreshUnavailable {
		b.Fatalf("unexpected status: %s", benchReceiptSink.Status)
	}
}

func BenchmarkRefreshSource_Unverifiable(b *testing.B) {
	payload := []byte("package example\n")
	pin := PinnedSource{
		Repository: "org/repo",
		Bytes:      payload,
		Digest:     DigestSource(payload),
	}
	obs := SourceObservation{
		ObservedAt: "2026-09-05T12:00:00Z",
		Bytes:      payload,
		Digest:     "sha256:0000000000000000000000000000000000000000000000000000000000000000",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReceiptSink = RefreshSource(pin, obs)
	}
	if benchReceiptSink.Status != RefreshUnverifiable {
		b.Fatalf("unexpected status: %s", benchReceiptSink.Status)
	}
}

func generateBenchRegistry(n int) studymonitor.Registry {
	repos := make([]studymonitor.Repository, n)
	baseDate := time.Date(2026, 9, 5, 0, 0, 0, 0, time.UTC)
	for i := 0; i < n; i++ {
		status := "active"
		if i%7 == 0 {
			status = "paused"
		}
		daysAgo := (i * 3) % 90
		lastChecked := baseDate.AddDate(0, 0, -daysAgo).Format("2006-01-02")
		if i%13 == 0 {
			lastChecked = "malformed-date"
		}
		repos[i] = studymonitor.Repository{
			Repository:  fmt.Sprintf("repo-%05d", i),
			Status:      status,
			Priority:    (i % 5) + 1,
			LastChecked: lastChecked,
		}
	}
	return studymonitor.Registry{Repositories: repos}
}

func BenchmarkSelectDue_SmallRegistry(b *testing.B) {
	reg := generateBenchRegistry(10)
	now := time.Date(2026, 9, 5, 0, 0, 0, 0, time.UTC)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReposSink = SelectDue(reg, now, 14, 5)
	}
}

func BenchmarkSelectDue_MediumRegistry(b *testing.B) {
	reg := generateBenchRegistry(100)
	now := time.Date(2026, 9, 5, 0, 0, 0, 0, time.UTC)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReposSink = SelectDue(reg, now, 14, 20)
	}
}

func BenchmarkSelectDue_LargeRegistry(b *testing.B) {
	reg := generateBenchRegistry(1000)
	now := time.Date(2026, 9, 5, 0, 0, 0, 0, time.UTC)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReposSink = SelectDue(reg, now, 14, 50)
	}
}

func BenchmarkReconciliationBatch(b *testing.B) {
	const batchSize = 40
	type item struct {
		pin PinnedSource
		obs SourceObservation
	}
	batch := make([]item, batchSize)
	for i := 0; i < batchSize; i++ {
		payload := []byte(fmt.Sprintf("package component%d\n", i))
		digest := DigestSource(payload)
		pin := PinnedSource{
			Repository:    fmt.Sprintf("org/repo-%d", i),
			URL:           fmt.Sprintf("https://example.com/org/repo-%d", i),
			Revision:      "v1.0.0",
			Bytes:         payload,
			Digest:        digest,
			ObservationID: fmt.Sprintf("obs-pin-%d", i),
			DecisionID:    fmt.Sprintf("dec-pin-%d", i),
		}

		var obs SourceObservation
		switch i % 4 {
		case 0:
			// Unchanged
			obs = SourceObservation{
				ObservedAt:    "2026-09-05T12:00:00Z",
				URL:           pin.URL,
				Revision:      pin.Revision,
				Bytes:         payload,
				Digest:        digest,
				ObservationID: fmt.Sprintf("obs-now-%d", i),
				DecisionID:    fmt.Sprintf("dec-now-%d", i),
			}
		case 1:
			// Moved
			obs = SourceObservation{
				ObservedAt:    "2026-09-05T12:00:00Z",
				URL:           fmt.Sprintf("https://example.com/org/repo-%d-new", i),
				Revision:      "v1.0.1",
				Bytes:         payload,
				Digest:        digest,
				ObservationID: fmt.Sprintf("obs-now-%d", i),
				DecisionID:    fmt.Sprintf("dec-now-%d", i),
			}
		case 2:
			// Changed
			driftBytes := []byte(fmt.Sprintf("package component%d\n// drifted\n", i))
			obs = SourceObservation{
				ObservedAt:    "2026-09-05T12:00:00Z",
				URL:           pin.URL,
				Revision:      "v1.1.0",
				Bytes:         driftBytes,
				Digest:        DigestSource(driftBytes),
				ObservationID: fmt.Sprintf("obs-now-%d", i),
				DecisionID:    fmt.Sprintf("dec-now-%d", i),
			}
		default:
			// Unavailable
			obs = SourceObservation{
				ObservedAt:  "2026-09-05T12:00:00Z",
				Unavailable: true,
			}
		}
		batch[i] = item{pin: pin, obs: obs}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range batch {
			benchReceiptSink = RefreshSource(batch[j].pin, batch[j].obs)
		}
	}
}

func TestBenchmarkSuiteSanity(t *testing.T) {
	reg := generateBenchRegistry(20)
	now := time.Date(2026, 9, 5, 0, 0, 0, 0, time.UTC)
	due := SelectDue(reg, now, 14, 5)
	if len(due) == 0 || len(due) > 5 {
		t.Fatalf("unexpected due repos length: %d", len(due))
	}
}
