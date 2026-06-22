package gateway

// metrics_blob_test.go — the content-addressed blob store (internal/blob) footprint
// family on /metrics. The store is the ONE CAS the vDSO tier-2 cache and the
// context-MMU page-out share; it kept concurrency-safe KPI taps but emitted no
// metrics, so its resident footprint, content-dedup, and byte-bound eviction were
// invisible to a scrape. The values come from the process-global store and may carry
// counts from sibling tests, so the contract is "these families are emitted" — except
// the live-read assertion below, which stores a unique large blob and confirms the
// resident-blobs gauge reflects it.

import (
	"bytes"
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/blob"
)

func TestMetricsExposesBlobStats(t *testing.T) {
	srv := newTestServer(t)

	for _, want := range []string{
		"# TYPE fak_blob_puts_total counter",
		"fak_blob_puts_total ",
		"# TYPE fak_blob_dedup_hits_total counter",
		"# TYPE fak_blob_resolves_total counter",
		"# TYPE fak_blob_resident_blobs gauge",
		"fak_blob_resident_blobs ",
		"# TYPE fak_blob_resident_bytes gauge",
		"# TYPE fak_blob_evicted_total counter",
		"# TYPE fak_blob_max_bytes gauge",
		"# TYPE fak_blob_dedup_ratio gauge",
		"fak_blob_dedup_ratio ",
	} {
		if text := srv.renderMetrics(); !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	// Live read: store a unique blob LARGER than InlineMax so it lands in the CAS (not
	// inline), then confirm the resident-blobs gauge rose to reflect it. Pin it so the
	// byte bound cannot evict it out from under the assertion.
	before := blob.Default.Len()
	payload := bytes.Repeat([]byte("fak-blob-metric-witness-"), 64) // > InlineMax
	ref, err := blob.Default.Put(context.Background(), payload)
	if err != nil {
		t.Fatalf("blob put: %v", err)
	}
	blob.Default.Pin(ref.Digest)
	t.Cleanup(func() { blob.Default.Unpin(ref.Digest) })
	if got := blob.Default.Len(); got <= before {
		t.Fatalf("resident blob count did not rise after a CAS put: before=%d after=%d", before, got)
	}

	text := srv.renderMetrics()
	line := metricLine(text, "fak_blob_resident_blobs")
	if line == "" {
		t.Fatalf("no fak_blob_resident_blobs line:\n%s", text)
	}
	n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "fak_blob_resident_blobs")))
	if err != nil {
		t.Fatalf("parse %q: %v", line, err)
	}
	if n < 1 {
		t.Fatalf("fak_blob_resident_blobs = %d, want >= 1 after a pinned CAS put", n)
	}
}

// metricLine returns the first non-comment exposition line for a metric name.
func metricLine(text, name string) string {
	for _, ln := range strings.Split(text, "\n") {
		if strings.HasPrefix(ln, name+" ") {
			return ln
		}
	}
	return ""
}
