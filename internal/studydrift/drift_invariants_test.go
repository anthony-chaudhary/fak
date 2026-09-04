package studydrift

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

// TestDriftInvariantsDigestDeterministic asserts standard prefix and SHA-256 formatting invariants.
func TestDriftInvariantsDigestDeterministic(t *testing.T) {
	data := []byte("deterministic content payload")
	d1 := DigestSource(data)
	d2 := DigestSource(data)
	if d1 != d2 {
		t.Fatalf("DigestSource non-deterministic: %s vs %s", d1, d2)
	}
	if !strings.HasPrefix(d1, "sha256:") {
		t.Fatalf("DigestSource missing sha256: prefix: %s", d1)
	}
	hexPart := strings.TrimPrefix(d1, "sha256:")
	if len(hexPart) != 64 {
		t.Fatalf("DigestSource hex length = %d, want 64", len(hexPart))
	}
	for _, r := range hexPart {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			t.Fatalf("DigestSource non-hex character %c in %s", r, hexPart)
		}
	}
}

// TestDriftInvariantsCorruptPinRefusal verifies fail-closed refusal when the pin itself is corrupted.
func TestDriftInvariantsCorruptPinRefusal(t *testing.T) {
	pinBytes := []byte("pinned content")
	corruptPin := PinnedSource{
		Repository:    "acme/lib",
		URL:           "https://acme.test/lib",
		Revision:      "v1.0",
		Bytes:         pinBytes,
		Digest:        "sha256:0000000000000000000000000000000000000000000000000000000000000000",
		ObservationID: "obs-1",
		DecisionID:    "dec-1",
	}

	obsBytes := []byte("pinned content")
	obs := SourceObservation{
		ObservedAt:    "2026-09-04T12:00:00Z",
		URL:           "https://acme.test/lib",
		Revision:      "v1.0",
		Bytes:         obsBytes,
		Digest:        DigestSource(obsBytes),
		ObservationID: "obs-2",
		DecisionID:    "dec-2",
	}

	receipt := RefreshSource(corruptPin, obs)
	if receipt.Status != RefreshUnverifiable {
		t.Fatalf("expected RefreshUnverifiable for corrupted pin, got %s", receipt.Status)
	}
	if receipt.Canonical != nil {
		t.Fatalf("expected nil Canonical on unverifiable status, got %+v", receipt.Canonical)
	}
}

// TestDriftInvariantsCorruptObservationRefusal verifies fail-closed refusal when observation bytes do not match digest.
func TestDriftInvariantsCorruptObservationRefusal(t *testing.T) {
	pinBytes := []byte("clean content")
	pin := PinnedSource{
		Repository:    "acme/lib",
		URL:           "https://acme.test/lib",
		Revision:      "v1.0",
		Bytes:         pinBytes,
		Digest:        DigestSource(pinBytes),
		ObservationID: "obs-1",
		DecisionID:    "dec-1",
	}

	tamperedObs := SourceObservation{
		ObservedAt:    "2026-09-04T12:00:00Z",
		URL:           "https://acme.test/lib",
		Revision:      "v1.0",
		Bytes:         []byte("tampered content"),
		Digest:        DigestSource(pinBytes), // Mismatched digest
		ObservationID: "obs-2",
		DecisionID:    "dec-2",
	}

	receipt := RefreshSource(pin, tamperedObs)
	if receipt.Status != RefreshUnverifiable {
		t.Fatalf("expected RefreshUnverifiable for tampered observation, got %s", receipt.Status)
	}
	if receipt.Canonical != nil {
		t.Fatalf("expected nil Canonical on unverifiable status, got %+v", receipt.Canonical)
	}
}

// TestDriftInvariantsTimestampMalformed verifies fail-closed refusal on invalid or empty timestamps.
func TestDriftInvariantsTimestampMalformed(t *testing.T) {
	pinBytes := []byte("clean content")
	pin := PinnedSource{
		Bytes:  pinBytes,
		Digest: DigestSource(pinBytes),
	}

	invalidTimestamps := []string{
		"",
		"invalid-date",
		"2026-09-04",
		"04-09-2026T12:00:00Z",
	}

	for _, ts := range invalidTimestamps {
		obs := SourceObservation{
			ObservedAt: ts,
			Bytes:      pinBytes,
			Digest:     DigestSource(pinBytes),
		}
		receipt := RefreshSource(pin, obs)
		if receipt.Status != RefreshUnverifiable {
			t.Fatalf("expected RefreshUnverifiable for timestamp %q, got %s", ts, receipt.Status)
		}
		if receipt.Canonical != nil {
			t.Fatalf("expected nil Canonical for timestamp %q", ts)
		}
	}
}

// TestDriftInvariantsMissingAuditIDsOnDrift verifies that content changes require explicit observation and decision IDs.
func TestDriftInvariantsMissingAuditIDsOnDrift(t *testing.T) {
	pinBytes := []byte("v1 content")
	pin := PinnedSource{
		Repository:    "acme/lib",
		Bytes:         pinBytes,
		Digest:        DigestSource(pinBytes),
		ObservationID: "obs-1",
		DecisionID:    "dec-1",
	}

	newBytes := []byte("v2 content drifted")
	newDigest := DigestSource(newBytes)

	// Missing ObservationID
	obsNoObsID := SourceObservation{
		ObservedAt: "2026-09-04T12:00:00Z",
		Bytes:      newBytes,
		Digest:     newDigest,
		DecisionID: "dec-2",
	}
	r1 := RefreshSource(pin, obsNoObsID)
	if r1.Status != RefreshUnverifiable || r1.Canonical != nil {
		t.Fatalf("expected unverifiable without observation ID, got %+v", r1)
	}

	// Missing DecisionID
	obsNoDecID := SourceObservation{
		ObservedAt:    "2026-09-04T12:00:00Z",
		Bytes:         newBytes,
		Digest:        newDigest,
		ObservationID: "obs-2",
	}
	r2 := RefreshSource(pin, obsNoDecID)
	if r2.Status != RefreshUnverifiable || r2.Canonical != nil {
		t.Fatalf("expected unverifiable without decision ID, got %+v", r2)
	}
}

// TestDriftInvariantsMovedSemantics verifies that URL or Revision changes without byte drift yield RefreshMoved.
func TestDriftInvariantsMovedSemantics(t *testing.T) {
	bytes := []byte("constant byte content")
	digest := DigestSource(bytes)

	pin := PinnedSource{
		Repository:    "acme/lib",
		URL:           "https://old.url/lib",
		Revision:      "v1.0.0",
		Bytes:         bytes,
		Digest:        digest,
		ObservationID: "obs-1",
		DecisionID:    "dec-1",
	}

	// Only URL changed
	obsURL := SourceObservation{
		ObservedAt:    "2026-09-04T12:00:00Z",
		URL:           "https://new.url/lib",
		Revision:      "v1.0.0",
		Bytes:         bytes,
		Digest:        digest,
		ObservationID: "obs-2",
		DecisionID:    "dec-2",
	}
	r1 := RefreshSource(pin, obsURL)
	if r1.Status != RefreshMoved {
		t.Fatalf("expected RefreshMoved for URL change, got %s", r1.Status)
	}
	if r1.Canonical == nil || r1.Canonical.URL != "https://new.url/lib" {
		t.Fatalf("expected Canonical with updated URL: %+v", r1.Canonical)
	}
	if r1.Supersedes != nil {
		t.Fatalf("RefreshMoved must not record Supersedes: %+v", r1.Supersedes)
	}

	// Only Revision changed
	obsRev := SourceObservation{
		ObservedAt:    "2026-09-04T12:00:00Z",
		URL:           "https://old.url/lib",
		Revision:      "v1.0.1",
		Bytes:         bytes,
		Digest:        digest,
		ObservationID: "obs-3",
		DecisionID:    "dec-3",
	}
	r2 := RefreshSource(pin, obsRev)
	if r2.Status != RefreshMoved {
		t.Fatalf("expected RefreshMoved for revision change, got %s", r2.Status)
	}
	if r2.Canonical == nil || r2.Canonical.Revision != "v1.0.1" {
		t.Fatalf("expected Canonical with updated revision: %+v", r2.Canonical)
	}
}

// TestDriftInvariantsUnavailableSemantics verifies that unavailable observations short-circuit fail-closed.
func TestDriftInvariantsUnavailableSemantics(t *testing.T) {
	bytes := []byte("baseline bytes")
	pin := PinnedSource{
		Repository: "acme/lib",
		Bytes:      bytes,
		Digest:     DigestSource(bytes),
	}

	obs := SourceObservation{
		ObservedAt:  "2026-09-04T12:00:00Z",
		Unavailable: true,
	}

	receipt := RefreshSource(pin, obs)
	if receipt.Status != RefreshUnavailable {
		t.Fatalf("expected RefreshUnavailable, got %s", receipt.Status)
	}
	if receipt.Canonical != nil {
		t.Fatalf("unavailable must never return canonical pin: %+v", receipt.Canonical)
	}
}

// TestDriftInvariantsSelectDueInvariants asserts ordering, filtering, and limit contracts for SelectDue.
func TestDriftInvariantsSelectDueInvariants(t *testing.T) {
	now := time.Date(2026, 9, 4, 0, 0, 0, 0, time.UTC)

	reg := studymonitor.Registry{
		Repositories: []studymonitor.Repository{
			{Repository: "z-paused", Status: "paused", Priority: 1, LastChecked: "2026-01-01"},
			{Repository: "b-active-p1-old", Status: "active", Priority: 1, LastChecked: "2026-07-01"},
			{Repository: "a-active-p1-old", Status: "active", Priority: 1, LastChecked: "2026-07-01"},
			{Repository: "c-active-p1-recent", Status: "active", Priority: 1, LastChecked: "2026-09-01"},
			{Repository: "d-active-p2-overdue", Status: "active", Priority: 2, LastChecked: "2026-06-01"},
			{Repository: "e-active-p1-invalid-date", Status: "active", Priority: 1, LastChecked: "malformed"},
		},
	}

	// 1. Non-positive limit yields nil.
	if got := SelectDue(reg, now, 14, 0); got != nil {
		t.Fatalf("limit 0 must return nil, got %v", got)
	}
	if got := SelectDue(reg, now, 14, -3); got != nil {
		t.Fatalf("negative limit must return nil, got %v", got)
	}

	// 2. Select with limit 10 (dueDays = 14 days cutoff is 2026-08-21).
	// c-active-p1-recent (2026-09-01) is after cutoff -> NOT due.
	// z-paused is paused -> NOT due.
	// e-active-p1-invalid-date is malformed -> treated as due.
	// Order should be:
	// Priority 1:
	//   e-active-p1-invalid-date (LastChecked: "malformed")
	//   a-active-p1-old (LastChecked: "2026-07-01", name "a" < "b")
	//   b-active-p1-old (LastChecked: "2026-07-01", name "b")
	// Priority 2:
	//   d-active-p2-overdue (LastChecked: "2026-06-01")
	due := SelectDue(reg, now, 14, 10)
	var gotNames []string
	for _, r := range due {
		gotNames = append(gotNames, r.Repository)
	}
	wantNames := []string{
		"a-active-p1-old",
		"b-active-p1-old",
		"e-active-p1-invalid-date",
		"d-active-p2-overdue",
	}
	// Let's verify each element according to the sort comparator:
	// If priority differs: due[i].Priority < due[j].Priority
	// If LastChecked differs: due[i].LastChecked < due[j].LastChecked ("2026-07-01" < "malformed")
	// If LastChecked same: due[i].Repository < due[j].Repository ("a..." < "b...")
	// So for Priority 1: "2026-07-01" comes before "malformed". "a-active-p1-old" before "b-active-p1-old".
	if !reflect.DeepEqual(gotNames, wantNames) {
		t.Fatalf("got due repositories %v, want %v", gotNames, wantNames)
	}

	// 3. Limit bounds the slice strictly.
	dueBounded := SelectDue(reg, now, 14, 2)
	if len(dueBounded) != 2 {
		t.Fatalf("expected exactly 2 items, got %d", len(dueBounded))
	}
	if dueBounded[0].Repository != "a-active-p1-old" || dueBounded[1].Repository != "b-active-p1-old" {
		t.Fatalf("unexpected bounded slice: %v", dueBounded)
	}
}

// TestDriftInvariantsMemoryIsolation ensures that byte arrays are deeply cloned and never alias caller storage.
func TestDriftInvariantsMemoryIsolation(t *testing.T) {
	origPinBytes := []byte("orig pin bytes")
	pin := PinnedSource{
		Repository: "acme/lib",
		Bytes:      origPinBytes,
		Digest:     DigestSource(origPinBytes),
	}

	origObsBytes := []byte("orig obs bytes")
	obs := SourceObservation{
		ObservedAt:    "2026-09-04T12:00:00Z",
		Bytes:         origObsBytes,
		Digest:        DigestSource(origObsBytes),
		ObservationID: "obs-2",
		DecisionID:    "dec-2",
	}

	receipt := RefreshSource(pin, obs)
	// Mutate the original slices
	origPinBytes[0] = 'Z'
	origObsBytes[0] = 'Z'

	if receipt.Prior.Bytes[0] == 'Z' {
		t.Fatal("receipt.Prior.Bytes aliased input pin slice")
	}
	if receipt.Observation.Bytes[0] == 'Z' {
		t.Fatal("receipt.Observation.Bytes aliased input observation slice")
	}
	if receipt.Canonical.Bytes[0] == 'Z' {
		t.Fatal("receipt.Canonical.Bytes aliased input observation slice")
	}
}
