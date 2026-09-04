package nativeperfartifact

import (
	"testing"
	"time"
)

// Invariant: Native performance artifact indexing must provide deterministic locator resolution by correlation key.
// Guard: Resolve returns ErrNotFound for expired or unregistered correlation keys.

func TestNativePerfArtifactLifecycle(t *testing.T) {
	t.Parallel()

	now := time.Now().UTC()
	index, err := NewIndex(5)
	if err != nil {
		t.Fatalf("failed creating index: %v", err)
	}

	const corrKey = "npc1_0123456789abcdef0123456789abcdef"
	art := ready(KindReceipt, "https://example.test/receipt.json", now.Add(time.Hour))
	if err := index.Add(Record{CorrelationKey: corrKey, Engine: "fak-native", Artifacts: []Artifact{art}}); err != nil {
		t.Fatalf("failed adding record: %v", err)
	}

	got, err := index.Resolve(corrKey, KindReceipt, now)
	if err != nil {
		t.Fatalf("failed resolving artifact: %v", err)
	}
	if got.Locator != art.Locator {
		t.Fatalf("expected locator %s, got %s", art.Locator, got.Locator)
	}
}
