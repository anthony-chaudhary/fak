package docrender

import (
	"testing"
)

// Invariant: Markdown doc rendering scanning must accurately identify alternate thematic breaks without mixing markers.
// Guard: Scan returns items only when consistent thematic breaks are present.

func TestDocRenderLifecycle(t *testing.T) {
	t.Parallel()

	items := Scan("***")
	if len(items) != 1 {
		t.Fatalf("expected 1 item from Scan, got %d", len(items))
	}
	if items[0].Construct != "alternate thematic break" {
		t.Fatalf("unexpected construct: %s", items[0].Construct)
	}
}
