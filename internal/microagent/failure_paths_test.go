package microagent

import (
	"strings"
	"testing"
	"time"
)

func TestCompatibilityErrorsNameRecovery(t *testing.T) {
	_, _, err := ComposeCompatible(nil, CompatibilityConfig{})
	for _, want := range []string{"invalid compatibility config", "set max_batch", "max_padding"} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("config error=%v missing %q", err, want)
		}
	}
	_, _, err = ComposeCompatible([]CompatibleWork{{}}, CompatibilityConfig{MaxBatch: 1, MaxQueuePerClass: 1, Now: time.Now()})
	for _, want := range []string{"requires id and positive tokens", "assign a nonempty id", "tokens > 0"} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("work error=%v missing %q", err, want)
		}
	}
}
