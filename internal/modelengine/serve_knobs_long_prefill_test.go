package modelengine

import (
	"bytes"
	"log"
	"strings"
	"testing"
)

// buildWithLog builds a fresh Engine scheduler and returns the armed long
// prefill threshold plus anything the serve-knob wiring logged.
func buildWithLog(t *testing.T) (int, string) {
	t.Helper()
	var buf bytes.Buffer
	prev, prevFlags := log.Writer(), log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	defer func() {
		log.SetOutput(prev)
		log.SetFlags(prevFlags)
	}()
	return New().nativeScheduler().longPrefillTokenThreshold, buf.String()
}

// TestServeKnobLongPrefillTokenThresholdReachesScheduler pins the serve-path
// wiring of the long-prefill fairness cap: unset, empty and zero leave it off
// silently; negative or unparsable values and values below the resident
// minimum chunk leave it off and log a refusal naming the env var; a valid
// value arms it verbatim.
func TestServeKnobLongPrefillTokenThresholdReachesScheduler(t *testing.T) {
	clearServeKnobEnv(t)
	for _, raw := range []string{"", "  ", "0"} {
		t.Setenv(envNativeLongPrefillTokenThreshold, raw)
		got, logged := buildWithLog(t)
		if got != 0 {
			t.Fatalf("%s=%q longPrefillTokenThreshold = %d, want 0 (off)", envNativeLongPrefillTokenThreshold, raw, got)
		}
		if strings.Contains(logged, envNativeLongPrefillTokenThreshold) {
			t.Fatalf("%s=%q logged %q, want silence for the disabled value", envNativeLongPrefillTokenThreshold, raw, logged)
		}
	}
	for _, raw := range []string{"-5", "abc", "8"} {
		t.Setenv(envNativeLongPrefillTokenThreshold, raw)
		got, logged := buildWithLog(t)
		if got != 0 {
			t.Fatalf("%s=%q longPrefillTokenThreshold = %d, want 0 (refused)", envNativeLongPrefillTokenThreshold, raw, got)
		}
		if !strings.Contains(logged, envNativeLongPrefillTokenThreshold) || !strings.Contains(logged, "ignored") {
			t.Fatalf("%s=%q logged %q, want a refusal naming the env var", envNativeLongPrefillTokenThreshold, raw, logged)
		}
	}
	t.Setenv(envNativeLongPrefillTokenThreshold, "1024")
	got, logged := buildWithLog(t)
	if got != 1024 {
		t.Fatalf("%s=1024 longPrefillTokenThreshold = %d, want 1024", envNativeLongPrefillTokenThreshold, got)
	}
	if logged != "" {
		t.Fatalf("%s=1024 logged %q, want silence", envNativeLongPrefillTokenThreshold, logged)
	}
}
