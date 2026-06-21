package agent

import "testing"

// inkernel_stop_test.go proves the in-kernel decode path honors per-request Stop
// sequences (the docs gap: "TopP/Stop are accepted but not honored here"). The
// decode loop's string-suffix stop is factored into the pure helper checkStop so it
// can be witnessed without booting a weighted model (the model.Model path OOMs under
// WSL — see the weight-backed-test memory). The helper is exactly what the loop
// calls after each appended piece, so a green here pins the served behavior.

func TestCheckStopMatchesSuffixAndTrims(t *testing.T) {
	// The accumulated text ends with a stop string => stop, and the emitted text has
	// the stop suffix trimmed (an OpenAI/Anthropic stop sequence is NOT echoed back).
	out, hit := checkStop("hello world\n\n", []string{"\n\n", "STOP"})
	if !hit {
		t.Fatalf("checkStop should fire when text ends with a stop sequence")
	}
	if out != "hello world" {
		t.Fatalf("checkStop should trim the matched stop suffix, got %q", out)
	}
}

func TestCheckStopPrefersLongestMatch(t *testing.T) {
	// When two stops both match at the tail, the LONGER one wins so the trim is
	// maximal (a caller passing both "P" and "STOP" gets "STOP" trimmed, not "P").
	out, hit := checkStop("done STOP", []string{"P", "STOP"})
	if !hit {
		t.Fatalf("checkStop should fire")
	}
	if out != "done " {
		t.Fatalf("checkStop should trim the longest matching stop, got %q", out)
	}
}

func TestCheckStopNoMatch(t *testing.T) {
	out, hit := checkStop("hello world", []string{"\n\n", "STOP"})
	if hit {
		t.Fatalf("checkStop must not fire when no stop sequence matches the tail")
	}
	if out != "hello world" {
		t.Fatalf("checkStop must return the text unchanged on no match, got %q", out)
	}
}

func TestCheckStopEmptyStopsNeverFires(t *testing.T) {
	// No stop sequences (the default in-kernel path) => never a string-stop; the
	// loop falls through to token-ID stops / maxNew exactly as before the seam.
	out, hit := checkStop("anything at all", nil)
	if hit {
		t.Fatalf("checkStop must not fire with no stop sequences (pre-seam behavior)")
	}
	if out != "anything at all" {
		t.Fatalf("checkStop must pass text through unchanged, got %q", out)
	}
}

func TestCheckStopIgnoresEmptyStopString(t *testing.T) {
	// An empty stop string would suffix-match every text; it must be ignored so a
	// caller forwarding a sloppy [""] does not truncate every turn to nothing.
	out, hit := checkStop("hello", []string{""})
	if hit {
		t.Fatalf("an empty stop string must be ignored, not match everything")
	}
	if out != "hello" {
		t.Fatalf("text must be unchanged, got %q", out)
	}
}
