package gateway

import (
	"strings"
	"testing"
)

// TestStopHoldbackBufferSplitsAcrossChunks verifies the checkable step of #10719:
// given chunked stream ["hello world ST", "OP"] and stop string "STOP",
// the client receives only "hello world " and zero bytes of "ST" are leaked across SSE chunks.
func TestStopHoldbackBufferSplitsAcrossChunks(t *testing.T) {
	buf := NewStopHoldbackBuffer([]string{"STOP"})

	var emitted []string

	// Chunk 1: "hello world ST"
	out1 := buf.Append("hello world ST")
	if out1 != "" {
		emitted = append(emitted, out1)
	}

	// At this point, "ST" must be held back, so out1 contains zero bytes of "ST"
	if strings.Contains(out1, "ST") {
		t.Fatalf("partial stop string 'ST' leaked in chunk 1: %q", out1)
	}

	// Chunk 2: "OP" -> forms full "STOP"
	out2 := buf.Append("OP")
	if out2 != "" {
		emitted = append(emitted, out2)
	}
	if strings.Contains(out2, "STOP") || strings.Contains(out2, "ST") || strings.Contains(out2, "OP") {
		t.Fatalf("stop marker leaked in chunk 2: %q", out2)
	}

	if !buf.Stopped() {
		t.Fatalf("expected buf.Stopped() to be true")
	}
	if buf.MatchedStop() != "STOP" {
		t.Fatalf("matched stop = %q, want 'STOP'", buf.MatchedStop())
	}

	// Final flush on stream close: should emit nothing because stop matched
	outFinal := buf.Flush()
	if outFinal != "" {
		t.Fatalf("flush emitted %q, want empty string after stop match", outFinal)
	}

	joined := strings.Join(emitted, "")
	if joined != "hello world " {
		t.Fatalf("total emitted = %q, want 'hello world '", joined)
	}
}

// TestStopHoldbackBufferNoStopFlushesAll verifies that when no stop string matches,
// all buffered characters are completely emitted through Append and Flush.
func TestStopHoldbackBufferNoStopFlushesAll(t *testing.T) {
	buf := NewStopHoldbackBuffer([]string{"STOP"})

	chunks := []string{"hello ", "world ", "ST", "EP ", "ahead"}
	var emitted []string

	for _, c := range chunks {
		if out := buf.Append(c); out != "" {
			emitted = append(emitted, out)
		}
	}
	if out := buf.Flush(); out != "" {
		emitted = append(emitted, out)
	}

	joined := strings.Join(emitted, "")
	want := "hello world STEP ahead"
	if joined != want {
		t.Fatalf("total emitted = %q, want %q", joined, want)
	}
	if buf.Stopped() {
		t.Fatalf("buffer should not be stopped")
	}
}

// TestStopHoldbackBufferMultipleStops verifies handling of multiple stop sequences
// with varying lengths, picking the earliest match.
func TestStopHoldbackBufferMultipleStops(t *testing.T) {
	buf := NewStopHoldbackBuffer([]string{"<|im_end|>", "</tool_call>"})

	out1 := buf.Append("processing data: <")
	out2 := buf.Append("/tool_call> suffix")

	var emitted []string
	if out1 != "" {
		emitted = append(emitted, out1)
	}
	if out2 != "" {
		emitted = append(emitted, out2)
	}
	if outFinal := buf.Flush(); outFinal != "" {
		emitted = append(emitted, outFinal)
	}

	joined := strings.Join(emitted, "")
	want := "processing data: "
	if joined != want {
		t.Fatalf("total emitted = %q, want %q", joined, want)
	}
	if !buf.Stopped() {
		t.Fatalf("expected Stopped() == true")
	}
	if buf.MatchedStop() != "</tool_call>" {
		t.Fatalf("matched stop = %q, want '</tool_call>'", buf.MatchedStop())
	}
}
