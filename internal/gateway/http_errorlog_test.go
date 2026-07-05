package gateway

import (
	"fmt"
	"log"
	"strings"
	"testing"
)

// TestLogfWriterRoutesThroughLogf pins the #2772 fix: net/http's ErrorLog must drain into the
// gateway's logf sink, not os.Stderr. A single (possibly multi-line) message — the shape of a
// net/http panic-recovery dump — is forwarded verbatim minus the trailing newline that log adds.
func TestLogfWriterRoutesThroughLogf(t *testing.T) {
	var got []string
	w := logfWriter{logf: func(format string, args ...any) {
		got = append(got, fmt.Sprintf(format, args...))
	}}
	el := log.New(w, "", 0)

	el.Printf("http: panic serving 127.0.0.1:1: boom\ngoroutine 1 [running]:\nmain.h()")

	if len(got) != 1 {
		t.Fatalf("logf calls = %d, want 1 (one whole message per ErrorLog line)", len(got))
	}
	if !strings.Contains(got[0], "panic serving") || !strings.Contains(got[0], "goroutine 1") {
		t.Fatalf("message not forwarded verbatim: %q", got[0])
	}
	if strings.HasSuffix(got[0], "\n") {
		t.Fatalf("trailing newline not trimmed: %q", got[0])
	}
}

// TestLogfWriterNilLogfIsSilentNoOp is the guard-default path: when --log is off, s.logf is a
// no-op and (belt-and-braces) may be nil. The ErrorLog write must then drop the bytes and emit
// nothing — that is what keeps a recovered panic OFF the wrapped agent's controlling TTY.
func TestLogfWriterNilLogfIsSilentNoOp(t *testing.T) {
	w := logfWriter{logf: nil}
	msg := []byte("http: panic serving X: boom\n")
	n, err := w.Write(msg)
	if err != nil {
		t.Fatalf("Write err = %v, want nil", err)
	}
	if n != len(msg) {
		t.Fatalf("n = %d, want %d (must report all bytes consumed so log never errors)", n, len(msg))
	}
}
