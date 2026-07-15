package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

func TestCtxRestoreTombstoneIDs(t *testing.T) {
	idA := strings.Repeat("a", 64)
	idB := strings.Repeat("b", 64)
	text := "[fak] originating task: id=" + idA + " x; id=short; id=" + idB + "; id=" + idA
	got := CtxRestoreTombstoneIDs(text)
	if len(got) != 2 || got[0] != idA || got[1] != idB {
		t.Fatalf("ids=%v", got)
	}
}

func TestCtxRestoreBounded(t *testing.T) {
	srv := newTestServer(t)
	const trace = "worker-bounded"
	a := []byte("alpha")
	b := []byte("bravo-long")
	idA, idB := ctxplan.Digest(a), ctxplan.Digest(b)
	srv.stashRestore(trace, idA, "alpha", a)
	srv.stashRestore(trace, idB, "bravo", b)

	got := srv.restoreContextBounded("", CtxRestoreBoundedRequest{TraceID: trace, IDs: []string{idA, idB}, ByteBudget: len(a)})
	if got.UsedBytes != len(a) || got.Elided != 1 || len(got.Spans) != 2 {
		t.Fatalf("got=%+v", got)
	}
	if got.Spans[0].Status != "RESTORED" || got.Spans[0].Bytes != string(a) || got.Spans[1].Status != "ELIDED" {
		t.Fatalf("spans=%+v", got.Spans)
	}
}

func TestCtxRestoreMissIsStructured(t *testing.T) {
	srv := newTestServer(t)
	got := srv.restoreContextBounded("", CtxRestoreBoundedRequest{TraceID: "missing", IDs: []string{strings.Repeat("c", 64)}, ByteBudget: 100})
	if got.Misses != 1 || len(got.Spans) != 1 || got.Spans[0].Status != "MISS" {
		t.Fatalf("got=%+v", got)
	}
}

func TestCtxRestoreMissSealedStaysTrustGated(t *testing.T) {
	srv := newTestServer(t)
	const trace = "worker-sealed"
	data := []byte("secret")
	id := ctxplan.Digest(data)
	srv.stashRestore(trace, id, "secret", data)
	srv.ctxRestoreMu.Lock()
	srv.ctxRestore[trace].entries[0].sealed = true
	srv.ctxRestoreMu.Unlock()

	got := srv.restoreContextBounded("", CtxRestoreBoundedRequest{TraceID: trace, IDs: []string{id}, ByteBudget: 100})
	if got.Refused != 1 || got.Spans[0].Status != "REFUSED" || got.Spans[0].Bytes != "" {
		t.Fatalf("trust gate bypassed: %+v", got)
	}
}
