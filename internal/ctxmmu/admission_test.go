package ctxmmu

import (
	"strings"
	"testing"
	"unsafe"
)

// alignedBytes returns an n-byte slice whose first element starts on a
// pageAlign boundary, by over-allocating and trimming to the aligned offset.
func alignedBytes(n, pageAlign int) []byte {
	buf := make([]byte, n+pageAlign)
	base := uintptr(unsafe.Pointer(&buf[0]))
	off := int((uintptr(pageAlign) - base%uintptr(pageAlign)) % uintptr(pageAlign))
	for i := 0; i < n; i++ {
		buf[off+i] = byte('a' + i%26)
	}
	return buf[off : off+n]
}

type stubPager struct {
	called     bool
	aligned    bool
	admitPages int
	admitTok   int
}

func (s *stubPager) AdmitRaw(raw []byte) (uintptr, []int, int, bool, error) {
	s.called = true
	if !s.aligned {
		return 0, []int{1, 2}, s.admitPages, false, nil
	}
	return 0xdead, []int{1, 2}, s.admitPages, true, nil
}

func TestAdmitFallbackOnEmptyPayload(t *testing.T) {
	z := NewZeroCopyAdmission(nil, &stubPager{aligned: true}, 0)
	rec, err := z.Admit(ToolObservation{Name: "x", Bytes: nil})
	if err != nil {
		t.Fatalf("Admit returned error: %v", err)
	}
	if rec.ZeroCopy || !rec.Fallback || rec.FallbackReason != FallbackReasonEmptyPayload {
		t.Fatalf("want fallback empty_payload, got %+v", rec)
	}
	if rec.Tokens != 0 {
		t.Fatalf("want 0 tokens for empty, got %d", rec.Tokens)
	}
}

func TestAdmitFallbackUnsupportedMIME(t *testing.T) {
	z := NewZeroCopyAdmission(nil, &stubPager{aligned: true}, 4096)
	rec, err := z.Admit(ToolObservation{Name: "x", Bytes: alignedBytes(32, 4096), MIME: "application/gzip"})
	if err != nil {
		t.Fatalf("Admit returned error: %v", err)
	}
	if rec.FallbackReason != FallbackReasonUnsupportedEncoding || rec.ZeroCopy {
		t.Fatalf("want unsupported_encoding fallback, got %+v", rec)
	}
}

func TestAdmitFallbackNonPageAligned(t *testing.T) {
	z := NewZeroCopyAdmission(nil, &stubPager{aligned: false}, 4096)
	raw := []byte("not aligned")
	rec, err := z.Admit(ToolObservation{Name: "x", Bytes: raw})
	if err != nil {
		t.Fatalf("Admit returned error: %v", err)
	}
	if rec.FallbackReason != FallbackReasonNonPageAligned || rec.ZeroCopy || !rec.Fallback {
		t.Fatalf("want non_page_aligned fallback, got %+v", rec)
	}
	if rec.Tokens != len(fallbackTokens(raw)) {
		t.Fatalf("token count mismatch: %d", rec.Tokens)
	}
}

func TestAdmitZeroCopyFastPath(t *testing.T) {
	sp := &stubPager{aligned: true, admitPages: 3}
	z := NewZeroCopyAdmission(nil, sp, 4096)
	raw := alignedBytes(4096, 4096)
	rec, err := z.Admit(ToolObservation{Name: "x", Bytes: raw})
	if err != nil {
		t.Fatalf("Admit returned error: %v", err)
	}
	if !sp.called {
		t.Fatalf("pager was not called on fast path")
	}
	if !rec.Admitted || !rec.ZeroCopy || rec.Fallback {
		t.Fatalf("want zero-copy admit, got %+v", rec)
	}
	if rec.NodeRef != 0xdead || rec.Pages != 3 || rec.Tokens != 2 {
		t.Fatalf("unexpected receipt: %+v", rec)
	}
	if len(rec.Digest) != 64 {
		t.Fatalf("digest not sha256 hex: %q", rec.Digest)
	}
}

func TestAdmitFastPathWithoutPagerFallsBack(t *testing.T) {
	z := NewZeroCopyAdmission(nil, nil, 4096)
	rec, _ := z.Admit(ToolObservation{Bytes: alignedBytes(4096, 4096)})
	if !rec.Fallback || rec.ZeroCopy {
		t.Fatalf("nil pager must fall back, got %+v", rec)
	}
}

func TestAdmitCustomFallbackWins(t *testing.T) {
	z := NewZeroCopyAdmission(nil, nil, 4096)
	z.SetFallback(func(b []byte) []int { return []int{7, 8, 9} })
	rec, _ := z.Admit(ToolObservation{Bytes: alignedBytes(4096, 4096)})
	if rec.Tokens != 3 {
		t.Fatalf("custom fallback not used: %+v", rec)
	}
}

func TestAdmitNilGateNeverPanics(t *testing.T) {
	var z *ZeroCopyAdmission
	rec, err := z.Admit(ToolObservation{Bytes: []byte("abc")})
	if err != nil || !rec.Fallback {
		t.Fatalf("nil gate should degrade to fallback, got %+v err=%v", rec, err)
	}
}

func TestUnsupportedMIMENormalizes(t *testing.T) {
	if !unsupportedMIME("Application/GZIP; charset=binary") {
		t.Fatalf("expected case-insensitive + parameter-stripped match")
	}
	if unsupportedMIME("text/plain") {
		t.Fatalf("text/plain must be supported")
	}
}

func TestDigestHexMatchesKnownVector(t *testing.T) {
	got := digestHex([]byte("abc"))
	const want = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	if !strings.EqualFold(got, want) {
		t.Fatalf("digest mismatch: %s", got)
	}
}
