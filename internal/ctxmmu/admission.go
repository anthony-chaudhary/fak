package ctxmmu

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"strings"
	"unsafe"
)

// Admission fallback reasons form a small closed set: a caller can switch on
// them without parsing free text, and each names exactly one failed fast-path
// precondition.
const (
	// FallbackReasonEmptyPayload is set when the observation carried no bytes.
	FallbackReasonEmptyPayload = "empty_payload"
	// FallbackReasonNonPageAligned is set when the payload did not start on the
	// configured page boundary.
	FallbackReasonNonPageAligned = "non_page_aligned"
	// FallbackReasonUnsupportedEncoding is set when the observation declared a
	// MIME type the zero-copy path does not admit.
	FallbackReasonUnsupportedEncoding = "unsupported_encoding"
)

// unsupportedMIMEs is the deterministic set of encodings the zero-copy bind
// refuses. They are container/transport encodings whose bytes are not the raw
// payload a KV page should cache; invalid UTF-8 is explicitly NOT in this set -
// arbitrary binary tool output is admitted because the pager only folds bytes,
// it never interprets them.
var unsupportedMIMEs = map[string]bool{
	"application/gzip":    true,
	"application/x-gzip":  true,
	"application/zip":     true,
	"application/x-tar":   true,
	"application/x-bzip2": true,
	"application/x-xz":    true,
	"application/zstd":    true,
}

// DefaultAdmissionAlign is the page granularity the admission gate assumes when
// a caller does not name one.
const DefaultAdmissionAlign = 4096

// ObservationPager is the narrow seam the admission gate binds through. It is
// satisfied by radixpager.RadixKVPager; defining it here (rather than importing
// internal/radixkv, which already imports this package) breaks the would-be
// import cycle while keeping the gate testable with a stub.
type ObservationPager interface {
	AdmitRaw(raw []byte) (nodeRef uintptr, tokens []int, pages int, zeroCopy bool, err error)
}

// ToolObservation is a raw tool observation (file read / grep / diff output)
// whose Bytes already live in host or unified memory. Admit binds them without a
// string serialization on the fast path. (The name is ToolObservation rather
// than Observation because disposition.go already owns Observation for the
// #1598 disposition-minting vocabulary.)
type ToolObservation struct {
	Name  string
	Bytes []byte
	MIME  string
}

// AdmissionReceipt describes the outcome of admitting one observation.
// ZeroCopy is true iff the observation was paged without a copy; Fallback is
// true with FallbackReason set iff the fallback path ran.
type AdmissionReceipt struct {
	Admitted       bool
	ZeroCopy       bool
	Fallback       bool
	FallbackReason string
	Bytes          int
	Tokens         int
	Pages          int
	Alignment      int
	Digest         string
	NodeRef        uintptr
}

// ZeroCopyAdmission is the write-time admission gate for tool observations. Its
// fast path binds an aligned, supported payload through the pager with no copy;
// when a precondition fails it tokenizes through fallback (or a self-contained
// byte->token fold) so it always returns a usable token sequence.
type ZeroCopyAdmission struct {
	mmu      *MMU
	pager    ObservationPager
	align    int
	fallback func([]byte) []int
}

// NewZeroCopyAdmission builds an admission gate over an MMU and a pager. A
// non-positive pageAlign defaults to 4096. Either argument may be nil; the gate
// degrades to the fallback path rather than panicking.
func NewZeroCopyAdmission(m *MMU, pager ObservationPager, pageAlign int) *ZeroCopyAdmission {
	if pageAlign <= 0 {
		pageAlign = DefaultAdmissionAlign
	}
	return &ZeroCopyAdmission{mmu: m, pager: pager, align: pageAlign}
}

// SetFallback installs a standard-tokenizer fallback invoked when the fast path
// cannot run. A nil fallback selects the package's deterministic byte->token
// fold instead.
func (z *ZeroCopyAdmission) SetFallback(f func([]byte) []int) {
	if z != nil {
		z.fallback = f
	}
}

// Pager returns the pager backing this gate.
func (z *ZeroCopyAdmission) Pager() ObservationPager {
	if z == nil {
		return nil
	}
	return z.pager
}

// fallbackTokens is the self-contained deterministic byte->token mapping used
// when no fallback tokenizer is installed: each 4-byte little-endian word becomes
// one token id, with a trailing partial word zero-padded. It never panics and
// always returns a usable (possibly empty-for-empty-input) sequence.
func fallbackTokens(b []byte) []int {
	if len(b) == 0 {
		return nil
	}
	n := (len(b) + 3) / 4
	out := make([]int, 0, n)
	var word [4]byte
	for i := 0; i < len(b); i += 4 {
		word = [4]byte{}
		copy(word[:], b[i:min(i+4, len(b))])
		out = append(out, int(binary.LittleEndian.Uint32(word[:])))
	}
	return out
}

// Admit attempts the zero-copy bind and falls back on any failed precondition.
// It never panics and never returns an error for a fallback (fallback is a
// successful degradation, not a failure).
func (z *ZeroCopyAdmission) Admit(obs ToolObservation) (AdmissionReceipt, error) {
	if z == nil {
		return AdmissionReceipt{Fallback: true, FallbackReason: FallbackReasonEmptyPayload, Bytes: len(obs.Bytes), Tokens: len(fallbackTokens(obs.Bytes))}, nil
	}
	rec := AdmissionReceipt{Bytes: len(obs.Bytes), Alignment: z.align, Digest: digestHex(obs.Bytes)}

	reason := ""
	switch {
	case len(obs.Bytes) == 0:
		reason = FallbackReasonEmptyPayload
	case unsupportedMIME(obs.MIME):
		reason = FallbackReasonUnsupportedEncoding
	case !aligned(obs.Bytes, z.align):
		reason = FallbackReasonNonPageAligned
	}

	if reason == "" && z.pager != nil {
		nodeRef, tokens, pages, zeroCopy, err := z.pager.AdmitRaw(obs.Bytes)
		if err == nil && zeroCopy {
			rec.Admitted = true
			rec.ZeroCopy = true
			rec.Tokens = len(tokens)
			rec.Pages = pages
			rec.NodeRef = nodeRef
			return rec, nil
		}
		if err != nil {
			reason = FallbackReasonUnsupportedEncoding
		} else if !zeroCopy {
			reason = FallbackReasonNonPageAligned
		}
	}
	if reason == "" {
		reason = FallbackReasonUnsupportedEncoding
	}

	tokens := z.tokenizeFallback(obs.Bytes)
	rec.Fallback = true
	rec.FallbackReason = reason
	rec.ZeroCopy = false
	rec.Tokens = len(tokens)
	rec.Pages = (len(obs.Bytes) + z.align - 1) / z.align
	return rec, nil
}

// tokenizeFallback runs the installed fallback tokenizer, or the package's
// deterministic byte->token fold when none is set.
func (z *ZeroCopyAdmission) tokenizeFallback(b []byte) []int {
	if z.fallback != nil {
		if toks := z.fallback(b); toks != nil {
			return toks
		}
	}
	return fallbackTokens(b)
}

// unsupportedMIME reports whether mime names an encoding the zero-copy path
// refuses. Matching is case-insensitive and ignores any parameters after ';'.
func unsupportedMIME(mime string) bool {
	if mime == "" {
		return false
	}
	if i := strings.IndexByte(mime, ';'); i >= 0 {
		mime = mime[:i]
	}
	return unsupportedMIMEs[strings.ToLower(strings.TrimSpace(mime))]
}

// aligned reports whether b is non-empty and starts on an align-byte boundary.
func aligned(b []byte, align int) bool {
	if len(b) == 0 {
		return false
	}
	if align <= 0 {
		align = DefaultAdmissionAlign
	}
	return (uintptr(unsafe.Pointer(&b[0])) % uintptr(align)) == 0
}

// digestHex returns the lowercase hex sha256 of b, or "" for empty input.
func digestHex(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
