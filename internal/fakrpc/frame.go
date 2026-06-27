package fakrpc

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// FAKRES frame sentinels. The grammar (one header line, a raw body, one footer line) is
// the same one tools/dgx_witness_run.sh already emits on the GPU boxes, so this codec is
// wire-compatible with a frame the fleet is already producing.
const (
	frameHeaderPrefix = "<<<FAKRES "
	frameFooterPrefix = "<<<ENDFAKRES "
	frameSentinelEnd  = ">>>"
)

// Frame is a decoded FAKRES result: the worker's return code, the verified body bytes,
// and the nonce that ties it to its Request.
type Frame struct {
	// Nonce ties the frame to its Request (and is asserted identical in both sentinels).
	Nonce string
	// RC is the worker's exit code: 0 is success, non-zero a typed terminal failure.
	RC int
	// Body is the raw, length- and sha-verified result payload.
	Body []byte
}

// OK reports whether the worker reported success (rc == 0).
func (f Frame) OK() bool { return f.RC == 0 }

// Frame errors. They are sentinels so a caller can branch with errors.Is; every one of
// them means "do not trust this transfer".
var (
	// ErrNoHeader is returned when no FAKRES header sentinel is present at all.
	ErrNoHeader = errors.New("fakrpc: missing FAKRES header sentinel")
	// ErrBadHeader is returned when the header sentinel is present but malformed (missing
	// a field, a non-integer rc/len, or no closing >>>).
	ErrBadHeader = errors.New("fakrpc: malformed FAKRES header")
	// ErrNoFooter is returned when the body is full-length but the ENDFAKRES footer is
	// absent or misplaced where the framed len ends (a transfer truncated at or after the
	// body). A body SHORTER than the framed len is caught earlier as ErrLenMismatch.
	ErrNoFooter = errors.New("fakrpc: missing ENDFAKRES footer where the framed len ends")
	// ErrLenMismatch is returned when fewer body bytes are present than the framed len —
	// a split or truncated transfer.
	ErrLenMismatch = errors.New("fakrpc: body shorter than the framed len (truncated transfer)")
	// ErrShaMismatch is returned when the body's sha256 does not match the framed sha — a
	// corrupt transfer, even if the length lined up.
	ErrShaMismatch = errors.New("fakrpc: body sha256 does not match the framed sha (corrupt transfer)")
	// ErrNonceMismatch is returned when the header and footer nonces disagree — a spliced
	// or interleaved transfer.
	ErrNonceMismatch = errors.New("fakrpc: header and footer nonces disagree")
)

// EncodeFrame wraps body in a FAKRES frame: a header carrying the nonce, return code,
// body sha256, and body length; the raw body; and a footer repeating the nonce. The
// output is byte-identical to the deployed shell emitter, so a Go worker and a shell
// witness runner produce the same wire. The body is carried raw (not base64): DecodeFrame
// reads it by length, so the len+sha already give truncation/corruption defense without
// the 33% base64 bloat.
//
// The nonce must be a frame-safe token (see ValidNonce); a nonce with whitespace would
// split the header field and yield a frame DecodeFrame cannot parse. Callers that did not
// already gate on Request.Validate should call ValidNonce first.
func EncodeFrame(nonce string, rc int, body []byte) []byte {
	sum := sha256.Sum256(body)
	var b bytes.Buffer
	fmt.Fprintf(&b, "%snonce=%s rc=%d sha=%s len=%d%s\n",
		frameHeaderPrefix, nonce, rc, hex.EncodeToString(sum[:]), len(body), frameSentinelEnd)
	b.Write(body)
	b.WriteByte('\n')
	fmt.Fprintf(&b, "%snonce=%s%s\n", frameFooterPrefix, nonce, frameSentinelEnd)
	return b.Bytes()
}

// DecodeFrame parses and VERIFIES a FAKRES frame out of data, returning the body only if
// every integrity check passes: the body is exactly the framed length, its sha256 matches
// the framed sha, the footer is present, and both sentinels carry the same nonce. Any
// failure returns a typed error and a zero Frame — a truncated or corrupt transfer is
// rejected, never trusted.
//
// data may carry leading noise before the header (stale transcript echoes under load,
// invariant 3): DecodeFrame scans to the first header sentinel and reads the body by the
// framed length, so an embedded newline or even an embedded sentinel string in the body
// does not confuse it.
func DecodeFrame(data []byte) (Frame, error) {
	start := bytes.Index(data, []byte(frameHeaderPrefix))
	if start < 0 {
		return Frame{}, ErrNoHeader
	}
	nlOff := bytes.IndexByte(data[start:], '\n')
	if nlOff < 0 {
		return Frame{}, ErrBadHeader
	}
	headerLine := strings.TrimRight(string(data[start:start+nlOff]), "\r")
	nonce, rc, sha, length, err := parseFrameHeader(headerLine)
	if err != nil {
		return Frame{}, err
	}

	bodyStart := start + nlOff + 1
	// Compare against the remaining byte count rather than bodyStart+length: a hostile
	// header can set len to near max-int, and bodyStart+length would overflow to a
	// negative value that slips past a naive bound and then panics the slice.
	if length > len(data)-bodyStart {
		return Frame{}, ErrLenMismatch
	}
	body := data[bodyStart : bodyStart+length]

	sum := sha256.Sum256(body)
	if !strings.EqualFold(hex.EncodeToString(sum[:]), sha) {
		return Frame{}, ErrShaMismatch
	}

	footerNonce, err := parseFrameFooter(string(data[bodyStart+length:]))
	if err != nil {
		return Frame{}, err
	}
	if footerNonce != nonce {
		return Frame{}, ErrNonceMismatch
	}
	return Frame{Nonce: nonce, RC: rc, Body: body}, nil
}

// parseFrameHeader pulls the nonce/rc/sha/len fields out of a header line of the form
// "<<<FAKRES nonce=N rc=R sha=H len=L>>>". It is strict: all four fields are required and
// rc/len must be integers, so a mangled header is rejected rather than half-trusted.
func parseFrameHeader(line string) (nonce string, rc int, sha string, length int, err error) {
	inner, ok := strings.CutPrefix(line, frameHeaderPrefix)
	if !ok {
		return "", 0, "", 0, ErrNoHeader
	}
	inner, ok = strings.CutSuffix(inner, frameSentinelEnd)
	if !ok {
		return "", 0, "", 0, ErrBadHeader
	}
	fields := map[string]string{}
	for _, tok := range strings.Fields(inner) {
		k, v, ok := strings.Cut(tok, "=")
		if !ok {
			return "", 0, "", 0, fmt.Errorf("%w: token %q is not key=value", ErrBadHeader, tok)
		}
		fields[k] = v
	}
	nonce, sha = fields["nonce"], fields["sha"]
	if nonce == "" || sha == "" || fields["rc"] == "" || fields["len"] == "" {
		return "", 0, "", 0, fmt.Errorf("%w: missing a required field in %q", ErrBadHeader, line)
	}
	if rc, err = strconv.Atoi(fields["rc"]); err != nil {
		return "", 0, "", 0, fmt.Errorf("%w: rc is not an integer: %v", ErrBadHeader, err)
	}
	if length, err = strconv.Atoi(fields["len"]); err != nil {
		return "", 0, "", 0, fmt.Errorf("%w: len is not an integer: %v", ErrBadHeader, err)
	}
	if length < 0 {
		return "", 0, "", 0, fmt.Errorf("%w: len is negative", ErrBadHeader)
	}
	return nonce, rc, sha, length, nil
}

// parseFrameFooter verifies that rest begins with the single framing newline separator
// then the footer sentinel "<<<ENDFAKRES nonce=N>>>", and returns N. If the footer is not
// exactly where the framed length put it, the body length was wrong (cut short or
// over-claimed) — reported as ErrNoFooter.
//
// It parses the footer line the SAME way parseFrameHeader parses the header — strip the
// prefix, then strip the LAST sentinel — so a nonce that itself contains the sentinel
// string ">>>" delimits identically in both places and round-trips rather than being
// truncated at the first ">>>".
func parseFrameFooter(rest string) (string, error) {
	rest = strings.TrimPrefix(rest, "\r")
	if !strings.HasPrefix(rest, "\n") {
		return "", ErrNoFooter
	}
	line := rest[1:]
	if nl := strings.IndexByte(line, '\n'); nl >= 0 {
		line = line[:nl]
	}
	line = strings.TrimRight(line, "\r")
	inner, ok := strings.CutPrefix(line, frameFooterPrefix)
	if !ok {
		return "", ErrNoFooter
	}
	inner, ok = strings.CutSuffix(inner, frameSentinelEnd)
	if !ok {
		return "", ErrNoFooter
	}
	nonce, ok := strings.CutPrefix(inner, "nonce=")
	if !ok || nonce == "" {
		return "", fmt.Errorf("%w: footer has no nonce", ErrNoFooter)
	}
	return nonce, nil
}
