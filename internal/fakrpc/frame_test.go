package fakrpc

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"testing"
)

// shellFrame builds a FAKRES frame the exact way tools/dgx_witness_run.sh does, computing
// the sha independently of the package under test. It is the ground truth for both the
// "EncodeFrame matches the deployed grammar" check and the "DecodeFrame reads a
// foreign-produced frame" check.
func shellFrame(nonce string, rc int, body []byte) string {
	sum := sha256.Sum256(body)
	return fmt.Sprintf("<<<FAKRES nonce=%s rc=%d sha=%s len=%d>>>\n", nonce, rc, hex.EncodeToString(sum[:]), len(body)) +
		string(body) + "\n" +
		fmt.Sprintf("<<<ENDFAKRES nonce=%s>>>\n", nonce)
}

func TestEncodeFrameMatchesDeployedGrammar(t *testing.T) {
	body := []byte(`{"schema":"glm-gpu-witness/1","ok":true}`)
	got := EncodeFrame("bell0042", 0, body)
	want := shellFrame("bell0042", 0, body)
	if string(got) != want {
		t.Fatalf("EncodeFrame is not byte-identical to the deployed shell emitter:\n got=%q\nwant=%q", got, want)
	}
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	body := []byte("hello, disaggregated world")
	frame := EncodeFrame("n7", 3, body)
	got, err := DecodeFrame(frame)
	if err != nil {
		t.Fatalf("DecodeFrame: %v", err)
	}
	if got.Nonce != "n7" || got.RC != 3 || !bytes.Equal(got.Body, body) {
		t.Fatalf("round-trip mismatch: %+v", got)
	}
	if got.OK() {
		t.Fatal("OK() should be false for a non-zero rc")
	}
}

func TestDecodeReadsForeignFrame(t *testing.T) {
	// A frame produced by the shell emitter (not by EncodeFrame) must decode and verify.
	body := []byte(`{"compact":"json","n":1}`)
	got, err := DecodeFrame([]byte(shellFrame("glmw9", 0, body)))
	if err != nil {
		t.Fatalf("DecodeFrame of a shell-produced frame: %v", err)
	}
	if !got.OK() || got.Nonce != "glmw9" || !bytes.Equal(got.Body, body) {
		t.Fatalf("foreign-frame decode mismatch: %+v", got)
	}
}

func TestDecodeBodyWithEmbeddedSentinelAndNewlines(t *testing.T) {
	// Read-by-length must survive a body that itself contains newlines and even the
	// footer sentinel string — the len, not a delimiter scan, bounds the body.
	body := []byte("line1\nline2\n<<<ENDFAKRES nonce=decoy>>>\ntail")
	frame := EncodeFrame("real", 0, body)
	got, err := DecodeFrame(frame)
	if err != nil {
		t.Fatalf("DecodeFrame with embedded sentinel: %v", err)
	}
	if got.Nonce != "real" || !bytes.Equal(got.Body, body) {
		t.Fatalf("embedded-sentinel body corrupted: nonce=%q bodyEqual=%v", got.Nonce, bytes.Equal(got.Body, body))
	}
}

func TestDecodeNonceContainingSentinel(t *testing.T) {
	// EncodeFrame is the low-level primitive below Request.Validate; if a caller hands it a
	// nonce containing the sentinel ">>>", the header (strips the LAST ">>>") and the footer
	// (parsed the same way) must still agree, so the frame round-trips rather than failing
	// ErrNonceMismatch. (Request.Validate rejects such a nonce up front; this pins the
	// primitive's symmetry independently.)
	for _, n := range []string{"a>>>b", "ends>>>", ">>>starts"} {
		got, err := DecodeFrame(EncodeFrame(n, 0, []byte("hi")))
		if err != nil {
			t.Fatalf("DecodeFrame for sentinel nonce %q: %v", n, err)
		}
		if got.Nonce != n {
			t.Fatalf("sentinel-nonce round-trip mismatch: got %q want %q", got.Nonce, n)
		}
	}
}

func TestDecodeSkipsLeadingNoise(t *testing.T) {
	// Under load the transcript floods with stale echoes (invariant 3); the real frame may
	// be preceded by noise. DecodeFrame scans to the first header sentinel.
	body := []byte("payload")
	noisy := append([]byte("stale echo line\nanother throttled line\n"), EncodeFrame("n2", 0, body)...)
	got, err := DecodeFrame(noisy)
	if err != nil {
		t.Fatalf("DecodeFrame with leading noise: %v", err)
	}
	if !bytes.Equal(got.Body, body) {
		t.Fatalf("body mismatch after skipping noise: %q", got.Body)
	}
}

func TestDecodeRejectsTruncatedBody(t *testing.T) {
	frame := EncodeFrame("n3", 0, []byte("0123456789"))
	// Chop the last 5 bytes: the declared len no longer fits in the data.
	truncated := frame[:len(frame)-5]
	if _, err := DecodeFrame(truncated); !errors.Is(err, ErrLenMismatch) && !errors.Is(err, ErrNoFooter) {
		t.Fatalf("truncated frame = %v, want ErrLenMismatch or ErrNoFooter", err)
	}
}

func TestDecodeRejectsCorruptBody(t *testing.T) {
	body := []byte("the quick brown fox")
	frame := EncodeFrame("n4", 0, body)
	// Flip one body byte in place — len still matches, sha no longer does.
	bodyAt := bytes.IndexByte(frame, '\n') + 1
	frame[bodyAt] ^= 0xFF
	if _, err := DecodeFrame(frame); !errors.Is(err, ErrShaMismatch) {
		t.Fatalf("corrupt body = %v, want ErrShaMismatch", err)
	}
}

func TestDecodeRejectsNonceMismatch(t *testing.T) {
	body := []byte("x")
	sum := sha256.Sum256(body)
	// Header says nonce=a, footer says nonce=b — a spliced transfer.
	spliced := fmt.Sprintf("<<<FAKRES nonce=a rc=0 sha=%s len=%d>>>\n", hex.EncodeToString(sum[:]), len(body)) +
		string(body) + "\n<<<ENDFAKRES nonce=b>>>\n"
	if _, err := DecodeFrame([]byte(spliced)); !errors.Is(err, ErrNonceMismatch) {
		t.Fatalf("nonce-mismatch frame = %v, want ErrNonceMismatch", err)
	}
}

func TestDecodeRejectsMissingHeader(t *testing.T) {
	if _, err := DecodeFrame([]byte("just some text, no frame here\n")); !errors.Is(err, ErrNoHeader) {
		t.Fatalf("headerless input = %v, want ErrNoHeader", err)
	}
}

func TestDecodeRejectsMissingFooter(t *testing.T) {
	body := []byte("body")
	sum := sha256.Sum256(body)
	// Correct header + body, but no footer line at all.
	noFooter := fmt.Sprintf("<<<FAKRES nonce=n rc=0 sha=%s len=%d>>>\n", hex.EncodeToString(sum[:]), len(body)) + string(body) + "\n"
	if _, err := DecodeFrame([]byte(noFooter)); !errors.Is(err, ErrNoFooter) {
		t.Fatalf("footerless frame = %v, want ErrNoFooter", err)
	}
}

func TestDecodeRejectsMalformedHeader(t *testing.T) {
	cases := []string{
		"<<<FAKRES nonce=n rc=zero sha=ab len=1>>>\nx\n<<<ENDFAKRES nonce=n>>>\n",   // rc not an int
		"<<<FAKRES nonce=n rc=0 sha=ab len=notnum>>>\nx\n<<<ENDFAKRES nonce=n>>>\n", // len not an int
		"<<<FAKRES nonce=n rc=0 len=1>>>\nx\n<<<ENDFAKRES nonce=n>>>\n",             // missing sha
		"<<<FAKRES nonce=n rc=0 sha=ab len=1\nx\n<<<ENDFAKRES nonce=n>>>\n",         // no closing >>>
	}
	for i, c := range cases {
		if _, err := DecodeFrame([]byte(c)); !errors.Is(err, ErrBadHeader) {
			t.Fatalf("case %d malformed header = %v, want ErrBadHeader", i, err)
		}
	}
}

func TestDecodeToleratesCRLF(t *testing.T) {
	// A text-only bridge / Windows control plane may rewrite the framing line endings to
	// CRLF. The body is sha-protected so it must be exact, but the header/footer framing
	// lines tolerate a trailing \r.
	body := []byte(`{"ok":true}`)
	sum := sha256.Sum256(body)
	crlf := fmt.Sprintf("<<<FAKRES nonce=c1 rc=0 sha=%s len=%d>>>\r\n", hex.EncodeToString(sum[:]), len(body)) +
		string(body) + "\r\n<<<ENDFAKRES nonce=c1>>>\r\n"
	got, err := DecodeFrame([]byte(crlf))
	if err != nil {
		t.Fatalf("DecodeFrame of CRLF frame: %v", err)
	}
	if got.Nonce != "c1" || !bytes.Equal(got.Body, body) {
		t.Fatalf("CRLF decode mismatch: %+v", got)
	}
}

func TestDecodeRejectsOverflowLen(t *testing.T) {
	// A hostile header with a near-max len must not overflow the bound check into a
	// negative slice index and panic — it must be rejected as a truncated transfer.
	for _, n := range []string{"9223372036854775807", "9999999999999999999"} {
		in := "<<<FAKRES nonce=n rc=0 sha=ab len=" + n + ">>>\nx\n<<<ENDFAKRES nonce=n>>>\n"
		_, err := DecodeFrame([]byte(in))
		if err == nil {
			t.Fatalf("len=%s should be rejected, got nil error", n)
		}
	}
}

func TestDecodeEmptyBody(t *testing.T) {
	frame := EncodeFrame("z", 0, nil)
	got, err := DecodeFrame(frame)
	if err != nil {
		t.Fatalf("DecodeFrame of empty body: %v", err)
	}
	if len(got.Body) != 0 || !got.OK() {
		t.Fatalf("empty-body decode mismatch: %+v", got)
	}
}

func TestEncodeFrameNoEmbeddedHeaderConfusesScan(t *testing.T) {
	// Sanity: the footer prefix must not be a false positive for the header scan.
	if strings.Contains(frameFooterPrefix, frameHeaderPrefix) {
		t.Fatal("footer prefix contains header prefix — DecodeFrame's scan could mis-anchor")
	}
}
