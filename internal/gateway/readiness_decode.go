package gateway

import (
	"strings"
	"sync"
	"unicode"
)

// readiness_decode.go is the #4247 output-coherence gate on the local-model
// readiness surface. #3186 fixed a CUDA cache-alias defect that made a
// device-served model decode a single repeated token ("!!!!") while every
// liveness signal stayed green: the process was up, the HTTP listener was bound,
// /healthz reported ok:true — and the model was emitting garbage. Liveness is not
// coherence. This file lets the host run a deterministic fixed-prompt decode at
// boot and, when the result is a degenerate shape, qualify /healthz ok:false so a
// watchdog does not route real work to a coherent-looking-but-broken serve.
//
// The policy lives HERE, in the gateway, deliberately apart from internal/compute
// (the issue's explicit constraint): a backend arithmetic/cache fix must not be
// able to silently absorb — or silently weaken — the gateway's readiness gate.

// degenerateKind names why a startup decode was rejected, or "" (decodeCoherent)
// when the output is benign. It is surfaced verbatim as the /healthz
// degenerate_decode.kind so an operator sees which shape tripped the gate.
type degenerateKind string

const (
	decodeCoherent      degenerateKind = ""
	decodeEmpty         degenerateKind = "empty"
	decodePunctuation   degenerateKind = "punctuation_only"
	decodeRepeatedToken degenerateKind = "repeated_single_token"

	// repeatThreshold is the minimum unit count before an all-identical decode is
	// judged degenerate. It keeps legitimately short benign output ("OK", "Hi",
	// "42", "hmm") ready while rejecting the "the the the the" / "aaaaaaaa" shapes
	// the #3186 failure produced. 4 is comfortably below any real fixed-prompt
	// answer and above every benign short reply.
	repeatThreshold = 4
)

// classifyDecode judges the decoded text of the deterministic startup probe. It
// returns decodeCoherent for benign output — including short small-model output —
// and a specific degenerate kind for the three rejected shapes the issue names:
// empty, punctuation-only, and a single token/rune repeated. Pure function of the
// text, so it is exhaustively table-testable without a live model.
func classifyDecode(text string) degenerateKind {
	trimmed := strings.TrimSpace(text)
	switch {
	case trimmed == "":
		return decodeEmpty
	case punctuationOnly(trimmed):
		return decodePunctuation
	case repeatedSingleToken(trimmed):
		return decodeRepeatedToken
	default:
		return decodeCoherent
	}
}

// punctuationOnly reports whether text carries no letter and no digit — i.e. it
// is entirely punctuation, symbols, and spaces. The #3186 "!" decode is the
// canonical case; "...", "?!?!", and "   ---   " are the same failure shape.
func punctuationOnly(text string) bool {
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

// repeatedSingleToken reports whether text is one unit repeated: either
// whitespace-separated tokens that are all identical ("the the the the"), or a
// single rune repeated with no whitespace ("aaaaaaaa"). Both require at least
// repeatThreshold units so benign short output is never rejected. Punctuation-
// only repeats ("!!!!") are already caught by punctuationOnly upstream.
func repeatedSingleToken(text string) bool {
	if fields := strings.Fields(text); len(fields) >= repeatThreshold {
		if allSameString(fields) {
			return true
		}
	}
	if runes := []rune(text); len(runes) >= repeatThreshold {
		if allSameRune(runes) {
			return true
		}
	}
	return false
}

func allSameString(xs []string) bool {
	for _, x := range xs[1:] {
		if x != xs[0] {
			return false
		}
	}
	return true
}

func allSameRune(rs []rune) bool {
	for _, r := range rs[1:] {
		if r != rs[0] {
			return false
		}
	}
	return true
}

// startupDecodeProbe is the gateway's record of the boot-time fixed-prompt decode
// verdict. Zero value == not probed (a proxy/mock serve runs no local decode, so
// the gate stays silent and /healthz is unaffected). Guarded by its own mutex so
// the health read never contends with the one-time boot set. Mirrors
// servedFailure's self-contained-verdict shape (served_failure.go).
type startupDecodeProbe struct {
	mu     sync.Mutex
	probed bool
	kind   degenerateKind
	sample string // bounded snapshot of the decoded output, for the /healthz reason
}

// set records the verdict for a decoded startup-probe output. A coherent decode
// is still recorded (probed=true, kind="") so a reader can tell "probed and
// clean" from "never probed".
func (p *startupDecodeProbe) set(text string) {
	kind := classifyDecode(text)
	p.mu.Lock()
	defer p.mu.Unlock()
	p.probed, p.kind, p.sample = true, kind, boundedSample(text)
}

// degenerate returns the rejecting kind and the sampled output when the boot
// probe found a degenerate decode; ok=false when the probe was never run or the
// decode was benign.
func (p *startupDecodeProbe) degenerate() (kind degenerateKind, sample string, ok bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.probed || p.kind == decodeCoherent {
		return decodeCoherent, "", false
	}
	return p.kind, p.sample, true
}

// boundedSample trims a decoded output to a short, single-line snippet safe to
// surface on /healthz: the probe prompt is fixed and code-authored (never client
// payload), so echoing a bounded prefix is safe and makes the failure legible.
func boundedSample(text string) string {
	const max = 80
	s := strings.ReplaceAll(strings.TrimSpace(text), "\n", " ")
	if len(s) > max {
		return s[:max]
	}
	return s
}

// SetStartupDecodeProbe records the decoded output of the host's deterministic
// fixed-prompt readiness probe against the local (in-kernel) model at boot. The
// host owns the model turn (it runs the decode); the gateway owns the coherence
// POLICY (classifyDecode) and the readiness surface (/healthz). A degenerate
// verdict qualifies /healthz ok:false until the process restarts and re-probes.
// Safe on a nil Server and for concurrent use. Passing benign output — or never
// calling this at all, as a proxy/mock serve does — leaves readiness unaffected.
func (s *Server) SetStartupDecodeProbe(text string) {
	if s == nil {
		return
	}
	s.startupDecode.set(text)
}
