package fakrpc

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode"
)

// Kind is the operation a Request asks a resident worker to run. The set is closed: a
// worker rejects an unknown kind rather than guessing what to do with it.
type Kind string

const (
	// KindTurn is a raw model call against the resident serve — the worker forwards the
	// payload to its local `fak serve` and frames the completion.
	KindTurn Kind = "turn"

	// KindDevTurn is a KERNEL-ADJUDICATED dev turn: the worker shells
	// `fak guard --remote-serve <serve> -- <agent> -p '<task>'`, so the agent runs on the
	// box, every tool call crosses the capability floor locally, and inference rides the
	// resident serve. The framed result carries the guard exit summary (allowed/denied +
	// journal pointer) — it IS the auditable witness. This is what makes the mechanism a
	// disaggregated kernel, not a remote curl. (Depends on `fak guard --remote-serve`.)
	KindDevTurn Kind = "dev-turn"
)

// knownKinds is the closed vocabulary Validate enforces; keep it in sync with the Kind
// constants above.
var knownKinds = map[Kind]bool{KindTurn: true, KindDevTurn: true}

// Request is the one-line envelope a caller drops into a worker's spool. It is
// transport-neutral: the bridge (a chat channel, a queue, any one-line-at-a-time pipe)
// carries only a pointer to it; the worker reads the spooled request, runs the kind
// against its resident serve, and writes a FAKRES-framed result (see EncodeFrame).
type Request struct {
	// Nonce uniquely identifies this request across the spool. It defeats stale-tail
	// replay (invariant 3) and is the idempotency key (done/<nonce> ⇒ skip, invariant 4),
	// so it must be unique per request and a frame-safe token (see ValidNonce).
	Nonce string `json:"nonce"`

	// Kind selects the operation; see the Kind constants.
	Kind Kind `json:"kind"`

	// Model is the upstream model id to run the turn against. Empty means the worker's
	// configured default.
	Model string `json:"model,omitempty"`

	// Payload is the kind-specific body: the raw prompt/messages for KindTurn, or the
	// agent task string for KindDevTurn.
	Payload string `json:"payload"`

	// Params carries optional, kind-specific knobs (e.g. {"max_tokens":"512"}). It is a
	// flat string map so it survives a text-only bridge without numeric/type ambiguity.
	Params map[string]string `json:"params,omitempty"`
}

// Envelope errors. They are sentinels so a caller can branch with errors.Is.
var (
	// ErrEmptyNonce is returned when a nonce is empty or all-whitespace — there would be
	// no idempotency key and no way to address the result.
	ErrEmptyNonce = errors.New("fakrpc: request nonce is empty")

	// ErrNonceNotToken is returned when a nonce contains a character that would corrupt
	// the single-token nonce field of a FAKRES frame: any Unicode whitespace (the header
	// is split with strings.Fields, which honors the full unicode.IsSpace set, not just
	// ASCII space/tab) or a frame-sentinel character ('<'/'>').
	ErrNonceNotToken = errors.New("fakrpc: request nonce is not a frame-safe token")

	// ErrUnknownKind is returned when a Request kind is outside the closed vocabulary.
	ErrUnknownKind = errors.New("fakrpc: unknown request kind")
)

// ValidNonce reports whether nonce is a frame-safe token: non-empty and free of any
// Unicode whitespace or frame-sentinel character ('<'/'>'). A nonce that fails this can
// corrupt the single-token nonce field of a FAKRES header (the header is split with
// strings.Fields and the sentinels are '<<<'/'>>>'), so both Request.Validate and a
// direct EncodeFrame caller gate on it: a passing nonce is guaranteed to survive an
// EncodeFrame→DecodeFrame round-trip.
func ValidNonce(nonce string) error {
	if strings.TrimSpace(nonce) == "" {
		return ErrEmptyNonce
	}
	if i := strings.IndexFunc(nonce, func(r rune) bool {
		return unicode.IsSpace(r) || r == '<' || r == '>'
	}); i >= 0 {
		return fmt.Errorf("%w: %q", ErrNonceNotToken, nonce)
	}
	return nil
}

// Validate reports whether the request is well-formed enough to enqueue: a frame-safe
// nonce (see ValidNonce) and a kind from the closed vocabulary. A worker calls this before
// acting so a malformed or unknown-kind request is refused up front, not run on a guess.
func (r Request) Validate() error {
	if err := ValidNonce(r.Nonce); err != nil {
		return err
	}
	if !knownKinds[r.Kind] {
		return fmt.Errorf("%w: %q", ErrUnknownKind, r.Kind)
	}
	return nil
}

// Encode marshals the request to compact, single-line JSON — the form that survives a
// line-oriented bridge intact. It refuses a malformed request rather than spooling one a
// worker would only reject later.
func (r Request) Encode() ([]byte, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(r)
}

// DecodeRequest parses a spooled request line and validates it, so a worker refuses a
// malformed or unknown-kind request up front rather than acting on a guess.
func DecodeRequest(b []byte) (Request, error) {
	var r Request
	if err := json.Unmarshal(b, &r); err != nil {
		return Request{}, fmt.Errorf("fakrpc: decode request: %w", err)
	}
	if err := r.Validate(); err != nil {
		return Request{}, err
	}
	return r, nil
}
