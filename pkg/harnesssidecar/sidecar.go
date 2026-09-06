// Package harnesssidecar provides a bounded, fail-closed local transport for
// language-neutral harness extensions.
package harnesssidecar

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"
)

// ProtocolVersion identifies the supported wire protocol specification for harness sidecars.
const ProtocolVersion = "fak.harness-sidecar/v1"

// ErrProtocol is returned when a wire payload or message sequence violates protocol semantics.
var ErrProtocol = errors.New("harness sidecar protocol violation")

// ErrWidening is returned when a peer requests or offers capabilities exceeding the allowed contract.
var ErrWidening = errors.New("harness sidecar capability widening")

// ErrClosed is returned when operations are attempted on a closed sidecar transport.
var ErrClosed = errors.New("harness sidecar closed")

// Limits defines bounds and timeouts enforced on sidecar communication frames and concurrency.
type Limits struct {
	// MaxFrame is the maximum allowed size in bytes for an incoming or outgoing frame.
	MaxFrame uint32 `json:"max_frame"`
	// MaxInflight is the maximum number of concurrent requests allowed to be processed.
	MaxInflight int `json:"max_inflight"`
	// CancelGrace is the time window allowed for a canceled request to terminate before hard teardown.
	CancelGrace time.Duration `json:"cancel_grace"`
}

func (l Limits) normalized() Limits {
	if l.MaxFrame == 0 {
		l.MaxFrame = 1 << 20
	}
	if l.MaxInflight == 0 {
		l.MaxInflight = 16
	}
	if l.CancelGrace == 0 {
		l.CancelGrace = time.Second
	}
	return l
}

// Validate checks whether all limit parameters meet minimum non-zero invariants.
func (l Limits) Validate() error {
	if l.MaxFrame == 0 {
		return fmt.Errorf("%w: invalid max_frame", ErrProtocol)
	}
	if l.MaxInflight <= 0 {
		return fmt.Errorf("%w: invalid max_inflight", ErrProtocol)
	}
	if l.CancelGrace <= 0 {
		return fmt.Errorf("%w: invalid cancel_grace", ErrProtocol)
	}
	return nil
}

// Identity contains metadata uniquely identifying a sidecar participant and its contract digest.
type Identity struct {
	// Name is the logical identifier of the sidecar peer (e.g. "python-worker", "host").
	Name string `json:"name"`
	// Version is the semantic or build version of the sidecar implementation.
	Version string `json:"version"`
	// Digest is the sha256 hex string validating the declared capabilities under ProtocolVersion.
	Digest string `json:"digest"`
}

// Handshake represents the initial negotiation frame exchanged between the host and sidecar.
type Handshake struct {
	// Protocol is the wire protocol version string (must match ProtocolVersion).
	Protocol string `json:"protocol"`
	// Identity specifies the peer's identifying metadata and contract digest.
	Identity Identity `json:"identity"`
	// Capabilities lists the capability tokens negotiated for this session.
	Capabilities []string `json:"capabilities"`
	// Limits specifies the operating boundaries requested or enforced by the peer.
	Limits Limits `json:"limits"`
}

// Request is an invocation frame dispatched from the host to the sidecar server.
type Request struct {
	// ID is the correlation identifier unique to this request.
	ID string `json:"id"`
	// Method is the RPC method or tool name being invoked.
	Method string `json:"method"`
	// Capability is the optional required capability name for gated operations.
	Capability string `json:"capability,omitempty"`
	// CapabilityToken is the security bearer token presented to authorize the capability.
	CapabilityToken string `json:"capability_token,omitempty"`
	// Payload is the method-specific JSON input arguments.
	Payload json.RawMessage `json:"payload,omitempty"`
	// DeadlineUnixNano is an optional absolute Unix timestamp in nanoseconds after which the call expires.
	DeadlineUnixNano int64 `json:"deadline_unix_nano,omitempty"`
}

// Response is an execution result or streaming event sent from the sidecar server to the caller.
type Response struct {
	// ID is the correlation identifier matching the originating Request.
	ID string `json:"id"`
	// Payload is the optional intermediate or final JSON result data.
	Payload json.RawMessage `json:"payload,omitempty"`
	// Error contains the failure description if the request was denied or faulted.
	Error string `json:"error,omitempty"`
	// Done indicates whether this response terminates the request stream.
	Done bool `json:"done"`
}

type frame struct {
	Kind      string     `json:"kind"`
	Handshake *Handshake `json:"handshake,omitempty"`
	Request   *Request   `json:"request,omitempty"`
	Response  *Response  `json:"response,omitempty"`
	CancelID  string     `json:"cancel_id,omitempty"`
}

// ContractDigest computes a deterministic sha256 hex digest for a set of capabilities under ProtocolVersion.
func ContractDigest(capabilities []string) string {
	blob, _ := json.Marshal(struct {
		Protocol     string   `json:"protocol"`
		Capabilities []string `json:"capabilities"`
	}{ProtocolVersion, capabilities})
	sum := sha256.Sum256(blob)
	return hex.EncodeToString(sum[:])
}

// ValidateHandshake verifies that a peer's handshake matches the expected protocol, identity, and capabilities.
// It fails closed if the protocol mismatches, identity fields are incomplete, the digest does not match,
// duplicate capabilities exist, or unadvertised capabilities (widening) are claimed.
func ValidateHandshake(want, got Handshake) error {
	if got.Protocol != ProtocolVersion || got.Protocol != want.Protocol {
		return fmt.Errorf("%w: version %q", ErrProtocol, got.Protocol)
	}
	if got.Identity.Name == "" || got.Identity.Version == "" || got.Identity.Digest == "" {
		return fmt.Errorf("%w: incomplete identity", ErrProtocol)
	}
	if got.Identity.Digest != ContractDigest(got.Capabilities) {
		return fmt.Errorf("%w: digest mismatch", ErrProtocol)
	}
	allowed := map[string]bool{}
	for _, c := range want.Capabilities {
		allowed[c] = true
	}
	seen := map[string]bool{}
	for _, c := range got.Capabilities {
		if !allowed[c] {
			return fmt.Errorf("%w: %s", ErrWidening, c)
		}
		if seen[c] {
			return fmt.Errorf("%w: duplicate capability %s", ErrProtocol, c)
		}
		seen[c] = true
	}
	return nil
}

// Codec provides length-prefixed JSON framing over an underlying io.Reader and io.Writer.
type Codec struct {
	r   *bufio.Reader
	w   io.Writer
	max uint32
	mu  sync.Mutex
}

// NewCodec constructs a length-prefixed JSON codec bounded by maxFrame bytes.
// If maxFrame is 0, a 1MB default limit is applied.
func NewCodec(r io.Reader, w io.Writer, maxFrame uint32) *Codec {
	if maxFrame == 0 {
		maxFrame = 1 << 20
	}
	return &Codec{r: bufio.NewReader(r), w: w, max: maxFrame}
}

// Write encodes v as JSON and writes it prefixed with a 4-byte big-endian length header.
func (c *Codec) Write(v any) error {
	blob, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if uint32(len(blob)) > c.max {
		return fmt.Errorf("%w: frame too large", ErrProtocol)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	var head [4]byte
	binary.BigEndian.PutUint32(head[:], uint32(len(blob)))
	if _, err = c.w.Write(head[:]); err != nil {
		return err
	}
	_, err = c.w.Write(blob)
	return err
}

// Read reads a 4-byte big-endian length header, reads the bounded frame payload,
// and decodes it as strict JSON into v with unknown fields disallowed.
func (c *Codec) Read(v any) error {
	var head [4]byte
	if _, err := io.ReadFull(c.r, head[:]); err != nil {
		return err
	}
	n := binary.BigEndian.Uint32(head[:])
	if n == 0 || n > c.max {
		return fmt.Errorf("%w: invalid frame length %d", ErrProtocol, n)
	}
	blob := make([]byte, n)
	if _, err := io.ReadFull(c.r, blob); err != nil {
		return err
	}
	dec := json.NewDecoder(bytesReader(blob))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("%w: malformed frame: %v", ErrProtocol, err)
	}
	return nil
}

type byteReader struct{ b []byte }

func bytesReader(b []byte) *byteReader { return &byteReader{b: b} }
func (r *byteReader) Read(p []byte) (int, error) {
	if len(r.b) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.b)
	r.b = r.b[n:]
	return n, nil
}

// Authorizer validates whether an invocation with a specific capability and bearer token is permitted.
type Authorizer interface {
	// Authorize evaluates capability access under ctx, returning an error if unauthorized.
	Authorize(context.Context, string, string) error
}

// AuthorizerFunc allows ordinary functions to satisfy the Authorizer interface.
type AuthorizerFunc func(context.Context, string, string) error

// Authorize calls f(ctx, capability, token).
func (f AuthorizerFunc) Authorize(ctx context.Context, capability, token string) error {
	return f(ctx, capability, token)
}

// Handler processes incoming requests and streams intermediate and final results.
type Handler interface {
	// Handle executes method with payload, calling send to stream intermediate payloads.
	Handle(context.Context, string, json.RawMessage, func(json.RawMessage) error) error
}

// HandlerFunc allows ordinary functions to satisfy the Handler interface.
type HandlerFunc func(context.Context, string, json.RawMessage, func(json.RawMessage) error) error

// Handle calls f(c, m, p, s).
func (f HandlerFunc) Handle(c context.Context, m string, p json.RawMessage, s func(json.RawMessage) error) error {
	return f(c, m, p, s)
}

// Server manages a sidecar peer lifecycle, concurrency limits, cancellation, and dispatch.
type Server struct {
	codec           *Codec
	local, expected Handshake
	handler         Handler
	authorizer      Authorizer
	limits          Limits
	mu              sync.Mutex
	active          map[string]context.CancelFunc
	sem             chan struct{}
	closed          chan struct{}
	once            sync.Once
}

// NewServer creates a Server that processes requests without token-based capability authorization.
func NewServer(r io.Reader, w io.Writer, local, expected Handshake, h Handler) *Server {
	return NewAuthorizedServer(r, w, local, expected, h, nil)
}

// NewAuthorizedServer creates a Server configured with an Authorizer for gated capability enforcement.
func NewAuthorizedServer(r io.Reader, w io.Writer, local, expected Handshake, h Handler, auth Authorizer) *Server {
	l := local.Limits.normalized()
	return &Server{codec: NewCodec(r, w, l.MaxFrame), local: local, expected: expected, handler: h,
		authorizer: auth, limits: l, active: map[string]context.CancelFunc{}, sem: make(chan struct{}, l.MaxInflight), closed: make(chan struct{})}
}

// Serve performs the initial handshake exchange and enters the dispatch loop until EOF or error.
func (s *Server) Serve(ctx context.Context) error {
	defer s.closeAll()
	var hello frame
	if err := s.codec.Read(&hello); err != nil {
		return err
	}
	if hello.Kind != "handshake" || hello.Handshake == nil {
		return fmt.Errorf("%w: handshake required", ErrProtocol)
	}
	if err := ValidateHandshake(s.expected, *hello.Handshake); err != nil {
		return err
	}
	if err := s.codec.Write(frame{Kind: "handshake", Handshake: &s.local}); err != nil {
		return err
	}
	for {
		var f frame
		if err := s.codec.Read(&f); err != nil {
			return err
		}
		switch f.Kind {
		case "request":
			if f.Request == nil {
				return ErrProtocol
			}
			select {
			case s.sem <- struct{}{}:
				go s.handle(ctx, *f.Request)
			default:
				return fmt.Errorf("%w: inflight limit", ErrProtocol)
			}
		case "cancel":
			s.cancel(f.CancelID)
		default:
			return fmt.Errorf("%w: unexpected %s", ErrProtocol, f.Kind)
		}
	}
}
func (s *Server) handle(parent context.Context, r Request) {
	if r.Capability != "" {
		if s.authorizer == nil || r.CapabilityToken == "" {
			_ = s.codec.Write(frame{Kind: "response", Response: &Response{ID: r.ID, Error: "capability denied", Done: true}})
			<-s.sem
			return
		}
		if err := s.authorizer.Authorize(parent, r.Capability, r.CapabilityToken); err != nil {
			_ = s.codec.Write(frame{Kind: "response", Response: &Response{ID: r.ID, Error: "capability denied", Done: true}})
			<-s.sem
			return
		}
	}
	defer func() { <-s.sem; s.mu.Lock(); delete(s.active, r.ID); s.mu.Unlock() }()
	ctx, cancel := context.WithCancel(parent)
	if r.DeadlineUnixNano > 0 {
		ctx, cancel = context.WithDeadline(parent, time.Unix(0, r.DeadlineUnixNano))
	}
	s.mu.Lock()
	if _, exists := s.active[r.ID]; exists {
		s.mu.Unlock()
		cancel()
		_ = s.codec.Write(frame{Kind: "response", Response: &Response{ID: r.ID, Error: "duplicate id", Done: true}})
		return
	}
	s.active[r.ID] = cancel
	s.mu.Unlock()
	defer cancel()
	send := func(p json.RawMessage) error {
		return s.codec.Write(frame{Kind: "response", Response: &Response{ID: r.ID, Payload: p}})
	}
	err := s.handler.Handle(ctx, r.Method, r.Payload, send)
	resp := Response{ID: r.ID, Done: true}
	if err != nil {
		resp.Error = err.Error()
	}
	_ = s.codec.Write(frame{Kind: "response", Response: &resp})
}
func (s *Server) cancel(id string) {
	s.mu.Lock()
	cancel := s.active[id]
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}
func (s *Server) closeAll() {
	s.once.Do(func() {
		close(s.closed)
		s.mu.Lock()
		defer s.mu.Unlock()
		for _, cancel := range s.active {
			cancel()
		}
	})
}
