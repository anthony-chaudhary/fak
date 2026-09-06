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

const ProtocolVersion = "fak.harness-sidecar/v1"

var ErrProtocol = errors.New("harness sidecar protocol violation")
var ErrWidening = errors.New("harness sidecar capability widening")
var ErrClosed = errors.New("harness sidecar closed")

type Limits struct {
	MaxFrame    uint32        `json:"max_frame"`
	MaxInflight int           `json:"max_inflight"`
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

type Identity struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

type Handshake struct {
	Protocol     string   `json:"protocol"`
	Identity     Identity `json:"identity"`
	Capabilities []string `json:"capabilities"`
	Limits       Limits   `json:"limits"`
}

type Request struct {
	ID               string          `json:"id"`
	Method           string          `json:"method"`
	Capability       string          `json:"capability,omitempty"`
	CapabilityToken  string          `json:"capability_token,omitempty"`
	Payload          json.RawMessage `json:"payload,omitempty"`
	DeadlineUnixNano int64           `json:"deadline_unix_nano,omitempty"`
}

type Response struct {
	ID      string          `json:"id"`
	Payload json.RawMessage `json:"payload,omitempty"`
	Error   string          `json:"error,omitempty"`
	Done    bool            `json:"done"`
}

type frame struct {
	Kind      string     `json:"kind"`
	Handshake *Handshake `json:"handshake,omitempty"`
	Request   *Request   `json:"request,omitempty"`
	Response  *Response  `json:"response,omitempty"`
	CancelID  string     `json:"cancel_id,omitempty"`
}

func ContractDigest(capabilities []string) string {
	blob, _ := json.Marshal(struct {
		Protocol     string   `json:"protocol"`
		Capabilities []string `json:"capabilities"`
	}{ProtocolVersion, capabilities})
	sum := sha256.Sum256(blob)
	return hex.EncodeToString(sum[:])
}

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

type Codec struct {
	r   *bufio.Reader
	w   io.Writer
	max uint32
	mu  sync.Mutex
}

func NewCodec(r io.Reader, w io.Writer, maxFrame uint32) *Codec {
	if maxFrame == 0 {
		maxFrame = 1 << 20
	}
	return &Codec{r: bufio.NewReader(r), w: w, max: maxFrame}
}
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

type Authorizer interface {
	Authorize(context.Context, string, string) error
}
type AuthorizerFunc func(context.Context, string, string) error

func (f AuthorizerFunc) Authorize(ctx context.Context, capability, token string) error {
	return f(ctx, capability, token)
}

type Handler interface {
	Handle(context.Context, string, json.RawMessage, func(json.RawMessage) error) error
}
type HandlerFunc func(context.Context, string, json.RawMessage, func(json.RawMessage) error) error

func (f HandlerFunc) Handle(c context.Context, m string, p json.RawMessage, s func(json.RawMessage) error) error {
	return f(c, m, p, s)
}

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

func NewServer(r io.Reader, w io.Writer, local, expected Handshake, h Handler) *Server {
	return NewAuthorizedServer(r, w, local, expected, h, nil)
}

func NewAuthorizedServer(r io.Reader, w io.Writer, local, expected Handshake, h Handler, auth Authorizer) *Server {
	l := local.Limits.normalized()
	return &Server{codec: NewCodec(r, w, l.MaxFrame), local: local, expected: expected, handler: h,
		authorizer: auth, limits: l, active: map[string]context.CancelFunc{}, sem: make(chan struct{}, l.MaxInflight), closed: make(chan struct{})}
}
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
