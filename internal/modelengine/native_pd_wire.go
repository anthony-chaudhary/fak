package modelengine

// native_pd_wire.go — the two-process native P/D handoff wire.
//
// NativePDCluster.Admit runs prefill, KV transfer, and decode admission inside
// ONE process. This file splits that admission across two OS processes so a
// real prefill node and a real decode node can run as separate `fak serve pd`
// roles (cmd/fak/serve_pd.go):
//
//   - PrefillForWire is the prefill half of Admit: prompt prefill on the
//     cluster's prefill worker, paged-KV wire frame + prefill logits out.
//   - PDHandoffTransport is the narrow data-plane seam between the processes;
//     PDWireClient is the shipped TCP implementation.
//   - PDWireServer is the decode-side receiver path: it ingests each frame
//     into a NativePDCluster decode worker and admits it through
//     AdmitImported — decode proceeds through continuous batching with NO
//     re-prefill.
//
// Protocol shape extends fak-pdbench/v1 (cmd/pdbench, the physical dual-eGPU
// proof): length-prefixed frames with byte limits, a JSON header plus
// acknowledgement BEFORE the payload, closed wire error codes, and typed
// sentinels. The KV payload itself is the native paged-KV wire frame
// (model.MarshalPagedKVTransfer, serializer fak-paged-kv-f32-v1), so the bytes
// crossing the socket are exactly the bytes the in-process Admit moves.
//
// Evidence scope: [SW-VERIFIED] loopback/TCP. Cross-host physical P/D receipts
// live in the companion repo under schema fak-pdbench/v1; nothing here claims
// multi-GPU or hardware performance.

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// PDWireProtocol is the version token every P/D wire header carries. It extends
// the fak-pdbench/v1 envelope over the native paged-KV serializer rather than
// diverging from it.
const PDWireProtocol = "fak-pd-wire/v1"

// PDGenTokens is the default per-request decode budget the P/D front door
// inherits from the native scheduler: a lane retires at EOS or after this many
// generated tokens, imported lanes included. A request-level max_tokens rides
// the wire as GenBudget and narrows the budget for that lane only.
const PDGenTokens = genTokens

// Closed wire error codes, mirroring the fak-pdbench/v1 vocabulary.
const (
	pdWireCodeProtocol = "protocol"
	pdWireCodeIdentity = "identity"
	pdWireCodeRuntime  = "runtime"
)

// Per-frame byte limits. They mirror the fak-pdbench/v1 bounds so a bench
// receipt and a serve handoff stay comparable; the KV frame limit is sized for
// real-model paged spans (host f32, block-granular). The 2 GiB cap covers the
// mix-extension planning bound (8192 tokens x 240 KiB/token ~= 1.92 GiB) that
// the 512 MiB pin-era cap refused at cold-2048 prompts.
const (
	pdWireMaxHeaderBytes  = uint64(1 << 20)
	pdWireMaxKVFrameBytes = uint64(2 << 30)
	pdWireMaxLogitsBytes  = uint64(16 << 20)
	pdWireMaxReplyBytes   = uint64(1 << 20)
	// PDWireMaxPromptTokens caps the prompt a single handoff may carry, the same
	// bound the physical proof used.
	PDWireMaxPromptTokens = 8192
)

var (
	// ErrPDWireProtocol marks a frame that violates the P/D wire contract
	// (version, framing, declared lengths, finite logits). Callers may use
	// errors.Is.
	ErrPDWireProtocol = errors.New("modelengine: P/D wire protocol violation")
	// ErrPDWireIdentity marks a producer/consumer identity mismatch (different
	// model configuration or vocabulary across the two roles). Callers may use
	// errors.Is.
	ErrPDWireIdentity = errors.New("modelengine: P/D wire identity mismatch")
	// ErrPDStreamConsumerStopped marks a ShipStream whose token sink refused a
	// token (the front-door response writer failed or the client left). The
	// handoff is aborted WITHOUT a peer retry: the caller's request is over.
	// Callers may use errors.Is.
	ErrPDStreamConsumerStopped = errors.New("modelengine: P/D stream consumer stopped")
)

// PDShipRequest is one prefill-to-decode handoff across the wire: the paged-KV
// wire frame built by PrefillForWire plus the prefill logits for the first
// generated token and the prompt IDs they were computed from.
type PDShipRequest struct {
	KVFrame []byte
	Logits  []float32
	Prompt  []int
	Tool    string
	// GenBudget is the imported lane's generated-token retirement budget; zero
	// keeps the engine-wide default. The front door sets it from max_tokens.
	GenBudget int
}

// PDShipResult is the decode worker's answer: the tokens its continuous-
// batching scheduler generated from the imported KV, with no re-prefill.
type PDShipResult struct {
	Tokens       []int
	DecodeWorker string
	SpanDigest   string
	BytesMoved   int64
}

// PDHandoffTransport is the narrow two-process P/D data-plane seam: one
// request out, one reply in. TCP is the shipped implementation; an RDMA/UCX
// backend implements the same contract without touching admission.
type PDHandoffTransport interface {
	Ship(ctx context.Context, req PDShipRequest) (PDShipResult, error)
}

// PDTokenSink receives generated tokens as the decode lane produces them, one
// call per token, in generation order. Returning a non-nil error aborts the
// handoff: the wire client stops reading, tears down the connection, and
// surfaces ErrPDStreamConsumerStopped so no peer retry is spent on a consumer
// that stopped listening.
type PDTokenSink func(token int) error

// PDHandoffTransportStream extends the handoff seam with incremental delivery:
// ShipStream streams each generated token to sink as the decode lane emits it
// and returns the full result after the terminal frame. The front door uses it
// for SSE responses; plain Ship keeps the single-reply shape.
type PDHandoffTransportStream interface {
	PDHandoffTransport
	ShipStream(ctx context.Context, req PDShipRequest, sink PDTokenSink) (PDShipResult, error)
}

// pdWireHeader is the JSON metadata frame that opens every handoff. It carries
// only identities and declared lengths — the KV bytes themselves follow in
// their own frames after the acknowledgement.
type pdWireHeader struct {
	Protocol     string `json:"protocol"`
	ConfigSHA256 string `json:"config_sha256"`
	VocabSize    int    `json:"vocab_size"`
	PromptIDs    []int  `json:"prompt_ids"`
	PromptSHA256 string `json:"prompt_sha256"`
	KVFrameBytes int    `json:"kv_frame_bytes"`
	LogitsBytes  int    `json:"logits_bytes"`
	// StreamTokens asks for incremental delivery: one token frame per generated
	// token, then a terminal reply frame. Absent (the historical shape) keeps
	// the single-reply exchange, so an older peer still pairs correctly.
	StreamTokens bool `json:"stream_tokens,omitempty"`
	// GenBudget is the imported lane's generated-token retirement budget; zero
	// keeps the engine-wide default.
	GenBudget int `json:"gen_budget,omitempty"`
}

// Closed P/D stream-frame kinds. A streaming reply is a sequence of token
// frames followed by exactly one reply frame; clients branch on kind, never on
// prose.
const (
	pdWireFrameToken = "token"
	pdWireFrameReply = "reply"
)

// pdWireStreamFrame is one frame of a streaming reply: either a single
// generated token or the terminal reply carrying the handoff metadata. The
// legacy non-streaming reply keeps its own pdWireReply shape byte-for-byte.
type pdWireStreamFrame struct {
	Kind         string `json:"kind"`
	Token        int    `json:"token,omitempty"`
	Protocol     string `json:"protocol,omitempty"`
	DecodeWorker string `json:"decode_worker,omitempty"`
	SpanDigest   string `json:"span_digest,omitempty"`
	BytesMoved   int64  `json:"bytes_moved,omitempty"`
	ErrorCode    string `json:"error_code,omitempty"`
	Error        string `json:"error,omitempty"`
}

// pdWireAck is the receiver's verdict on the header, written before any
// payload frame is sent. A negative ack (OK=false) carries a closed error
// code and terminates the handoff: no payload frames follow it.
type pdWireAck struct {
	OK    bool   `json:"ok"`
	Code  string `json:"code,omitempty"`
	Error string `json:"error,omitempty"`
}

// pdWireReply is the terminal frame: the generated tokens on success, or a
// closed error code plus detail on failure.
type pdWireReply struct {
	Protocol     string `json:"protocol"`
	DecodeWorker string `json:"decode_worker"`
	Tokens       []int  `json:"tokens"`
	SpanDigest   string `json:"span_digest"`
	BytesMoved   int64  `json:"bytes_moved"`
	ErrorCode    string `json:"error_code,omitempty"`
	Error        string `json:"error,omitempty"`
}

// pdWireConfigDigest is the stable model-configuration identity both roles
// bind their handshake to: the SHA-256 of the canonical JSON encoding of the
// loaded model.Config. Two roles loading different checkpoints (or different
// quant geometries) never share a digest, so a mispaired pair fails closed at
// the header instead of decoding against the wrong weights.
func pdWireConfigDigest(cfg model.Config) string {
	raw, err := json.Marshal(cfg)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// pdWirePromptDigest digests a prompt the same way NativePDPrefixKey does
// (little-endian token stream), so the header digest and the residency key
// agree on one prompt identity.
func pdWirePromptDigest(ids []int) string {
	h := sha256.New()
	var buf [8]byte
	for _, id := range ids {
		binary.LittleEndian.PutUint64(buf[:], uint64(int64(id)))
		_, _ = h.Write(buf[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// pdWireWriteFrame writes an 8-byte big-endian length prefix then the payload,
// refusing payloads above limit — the fak-pdbench/v1 framing discipline.
func pdWireWriteFrame(w io.Writer, payload []byte, limit uint64) error {
	if uint64(len(payload)) > limit {
		return fmt.Errorf("%w: frame is %d bytes, limit %d", ErrPDWireProtocol, len(payload), limit)
	}
	var hdr [8]byte
	binary.BigEndian.PutUint64(hdr[:], uint64(len(payload)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if len(payload) > 0 {
		if _, err := w.Write(payload); err != nil {
			return err
		}
	}
	return nil
}

// pdWireReadFrame reads one length-prefixed frame, bounded by limit so a
// corrupt or hostile peer cannot declare a multi-gigabyte frame.
func pdWireReadFrame(r io.Reader, limit uint64) ([]byte, error) {
	var hdr [8]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint64(hdr[:])
	if n > limit {
		return nil, fmt.Errorf("%w: frame declares %d bytes, limit %d", ErrPDWireProtocol, n, limit)
	}
	buf := make([]byte, n)
	if n > 0 {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
	}
	return buf, nil
}

// pdWireEncodeFloat32 / pdWireDecodeFloat32 carry the prefill logits as
// little-endian f32, the same encoding the paged-KV block bytes use.
func pdWireEncodeFloat32(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func pdWireDecodeFloat32(b []byte) ([]float32, error) {
	if len(b) == 0 || len(b)%4 != 0 {
		return nil, fmt.Errorf("%w: malformed logits frame (%d bytes)", ErrPDWireProtocol, len(b))
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v, nil
}

// PrefillForWire runs the prefill half of Admit on this cluster's prefill
// worker and returns the paged-KV wire frame plus the prefill logits for the
// first generated token, ready to ship to a remote decode role. The decode
// half (UnmarshalPagedKVTransfer + AdmitImported) runs in the peer process —
// see PDWireServer. It is the two-process split of Admit's prefill boundary:
// the same session shape, the same paged serializer, the same transfer
// descriptor construction; only the socket moves instead of the in-process
// transport. The transfer descriptor's Owner stays empty here: the receiving
// decode worker stamps its own name onto the span it takes ownership of.
func (c *NativePDCluster) PrefillForWire(ctx context.Context, req NativePDRequest) (frame []byte, logits []float32, err error) {
	if c == nil || c.m == nil || c.prefill == nil {
		return nil, nil, errors.New("modelengine: native P/D cluster is not initialized")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	prompt := c.promptFor(ctx, req)
	if len(prompt) == 0 {
		return nil, nil, errors.New("modelengine: native P/D request has empty prompt")
	}
	prefillSess := c.newPrefillSession()
	logits = prefillSess.Prefill(prompt)
	srcPool := model.NewPagedKVPoolWithRaw(c.m.Cfg, nativePDDefaultBlockTokens)
	src, perr := model.KVCacheToPaged(srcPool, prefillSess.Cache)
	prefillSess.Close()
	if perr != nil {
		return nil, nil, perr
	}
	defer src.Free()
	transfer := c.transferDescriptor(req, "", len(prompt))
	frame, merr := model.MarshalPagedKVTransfer(src, transfer, 0, len(prompt))
	if merr != nil {
		return nil, nil, merr
	}
	c.prefill.addPrefill(transfer.BytesMoved)
	return frame, logits, nil
}

// newPrefillSession mints the prefill-half session with the cluster's optional
// session decorator applied (e.g. a device backend bound by a serve role).
// Without a decorator it is byte-for-byte the historical m.NewSession().
func (c *NativePDCluster) newPrefillSession() *model.Session {
	sess := c.m.NewSession()
	if c.sessionHook != nil {
		c.sessionHook(sess)
	}
	return sess
}

// PDWireClient dials a decode role's listener and performs one handoff per
// Ship call: a fresh connection per handoff, no session state retained on the
// wire. The client derives the handshake identity from its own cluster, so a
// prefill role always presents the configuration it actually prefilled under.
type PDWireClient struct {
	cluster     *NativePDCluster
	addr        string
	dialTimeout time.Duration
}

// NewPDWireClient builds the TCP handoff transport toward addr. dialTimeout
// bounds the connect; per-handoff deadlines come from the caller's context.
func NewPDWireClient(cluster *NativePDCluster, addr string, dialTimeout time.Duration) *PDWireClient {
	if dialTimeout <= 0 {
		dialTimeout = 30 * time.Second
	}
	return &PDWireClient{cluster: cluster, addr: addr, dialTimeout: dialTimeout}
}

// Ship performs one prefill-to-decode handoff over TCP and returns the decode
// worker's generated tokens. A cancelled context interrupts the socket and
// surfaces the context error; a mispaired or malformed peer surfaces
// ErrPDWireIdentity / ErrPDWireProtocol.
func (c *PDWireClient) Ship(ctx context.Context, req PDShipRequest) (PDShipResult, error) {
	return c.ship(ctx, req, nil)
}

// ShipStream is Ship with incremental delivery: every generated token reaches
// sink in generation order as the decode lane produces it, and the returned
// result carries the accumulated token list plus the terminal metadata. A
// sink error aborts the handoff with ErrPDStreamConsumerStopped.
func (c *PDWireClient) ShipStream(ctx context.Context, req PDShipRequest, sink PDTokenSink) (PDShipResult, error) {
	if sink == nil {
		return PDShipResult{}, errors.New("modelengine: P/D wire stream requires a token sink")
	}
	return c.ship(ctx, req, sink)
}

// ship runs the shared handshake: header, acknowledgement, payload frames —
// then either the legacy single reply (sink nil) or the streamed token frames
// plus terminal reply.
func (c *PDWireClient) ship(ctx context.Context, req PDShipRequest, sink PDTokenSink) (PDShipResult, error) {
	if c == nil || c.cluster == nil || c.cluster.m == nil {
		return PDShipResult{}, errors.New("modelengine: P/D wire client has no cluster")
	}
	m := c.cluster.m
	if len(req.Prompt) == 0 || len(req.Prompt) > PDWireMaxPromptTokens {
		return PDShipResult{}, fmt.Errorf("%w: prompt must carry 1..%d tokens", ErrPDWireProtocol, PDWireMaxPromptTokens)
	}
	if len(req.Logits) != m.Cfg.VocabSize {
		return PDShipResult{}, fmt.Errorf("%w: logits carry %d values, model vocabulary is %d", ErrPDWireProtocol, len(req.Logits), m.Cfg.VocabSize)
	}
	if len(req.KVFrame) == 0 || uint64(len(req.KVFrame)) > pdWireMaxKVFrameBytes {
		return PDShipResult{}, fmt.Errorf("%w: KV frame must be 1..%d bytes", ErrPDWireProtocol, pdWireMaxKVFrameBytes)
	}
	if req.GenBudget < 0 || req.GenBudget > PDWireMaxPromptTokens {
		return PDShipResult{}, fmt.Errorf("%w: gen budget must be 0..%d", ErrPDWireProtocol, PDWireMaxPromptTokens)
	}
	header := pdWireHeader{
		Protocol:     PDWireProtocol,
		ConfigSHA256: pdWireConfigDigest(m.Cfg),
		VocabSize:    m.Cfg.VocabSize,
		PromptIDs:    append([]int(nil), req.Prompt...),
		PromptSHA256: pdWirePromptDigest(req.Prompt),
		KVFrameBytes: len(req.KVFrame),
		LogitsBytes:  4 * len(req.Logits),
		StreamTokens: sink != nil,
		GenBudget:    req.GenBudget,
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return PDShipResult{}, fmt.Errorf("%w: encode header: %v", ErrPDWireProtocol, err)
	}

	connCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	stop := context.AfterFunc(ctx, func() {
		cancel()
	})
	defer stop()

	dialer := net.Dialer{Timeout: c.dialTimeout}
	conn, err := dialer.DialContext(connCtx, "tcp", c.addr)
	if err != nil {
		return PDShipResult{}, fmt.Errorf("modelengine: P/D wire dial %s: %w", c.addr, err)
	}
	defer conn.Close()
	interruptConnOnCancel(connCtx, conn)

	if err := pdWireWriteFrame(conn, headerJSON, pdWireMaxHeaderBytes); err != nil {
		return PDShipResult{}, fmt.Errorf("%w: send header: %v", ErrPDWireProtocol, err)
	}
	ackRaw, err := pdWireReadFrame(conn, pdWireMaxReplyBytes)
	if err != nil {
		return PDShipResult{}, preferContextError(connCtx, fmt.Errorf("%w: read ack: %v", ErrPDWireProtocol, err))
	}
	var ack pdWireAck
	if err := json.Unmarshal(ackRaw, &ack); err != nil {
		return PDShipResult{}, fmt.Errorf("%w: malformed receiver acknowledgement", ErrPDWireProtocol)
	}
	if !ack.OK {
		return PDShipResult{}, decodePDWireError(ack.Code, ack.Error)
	}
	logitsWire := pdWireEncodeFloat32(req.Logits)
	if err := pdWireWriteFrame(conn, req.KVFrame, pdWireMaxKVFrameBytes); err != nil {
		return PDShipResult{}, preferContextError(connCtx, fmt.Errorf("%w: send KV frame: %v", ErrPDWireProtocol, err))
	}
	if err := pdWireWriteFrame(conn, logitsWire, pdWireMaxLogitsBytes); err != nil {
		return PDShipResult{}, preferContextError(connCtx, fmt.Errorf("%w: send logits frame: %v", ErrPDWireProtocol, err))
	}
	if sink == nil {
		return c.readLegacyReply(conn, connCtx)
	}
	return c.readStreamedReply(conn, connCtx, sink)
}

// readLegacyReply consumes the historical single-reply exchange.
func (c *PDWireClient) readLegacyReply(conn net.Conn, connCtx context.Context) (PDShipResult, error) {
	replyRaw, err := pdWireReadFrame(conn, pdWireMaxReplyBytes)
	if err != nil {
		return PDShipResult{}, preferContextError(connCtx, fmt.Errorf("%w: read reply: %v", ErrPDWireProtocol, err))
	}
	var reply pdWireReply
	if err := json.Unmarshal(replyRaw, &reply); err != nil {
		return PDShipResult{}, fmt.Errorf("%w: malformed reply JSON", ErrPDWireProtocol)
	}
	if reply.Error != "" {
		return PDShipResult{}, decodePDWireError(reply.ErrorCode, reply.Error)
	}
	if reply.Protocol != PDWireProtocol {
		return PDShipResult{}, fmt.Errorf("%w: reply version %q", ErrPDWireProtocol, reply.Protocol)
	}
	return PDShipResult{
		Tokens:       append([]int(nil), reply.Tokens...),
		DecodeWorker: reply.DecodeWorker,
		SpanDigest:   reply.SpanDigest,
		BytesMoved:   reply.BytesMoved,
	}, nil
}

// readStreamedReply consumes the streaming reply: one token frame per
// generated token, then the terminal reply frame carrying the metadata. The
// accumulated tokens mirror what the sink received.
func (c *PDWireClient) readStreamedReply(conn net.Conn, connCtx context.Context, sink PDTokenSink) (PDShipResult, error) {
	var res PDShipResult
	for {
		raw, err := pdWireReadFrame(conn, pdWireMaxReplyBytes)
		if err != nil {
			return PDShipResult{}, preferContextError(connCtx, fmt.Errorf("%w: read stream frame: %v", ErrPDWireProtocol, err))
		}
		var fr pdWireStreamFrame
		if err := json.Unmarshal(raw, &fr); err != nil {
			return PDShipResult{}, fmt.Errorf("%w: malformed stream frame JSON", ErrPDWireProtocol)
		}
		switch fr.Kind {
		case pdWireFrameToken:
			res.Tokens = append(res.Tokens, fr.Token)
			if err := sink(fr.Token); err != nil {
				// The consumer stopped listening: tear the socket down so the
				// peer stops decoding, and surface the closed sentinel so no
				// peer retry is spent on a dead request.
				_ = conn.Close()
				return PDShipResult{}, fmt.Errorf("%w: %v", ErrPDStreamConsumerStopped, err)
			}
		case pdWireFrameReply:
			if fr.Error != "" {
				return PDShipResult{}, decodePDWireError(fr.ErrorCode, fr.Error)
			}
			if fr.Protocol != PDWireProtocol {
				return PDShipResult{}, fmt.Errorf("%w: reply version %q", ErrPDWireProtocol, fr.Protocol)
			}
			res.DecodeWorker = fr.DecodeWorker
			res.SpanDigest = fr.SpanDigest
			res.BytesMoved = fr.BytesMoved
			return res, nil
		default:
			return PDShipResult{}, fmt.Errorf("%w: unknown stream frame kind %q", ErrPDWireProtocol, fr.Kind)
		}
	}
}

// PDWireServer is the decode-side receiver path over a NativePDCluster: it
// accepts one connection per handoff, validates the header against the
// cluster's own model identity, ingests the paged-KV frame into the routed
// decode worker's pool, admits the imported cache through AdmitImported (no
// re-prefill), and replies with the generated tokens. Configure it once and
// do not mutate it while it serves.
type PDWireServer struct {
	cluster  *NativePDCluster
	timeout  time.Duration // per-handoff bound; zero = unbounded
	listener net.Listener
	mu       sync.Mutex
	wg       sync.WaitGroup
	closed   bool
	conns    map[net.Conn]context.CancelFunc
}

// NewPDWireServer builds the receiver path over cluster. Every accepted
// connection is served until the listener is closed.
func NewPDWireServer(cluster *NativePDCluster) *PDWireServer {
	return &PDWireServer{cluster: cluster}
}

// SetHandoffTimeout bounds how long one admitted handoff may take before the
// server aborts it and frees the received KV. Zero (the default) leaves the
// handoff bounded only by the client's context.
func (s *PDWireServer) SetHandoffTimeout(d time.Duration) {
	s.mu.Lock()
	s.timeout = d
	s.mu.Unlock()
}

// Listen binds the server's TCP listener. The caller serves it with Serve and
// ends the accept loop with Close.
func (s *PDWireServer) Listen(addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("modelengine: P/D wire listen %s: %w", addr, err)
	}
	s.listener = ln
	return nil
}

// Addr reports the bound listen address (useful for tests that bind :0).
func (s *PDWireServer) Addr() net.Addr {
	if s.listener == nil {
		return nil
	}
	return s.listener.Addr()
}

// Serve runs the accept loop until the listener is closed. It returns nil on a
// clean close.
func (s *PDWireServer) Serve() error {
	if s == nil || s.cluster == nil || s.cluster.m == nil || s.listener == nil {
		return errors.New("modelengine: P/D wire server is not initialized")
	}
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		ctx, cancel := context.WithCancel(context.Background())
		s.mu.Lock()
		if s.closed {
			s.mu.Unlock()
			cancel()
			_ = conn.Close()
			return nil
		}
		if s.conns == nil {
			s.conns = make(map[net.Conn]context.CancelFunc)
		}
		s.conns[conn] = cancel
		// Register under the close mutex: no Add can race a zero-count Wait.
		s.wg.Add(1)
		s.mu.Unlock()
		go func() {
			defer s.wg.Done()
			defer func() {
				cancel()
				_ = conn.Close()
				s.mu.Lock()
				delete(s.conns, conn)
				s.mu.Unlock()
			}()
			s.handleConn(ctx, conn)
		}()
	}
}

// Close stops accepting, cancels and closes accepted connections, and waits for
// their handoffs to retire. Idle clients cannot hold shutdown open.
// Idempotent.
func (s *PDWireServer) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		s.wg.Wait()
		return nil
	}
	s.closed = true
	var lnErr error
	if s.listener != nil {
		lnErr = s.listener.Close()
	}
	for conn, cancel := range s.conns {
		cancel()
		_ = conn.Close()
	}
	s.mu.Unlock()
	s.wg.Wait()
	return lnErr
}

// handleConn serves exactly one handoff on conn. Any failure replies with a
// closed error code when the wire still allows it, and frees KV the server
// came to own.
func (s *PDWireServer) handleConn(ctx context.Context, conn net.Conn) {
	stop := context.AfterFunc(ctx, func() {
		_ = conn.SetDeadline(time.Now())
	})
	defer stop()
	if err := s.serveHandoff(ctx, conn); err != nil {
		// A failed read or reply may leave the client without a diagnostic.
		// Error text may contain protocol strings or prompt IDs from the peer;
		// only closed classifications belong in the receiver's normal log.
		log.Printf("modelengine: P/D wire handoff failed: peer=%q code=%s class=%s", conn.RemoteAddr(), pdWireErrorCode(err), pdWireErrorClass(err))
	}
}

// pdWireErrorClass records useful transport distinctions without echoing
// request-derived error text. All returned values are a closed vocabulary.
func pdWireErrorClass(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "deadline_exceeded"
	case errors.Is(err, io.ErrUnexpectedEOF):
		return "truncated_frame"
	case errors.Is(err, io.EOF):
		return "eof"
	default:
		return pdWireErrorCode(err)
	}
}

func (s *PDWireServer) serveHandoff(ctx context.Context, conn net.Conn) error {
	headerRaw, err := pdWireReadFrame(conn, pdWireMaxHeaderBytes)
	if err != nil {
		return preferContextError(ctx, err)
	}
	var header pdWireHeader
	if err := json.Unmarshal(headerRaw, &header); err != nil {
		return s.replyAckError(conn, ctx, fmt.Errorf("%w: malformed header JSON", ErrPDWireProtocol))
	}
	if err := s.validateHeader(&header); err != nil {
		return s.replyAckError(conn, ctx, err)
	}
	ackJSON, _ := json.Marshal(pdWireAck{OK: true})
	if err := pdWireWriteFrame(conn, ackJSON, pdWireMaxReplyBytes); err != nil {
		return preferContextError(ctx, err)
	}
	kvFrame, err := pdWireReadFrame(conn, pdWireMaxKVFrameBytes)
	if err != nil {
		return preferContextError(ctx, err)
	}
	logitsRaw, err := pdWireReadFrame(conn, pdWireMaxLogitsBytes)
	if err != nil {
		return preferContextError(ctx, err)
	}
	logits, err := pdWireDecodeFloat32(logitsRaw)
	if err != nil {
		return s.replyError(conn, ctx, err)
	}
	for _, x := range logits {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return s.replyError(conn, ctx, fmt.Errorf("%w: non-finite prefill logit", ErrPDWireProtocol))
		}
	}

	laneCtx := ctx
	if s.handoffTimeout() > 0 {
		var cancelFn context.CancelFunc
		laneCtx, cancelFn = context.WithTimeout(ctx, s.handoffTimeout())
		defer cancelFn()
	}

	var observe func(int) error
	if header.StreamTokens {
		observe = func(tok int) error {
			frame := pdWireStreamFrame{Kind: pdWireFrameToken, Token: tok}
			raw, jerr := json.Marshal(frame)
			if jerr != nil {
				return jerr
			}
			return pdWireWriteFrame(conn, raw, pdWireMaxReplyBytes)
		}
	}

	res, err := s.admitImported(laneCtx, &header, kvFrame, logits, observe)
	if err != nil {
		return s.replyError(conn, ctx, err)
	}
	if header.StreamTokens {
		// The tokens already flowed as frames; the terminal frame carries only
		// the handoff metadata.
		reply := pdWireStreamFrame{
			Kind:         pdWireFrameReply,
			Protocol:     PDWireProtocol,
			DecodeWorker: res.DecodeWorker,
			SpanDigest:   res.SpanDigest,
			BytesMoved:   res.BytesMoved,
		}
		replyJSON, jerr := json.Marshal(reply)
		if jerr != nil {
			return jerr
		}
		return preferContextError(ctx, pdWireWriteFrame(conn, replyJSON, pdWireMaxReplyBytes))
	}
	reply := pdWireReply{
		Protocol:     PDWireProtocol,
		DecodeWorker: res.DecodeWorker,
		Tokens:       res.Tokens,
		SpanDigest:   res.SpanDigest,
		BytesMoved:   res.BytesMoved,
	}
	replyJSON, jerr := json.Marshal(reply)
	if jerr != nil {
		return jerr
	}
	return preferContextError(ctx, pdWireWriteFrame(conn, replyJSON, pdWireMaxReplyBytes))
}

// validateHeader checks the header against this cluster's own identity before
// any payload byte is spent. Every refusal is typed: protocol shape problems
// wrap ErrPDWireProtocol, identity mismatches wrap ErrPDWireIdentity.
func (s *PDWireServer) validateHeader(h *pdWireHeader) error {
	if h.Protocol != PDWireProtocol {
		return fmt.Errorf("%w: version %q", ErrPDWireProtocol, h.Protocol)
	}
	m := s.cluster.m
	if h.ConfigSHA256 != pdWireConfigDigest(m.Cfg) {
		return fmt.Errorf("%w: producer and consumer model configurations differ", ErrPDWireIdentity)
	}
	if h.VocabSize != m.Cfg.VocabSize {
		return fmt.Errorf("%w: producer vocabulary %d, consumer %d", ErrPDWireIdentity, h.VocabSize, m.Cfg.VocabSize)
	}
	if len(h.PromptIDs) == 0 || len(h.PromptIDs) > PDWireMaxPromptTokens {
		return fmt.Errorf("%w: prompt must carry 1..%d tokens", ErrPDWireProtocol, PDWireMaxPromptTokens)
	}
	for _, id := range h.PromptIDs {
		if id < 0 || id >= m.Cfg.VocabSize {
			return fmt.Errorf("%w: token ID %d outside model vocabulary [0,%d)", ErrPDWireProtocol, id, m.Cfg.VocabSize)
		}
	}
	if pdWirePromptDigest(h.PromptIDs) != h.PromptSHA256 {
		return fmt.Errorf("%w: prompt token digest", ErrPDWireIdentity)
	}
	if h.KVFrameBytes < 1 || uint64(h.KVFrameBytes) > pdWireMaxKVFrameBytes {
		return fmt.Errorf("%w: declared KV frame length %d", ErrPDWireProtocol, h.KVFrameBytes)
	}
	if h.LogitsBytes < 4 || h.LogitsBytes%4 != 0 || uint64(h.LogitsBytes) > pdWireMaxLogitsBytes {
		return fmt.Errorf("%w: declared logits length %d", ErrPDWireProtocol, h.LogitsBytes)
	}
	if h.LogitsBytes != 4*h.VocabSize {
		return fmt.Errorf("%w: logits frame %d bytes, want %d for vocabulary %d", ErrPDWireProtocol, h.LogitsBytes, 4*h.VocabSize, h.VocabSize)
	}
	// The lane retirement budget is bounded like the prompt itself: a hostile
	// or misconfigured producer cannot force unbounded decode work on this
	// role. Zero keeps the engine-wide default.
	if h.GenBudget < 0 || h.GenBudget > PDWireMaxPromptTokens {
		return fmt.Errorf("%w: gen budget %d outside 0..%d", ErrPDWireProtocol, h.GenBudget, PDWireMaxPromptTokens)
	}
	return nil
}

// admitImported is the decode half of the two-process Admit: ingest the paged
// frame into the routed decode worker's pool, admit the imported cache through
// AdmitImported (no re-prefill), and drain the generated tokens. It mirrors
// NativePDCluster.Admit's post-transfer bookkeeping: ownership stamped onto
// the receiving worker, import recorded before admission, residency observed
// after it. When observe is non-nil (streaming handoff) every generated token
// is forwarded to it BEFORE the drain continues; an observe failure aborts the
// handoff and retires the lane through the connection context.
func (s *PDWireServer) admitImported(ctx context.Context, header *pdWireHeader, kvFrame []byte, logits []float32, observe func(int) error) (PDShipResult, error) {
	c := s.cluster
	prompt := append([]int(nil), header.PromptIDs...)
	segments := []string{NativePDPrefixKey(prompt)}
	decode, _, _ := c.routeDecode(segments)

	receipt, err := model.UnmarshalPagedKVTransfer(decode.pool, kvFrame)
	if err != nil {
		return PDShipResult{}, fmt.Errorf("modelengine: P/D wire ingest: %w", err)
	}
	freed := false
	freeReceipt := func() {
		if !freed {
			freed = true
			receipt.KV.Free()
		}
	}
	defer freeReceipt()

	cache := receipt.KV.ToKVCache(c.m.Cfg)
	tool := "pd_serve"
	call := &abi.ToolCall{Tool: tool}
	r, err := decode.sched.AdmitImported(ctx, call, ImportedSequence{
		Session:   &model.Session{M: c.m, Cache: cache},
		Logits:    append([]float32(nil), logits...),
		Prompt:    prompt,
		PromptLen: len(prompt),
		Tool:      tool,
		GenBudget: header.GenBudget,
	})
	if err != nil {
		return PDShipResult{}, fmt.Errorf("modelengine: P/D wire import admission: %w", err)
	}

	// Stamp the receiving worker's ownership and the wire backend label, then
	// record the import exactly as the in-process Admit does.
	if receipt.Transfer.Owner == "" {
		receipt.Transfer.Owner = decode.name
	}
	if receipt.Transfer.Backend == "" {
		receipt.Transfer.Backend = string(model.KVTransferBackendTCP)
	}
	c.observeTransferBackend(receipt.Transfer.Backend)
	decode.addImport(receipt)

	tokens := make([]int, 0, PDGenTokens)
	for tok := range r.Tokens() {
		if observe != nil {
			if err := observe(tok.ID); err != nil {
				// The consumer stopped reading: stop spending decode steps for
				// it. The import was already recorded, so the worker owns the
				// span (suppress the deferred free), and the lane retires when
				// this connection's context cancels.
				freed = true
				return PDShipResult{}, err
			}
		}
		tokens = append(tokens, tok.ID)
	}
	if _, err := r.Result(); err != nil {
		return PDShipResult{}, fmt.Errorf("modelengine: P/D wire decode lane: %w", err)
	}

	c.mu.Lock()
	c.residency.Observe(decode.name, segments)
	c.lastDecodeName = decode.name
	c.mu.Unlock()

	freed = true // the decode worker now owns the span (addImport stored it)
	return PDShipResult{
		Tokens:       tokens,
		DecodeWorker: decode.name,
		SpanDigest:   receipt.Transfer.SpanDigest,
		BytesMoved:   receipt.Transfer.BytesMoved,
	}, nil
}

func (s *PDWireServer) handoffTimeout() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.timeout
}

// replyError sends the closed-code error reply, matching the fak-pdbench/v1
// convention that a failed handoff still terminates with a structured frame
// whenever the socket allows it.
func (s *PDWireServer) replyError(conn net.Conn, ctx context.Context, err error) error {
	reply := pdWireReply{Protocol: PDWireProtocol, ErrorCode: pdWireErrorCode(err), Error: err.Error()}
	replyJSON, jerr := json.Marshal(reply)
	if jerr != nil {
		return jerr
	}
	if werr := pdWireWriteFrame(conn, replyJSON, pdWireMaxReplyBytes); werr != nil {
		return preferContextError(ctx, errors.Join(err, werr))
	}
	return err
}

// replyAckError sends the negative acknowledgement carrying the closed error
// code, ending the handoff before any payload byte moves.
func (s *PDWireServer) replyAckError(conn net.Conn, ctx context.Context, err error) error {
	ack := pdWireAck{OK: false, Code: pdWireErrorCode(err), Error: err.Error()}
	ackJSON, jerr := json.Marshal(ack)
	if jerr != nil {
		return jerr
	}
	if werr := pdWireWriteFrame(conn, ackJSON, pdWireMaxReplyBytes); werr != nil {
		return preferContextError(ctx, errors.Join(err, werr))
	}
	return err
}

func pdWireErrorCode(err error) string {
	switch {
	case errors.Is(err, ErrPDWireIdentity):
		return pdWireCodeIdentity
	case errors.Is(err, ErrPDWireProtocol):
		return pdWireCodeProtocol
	default:
		return pdWireCodeRuntime
	}
}

// decodePDWireError maps a remote closed code back onto the local sentinel,
// keeping errors.Is usable across the socket.
func decodePDWireError(code, detail string) error {
	switch code {
	case pdWireCodeIdentity:
		return fmt.Errorf("%w: remote: %s", ErrPDWireIdentity, detail)
	case pdWireCodeProtocol:
		return fmt.Errorf("%w: remote: %s", ErrPDWireProtocol, detail)
	default:
		return fmt.Errorf("modelengine: P/D wire remote runtime error: %s", detail)
	}
}

// interruptConnOnCancel arms a deadline the instant ctx is done, so a blocked
// socket read/write wakes and the handoff tears down — the cancellation
// cleanup the two-process topology owes a departed peer.
func interruptConnOnCancel(ctx context.Context, conn net.Conn) {
	_ = context.AfterFunc(ctx, func() {
		_ = conn.SetDeadline(time.Now())
	})
}

// preferContextError reports the context error when the context lost, so a
// cancelled handoff never masquerades as a transport fault.
func preferContextError(ctx context.Context, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	return err
}

// WithNativePDSessionHook decorates every session the cluster mints — the
// prefill-half session and every decode lane session (fresh and restored) —
// before use, e.g. to bind the device backend a serve role resolved. Set once
// through the constructor options; nil keeps the historical session shape.
func WithNativePDSessionHook(hook func(*model.Session)) NativePDOption {
	return func(c *NativePDCluster) {
		c.sessionHook = hook
	}
}
