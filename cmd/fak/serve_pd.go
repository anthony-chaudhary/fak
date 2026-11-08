package main

// `fak serve pd` is the two-process P/D disaggregation front door: one process
// runs the prefill half behind an OpenAI-compatible chat endpoint and ships
// the paged-KV frame to a peer process, whose decode workers admit the
// imported KV through AdmitImported and generate the reply tokens. Both roles
// load the same checkpoint; the wire handshake binds the two configurations
// together, so a mispaired pair fails closed at the header.
//
// Prefill role (front door):
//
//	fak serve pd --role prefill --gguf <path> --listen 127.0.0.1:8090 \
//	    --decode-addr 127.0.0.1:8091[,127.0.0.1:8092,...] \
//	    [--decode-control 127.0.0.1:8094[,127.0.0.1:8095,...]] \
//	    [--backend auto] [--q4k] [--tokenizer <path>]
//
// Decode role (KV receiver):
//
//	fak serve pd --role decode --gguf <path> --listen 127.0.0.1:8091 \
//	    [--decode-workers 2] [--backend auto] [--q4k] [--http-addr 127.0.0.1:8092]
//
// Each role's --listen is the plane that role owns: the prefill role listens
// for HTTP (chat completions + /healthz + /metrics), the decode role listens
// for the P/D wire data plane. The decode role may additionally expose an HTTP
// control plane (--http-addr) with /healthz + /metrics.
//
// The prefill role may front N decode peers: --decode-addr takes a comma list
// of wire addresses and --decode-control the parallel comma list of each
// peer's control-plane address. Every handoff routes to the peer with the
// fewest active imported lanes (scraped from its control plane /metrics),
// ties broken by lowest peer index, with one bounded retry on a different
// peer before the typed 502 handoff_failed.

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// Closed serve-pd configuration sentinels. Every operator-facing refusal in
// this verb wraps exactly one of them; callers may use errors.Is.
var (
	// ErrServePDRoleInvalid marks a --role value outside {prefill, decode}.
	ErrServePDRoleInvalid = errors.New(`fak serve pd: --role must be "prefill" or "decode"`)
	// ErrServePDMissingGGUF marks a role launched without --gguf.
	ErrServePDMissingGGUF = errors.New("fak serve pd: --gguf is required (both roles load the same checkpoint)")
	// ErrServePDMissingListen marks a role launched without --listen.
	ErrServePDMissingListen = errors.New("fak serve pd: --listen is required")
	// ErrServePDMissingDecodeAddr marks a prefill role with no --decode-addr to
	// ship its paged-KV frames to.
	ErrServePDMissingDecodeAddr = errors.New("fak serve pd: the prefill role requires --decode-addr (the decode role's wire listener)")
	// ErrServePDBadDecodeAddr marks a --decode-addr / --decode-control entry
	// that is not a host:port address.
	ErrServePDBadDecodeAddr = errors.New("fak serve pd: every --decode-addr and --decode-control entry must be a host:port address")
	// ErrServePDBadControlAddrs marks a --decode-control list that is neither
	// empty nor parallel to --decode-addr.
	ErrServePDBadControlAddrs = errors.New("fak serve pd: --decode-control must be empty or carry one address per --decode-addr entry")
	// ErrServePDBadDecodeWorkers marks a decode role whose --decode-workers is
	// below the one-worker minimum.
	ErrServePDBadDecodeWorkers = errors.New("fak serve pd: --decode-workers must be >= 1")
	// ErrServePDTokenizerUnavailable marks a prefill role whose checkpoint and
	// --tokenizer selection yield no usable tokenizer.
	ErrServePDTokenizerUnavailable = errors.New("fak serve pd: no usable tokenizer (pass --tokenizer or use a GGUF with an embedded BPE tokenizer)")
)

// servePDConfig is the fully parsed flag set for one serve-pd role.
type servePDConfig struct {
	Role          string
	GGUF          string
	Listen        string
	DecodeAddrs   []string // prefill role: the decode peers' wire listeners (peer index = position)
	ControlAddrs  []string // prefill role: parallel control-plane addresses; empty = no lane counters
	DecodeWorkers int      // decode role: worker count
	BackendName   string
	Q4K           bool
	TokenizerPath string
	HTTPAddr      string // decode role: optional HTTP control plane
}

// parseCSVHostPortList splits a comma list into trimmed host:port entries,
// dropping empties. An entry that does not split into host:port reports
// ErrServePDBadDecodeAddr wrapped with the offending value.
func parseCSVHostPortList(flag, raw string) ([]string, error) {
	var out []string
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if _, _, err := net.SplitHostPort(part); err != nil {
			return nil, fmt.Errorf("%w: %s entry %q is not host:port", ErrServePDBadDecodeAddr, flag, part)
		}
		out = append(out, part)
	}
	return out, nil
}

// validateServePD is the pure closed-contract check over the parsed flags. It
// reports the first violation as a wrapped sentinel; nil means bootable.
func validateServePD(cfg servePDConfig) error {
	switch cfg.Role {
	case "prefill":
	case "decode":
	default:
		return fmt.Errorf("%w: got %q", ErrServePDRoleInvalid, cfg.Role)
	}
	if cfg.GGUF == "" {
		return ErrServePDMissingGGUF
	}
	if cfg.Listen == "" {
		return ErrServePDMissingListen
	}
	switch cfg.Role {
	case "prefill":
		if len(cfg.DecodeAddrs) == 0 {
			return ErrServePDMissingDecodeAddr
		}
		if len(cfg.ControlAddrs) > 0 && len(cfg.ControlAddrs) != len(cfg.DecodeAddrs) {
			return fmt.Errorf("%w: %d control address(es) for %d decode peer(s)",
				ErrServePDBadControlAddrs, len(cfg.ControlAddrs), len(cfg.DecodeAddrs))
		}
	case "decode":
		if cfg.DecodeWorkers <= 0 {
			return ErrServePDBadDecodeWorkers
		}
	}
	return nil
}

// runServePD is the `fak serve pd` subverb entry point: parse, validate, boot
// the role, and block until a termination signal.
func runServePD(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("serve pd", flag.ContinueOnError)
	fs.SetOutput(stderr)
	role := fs.String("role", "", `P/D role: "prefill" (front door + prefill half) or "decode" (KV receiver)`)
	gguf := fs.String("gguf", "", "GGUF checkpoint path; both roles must load the same artifact")
	listen := fs.String("listen", "", "this role's listen address (prefill: HTTP front door; decode: P/D wire data plane)")
	decodeAddrs := fs.String("decode-addr", "", "comma-separated decode peer wire addresses (prefill role only)")
	controlAddrs := fs.String("decode-control", "", "comma-separated decode peer control-plane addresses, parallel to --decode-addr (prefill role only)")
	decodeWorkers := fs.Int("decode-workers", 1, "decode-role worker count (decode role only)")
	backendName := fs.String("backend", "", `device backend: one of the registered names, "auto", or "cpu"`)
	q4k := fs.Bool("q4k", os.Getenv("FAK_Q4K") != "", "load the resident Q4_K arm (default: FAK_Q4K env)")
	tokPath := fs.String("tokenizer", "", "explicit tokenizer.json or directory (prefill role; default: the GGUF's embedded tokenizer)")
	httpAddr := fs.String("http-addr", "", "optional HTTP control plane for the decode role (/healthz + /metrics)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	wireAddrs, err := parseCSVHostPortList("--decode-addr", *decodeAddrs)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	ctrlAddrs, err := parseCSVHostPortList("--decode-control", *controlAddrs)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	cfg := servePDConfig{
		Role:          *role,
		GGUF:          pathutil.ExpandTilde(*gguf),
		Listen:        *listen,
		DecodeAddrs:   wireAddrs,
		ControlAddrs:  ctrlAddrs,
		DecodeWorkers: *decodeWorkers,
		BackendName:   *backendName,
		Q4K:           *q4k,
		TokenizerPath: pathutil.ExpandTilde(*tokPath),
		HTTPAddr:      *httpAddr,
	}
	if err := validateServePD(cfg); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	return runServePDRole(stdout, stderr, cfg)
}

// runServePDRole boots one validated role. It blocks until a termination
// signal; 0 is a clean shutdown, 1 a runtime fault, 2 a configuration fault.
func runServePDRole(stdout, stderr io.Writer, cfg servePDConfig) int {
	backend, err := resolveServeChatBackend(cfg.BackendName)
	if err != nil {
		writeBackendUnavailableBail(stderr, "fak serve pd", cfg.BackendName)
		return 2
	}

	var m *fakmodel.Model
	var loadErr error
	if cfg.Q4K {
		m, loadErr = ggufload.LoadModelQ4K(cfg.GGUF)
	} else {
		m, loadErr = ggufload.LoadModelQuant(cfg.GGUF)
	}
	if loadErr != nil {
		fmt.Fprintf(stderr, "fak serve pd: model load %s: %v\n", cfg.GGUF, loadErr)
		return 1
	}
	defer m.CloseWeights()

	// The resolved device backend binds every session the cluster mints — the
	// prefill-half session and every decode lane session — symmetrically across
	// roles. A nil backend keeps the historical session shape byte-for-byte.
	var opts []modelengine.NativePDOption
	if backend != nil {
		q4k := cfg.Q4K
		opts = append(opts, modelengine.WithNativePDSessionHook(func(sess *fakmodel.Session) {
			sess.Backend = backend
			sess.Quant = true
			sess.Q4K = q4k
		}))
	}

	decodeWorkers := 1 // the prefill role carries one nominal (unused locally) decode worker
	if cfg.Role == "decode" {
		decodeWorkers = cfg.DecodeWorkers
	}
	cluster := modelengine.NewNativePDClusterWithResidency(m, decodeWorkers, nil, opts...)
	defer cluster.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if cfg.Role == "prefill" {
		return runServePDPrefillRole(ctx, stdout, stderr, cfg, cluster)
	}
	return runServePDDecodeRole(ctx, stdout, stderr, cfg, cluster)
}

// runServePDPrefillRole owns the HTTP front door: chat completions ride
// PrefillForWire + the wire transport, plus /healthz and /metrics.
func runServePDPrefillRole(ctx context.Context, stdout, stderr io.Writer, cfg servePDConfig, cluster *modelengine.NativePDCluster) int {
	var tok *tokenizer.Tokenizer
	var tokErr error
	if cfg.TokenizerPath != "" {
		tok, tokErr = loadExplicitServeTokenizer(cfg.TokenizerPath)
	} else {
		tok, tokErr = embeddedGGUFTokenizer(cfg.GGUF)
	}
	if tokErr != nil || tok == nil {
		if tokErr == nil {
			tokErr = errors.New("no tokenizer resolved")
		}
		fmt.Fprintf(stderr, "%v: %v\n", ErrServePDTokenizerUnavailable, tokErr)
		return 2
	}

	peers := make([]modelengine.PDDecodePeer, len(cfg.DecodeAddrs))
	for i, addr := range cfg.DecodeAddrs {
		peers[i] = modelengine.PDDecodePeer{WireAddr: addr}
		if i < len(cfg.ControlAddrs) {
			peers[i].ControlAddr = cfg.ControlAddrs[i]
		}
	}
	router := modelengine.NewPDPeerRouter(cluster, peers, nil)
	h := &servePDChatHandler{
		cluster:          cluster,
		ship:             router,
		encode:           tok.Encode,
		decode:           tok.Decode,
		newStreamDecoder: func() pdStreamDecoder { return tok.NewIncrementalDecoder() },
		modelLabel:       "fak-serve-pd",
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", h.handleChat)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = io.WriteString(w, "ok\n")
	})
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		_, _ = io.WriteString(w, cluster.Metrics())
		_, _ = io.WriteString(w, router.Metrics())
	})
	ln, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		fmt.Fprintf(stderr, "fak serve pd: listen %s: %v\n", cfg.Listen, err)
		return 1
	}
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	errCh := make(chan error, 1)
	go func() { errCh <- srv.Serve(ln) }()

	fmt.Fprintf(stdout, "fak serve pd: prefill role listening on %s; routing across %d decode peer(s) [%s]; vocab=%d\n",
		cfg.Listen, len(peers), strings.Join(cfg.DecodeAddrs, ", "), cluster.VocabSize())
	select {
	case <-ctx.Done():
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Fprintf(stderr, "fak serve pd: front door: %v\n", err)
			return 1
		}
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
	return 0
}

// runServePDDecodeRole owns the wire data plane: one accepted connection per
// handoff, ingested through AdmitImported with no re-prefill. An optional HTTP
// control plane (--http-addr) exposes /healthz + /metrics.
func runServePDDecodeRole(ctx context.Context, stdout, stderr io.Writer, cfg servePDConfig, cluster *modelengine.NativePDCluster) int {
	wire := modelengine.NewPDWireServer(cluster)
	if err := wire.Listen(cfg.Listen); err != nil {
		fmt.Fprintf(stderr, "fak serve pd: %v\n", err)
		return 1
	}
	wireErr := make(chan error, 1)
	go func() { wireErr <- wire.Serve() }()

	var httpErr chan error
	var srv *http.Server
	if cfg.HTTPAddr != "" {
		mux := http.NewServeMux()
		mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			_, _ = io.WriteString(w, "ok\n")
		})
		mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
			_, _ = io.WriteString(w, cluster.Metrics())
		})
		hl, err := net.Listen("tcp", cfg.HTTPAddr)
		if err != nil {
			fmt.Fprintf(stderr, "fak serve pd: listen %s: %v\n", cfg.HTTPAddr, err)
			return 1
		}
		srv = &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}
		httpErr = make(chan error, 1)
		go func() { httpErr <- srv.Serve(hl) }()
	}

	fmt.Fprintf(stdout, "fak serve pd: decode role listening on %s with %d worker(s)\n",
		cfg.Listen, cfg.DecodeWorkers)
	select {
	case <-ctx.Done():
	case err := <-wireErr:
		if err != nil {
			fmt.Fprintf(stderr, "fak serve pd: wire data plane: %v\n", err)
			return 1
		}
	}
	if httpErr != nil {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}
	_ = wire.Close()
	return 0
}

// ---- chat front door -------------------------------------------------------

const pdChatMaxBodyBytes = 1 << 20 // 1 MiB request cap

// Closed chat-front-door error codes. Clients branch on code, never prose.
const (
	pdChatCodeInvalidRequest = "invalid_request"
	pdChatCodePrefillFailed  = "prefill_failed"
	pdChatCodeHandoffFailed  = "handoff_failed"
	pdChatCodeInternal       = "internal"
)

type pdChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type pdChatRequest struct {
	Model     string          `json:"model"`
	Messages  []pdChatMessage `json:"messages"`
	Stream    bool            `json:"stream"`
	MaxTokens int             `json:"max_tokens"`
}

type pdChatChoice struct {
	Index        int           `json:"index"`
	Message      pdChatMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type pdChatReply struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []pdChatChoice `json:"choices"`
}

type pdChatErrorBody struct {
	Error struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

// SSE (stream:true) reply shape: OpenAI chat.completion.chunk frames, one per
// generated token, a final frame carrying finish_reason + usage, then the
// terminal "data: [DONE]" line. Clients that read usage (the h2h bench client)
// take it from any chunk that carries one.
type pdChatStreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type pdChatStreamChoice struct {
	Index        int               `json:"index"`
	Delta        pdChatStreamDelta `json:"delta"`
	FinishReason *string           `json:"finish_reason"` // null until the final frame
}

type pdChatStreamUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type pdChatStreamChunk struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []pdChatStreamChoice   `json:"choices"`
	Usage   *pdChatStreamUsage     `json:"usage,omitempty"`
	Error   *pdChatStreamErrorBody `json:"error,omitempty"`
}

type pdChatStreamErrorBody struct {
	Message string `json:"message"`
	Code    string `json:"code"`
}

// pdStreamDecoder turns generated token ids into safe-to-emit text deltas,
// holding bytes that end in a not-yet-complete UTF-8 code point. The loaded
// tokenizer's incremental decoder satisfies it; front-door tests inject a
// byte-level stand-in.
type pdStreamDecoder interface {
	Push(id int) (string, error)
	Finalize() (string, error)
}

// pdSSEWriter emits server-sent events on an http.ResponseWriter and reports
// write failures so a dead client stops the handoff instead of decoding into
// the void.
type pdSSEWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	started bool // the status line has committed with the first event
}

func newPDSSEResponse(w http.ResponseWriter) (*pdSSEWriter, error) {
	fl, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("response writer does not support flushing")
	}
	// Headers are set now but the status line only commits with the first
	// event, so a prefill or handoff failure BEFORE any token can still answer
	// with the typed JSON error body.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	return &pdSSEWriter{w: w, flusher: fl}, nil
}

// event writes one "data: <payload>\n\n" event and flushes it. The first
// event commits the 200 status line, arming started, which tells the caller
// no JSON error body can follow from here on.
func (s *pdSSEWriter) event(payload []byte) error {
	if !s.started {
		s.w.WriteHeader(http.StatusOK)
		s.started = true
	}
	if _, err := s.w.Write(append([]byte("data: "), payload...)); err != nil {
		return err
	}
	if _, err := s.w.Write([]byte("\n\n")); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

func (s *pdSSEWriter) chunk(chunk pdChatStreamChunk) error {
	raw, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	return s.event(raw)
}

func (s *pdSSEWriter) done() error { return s.event([]byte("[DONE]")) }

func writePDChatError(w http.ResponseWriter, status int, code, message string) {
	body := pdChatErrorBody{}
	body.Error.Code = code
	body.Error.Message = message
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// servePDChatHandler serves POST /v1/chat/completions on the prefill role. The
// encode/decode seams stand in for the loaded tokenizer so the contract is
// exercisable without a checkpoint; ship is the P/D handoff transport (the
// peer router in production); newStreamDecoder mints one incremental decoder
// per streamed request.
type servePDChatHandler struct {
	cluster          *modelengine.NativePDCluster
	ship             modelengine.PDHandoffTransportStream
	encode           func(string) ([]int, error)
	decode           func([]int) (string, error)
	newStreamDecoder func() pdStreamDecoder
	modelLabel       string

	mu    sync.Mutex
	reqNo uint64
}

func (h *servePDChatHandler) nextID() string {
	h.mu.Lock()
	h.reqNo++
	n := h.reqNo
	h.mu.Unlock()
	return fmt.Sprintf("pd-%d", n)
}

// handleChat runs one chat completion: tokenize the last user message,
// prefill, route the paged-KV frame to a decode peer, and detokenize the
// generated tokens. stream:true answers over SSE (handleChatStream); otherwise
// the whole reply arrives in one JSON body. Truncation honors max_tokens with
// finish_reason "length"; otherwise the lane retires with "stop".
func (h *servePDChatHandler) handleChat(w http.ResponseWriter, r *http.Request) {
	fail := func(status int, code, msg string) { writePDChatError(w, status, code, msg) }

	var req pdChatRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, pdChatMaxBodyBytes))
	if err := dec.Decode(&req); err != nil {
		fail(http.StatusBadRequest, pdChatCodeInvalidRequest, "request body is not valid chat-completions JSON")
		return
	}
	content := ""
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			content = req.Messages[i].Content
			break
		}
	}
	ids, err := h.encode(content)
	if err != nil {
		fail(http.StatusBadRequest, pdChatCodeInvalidRequest, "prompt could not be tokenized")
		return
	}
	if len(ids) == 0 {
		fail(http.StatusBadRequest, pdChatCodeInvalidRequest, "the last user message tokenizes to zero tokens")
		return
	}

	pdReq := modelengine.NativePDRequest{
		Call:             &abi.ToolCall{Tool: "chat_completions"},
		Prompt:           ids,
		ModelID:          h.modelLabel,
		TokenizerID:      "ggml-bpe",
		Taint:            abi.TaintTrusted,
		Scope:            abi.ScopeFleet,
		AdmissionVerdict: cachemeta.AdmissionAllow,
		AdmittedBy:       "fak-serve-pd",
	}
	frame, logits, err := h.cluster.PrefillForWire(r.Context(), pdReq)
	if err != nil {
		fail(http.StatusInternalServerError, pdChatCodePrefillFailed, "prefill half failed")
		return
	}
	// max_tokens rides the wire as the imported lane's retirement budget, so a
	// long answer stops burning decode steps at the caller's cap instead of
	// post-truncating a longer generation.
	shipReq := modelengine.PDShipRequest{KVFrame: frame, Logits: logits, Prompt: ids, GenBudget: req.MaxTokens}
	if req.Stream {
		h.handleChatStream(w, r, &req, ids, shipReq)
		return
	}
	res, err := h.ship.Ship(r.Context(), shipReq)
	if err != nil {
		fail(http.StatusBadGateway, pdChatCodeHandoffFailed, "the decode role refused or dropped the handoff")
		return
	}

	n := len(res.Tokens)
	finish := "stop"
	if req.MaxTokens > 0 && n > req.MaxTokens {
		n = req.MaxTokens
		finish = "length"
	}
	text, err := h.decode(res.Tokens[:n])
	if err != nil {
		fail(http.StatusInternalServerError, pdChatCodeInternal, "generated tokens could not be detokenized")
		return
	}
	reply := pdChatReply{
		ID:      h.nextID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   h.modelLabel,
		Choices: []pdChatChoice{{Index: 0, Message: pdChatMessage{Role: "assistant", Content: text}, FinishReason: finish}},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(reply)
}

// handleChatStream serves a stream:true completion over SSE: each generated
// token streams out as the decode lane produces it, and the final chunk
// carries finish_reason plus the usage block the bench clients read. A
// prefill or handoff failure BEFORE the first event still answers with the
// typed JSON error; a failure mid-stream emits an SSE error event and closes
// without [DONE], because the status line is already committed.
func (h *servePDChatHandler) handleChatStream(w http.ResponseWriter, r *http.Request, req *pdChatRequest, ids []int, shipReq modelengine.PDShipRequest) {
	if h.newStreamDecoder == nil {
		writePDChatError(w, http.StatusInternalServerError, pdChatCodeInternal, "no incremental decoder configured")
		return
	}
	sse, err := newPDSSEResponse(w)
	if err != nil {
		writePDChatError(w, http.StatusInternalServerError, pdChatCodeInternal, "response writer cannot stream")
		return
	}
	id := h.nextID()
	created := time.Now().Unix()
	incr := h.newStreamDecoder()
	limit := req.MaxTokens // 0 = no request-level cap

	emitted := 0
	firstDelta := true
	sink := func(tok int) error {
		delta, derr := incr.Push(tok)
		if derr != nil {
			return derr
		}
		if delta == "" || (limit > 0 && emitted >= limit) {
			return nil
		}
		chunk := pdChatStreamChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   h.modelLabel,
			Choices: []pdChatStreamChoice{{Index: 0, Delta: pdChatStreamDelta{Content: delta}}},
		}
		if firstDelta {
			chunk.Choices[0].Delta.Role = "assistant"
			firstDelta = false
		}
		emitted++
		return sse.chunk(chunk)
	}

	res, err := h.ship.ShipStream(r.Context(), shipReq, sink)
	if err != nil {
		if errors.Is(err, modelengine.ErrPDStreamConsumerStopped) {
			return // the response writer failed or the client left; nothing more can be said
		}
		if sse.started {
			_ = sse.chunk(pdChatStreamChunk{
				ID:      id,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   h.modelLabel,
				Error:   &pdChatStreamErrorBody{Message: "the decode role refused or dropped the handoff", Code: pdChatCodeHandoffFailed},
			})
			return
		}
		writePDChatError(w, http.StatusBadGateway, pdChatCodeHandoffFailed, "the decode role refused or dropped the handoff")
		return
	}

	// Flush any UTF-8 tail belonging to an already-counted token.
	if tail, ferr := incr.Finalize(); ferr == nil && tail != "" {
		_ = sse.chunk(pdChatStreamChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   h.modelLabel,
			Choices: []pdChatStreamChoice{{Index: 0, Delta: pdChatStreamDelta{Content: tail}}},
		})
	}

	completed := len(res.Tokens)
	finish := "stop"
	if limit > 0 && completed > limit {
		completed = limit
		finish = "length"
	}
	_ = sse.chunk(pdChatStreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   h.modelLabel,
		Choices: []pdChatStreamChoice{{Index: 0, FinishReason: &finish}},
		Usage:   &pdChatStreamUsage{PromptTokens: len(ids), CompletionTokens: completed, TotalTokens: len(ids) + completed},
	})
	_ = sse.done()
}
