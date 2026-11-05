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
//	    --decode-addr 127.0.0.1:8091 [--backend auto] [--q4k] [--tokenizer <path>]
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
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
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
	DecodeAddr    string // prefill role: the decode role's wire listener
	DecodeWorkers int    // decode role: worker count
	BackendName   string
	Q4K           bool
	TokenizerPath string
	HTTPAddr      string // decode role: optional HTTP control plane
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
		if cfg.DecodeAddr == "" {
			return ErrServePDMissingDecodeAddr
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
	decodeAddr := fs.String("decode-addr", "", "the decode role's wire address (prefill role only)")
	decodeWorkers := fs.Int("decode-workers", 1, "decode-role worker count (decode role only)")
	backendName := fs.String("backend", "", `device backend: one of the registered names, "auto", or "cpu"`)
	q4k := fs.Bool("q4k", os.Getenv("FAK_Q4K") != "", "load the resident Q4_K arm (default: FAK_Q4K env)")
	tokPath := fs.String("tokenizer", "", "explicit tokenizer.json or directory (prefill role; default: the GGUF's embedded tokenizer)")
	httpAddr := fs.String("http-addr", "", "optional HTTP control plane for the decode role (/healthz + /metrics)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	cfg := servePDConfig{
		Role:          *role,
		GGUF:          *gguf,
		Listen:        *listen,
		DecodeAddr:    *decodeAddr,
		DecodeWorkers: *decodeWorkers,
		BackendName:   *backendName,
		Q4K:           *q4k,
		TokenizerPath: *tokPath,
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

	ship := modelengine.NewPDWireClient(cluster, cfg.DecodeAddr, 10*time.Second)
	h := &servePDChatHandler{
		cluster:    cluster,
		ship:       ship,
		encode:     tok.Encode,
		decode:     tok.Decode,
		modelLabel: "fak-serve-pd",
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
	})
	ln, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		fmt.Fprintf(stderr, "fak serve pd: listen %s: %v\n", cfg.Listen, err)
		return 1
	}
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	errCh := make(chan error, 1)
	go func() { errCh <- srv.Serve(ln) }()

	fmt.Fprintf(stdout, "fak serve pd: prefill role listening on %s; shipping to %s; vocab=%d\n",
		cfg.Listen, cfg.DecodeAddr, cluster.VocabSize())
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
	pdChatCodeInvalidRequest    = "invalid_request"
	pdChatCodeStreamUnsupported = "stream_not_supported"
	pdChatCodePrefillFailed     = "prefill_failed"
	pdChatCodeHandoffFailed     = "handoff_failed"
	pdChatCodeInternal          = "internal"
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
// exercisable without a checkpoint; ship is the P/D wire transport.
type servePDChatHandler struct {
	cluster    *modelengine.NativePDCluster
	ship       modelengine.PDHandoffTransport
	encode     func(string) ([]int, error)
	decode     func([]int) (string, error)
	modelLabel string

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
// prefill, ship the paged-KV frame to the decode role, and detokenize the
// generated tokens. Truncation honors max_tokens with finish_reason "length";
// otherwise the lane retires with "stop".
func (h *servePDChatHandler) handleChat(w http.ResponseWriter, r *http.Request) {
	fail := func(status int, code, msg string) { writePDChatError(w, status, code, msg) }

	var req pdChatRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, pdChatMaxBodyBytes))
	if err := dec.Decode(&req); err != nil {
		fail(http.StatusBadRequest, pdChatCodeInvalidRequest, "request body is not valid chat-completions JSON")
		return
	}
	if req.Stream {
		fail(http.StatusBadRequest, pdChatCodeStreamUnsupported, "stream:true is not supported by fak serve pd")
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
	res, err := h.ship.Ship(r.Context(), modelengine.PDShipRequest{KVFrame: frame, Logits: logits, Prompt: ids})
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
