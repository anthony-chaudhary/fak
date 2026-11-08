package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

// fak-test:runtime fast est=1s lane=default
func TestServePDValidateClosedContract(t *testing.T) {
	validPrefill := servePDConfig{Role: "prefill", GGUF: "m.gguf", Listen: "127.0.0.1:8090", DecodeAddrs: []string{"127.0.0.1:8091"}}
	validMultiPeer := servePDConfig{
		Role:         "prefill",
		GGUF:         "m.gguf",
		Listen:       "127.0.0.1:8090",
		DecodeAddrs:  []string{"127.0.0.1:8091", "127.0.0.1:8092", "127.0.0.1:8093"},
		ControlAddrs: []string{"127.0.0.1:8094", "127.0.0.1:8095", "127.0.0.1:8096"},
	}
	validDecode := servePDConfig{Role: "decode", GGUF: "m.gguf", Listen: "127.0.0.1:8091", DecodeWorkers: 2}

	if err := validateServePD(validPrefill); err != nil {
		t.Fatalf("valid prefill config refused: %v", err)
	}
	if err := validateServePD(validMultiPeer); err != nil {
		t.Fatalf("valid multi-peer prefill config refused: %v", err)
	}
	if err := validateServePD(validDecode); err != nil {
		t.Fatalf("valid decode config refused: %v", err)
	}

	cases := []struct {
		name     string
		cfg      servePDConfig
		sentinel error
	}{
		{name: "bad role", cfg: servePDConfig{Role: "both", GGUF: "m.gguf", Listen: "l"}, sentinel: ErrServePDRoleInvalid},
		{name: "empty role", cfg: servePDConfig{}, sentinel: ErrServePDRoleInvalid},
		{name: "missing gguf", cfg: servePDConfig{Role: "prefill", Listen: "l", DecodeAddrs: []string{"127.0.0.1:8091"}}, sentinel: ErrServePDMissingGGUF},
		{name: "missing listen", cfg: servePDConfig{Role: "decode", GGUF: "m.gguf"}, sentinel: ErrServePDMissingListen},
		{name: "prefill missing decode addr", cfg: servePDConfig{Role: "prefill", GGUF: "m.gguf", Listen: "l"}, sentinel: ErrServePDMissingDecodeAddr},
		{name: "control list shorter than peers", cfg: servePDConfig{Role: "prefill", GGUF: "m.gguf", Listen: "l", DecodeAddrs: []string{"127.0.0.1:8091", "127.0.0.1:8092"}, ControlAddrs: []string{"127.0.0.1:8094"}}, sentinel: ErrServePDBadControlAddrs},
		{name: "control list longer than peers", cfg: servePDConfig{Role: "prefill", GGUF: "m.gguf", Listen: "l", DecodeAddrs: []string{"127.0.0.1:8091"}, ControlAddrs: []string{"127.0.0.1:8094", "127.0.0.1:8095"}}, sentinel: ErrServePDBadControlAddrs},
		{name: "decode zero workers", cfg: servePDConfig{Role: "decode", GGUF: "m.gguf", Listen: "l", DecodeWorkers: 0}, sentinel: ErrServePDBadDecodeWorkers},
		{name: "decode negative workers", cfg: servePDConfig{Role: "decode", GGUF: "m.gguf", Listen: "l", DecodeWorkers: -1}, sentinel: ErrServePDBadDecodeWorkers},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateServePD(tc.cfg)
			if !errors.Is(err, tc.sentinel) {
				t.Fatalf("validate = %v, want %v", err, tc.sentinel)
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestParseCSVHostPortList(t *testing.T) {
	addrs, err := parseCSVHostPortList("--decode-addr", "127.0.0.1:8091, 127.0.0.1:8092 ,,127.0.0.1:8093")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := []string{"127.0.0.1:8091", "127.0.0.1:8092", "127.0.0.1:8093"}
	if len(addrs) != len(want) {
		t.Fatalf("parsed %d addresses %v, want %v", len(addrs), addrs, want)
	}
	for i := range want {
		if addrs[i] != want[i] {
			t.Fatalf("address[%d] = %q, want %q", i, addrs[i], want[i])
		}
	}
	if got, err := parseCSVHostPortList("--decode-addr", ""); err != nil || len(got) != 0 {
		t.Fatalf("empty list = (%v, %v), want (nil, nil)", got, err)
	}
	if _, err := parseCSVHostPortList("--decode-addr", "not-a-port"); !errors.Is(err, ErrServePDBadDecodeAddr) {
		t.Fatalf("bad entry err = %v, want errors.Is(_, ErrServePDBadDecodeAddr)", err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRunServePDFlagRefusals(t *testing.T) {
	cases := []struct {
		name string
		argv []string
	}{
		{name: "no role", argv: []string{"--gguf", "m.gguf", "--listen", "127.0.0.1:1"}},
		{name: "bogus role", argv: []string{"--role", "midfill", "--gguf", "m.gguf", "--listen", "127.0.0.1:1"}},
		{name: "prefill without decode addr", argv: []string{"--role", "prefill", "--gguf", "m.gguf", "--listen", "127.0.0.1:1"}},
		{name: "malformed decode addr", argv: []string{"--role", "prefill", "--gguf", "m.gguf", "--listen", "127.0.0.1:1", "--decode-addr", "not-a-port"}},
		{name: "control list not parallel to peers", argv: []string{"--role", "prefill", "--gguf", "m.gguf", "--listen", "127.0.0.1:1", "--decode-addr", "127.0.0.1:8091,127.0.0.1:8092", "--decode-control", "127.0.0.1:8094"}},
		{name: "decode with zero workers", argv: []string{"--role", "decode", "--gguf", "m.gguf", "--listen", "127.0.0.1:1", "--decode-workers", "0"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var out, errOut bytes.Buffer
			rc := runServePD(&out, &errOut, tc.argv)
			if rc != 2 {
				t.Fatalf("runServePD(%v) = %d, want 2; stderr=%q", tc.argv, rc, errOut.String())
			}
			if errOut.Len() == 0 {
				t.Fatal("refusal printed nothing to stderr")
			}
		})
	}
}

// pdStubShip is the closed-contract stand-in for the P/D wire transport in
// front-door tests: it records the request and replies with canned tokens.
// shipErr fails before any token; stopAfter + midStreamErr deliver a prefix
// then fail like a dropped peer; a sink error is wrapped as
// ErrPDStreamConsumerStopped exactly like the wire client does.
type pdStubShip struct {
	t            *testing.T
	tokens       []int
	shipErr      error
	stopAfter    int   // deliver this many tokens then stop (0 = all)
	midStreamErr error // returned instead of success when stopAfter truncated
	lastReq      modelengine.PDShipRequest
	gotFrame     bool
}

func (s *pdStubShip) Ship(_ context.Context, req modelengine.PDShipRequest) (modelengine.PDShipResult, error) {
	s.lastReq = req
	s.gotFrame = len(req.KVFrame) > 0
	if s.shipErr != nil {
		return modelengine.PDShipResult{}, s.shipErr
	}
	return modelengine.PDShipResult{Tokens: append([]int(nil), s.tokens...), DecodeWorker: "decode-0"}, nil
}

func (s *pdStubShip) ShipStream(_ context.Context, req modelengine.PDShipRequest, sink modelengine.PDTokenSink) (modelengine.PDShipResult, error) {
	s.lastReq = req
	s.gotFrame = len(req.KVFrame) > 0
	if s.shipErr != nil {
		return modelengine.PDShipResult{}, s.shipErr
	}
	n := len(s.tokens)
	if s.stopAfter > 0 && s.stopAfter < n {
		n = s.stopAfter
	}
	for i := 0; i < n; i++ {
		if err := sink(s.tokens[i]); err != nil {
			return modelengine.PDShipResult{}, fmt.Errorf("%w: %v", modelengine.ErrPDStreamConsumerStopped, err)
		}
	}
	if s.midStreamErr != nil {
		return modelengine.PDShipResult{}, s.midStreamErr
	}
	return modelengine.PDShipResult{Tokens: append([]int(nil), s.tokens...), DecodeWorker: "decode-0"}, nil
}

// pdByteEncode / pdByteDecode map the synthetic 256-vocabulary onto raw bytes,
// keeping the front-door tests free of any real checkpoint.
func pdByteEncode(text string) ([]int, error) {
	ids := make([]int, 0, len(text))
	for _, b := range []byte(text) {
		ids = append(ids, int(b))
	}
	return ids, nil
}

func pdByteDecode(ids []int) (string, error) {
	b := make([]byte, len(ids))
	for i, id := range ids {
		b[i] = byte(id)
	}
	return string(b), nil
}

// pdByteStreamDecoder maps the synthetic 256-vocabulary onto raw bytes: every
// token id is one complete byte, so Push emits immediately and Finalize stays
// empty — the front-door streaming tests stay free of any real checkpoint.
type pdByteStreamDecoder struct {
	pending []byte
}

func (d *pdByteStreamDecoder) Push(id int) (string, error) {
	d.pending = append(d.pending, byte(id))
	out := string(d.pending)
	d.pending = nil
	return out, nil
}

func (d *pdByteStreamDecoder) Finalize() (string, error) { return "", nil }

func newPDTestHandler(t *testing.T, ship modelengine.PDHandoffTransportStream) (*servePDChatHandler, func()) {
	t.Helper()
	m := fakmodel.NewSynthetic(modelengine.SyntheticConfig())
	cluster := modelengine.NewNativePDCluster(m, 1)
	h := &servePDChatHandler{
		cluster:          cluster,
		ship:             ship,
		encode:           pdByteEncode,
		decode:           pdByteDecode,
		newStreamDecoder: func() pdStreamDecoder { return &pdByteStreamDecoder{} },
		modelLabel:       "pd-test",
	}
	return h, func() { cluster.Close() }
}

func postPDChat(t *testing.T, h *servePDChatHandler, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.handleChat(rec, req)
	return rec
}

func pdChatReplyOf(t *testing.T, rec *httptest.ResponseRecorder) pdChatReply {
	t.Helper()
	var rep pdChatReply
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("reply is not chat-completions JSON: %v (%s)", err, rec.Body.String())
	}
	return rep
}

func pdChatErrorCode(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var body pdChatErrorBody
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("error body is not JSON: %v (%s)", err, rec.Body.String())
	}
	return body.Error.Code
}

// fak-test:runtime fast est=2s lane=default
func TestServePDChatHandlerClosedCodes(t *testing.T) {
	t.Run("stream answers SSE not a refusal", func(t *testing.T) {
		want := []byte("ok!")
		ids := make([]int, len(want))
		for i, b := range want {
			ids[i] = int(b)
		}
		ship := &pdStubShip{t: t, tokens: ids}
		h, done := newPDTestHandler(t, ship)
		defer done()
		rec := postPDChat(t, h, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
		if rec.Code != http.StatusOK {
			t.Fatalf("stream = %d, want 200 (%s)", rec.Code, rec.Body.String())
		}
		if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
			t.Fatalf("Content-Type = %q, want text/event-stream", ct)
		}
		body := rec.Body.String()
		if !strings.Contains(body, "data: ") || !strings.Contains(body, "data: [DONE]") {
			t.Fatalf("SSE body missing data frames or [DONE]:\n%s", body)
		}
		if !strings.Contains(body, `"content":"o"`) || !strings.Contains(body, `"content":"k"`) || !strings.Contains(body, `"content":"!"`) {
			t.Fatalf("SSE body missing per-token deltas:\n%s", body)
		}
		if !strings.Contains(body, `"finish_reason":"stop"`) || !strings.Contains(body, `"usage"`) {
			t.Fatalf("SSE body missing final finish/usage chunk:\n%s", body)
		}
	})

	t.Run("malformed body", func(t *testing.T) {
		ship := &pdStubShip{t: t, tokens: []int{1}}
		h, done := newPDTestHandler(t, ship)
		defer done()
		rec := postPDChat(t, h, `{not json`)
		if rec.Code != http.StatusBadRequest || pdChatErrorCode(t, rec) != pdChatCodeInvalidRequest {
			t.Fatalf("malformed body = %d/%s, want 400/%s", rec.Code, pdChatErrorCode(t, rec), pdChatCodeInvalidRequest)
		}
	})

	t.Run("no user message", func(t *testing.T) {
		ship := &pdStubShip{t: t, tokens: []int{1}}
		h, done := newPDTestHandler(t, ship)
		defer done()
		rec := postPDChat(t, h, `{"messages":[{"role":"assistant","content":"hi"}]}`)
		if rec.Code != http.StatusBadRequest || pdChatErrorCode(t, rec) != pdChatCodeInvalidRequest {
			t.Fatalf("no-user-message = %d/%s, want 400/%s", rec.Code, pdChatErrorCode(t, rec), pdChatCodeInvalidRequest)
		}
	})

	t.Run("handoff failure is 502", func(t *testing.T) {
		ship := &pdStubShip{t: t, tokens: []int{1}, shipErr: errors.New("peer down")}
		h, done := newPDTestHandler(t, ship)
		defer done()
		rec := postPDChat(t, h, `{"messages":[{"role":"user","content":"hi"}]}`)
		if rec.Code != http.StatusBadGateway || pdChatErrorCode(t, rec) != pdChatCodeHandoffFailed {
			t.Fatalf("handoff failure = %d/%s, want 502/%s", rec.Code, pdChatErrorCode(t, rec), pdChatCodeHandoffFailed)
		}
	})

	t.Run("success with stop", func(t *testing.T) {
		want := []byte("ok!")
		ids := make([]int, len(want))
		for i, b := range want {
			ids[i] = int(b)
		}
		ship := &pdStubShip{t: t, tokens: ids}
		h, done := newPDTestHandler(t, ship)
		defer done()
		rec := postPDChat(t, h, `{"messages":[{"role":"user","content":"ping"}]}`)
		if rec.Code != http.StatusOK {
			t.Fatalf("success = %d, want 200 (%s)", rec.Code, rec.Body.String())
		}
		rep := pdChatReplyOf(t, rec)
		if len(rep.Choices) != 1 || rep.Choices[0].Message.Content != "ok!" || rep.Choices[0].FinishReason != "stop" {
			t.Fatalf("choices = %+v, want one assistant choice with content ok! and finish stop", rep.Choices)
		}
		if !ship.gotFrame || len(ship.lastReq.Prompt) == 0 {
			t.Fatalf("transport saw no paged frame or prompt: %+v", ship.lastReq)
		}
	})

	t.Run("max_tokens truncates with length", func(t *testing.T) {
		ship := &pdStubShip{t: t, tokens: []int{int('a'), int('b'), int('c'), int('d')}}
		h, done := newPDTestHandler(t, ship)
		defer done()
		rec := postPDChat(t, h, `{"messages":[{"role":"user","content":"ping"}],"max_tokens":2}`)
		if rec.Code != http.StatusOK {
			t.Fatalf("truncation = %d, want 200 (%s)", rec.Code, rec.Body.String())
		}
		rep := pdChatReplyOf(t, rec)
		if len(rep.Choices) != 1 || rep.Choices[0].Message.Content != "ab" || rep.Choices[0].FinishReason != "length" {
			t.Fatalf("choices = %+v, want truncated content ab with finish length", rep.Choices)
		}
	})
}

// fak-test:runtime medium est=5s lane=default
func TestServePDChatFrontDoorEndToEndOverWire(t *testing.T) {
	// Two clusters over one synthetic checkpoint: the prefill role's front
	// door ships its paged-KV frame to the decode role's wire server over a
	// real loopback connection — the production P2-1 call path.
	m := fakmodel.NewSynthetic(modelengine.SyntheticConfig())
	prefillCluster := modelengine.NewNativePDCluster(m, 1)
	defer prefillCluster.Close()
	decodeCluster := modelengine.NewNativePDCluster(m, 2)
	defer decodeCluster.Close()

	wire := modelengine.NewPDWireServer(decodeCluster)
	if err := wire.Listen("127.0.0.1:0"); err != nil {
		t.Fatalf("wire listen: %v", err)
	}
	go func() { _ = wire.Serve() }()
	defer wire.Close()

	client := modelengine.NewPDWireClient(prefillCluster, wire.Addr().String(), time.Second)
	h := &servePDChatHandler{
		cluster:    prefillCluster,
		ship:       client,
		encode:     pdByteEncode,
		decode:     pdByteDecode,
		modelLabel: "pd-e2e",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"ping"}]}`))
	req = req.WithContext(ctx)
	rec := httptest.NewRecorder()
	h.handleChat(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("front door = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	rep := pdChatReplyOf(t, rec)
	if len(rep.Choices) != 1 || rep.Choices[0].Message.Content == "" || rep.Choices[0].FinishReason != "stop" {
		t.Fatalf("choices = %+v, want a non-empty assistant reply finished by stop", rep.Choices)
	}
	// The decode role must have admitted the import without re-prefilling.
	worker, ok := decodeCluster.DecodeWorker("decode-0")
	if !ok {
		t.Fatal("decode-0 not found on the decode cluster")
	}
	stats := worker.Stats()
	if stats.RePrefillOnDecode != 0 {
		t.Fatalf("decode stats = %+v, want no decode-side prefill", stats)
	}
}
