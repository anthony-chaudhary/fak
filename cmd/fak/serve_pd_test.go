package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
	validPrefill := servePDConfig{Role: "prefill", GGUF: "m.gguf", Listen: "127.0.0.1:8090", DecodeAddr: "127.0.0.1:8091"}
	validDecode := servePDConfig{Role: "decode", GGUF: "m.gguf", Listen: "127.0.0.1:8091", DecodeWorkers: 2}

	if err := validateServePD(validPrefill); err != nil {
		t.Fatalf("valid prefill config refused: %v", err)
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
		{name: "missing gguf", cfg: servePDConfig{Role: "prefill", Listen: "l", DecodeAddr: "d"}, sentinel: ErrServePDMissingGGUF},
		{name: "missing listen", cfg: servePDConfig{Role: "decode", GGUF: "m.gguf"}, sentinel: ErrServePDMissingListen},
		{name: "prefill missing decode addr", cfg: servePDConfig{Role: "prefill", GGUF: "m.gguf", Listen: "l"}, sentinel: ErrServePDMissingDecodeAddr},
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
func TestRunServePDFlagRefusals(t *testing.T) {
	cases := []struct {
		name string
		argv []string
	}{
		{name: "no role", argv: []string{"--gguf", "m.gguf", "--listen", "127.0.0.1:1"}},
		{name: "bogus role", argv: []string{"--role", "midfill", "--gguf", "m.gguf", "--listen", "127.0.0.1:1"}},
		{name: "prefill without decode addr", argv: []string{"--role", "prefill", "--gguf", "m.gguf", "--listen", "127.0.0.1:1"}},
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
type pdStubShip struct {
	t        *testing.T
	tokens   []int
	shipErr  error
	lastReq  modelengine.PDShipRequest
	gotFrame bool
}

func (s *pdStubShip) Ship(_ context.Context, req modelengine.PDShipRequest) (modelengine.PDShipResult, error) {
	s.lastReq = req
	s.gotFrame = len(req.KVFrame) > 0
	if s.shipErr != nil {
		return modelengine.PDShipResult{}, s.shipErr
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

func newPDTestHandler(t *testing.T, ship modelengine.PDHandoffTransport) (*servePDChatHandler, func()) {
	t.Helper()
	m := fakmodel.NewSynthetic(modelengine.SyntheticConfig())
	cluster := modelengine.NewNativePDCluster(m, 1)
	h := &servePDChatHandler{
		cluster:    cluster,
		ship:       ship,
		encode:     pdByteEncode,
		decode:     pdByteDecode,
		modelLabel: "pd-test",
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
	t.Run("stream refused typed", func(t *testing.T) {
		ship := &pdStubShip{t: t, tokens: []int{1}}
		h, done := newPDTestHandler(t, ship)
		defer done()
		rec := postPDChat(t, h, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
		if rec.Code != http.StatusBadRequest || pdChatErrorCode(t, rec) != pdChatCodeStreamUnsupported {
			t.Fatalf("stream refusal = %d/%s, want 400/%s", rec.Code, pdChatErrorCode(t, rec), pdChatCodeStreamUnsupported)
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
