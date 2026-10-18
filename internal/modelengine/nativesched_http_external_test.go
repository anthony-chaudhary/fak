package modelengine_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type exactHTTPResolver struct{}

func (exactHTTPResolver) Resolve(_ context.Context, r abi.Ref) ([]byte, error) { return r.Inline, nil }
func (exactHTTPResolver) Put(_ context.Context, b []byte) (abi.Ref, error) {
	return abi.Ref{Kind: abi.RefInline, Inline: append([]byte(nil), b...), Len: int64(len(b)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent}, nil
}

type exactHTTPBackend struct{}

func (exactHTTPBackend) Resolver() abi.Resolver { return exactHTTPResolver{} }
func (exactHTTPBackend) Caps() []abi.Capability { return nil }

type exactHTTPAllow struct{}

func (exactHTTPAllow) Caps() []abi.Capability { return nil }
func (exactHTTPAllow) Adjudicate(context.Context, *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "exact-http-test"}
}

func exactHTTPTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tokens := make([]string, 256)
	extra := 256
	for b := 0; b < 256; b++ {
		r := b
		if !(b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174) {
			r = extra
			extra++
		}
		tokens[b] = string(rune(r))
	}
	tok, err := tokenizer.FromGGML(tokens, []string{tokens['x'] + " " + tokens['y']}, make([]int32, 256), "")
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// fak-test:runtime fast est=2s
func TestExactTwinLoopbackSyscall(t *testing.T) {
	for _, collision := range []bool{false, true} {
		name := "identical-requests"
		if collision {
			name = "truncated-prefix-distinct-bindings"
		}
		t.Run(name, func(t *testing.T) { testExactTwinLoopbackSyscall(t, collision) })
	}
}

func testExactTwinLoopbackSyscall(t *testing.T, collision bool) {
	// Kernel/global registries are isolated for this serial fixture; no persistent settings change.
	abi.ResetForTest()
	t.Cleanup(abi.ResetForTest)
	abi.RegisterRegionBackend(exactHTTPBackend{})
	abi.RegisterAdjudicator(0, exactHTTPAllow{})
	tok := exactHTTPTokenizer(t)
	args := `{"query":"` + strings.Repeat("a", 70) + `"}`
	requests := [2]gateway.SyscallRequest{
		{Tool: "allow_exact", Arguments: json.RawMessage(args), TraceID: "exact-http-first"},
		{Tool: "allow_exact", Arguments: json.RawMessage(args), TraceID: "exact-http-second"},
	}
	if collision {
		// Both independently tokenized calls are truncated inside the long tool
		// prefix. Distinct tool suffixes and raw arguments must retain their own
		// request/result binding even though the executed model prompt is equal.
		prefix := "allow_" + strings.Repeat("a", 64)
		requests[0].Tool, requests[1].Tool = prefix+"first", prefix+"second"
		requests[0].Arguments = json.RawMessage(`{"query":"first raw arguments"}`)
		requests[1].Arguments = json.RawMessage(`{"query":"second raw arguments"}`)
	}
	var bodies [2][]byte
	var ids []int
	for i, request := range requests {
		encoded, err := tok.Encode(request.Tool + " " + string(request.Arguments))
		if err != nil || len(encoded) < 64 {
			t.Fatalf("real tokenizer prompt len=%d err=%v", len(encoded), err)
		}
		if i == 0 {
			ids = append([]int(nil), encoded[:64]...)
		} else if !reflect.DeepEqual(ids, encoded[:64]) {
			t.Fatal("fixture prompts do not share exact executed prefix")
		}
		bodies[i], err = json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
	}
	type outcome struct {
		response gateway.SyscallResponse
		status   int
		err      error
	}
	run := func(enabled, cancelPeer bool) ([2]outcome, int64, uint64) {
		engine, await, release, stats, closeEngine := modelengine.ExactHTTPFixtureForTest(t, tok, enabled)
		defer release()
		// Own scheduler teardown before any fallible gateway/listener setup.
		defer func() {
			release()
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			if err := closeEngine(ctx); err != nil {
				t.Errorf("scheduler setup teardown: %v", err)
			}
		}()
		id := fmt.Sprintf("exact-http-%v-%v", enabled, cancelPeer)
		abi.RegisterEngine(id, engine)
		srv, err := gateway.New(gateway.Config{EngineID: id, Model: "qwen-exact-fixture", VDSO: false})
		if err != nil {
			t.Fatal(err)
		}
		defer srv.Close()
		ts := httptest.NewServer(srv.Handler())
		// Release execution before listener teardown even after an assertion fails.
		defer func() {
			release()
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			if err := closeEngine(ctx); err != nil {
				t.Errorf("scheduler teardown: %v", err)
			}
			ts.CloseClientConnections()
			ts.Close()
		}()
		client := ts.Client()
		client.Timeout = 3 * time.Second
		defer client.CloseIdleConnections()
		results := [2]chan outcome{make(chan outcome, 1), make(chan outcome, 1)}
		contexts := [2]context.Context{}
		cancels := [2]context.CancelFunc{}
		for i := range contexts {
			contexts[i], cancels[i] = context.WithCancel(context.Background())
			defer cancels[i]()
		}
		for i := range results {
			go func(i int) {
				var out outcome
				req, err := http.NewRequestWithContext(contexts[i], http.MethodPost, ts.URL+"/v1/fak/syscall", bytes.NewReader(bodies[i]))
				if err != nil {
					out.err = err
					results[i] <- out
					return
				}
				req.Header.Set("Content-Type", "application/json")
				resp, err := client.Do(req)
				if err != nil {
					out.err = err
					results[i] <- out
					return
				}
				defer resp.Body.Close()
				out.status = resp.StatusCode
				raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
				if err == nil {
					err = json.Unmarshal(raw, &out.response)
				}
				out.err = err
				results[i] <- out
			}(i)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := await(ctx, 2, ids[:64], 0); err != nil {
			t.Fatalf("two real HTTP admissions: %v", err)
		}
		if cancelPeer {
			cancels[1]()
			// Wait for cancellation to reach the real server-owned lane before
			// releasing model execution; a client-side error alone is insufficient.
			if err := await(ctx, 2, ids[:64], 1); err != nil {
				t.Fatalf("HTTP cancellation did not reach queued lane: %v", err)
			}
		}
		release()
		var out [2]outcome
		for i := range out {
			select {
			case out[i] = <-results[i]:
			case <-ctx.Done():
				t.Fatalf("HTTP request %d did not drain: %v", i, ctx.Err())
			}
		}
		chunks, copies := stats()
		return out, chunks, copies
	}
	off, offChunks, offCopies := run(false, false)
	on, onChunks, onCopies := run(true, false)
	for i := range off {
		if off[i].err != nil || on[i].err != nil || off[i].status != http.StatusOK || on[i].status != http.StatusOK {
			t.Fatalf("request %d off/on=%+v/%+v", i, off[i], on[i])
		}
		if off[i].response.Result == nil || !reflect.DeepEqual(off[i].response.Result, on[i].response.Result) {
			t.Fatalf("terminal content/status/usage mismatch: %+v/%+v", off[i], on[i])
		}
		meta := on[i].response.Result.Meta
		if on[i].response.Result.Status != "OK" || meta["input_tokens"] != "64" || meta["output_tokens"] != "16" || meta["detokenized"] != "true" {
			t.Fatalf("missing explicit successful usage/tokenizer evidence: %+v", on[i].response.Result)
		}
		if off[i].response.TraceID != requests[i].TraceID || on[i].response.TraceID != requests[i].TraceID {
			t.Fatalf("request %d trace binding changed", i)
		}
		var payload struct {
			Tool   string `json:"tool"`
			Engine string `json:"engine"`
			Tokens []int  `json:"generated_tokens"`
			Text   string `json:"generated_text"`
		}
		if err := json.Unmarshal([]byte(on[i].response.Result.Content), &payload); err != nil || payload.Engine != modelengine.EngineID || payload.Tool != requests[i].Tool || len(payload.Tokens) != 16 {
			t.Fatalf("missing actual model result: %+v err=%v", payload, err)
		}
		decoded, err := tok.Decode(payload.Tokens)
		if err != nil {
			t.Fatalf("independent tokenizer decode: %v", err)
		}
		// Match JSON's specified invalid-UTF8 replacement on arbitrary byte tokens.
		encodedText, err := json.Marshal(decoded)
		if err != nil {
			t.Fatal(err)
		}
		var wireText string
		if err := json.Unmarshal(encodedText, &wireText); err != nil {
			t.Fatal(err)
		}
		if payload.Text != wireText {
			t.Fatalf("generated text differs from actual tokenizer decode: %q/%q", payload.Text, wireText)
		}
	}
	if offChunks != 4 || onChunks != 2 || offCopies != 0 || onCopies != 1 {
		t.Fatalf("HTTP chunks off/on=%d/%d copies=%d/%d", offChunks, onChunks, offCopies, onCopies)
	}
	canceled, cancelChunks, cancelCopies := run(true, true)
	if cancelChunks != 2 || cancelCopies != 0 {
		t.Fatalf("canceled lane executed or copied: chunks=%d copies=%d", cancelChunks, cancelCopies)
	}
	if canceled[0].err != nil || !reflect.DeepEqual(canceled[0].response.Result, off[0].response.Result) {
		t.Fatalf("survivor diverged: %+v", canceled[0])
	}
	if !errors.Is(canceled[1].err, context.Canceled) {
		t.Fatalf("canceled HTTP request error=%v", canceled[1].err)
	}
}
