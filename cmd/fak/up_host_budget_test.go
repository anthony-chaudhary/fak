package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// hostMemoryBudgetArmer is the seam armHostMemoryBudget looks for on the
// turnkey planner (#13267). It is spelled out here, independently of up.go,
// so a signature drift on either side breaks this file at compile time.
type hostMemoryBudgetArmer interface {
	SetHostMemoryBudget(ceiling int64, used func() (int64, bool))
}

// Compile-time pin: the production turnkey planner satisfies the seam.
var _ hostMemoryBudgetArmer = (*agent.InKernelPlanner)(nil)

// hostBudgetRecordingPlanner is a minimal agent.Planner that also implements
// the host-memory seam and records every arm call.
type hostBudgetRecordingPlanner struct {
	calls   int
	ceiling int64
	used    func() (int64, bool)
}

func (p *hostBudgetRecordingPlanner) Model() string { return "host-budget-fake" }

func (p *hostBudgetRecordingPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, errors.New("hostBudgetRecordingPlanner: Complete not expected")
}

func (p *hostBudgetRecordingPlanner) SetHostMemoryBudget(ceiling int64, used func() (int64, bool)) {
	p.calls++
	p.ceiling = ceiling
	p.used = used
}

// hostBudgetPlainPlanner is an agent.Planner WITHOUT the host-memory seam.
type hostBudgetPlainPlanner struct{ completes int }

func (p *hostBudgetPlainPlanner) Model() string { return "host-budget-plain" }

func (p *hostBudgetPlainPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	p.completes++
	return nil, errors.New("hostBudgetPlainPlanner: Complete not expected")
}

type turnkeyInferenceErrorEnvelope struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func decodeTurnkeyInferenceErrorEnvelope(t *testing.T, raw []byte) turnkeyInferenceErrorEnvelope {
	t.Helper()
	var env turnkeyInferenceErrorEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatalf("decode error envelope: %v; body=%s", err, raw)
	}
	return env
}

func TestWriteTurnkeyInferenceErrorCapacityIs503RetryAfter(t *testing.T) {
	if turnkeyCapacityRetryAfterSeconds != "5" {
		t.Fatalf("turnkeyCapacityRetryAfterSeconds = %q, want \"5\"", turnkeyCapacityRetryAfterSeconds)
	}
	capErr := &agent.InKernelCapacityError{
		Want:  10 << 30,
		Avail: 1 << 30,
		Class: compute.MemoryKVCache,
		Scope: compute.MemoryScopeHost,
		Site:  "host-memory-precheck",
	}
	oomErr := &agent.InKernelOOMError{Bytes: 3 << 30, Class: compute.MemoryKVCache, Site: "kv-grow"}
	ctxErr := &agent.InKernelContextLengthError{PromptTokens: 4000, MaxNewTokens: 200, MaxContext: 4096}
	stallErr := metalgemm.MetalCommandBufferStallError{Operation: "prefill", WaitedMilliseconds: 1500, LimitMilliseconds: 1000}
	plainErr := errors.New("tokenizer exploded")

	tests := []struct {
		name       string
		err        error
		typedText  string // the typed error's own text the body must carry
		wantStatus int
		wantCode   string // "" = no JSON envelope expected
		wantType   string
		wantRetry  string // "-" = do not check; "" = must be absent
	}{
		{name: "capacity", err: capErr, typedText: capErr.Error(), wantStatus: http.StatusServiceUnavailable, wantCode: "in_kernel_oom", wantType: "server_error", wantRetry: "5"},
		{name: "wrapped capacity", err: fmt.Errorf("inkernel complete: %w", capErr), typedText: capErr.Error(), wantStatus: http.StatusServiceUnavailable, wantCode: "in_kernel_oom", wantType: "server_error", wantRetry: "5"},
		{name: "double wrapped capacity", err: fmt.Errorf("turn: %w", fmt.Errorf("admit: %w", capErr)), typedText: capErr.Error(), wantStatus: http.StatusServiceUnavailable, wantCode: "in_kernel_oom", wantType: "server_error", wantRetry: "5"},
		{name: "oom", err: oomErr, typedText: oomErr.Error(), wantStatus: http.StatusServiceUnavailable, wantCode: "in_kernel_oom", wantType: "server_error", wantRetry: "5"},
		{name: "wrapped oom", err: fmt.Errorf("decode: %w", oomErr), typedText: oomErr.Error(), wantStatus: http.StatusServiceUnavailable, wantCode: "in_kernel_oom", wantType: "server_error", wantRetry: "5"},
		{name: "context length stays 400", err: ctxErr, typedText: ctxErr.Error(), wantStatus: http.StatusBadRequest, wantCode: "context_length_exceeded", wantType: "invalid_request_error", wantRetry: ""},
		{name: "wrapped context length stays 400", err: fmt.Errorf("plan: %w", ctxErr), typedText: ctxErr.Error(), wantStatus: http.StatusBadRequest, wantCode: "context_length_exceeded", wantType: "invalid_request_error", wantRetry: ""},
		{name: "metal stall stays 503 stalled", err: fmt.Errorf("observe: %w", stallErr), wantStatus: http.StatusServiceUnavailable, wantCode: "metal_command_buffer_stalled", wantType: "server_error", wantRetry: "-"},
		{name: "plain error stays 500", err: plainErr, wantStatus: http.StatusInternalServerError, wantRetry: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			writeTurnkeyInferenceError(rec, tt.err)
			res := rec.Result()
			defer res.Body.Close()
			raw, err := io.ReadAll(res.Body)
			if err != nil {
				t.Fatal(err)
			}
			if res.StatusCode != tt.wantStatus {
				t.Fatalf("status = %d, want %d; body=%s", res.StatusCode, tt.wantStatus, raw)
			}
			if tt.wantRetry != "-" {
				if got := res.Header.Get("Retry-After"); got != tt.wantRetry {
					t.Fatalf("Retry-After = %q, want %q", got, tt.wantRetry)
				}
			}
			if tt.wantCode == "" {
				// Generic arm: the historical text/plain "inference error: ..." 500.
				want := "inference error: " + tt.err.Error()
				if !strings.Contains(string(raw), want) {
					t.Fatalf("plain 500 body = %q, want it to contain %q", raw, want)
				}
				if strings.Contains(string(raw), "in_kernel_oom") {
					t.Fatalf("plain error mislabeled as in_kernel_oom: %s", raw)
				}
				return
			}
			if ct := res.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
				t.Fatalf("Content-Type = %q, want application/json", ct)
			}
			env := decodeTurnkeyInferenceErrorEnvelope(t, raw)
			if env.Error.Code != tt.wantCode {
				t.Fatalf("error.code = %q, want %q; body=%s", env.Error.Code, tt.wantCode, raw)
			}
			if env.Error.Type != tt.wantType {
				t.Fatalf("error.type = %q, want %q; body=%s", env.Error.Type, tt.wantType, raw)
			}
			if strings.TrimSpace(env.Error.Message) == "" {
				t.Fatalf("error.message is empty; body=%s", raw)
			}
			if tt.typedText != "" && !strings.Contains(env.Error.Message, tt.typedText) {
				t.Fatalf("error.message = %q, want it to contain the typed error text %q", env.Error.Message, tt.typedText)
			}
		})
	}
}

func TestTurnkeyArmHostMemoryBudget(t *testing.T) {
	t.Run("zero limit is not armed", func(t *testing.T) {
		p := &hostBudgetRecordingPlanner{}
		(&turnkeyServer{planner: p}).armHostMemoryBudget(0)
		if p.calls != 0 {
			t.Fatalf("SetHostMemoryBudget calls = %d for limit 0, want 0 (no-op)", p.calls)
		}
	})

	t.Run("24GiB limit arms ceiling and live RSS probe", func(t *testing.T) {
		const limit = uint64(24) << 30
		p := &hostBudgetRecordingPlanner{}
		(&turnkeyServer{planner: p}).armHostMemoryBudget(limit)
		if p.calls != 1 {
			t.Fatalf("SetHostMemoryBudget calls = %d, want exactly 1", p.calls)
		}
		if p.ceiling != int64(limit) {
			t.Fatalf("ceiling = %d, want %d", p.ceiling, int64(limit))
		}
		if p.used == nil {
			t.Fatal("used probe is nil; an armed budget needs a live usage probe")
		}
		used, known := p.used()
		if known && used <= 0 {
			t.Fatalf("probe reported known=true with non-positive usage %d", used)
		}
		if runtime.GOOS != "darwin" {
			return
		}
		ref := platformCurrentRSS()
		if ref == 0 {
			t.Skip("darwin build has no OS RSS source (cgo off); probe liveness is not checkable here")
		}
		if !known || used <= 0 {
			t.Fatalf("darwin probe = (%d, %v), want known=true with a positive resident byte count", used, known)
		}
		// The probe must report THIS process's resident size: same order of
		// magnitude as the OS reading taken right after it.
		if u := uint64(used); u > 2*ref || 2*u < ref {
			t.Fatalf("probe used=%d is not this process's RSS (platformCurrentRSS=%d)", used, ref)
		}
		// And it must be LIVE, not a snapshot taken at arm time: touching a
		// fresh 256 MiB buffer must raise what the already-armed probe reports.
		before, _ := p.used()
		buf := make([]uint64, (256<<20)/8)
		x := uint64(0x9E3779B97F4A7C15)
		for i := range buf {
			x ^= x << 13
			x ^= x >> 7
			x ^= x << 17
			buf[i] = x
		}
		after, afterKnown := p.used()
		runtime.KeepAlive(buf)
		if !afterKnown {
			t.Fatal("probe went unknown after allocation")
		}
		if grew := after - before; grew < 64<<20 {
			t.Fatalf("probe grew %d bytes after touching 256 MiB (before=%d after=%d); want a live RSS probe", grew, before, after)
		}
	})

	t.Run("limits above MaxInt64 clamp", func(t *testing.T) {
		for _, limit := range []uint64{uint64(math.MaxInt64), uint64(math.MaxInt64) + 1, math.MaxUint64} {
			p := &hostBudgetRecordingPlanner{}
			(&turnkeyServer{planner: p}).armHostMemoryBudget(limit)
			if p.calls != 1 {
				t.Fatalf("limit %d: SetHostMemoryBudget calls = %d, want 1", limit, p.calls)
			}
			if p.ceiling != math.MaxInt64 {
				t.Fatalf("limit %d: ceiling = %d, want clamp to MaxInt64 (%d)", limit, p.ceiling, int64(math.MaxInt64))
			}
		}
	})

	t.Run("small limit passes through exactly", func(t *testing.T) {
		p := &hostBudgetRecordingPlanner{}
		(&turnkeyServer{planner: p}).armHostMemoryBudget(1)
		if p.calls != 1 || p.ceiling != 1 {
			t.Fatalf("limit 1: calls=%d ceiling=%d, want 1 call with ceiling 1", p.calls, p.ceiling)
		}
	})

	t.Run("planner without the seam is left alone", func(t *testing.T) {
		p := &hostBudgetPlainPlanner{}
		s := &turnkeyServer{planner: p}
		s.armHostMemoryBudget(24 << 30)
		if p.completes != 0 {
			t.Fatalf("arming touched the planner's Complete %d times", p.completes)
		}
		if s.planner != agent.Planner(p) {
			t.Fatal("arming replaced the server's planner")
		}
	})

	t.Run("nil planner is safe", func(t *testing.T) {
		(&turnkeyServer{}).armHostMemoryBudget(24 << 30)
	})

	t.Run("nil server is safe", func(t *testing.T) {
		var s *turnkeyServer
		s.armHostMemoryBudget(24 << 30)
		s.armHostMemoryBudget(0)
	})
}

func TestInKernelPlannerSatisfiesHostMemoryBudgetArmer(t *testing.T) {
	realPlanner := agent.NewInKernelPlanner(&model.Model{}, testProbeTokenizer(t), "native-up", false, nil, false)
	var planner agent.Planner = realPlanner
	if _, ok := planner.(hostMemoryBudgetArmer); !ok {
		t.Fatal("*agent.InKernelPlanner held as agent.Planner does not expose SetHostMemoryBudget; armHostMemoryBudget would silently no-op in production")
	}
	// Arming the real planner through the server seam must not panic, and a
	// later zero-limit call stays a no-op.
	s := &turnkeyServer{planner: planner}
	s.armHostMemoryBudget(24 << 30)
	s.armHostMemoryBudget(0)
	// Nil-receiver and disarm spellings the seam documents must be safe too.
	(*agent.InKernelPlanner)(nil).SetHostMemoryBudget(1<<30, func() (int64, bool) { return 1, true })
	realPlanner.SetHostMemoryBudget(0, nil)
}

// TestTurnkeyArmHostMemoryBudgetDeclinesOverHTTP drives a real (synthetic,
// CPU-only, backend-nil) in-kernel planner through the turnkey HTTP server:
// armed with a generous ceiling a request is served; re-armed with a ceiling
// below this process's RSS the same request is declined BEFORE allocation as
// 503 + Retry-After + in_kernel_oom instead of an opaque 500.
func TestTurnkeyArmHostMemoryBudgetDeclinesOverHTTP(t *testing.T) {
	if platformCurrentRSS() == 0 {
		t.Skip("no OS RSS source on this build; the host arm fails open by design")
	}
	tok := testProbeTokenizer(t)
	m := model.NewSynthetic(upContextSyntheticConfig())
	m.Quantize()
	const contextTokens = 256
	m.Cfg.MaxPositionEmbeddings = contextTokens
	planner := newTurnkeyInKernelPlanner(m, tok, "native-up", false, nil, false, contextTokens)
	plan := macfit.TurnkeyProfile{
		Tier:                macfit.ModelTier{ModelID: "native-up"},
		ContextBudgetTokens: contextTokens,
	}
	server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", false, planner)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Shutdown(context.Background())

	request := chatCompletionRequest{
		Model:     "native-up",
		Messages:  []chatCompletionMessage{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	}

	server.armHostMemoryBudget(1 << 50)
	if status, raw := postUpChat(t, server, request); status != http.StatusOK {
		t.Fatalf("generous host ceiling: status = %d, want 200; body=%s", status, raw)
	}

	server.armHostMemoryBudget(1) // a 1-byte ceiling is always below this process's RSS
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post("http://"+server.Addr()+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("host ceiling below RSS: status = %d, want 503; body=%s", resp.StatusCode, raw)
	}
	if got := resp.Header.Get("Retry-After"); got != turnkeyCapacityRetryAfterSeconds {
		t.Fatalf("Retry-After = %q, want %q", got, turnkeyCapacityRetryAfterSeconds)
	}
	env := decodeTurnkeyInferenceErrorEnvelope(t, raw)
	if env.Error.Code != "in_kernel_oom" || env.Error.Type != "server_error" {
		t.Fatalf("error envelope = %+v, want server_error/in_kernel_oom; body=%s", env.Error, raw)
	}
	if !strings.Contains(env.Error.Message, "capacity precheck refused") {
		t.Fatalf("error.message = %q, want the typed capacity refusal text", env.Error.Message)
	}
}

// TestRunTurnkeyUpArmHostMemoryBudgetAfterMemGuard pins the wiring in
// runTurnkeyUp: the host-memory arm is fed the SAME --max-rss value as the
// memory guard and is installed right after it.
func TestRunTurnkeyUpArmHostMemoryBudgetAfterMemGuard(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "up.go", nil, 0)
	if err != nil {
		t.Fatalf("parse up.go: %v", err)
	}
	var fn *ast.FuncDecl
	for _, decl := range file.Decls {
		if d, ok := decl.(*ast.FuncDecl); ok && d.Recv == nil && d.Name.Name == "runTurnkeyUp" {
			fn = d
			break
		}
	}
	if fn == nil || fn.Body == nil {
		t.Fatal("runTurnkeyUp not found in up.go")
	}
	type call struct {
		pos  token.Pos
		arg0 string
	}
	var memGuard, hostBudget []call
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		ce, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := ce.Fun.(*ast.SelectorExpr)
		if !ok || len(ce.Args) == 0 {
			return true
		}
		var buf bytes.Buffer
		if star, ok := ce.Args[0].(*ast.StarExpr); ok {
			if id, ok := star.X.(*ast.Ident); ok {
				buf.WriteString("*" + id.Name)
			}
		} else if id, ok := ce.Args[0].(*ast.Ident); ok {
			buf.WriteString(id.Name)
		}
		switch sel.Sel.Name {
		case "armMemGuard":
			memGuard = append(memGuard, call{ce.Pos(), buf.String()})
		case "armHostMemoryBudget":
			hostBudget = append(hostBudget, call{ce.Pos(), buf.String()})
		}
		return true
	})
	if len(memGuard) != 1 {
		t.Fatalf("runTurnkeyUp has %d armMemGuard calls, want 1", len(memGuard))
	}
	if len(hostBudget) != 1 {
		t.Fatalf("runTurnkeyUp has %d armHostMemoryBudget calls, want 1 (#13267 wiring)", len(hostBudget))
	}
	if hostBudget[0].pos <= memGuard[0].pos {
		t.Fatalf("armHostMemoryBudget at %s is not after armMemGuard at %s", fset.Position(hostBudget[0].pos), fset.Position(memGuard[0].pos))
	}
	if memGuard[0].arg0 == "" || hostBudget[0].arg0 != memGuard[0].arg0 {
		t.Fatalf("armHostMemoryBudget(%s) is not fed the same --max-rss value as armMemGuard(%s)", hostBudget[0].arg0, memGuard[0].arg0)
	}
	if memGuard[0].arg0 != "maxRSSCeiling" {
		t.Fatalf("armMemGuard first arg = %s, want maxRSSCeiling (the resolved --max-rss ceiling)", memGuard[0].arg0)
	}
}
