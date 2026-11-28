package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"weak"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

func primeE2EModelConfig() model.Config {
	return model.Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         260,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
		ModelType:         "llama",
	}
}

func newPrimeE2EQuantModel() *model.Model {
	m := model.NewSynthetic(primeE2EModelConfig())
	m.Quantize()
	return m
}

func newPrimeE2EDeviceGLMModel() *model.Model {
	cfg := tinyGLMDsaCfg()
	cfg.VocabSize = 260
	m := model.NewSyntheticGLMDsa(cfg)
	m.Quantize()
	return m
}

type multiTurnPrimeTestPlanner struct {
	planner  *InKernelPlanner
	toolPath string

	mu           sync.Mutex
	turns        int
	coldTurn     int
	reused       []int
	turn2Msgs    []Message
	turn2Content string
}

func (m *multiTurnPrimeTestPlanner) Model() string { return m.planner.Model() }

func (m *multiTurnPrimeTestPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	m.mu.Lock()
	turn := m.turns
	m.turns++
	m.mu.Unlock()

	if turn > 0 {
		// Wait for Turn 1's background prime so Turn 2 deterministically observes cached
		// token reuse; un-waited concurrent Complete vs prime safety is witnessed separately
		// in TestPrimeCacheConcurrentCompleteAndPrimeRaceFree.
		m.planner.WaitNextTurnPrime()
	}

	callOpts := append([]SampleOpt(nil), opts...)
	callOpts = append(callOpts, WithMaxTokens(4))
	comp, err := m.planner.Complete(ctx, messages, tools, callOpts...)
	if err != nil {
		return nil, err
	}

	cached := 0
	if comp.Usage.PromptTokensDetails != nil {
		cached = comp.Usage.PromptTokensDetails.CachedTokens
	}
	m.mu.Lock()
	if turn == 0 {
		m.coldTurn = cached
	} else {
		m.reused = append(m.reused, cached)
		if len(m.reused) == 1 {
			m.turn2Msgs = append([]Message(nil), messages...)
			m.turn2Content = comp.Message.Content
		}
	}
	m.mu.Unlock()

	if turn == 0 {
		args, err := json.Marshal(map[string]string{"file_path": m.toolPath})
		if err != nil {
			return nil, err
		}
		out := *comp
		out.FinishReason = "tool_calls"
		out.Message = Message{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{{
				ID:   "call_e2e_1",
				Type: "function",
				Function: Func{
					Name:      "Read",
					Arguments: string(args),
				},
			}},
		}
		return &out, nil
	}
	return comp, nil
}

// fak-test:runtime fast est=100ms lane=default
func TestPrimeCacheRuntimeConfigRoundTrip(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "")
	t.Setenv("FAK_INKERNEL_BATCH", "")
	t.Setenv("FAK_INKERNEL_KVMMU", "")

	t.Run("explicit config fields round-trip verbatim", func(t *testing.T) {
		m := newPrimeE2EQuantModel()
		tok := loadProbeTok(t)
		cfg := InKernelPlannerConfig{
			ContextTokens:             4096,
			CPUCacheBytes:             1 << 20,
			RadixBudgetTokens:         2048,
			BatchDecode:               true,
			QwenQ4KPrefillChunkTokens: 1024,
			IncrementalContext:        true,
			NextTurnPrimeEnabled:      true,
			NextTurnPrimeConfig:       NextTurnPrimeConfig{MaxConcurrent: 2, Timeout: 15 * time.Second},
			RadixSessionID:            "sess-e2e-explicit",
			EnableFingerprintIndex:    true,
			FingerprintChunkSize:      64,
			KVSpanEvict:               true,
		}

		p := NewInKernelPlannerWithConfig(m, tok, "e2e-config-explicit", false, nil, false, cfg)
		defer p.Close()

		got := p.RuntimeConfig()
		if !reflect.DeepEqual(got, cfg) {
			t.Fatalf("RuntimeConfig() mismatch:\n got: %+v\nwant: %+v", got, cfg)
		}
		if p.NextTurnPrimeRunner() == nil {
			t.Fatal("NextTurnPrimeRunner() = nil, want non-nil")
		}
		if !p.FingerprintIndexEnabled() {
			t.Fatal("FingerprintIndexEnabled() = false, want true")
		}
		if p.RadixSessionID() != "sess-e2e-explicit" {
			t.Fatalf("RadixSessionID() = %q, want %q", p.RadixSessionID(), "sess-e2e-explicit")
		}
		if gen, ok := p.RadixSessionGen("sess-e2e-explicit"); !ok || gen == 0 {
			t.Fatalf("RadixSessionGen = (%d, %v), want active non-zero generation", gen, ok)
		}
		if !p.BatchDecodeEnabled() {
			t.Fatal("BatchDecodeEnabled() = false, want true")
		}

		p2 := NewInKernelPlannerWithConfig(m, tok, "e2e-config-explicit-2", false, nil, false, got)
		defer p2.Close()
		if got2 := p2.RuntimeConfig(); !reflect.DeepEqual(got2, cfg) {
			t.Fatalf("second-hop RuntimeConfig() mismatch:\n got: %+v\nwant: %+v", got2, cfg)
		}

		p.Close()
		p.Close() // idempotent
		if p.NextTurnPrimeRunner() != nil {
			t.Fatal("NextTurnPrimeRunner() remained non-nil after Close()")
		}
		if !p.RadixSessionClosed("sess-e2e-explicit") {
			t.Fatal("RadixSessionClosed() = false after Close(), want true")
		}
		if gotNodes := p.SessionImmuneNodes(); gotNodes != 0 {
			t.Fatalf("SessionImmuneNodes() = %d after Close(), want 0", gotNodes)
		}
	})

	t.Run("defaults resolve cleanly and preserve negative radix budget", func(t *testing.T) {
		m := newPrimeE2EQuantModel()
		tok := loadProbeTok(t)
		p := NewInKernelPlannerWithConfig(m, tok, "e2e-config-defaults", false, nil, false, InKernelPlannerConfig{
			RadixBudgetTokens:      -1,
			NextTurnPrimeEnabled:   true,
			EnableFingerprintIndex: true,
		})
		defer p.Close()

		got := p.RuntimeConfig()
		if got.RadixBudgetTokens != -1 {
			t.Fatalf("RadixBudgetTokens = %d, want -1", got.RadixBudgetTokens)
		}
		if !got.NextTurnPrimeEnabled {
			t.Fatal("NextTurnPrimeEnabled = false, want true")
		}
		wantPrimeCfg := NextTurnPrimeConfig{
			MaxConcurrent: DefaultNextTurnPrimeConcurrency,
			Timeout:       DefaultNextTurnPrimeTimeout,
		}
		if got.NextTurnPrimeConfig != wantPrimeCfg {
			t.Fatalf("NextTurnPrimeConfig = %+v, want %+v", got.NextTurnPrimeConfig, wantPrimeCfg)
		}
		if !got.EnableFingerprintIndex || got.FingerprintChunkSize != radixkv.DefaultFingerprintChunkSize {
			t.Fatalf("Fingerprint config = (enabled=%v, chunk=%d), want (true, %d)",
				got.EnableFingerprintIndex, got.FingerprintChunkSize, radixkv.DefaultFingerprintChunkSize)
		}
		if got.QwenQ4KPrefillChunkTokens != inKernelQwenQ4KPrefillChunkTokens {
			t.Fatalf("QwenQ4KPrefillChunkTokens = %d, want default %d", got.QwenQ4KPrefillChunkTokens, inKernelQwenQ4KPrefillChunkTokens)
		}

		p2 := NewInKernelPlannerWithConfig(m, tok, "e2e-config-defaults-2", false, nil, false, got)
		defer p2.Close()
		if !reflect.DeepEqual(p2.RuntimeConfig(), got) {
			t.Fatalf("second-hop RuntimeConfig() mismatch: %+v != %+v", p2.RuntimeConfig(), got)
		}
	})
}

// fak-test:runtime fast est=1500ms lane=default
func TestPrimeCacheMultiTurnToolDispatchEndToEnd(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "")
	t.Setenv("FAK_INKERNEL_BATCH", "")

	root := t.TempDir()
	notePath := filepath.Join(root, "note.txt")
	if err := os.WriteFile(notePath, []byte("e2e prime witness content\n"), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	catalog, err := ArmCodeToolsWithOptions(CodeToolsOptions{
		Root:         root,
		EnableSkills: false,
	})
	if err != nil {
		t.Fatalf("ArmCodeToolsWithOptions: %v", err)
	}
	t.Cleanup(DisarmCodeTools)

	m := newPrimeE2EQuantModel()
	tok := loadProbeTok(t)
	const sid = "sess-loop-e2e"
	p := NewInKernelPlannerWithConfig(m, tok, "e2e-tool-dispatch", false, nil, false, InKernelPlannerConfig{
		ContextTokens:        32768,
		RadixBudgetTokens:    32768,
		NextTurnPrimeEnabled: true,
		NextTurnPrimeConfig: NextTurnPrimeConfig{
			MaxConcurrent: 2,
			Timeout:       10 * time.Second,
		},
		RadixSessionID: sid,
	})
	defer p.Close()

	var primeCompleted atomic.Int64
	runner := p.NextTurnPrimeRunner()
	if runner == nil {
		t.Fatal("expected wired NextTurnPrimeRunner")
	}
	runner.onResult = func(usable bool, perr error) {
		if usable && perr == nil {
			primeCompleted.Add(1)
		}
	}

	adapter := &multiTurnPrimeTestPlanner{
		planner:  p,
		toolPath: "note.txt",
	}
	var trace []traceEvent
	metrics, err := RunArm(
		context.Background(),
		adapter,
		"Read note.txt and summarize what you found.",
		true,
		4,
		&trace,
		append([]RunOption{WithToolCatalog(catalog)}, p.NextTurnPrimeRunOptions()...)...,
	)
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	p.WaitNextTurnPrime()

	if metrics.Turns != 2 {
		t.Fatalf("metrics.Turns = %d, want 2", metrics.Turns)
	}
	if st := p.NextTurnPrimeStats(); st.Started != 1 || st.Primed != 1 || st.Failed != 0 {
		t.Fatalf("NextTurnPrimeStats = %+v, want Started=1 Primed=1 Failed=0", st)
	}
	if primeCompleted.Load() != 1 {
		t.Fatalf("prime onResult completed = %d, want 1", primeCompleted.Load())
	}
	if adapter.coldTurn != 0 {
		t.Fatalf("turn 1 cached = %d, want 0", adapter.coldTurn)
	}
	if len(adapter.reused) != 1 || adapter.reused[0] == 0 {
		t.Fatalf("turn 2 cached tokens = %v, want positive reuse from NextTurnPrime", adapter.reused)
	}
	if immune := p.SessionImmuneNodes(); immune == 0 {
		t.Fatalf("SessionImmuneNodes() = %d, want > 0 for active %q", immune, sid)
	}
	if metrics.CachedTokens == 0 {
		t.Fatalf("ArmMetrics.CachedTokens = 0, want positive primed reuse")
	}

	// Verify the tool dispatch ran Read through the kernel and compare primed vs cold on Turn 2.
	sawDispatchedRead := false
	for _, ev := range trace {
		if ev.Tool == "Read" && (ev.Verdict == "ALLOW" || ev.Verdict == "TRANSFORM") {
			sawDispatchedRead = true
			break
		}
	}
	if !sawDispatchedRead {
		t.Fatalf("expected dispatched Read event (ALLOW/TRANSFORM) in trace, got %+v", trace)
	}
	toolMsgIdx := -1
	for i, msg := range adapter.turn2Msgs {
		if msg.Role == RoleTool {
			toolMsgIdx = i
			break
		}
	}
	if toolMsgIdx <= 0 {
		t.Fatalf("expected RoleTool message in turn2Msgs, got %+v", adapter.turn2Msgs)
	}
	if !strings.Contains(adapter.turn2Msgs[toolMsgIdx].Content, "e2e prime witness content") {
		t.Fatalf("turn 2 tool message missing Read fixture content: %q", adapter.turn2Msgs[toolMsgIdx].Content)
	}
	primeTurn2Enc, err := p.EncodePrompt(context.Background(), adapter.turn2Msgs[:toolMsgIdx], catalog)
	if err != nil {
		t.Fatalf("EncodePrompt: %v", err)
	}
	if cached := p.CachedPrefixLen(context.Background(), primeTurn2Enc.TokenIDs); cached != len(primeTurn2Enc.TokenIDs) {
		t.Fatalf("primed prefix cached %d/%d tokens", cached, len(primeTurn2Enc.TokenIDs))
	}
	cold := NewInKernelPlannerWithConfig(m, tok, "e2e-cold-ref", false, nil, false, InKernelPlannerConfig{
		ContextTokens: 32768,
	})
	defer cold.Close()
	coldComp, err := cold.Complete(context.Background(), adapter.turn2Msgs, catalog, WithMaxTokens(4))
	if err != nil {
		t.Fatalf("cold Complete: %v", err)
	}
	if coldComp.Message.Content != adapter.turn2Content {
		t.Fatalf("turn 2 primed decode diverged from cold baseline:\nprimed=%q\ncold=%q",
			adapter.turn2Content, coldComp.Message.Content)
	}

	// Verify device-serialized (p.devMu) + scoped PrimeRadixSession on GLM-DSA cpu-ref backend.
	be, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend not registered")
	}
	devPlanner := NewInKernelPlannerWithConfig(newPrimeE2EDeviceGLMModel(), tok, "e2e-dev-glm", false, be, false, InKernelPlannerConfig{
		RadixSessionID:    "sess-dev-glm",
		RadixBudgetTokens: 2048,
	})
	defer devPlanner.Close()

	scopedCtx := WithPrefixCacheIdentity(context.Background(), "tenant-dev", "agent-1")
	devMsgs := []Message{
		{Role: RoleSystem, Content: "Device serialized prime witness."},
		{Role: RoleUser, Content: "Confirm scoped device prime."},
	}
	devGen, ok := devPlanner.RadixSessionGen("sess-dev-glm")
	if !ok || devGen == 0 {
		t.Fatalf("device session gen = (%d, %v), want active", devGen, ok)
	}

	// Hold p.devMu briefly to prove PrimeRadixSession serializes cleanly behind devMu.
	devPlanner.devMu.Lock()
	done := make(chan struct{})
	var rec CachePrimeReceipt
	var covered int
	var primeErr error
	go func() {
		defer close(done)
		rec, covered, primeErr = devPlanner.PrimeRadixSessionMessages(scopedCtx, "sess-dev-glm", devGen, devMsgs, nil)
	}()
	select {
	case <-done:
		devPlanner.devMu.Unlock()
		t.Fatal("PrimeRadixSessionMessages returned while p.devMu was held")
	case <-time.After(30 * time.Millisecond):
	}
	devPlanner.devMu.Unlock()
	<-done

	if primeErr != nil {
		t.Fatalf("PrimeRadixSessionMessages: %v", primeErr)
	}
	if !rec.Usable() || covered == 0 || covered != rec.Tokens {
		t.Fatalf("scoped device prime = rec=%+v covered=%d, want usable and full session coverage", rec, covered)
	}
	if devPlanner.SessionImmuneNodes() == 0 {
		t.Fatal("expected immune nodes on scoped device radix tree")
	}
}

func commonPrefixLen(a, b []int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// fak-test:runtime fast est=2s lane=default
func TestPrimeCacheSessionRefsSurviveEvictionAndQuarantineBitExact(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "")
	t.Setenv("FAK_INKERNEL_KVMMU", "")

	t.Run("active session refs survive memory pressure while closed session and unreferenced leaves evict", func(t *testing.T) {
		m := newPrimeE2EQuantModel()
		tok := loadProbeTok(t)
		ctx := context.Background()

		sysMsg := Message{Role: RoleSystem, Content: "Shared system prefix for radix ref eviction witness."}
		activeMsgs := []Message{sysMsg, {Role: RoleUser, Content: "Alpha active session prefix kept alive during pressure."}}
		closedMsgs := []Message{sysMsg, {Role: RoleUser, Content: "Zebra closed session tail released and evicted on pressure."}}
		pressureC := []Message{{Role: RoleSystem, Content: "Disjoint pressure branch Charlie occupies budget."}}
		pressureD := []Message{{Role: RoleSystem, Content: "Disjoint pressure branch Delta triggers eviction."}}

		probe := NewInKernelPlannerWithConfig(m, tok, "e2e-probe-len", false, nil, false, InKernelPlannerConfig{})
		defer probe.Close()
		activeEnc, err := probe.EncodePrompt(ctx, activeMsgs, nil)
		if err != nil {
			t.Fatalf("EncodePrompt active: %v", err)
		}
		closedEnc, err := probe.EncodePrompt(ctx, closedMsgs, nil)
		if err != nil {
			t.Fatalf("EncodePrompt closed: %v", err)
		}
		sharedPrefixLen := commonPrefixLen(activeEnc.TokenIDs, closedEnc.TokenIDs)
		if sharedPrefixLen == 0 || sharedPrefixLen >= len(closedEnc.TokenIDs) {
			t.Fatalf("unexpected sharedPrefixLen=%d for closed len=%d", sharedPrefixLen, len(closedEnc.TokenIDs))
		}
		cEnc, err := probe.EncodePrompt(ctx, pressureC, nil)
		if err != nil {
			t.Fatalf("EncodePrompt C: %v", err)
		}
		dEnc, err := probe.EncodePrompt(ctx, pressureD, nil)
		if err != nil {
			t.Fatalf("EncodePrompt D: %v", err)
		}

		// Budget fits active branch + one pressure branch, so inserting closed + C + D
		// forces LRU eviction of unlocked/released leaves while active immune nodes survive.
		budget := len(activeEnc.TokenIDs) + len(dEnc.TokenIDs) + 4
		p := NewInKernelPlannerWithConfig(m, tok, "e2e-evict-immunity", false, nil, false, InKernelPlannerConfig{
			RadixBudgetTokens: budget,
			RadixSessionID:    "sess-active",
		})
		defer p.Close()

		genActive, ok := p.RadixSessionGen("sess-active")
		if !ok {
			t.Fatal("missing initial sess-active generation")
		}
		recA, covA, err := p.PrimeRadixSession(ctx, "sess-active", genActive, activeEnc.TokenIDs)
		if err != nil || !recA.Usable() || covA != len(activeEnc.TokenIDs) {
			t.Fatalf("PrimeRadixSession active: rec=%+v cov=%d err=%v", recA, covA, err)
		}

		genClosed := p.OpenRadixSession("sess-closed")
		recB, covB, err := p.PrimeRadixSession(ctx, "sess-closed", genClosed, closedEnc.TokenIDs)
		if err != nil || !recB.Usable() || covB != len(closedEnc.TokenIDs) {
			t.Fatalf("PrimeRadixSession closed: rec=%+v cov=%d err=%v", recB, covB, err)
		}
		if nodes := p.SessionImmuneNodes(); nodes == 0 {
			t.Fatal("SessionImmuneNodes() = 0 before release")
		}

		if !p.ReleaseRadixSession("sess-closed") {
			t.Fatal("ReleaseRadixSession(sess-closed) returned false")
		}
		if !p.RadixSessionClosed("sess-closed") {
			t.Fatal("RadixSessionClosed(sess-closed) = false after release")
		}
		// Late register with stale generation after close must be ignored.
		if late := p.RegisterRadixSessionLeaf(ctx, "sess-closed", genClosed, closedEnc.TokenIDs); late != 0 {
			t.Fatalf("late register after close covered %d tokens, want 0", late)
		}

		// Drive unreferenced disjoint pressure turns that exceed RadixBudgetTokens.
		if _, err := p.PrimePrefixState(ctx, cEnc.TokenIDs); err != nil {
			t.Fatalf("PrimePrefixState C: %v", err)
		}
		if _, err := p.PrimePrefixState(ctx, dEnc.TokenIDs); err != nil {
			t.Fatalf("PrimePrefixState D: %v", err)
		}

		if st := p.KVMemoryStats(); st.Evictions == 0 {
			t.Fatalf("expected budget evictions > 0 under RadixBudgetTokens=%d, stats=%+v", budget, st)
		}
		if gotActive := p.CachedPrefixLen(ctx, activeEnc.TokenIDs); gotActive != len(activeEnc.TokenIDs) {
			t.Fatalf("active session prefix evicted under pressure: %d/%d", gotActive, len(activeEnc.TokenIDs))
		}
		if gotClosed := p.CachedPrefixLen(ctx, closedEnc.TokenIDs); gotClosed > sharedPrefixLen {
			t.Fatalf("closed session tail survived eviction: cached %d, shared ancestor <= %d", gotClosed, sharedPrefixLen)
		}

		// Bit-exact greedy decode parity on the surviving immune prefix vs a cold baseline.
		warmComp, err := p.Complete(ctx, activeMsgs, nil, WithMaxTokens(6))
		if err != nil {
			t.Fatalf("warm Complete: %v", err)
		}
		if warmComp.Usage.PromptTokensDetails == nil || warmComp.Usage.PromptTokensDetails.CachedTokens != len(activeEnc.TokenIDs) {
			t.Fatalf("expected full cached hit %d on immune prefix, got %+v", len(activeEnc.TokenIDs), warmComp.Usage.PromptTokensDetails)
		}
		cold := NewInKernelPlannerWithConfig(m, tok, "e2e-cold-immune", false, nil, false, InKernelPlannerConfig{})
		defer cold.Close()
		coldComp, err := cold.Complete(ctx, activeMsgs, nil, WithMaxTokens(6))
		if err != nil {
			t.Fatalf("cold Complete: %v", err)
		}
		if warmComp.Message.Content != coldComp.Message.Content {
			t.Fatalf("immune prefix decode diverged from cold:\nwarm=%q\ncold=%q", warmComp.Message.Content, coldComp.Message.Content)
		}
	})

	t.Run("scoped quarantine evicts poisoned branch bit-exactly while preserving immune benign sibling", func(t *testing.T) {
		m := newPrimeE2EQuantModel()
		tok := loadProbeTok(t)
		p := NewInKernelPlannerWithConfig(m, tok, "e2e-quarantine", false, nil, false, InKernelPlannerConfig{
			RadixBudgetTokens: -1,
			RadixSessionID:    "sess-benign",
			KVSpanEvict:       true,
		})
		defer p.Close()

		tools := []ToolDef{{
			Type: "function",
			Function: ToolDefFunction{
				Name:        "fetch_url",
				Description: "fetch URL body",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"url":{"type":"string"}}}`),
			},
		}}
		const secret = "sk-abcdef0123456789abcdef0123"
		baseMsgs := []Message{
			{Role: RoleSystem, Content: "You inspect tool outputs safely."},
			{Role: RoleUser, Content: "Check config endpoints."},
			{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{{
					ID:       "call_fetch",
					Type:     "function",
					Function: Func{Name: "fetch_url", Arguments: `{"url":"https://example.invalid"}`},
				}},
			},
		}
		goodMsgs := append(append([]Message(nil), baseMsgs...), Message{
			Role: RoleTool, ToolCallID: "call_fetch", Name: "fetch_url",
			Content: `{"ok":true,"status":"healthy"}`,
		})
		badMsgs := append(append([]Message(nil), baseMsgs...), Message{
			Role: RoleTool, ToolCallID: "call_fetch", Name: "fetch_url",
			Content: `{"page":"api_key=` + secret + `"}`,
		})

		scopedCtx := WithPrefixCacheIdentity(context.Background(), "tenant-quarantine", "agent-q")
		basePrefixIDs, err := tok.Encode(renderTranscriptTools(baseMsgs, tools))
		if err != nil {
			t.Fatalf("Encode base transcript: %v", err)
		}
		goodEnc, err := p.EncodePrompt(scopedCtx, goodMsgs, tools)
		if err != nil {
			t.Fatalf("EncodePrompt good: %v", err)
		}
		badEnc, err := p.EncodePrompt(scopedCtx, badMsgs, tools)
		if err != nil {
			t.Fatalf("EncodePrompt bad: %v", err)
		}
		badEvictIDs, err := tok.Encode(renderTranscriptTools(badMsgs, tools))
		if err != nil {
			t.Fatalf("Encode bad transcript: %v", err)
		}
		sharedGoodBadPrefix := commonPrefixLen(goodEnc.TokenIDs, badEnc.TokenIDs)
		if sharedGoodBadPrefix < len(basePrefixIDs) || sharedGoodBadPrefix >= len(badEvictIDs) {
			t.Fatalf("unexpected sharedGoodBadPrefix=%d base=%d bad=%d", sharedGoodBadPrefix, len(basePrefixIDs), len(badEvictIDs))
		}

		genBenign, ok := p.RadixSessionGen("sess-benign")
		if !ok {
			t.Fatal("missing sess-benign generation")
		}
		if rec, cov, err := p.PrimeRadixSession(scopedCtx, "sess-benign", genBenign, goodEnc.TokenIDs); err != nil || !rec.Usable() || cov != len(goodEnc.TokenIDs) {
			t.Fatalf("PrimeRadixSession good: rec=%+v cov=%d err=%v", rec, cov, err)
		}

		// Even if another session holds refs on the poisoned path, quarantine policy eviction
		// must still evict the poisoned branch while preserving benign ancestors and siblings.
		genPoison := p.OpenRadixSession("sess-poison")
		if rec, cov, err := p.PrimeRadixSession(scopedCtx, "sess-poison", genPoison, badEnc.TokenIDs); err != nil || !rec.Usable() || cov != len(badEnc.TokenIDs) {
			t.Fatalf("PrimeRadixSession bad: rec=%+v cov=%d err=%v", rec, cov, err)
		}

		immuneBefore := p.SessionImmuneNodes()
		if immuneBefore == 0 {
			t.Fatal("expected immune nodes > 0 before quarantine")
		}

		qRec := p.QuarantineSessionTurn(scopedCtx, badMsgs, len(badMsgs)-1, tools)
		if qRec.RadixTokensFreed <= 0 {
			t.Fatalf("RadixTokensFreed = %d, want > 0", qRec.RadixTokensFreed)
		}
		if qRec.SpanPositionsFreed <= 0 || !qRec.RepositionExact {
			t.Fatalf("KVSpan eviction = (freed=%d exact=%v), want (>0, true)", qRec.SpanPositionsFreed, qRec.RepositionExact)
		}

		// Poisoned transcript prefix must no longer reach past the shared benign tool-output prefix.
		if gotBad := p.CachedPrefixLen(scopedCtx, badEvictIDs); gotBad > sharedGoodBadPrefix {
			t.Fatalf("poisoned branch survived quarantine: cached %d > shared %d (full %d)", gotBad, sharedGoodBadPrefix, len(badEvictIDs))
		}
		// Benign shared transcript prefix and benign sibling branch must stay intact and immune.
		if gotBase := p.CachedPrefixLen(scopedCtx, basePrefixIDs); gotBase != len(basePrefixIDs) {
			t.Fatalf("shared base prefix damaged: cached %d/%d", gotBase, len(basePrefixIDs))
		}
		if gotGood := p.CachedPrefixLen(scopedCtx, goodEnc.TokenIDs); gotGood != len(goodEnc.TokenIDs) {
			t.Fatalf("benign sibling evicted by poison quarantine: cached %d/%d", gotGood, len(goodEnc.TokenIDs))
		}
		if p.SessionImmuneNodes() == 0 {
			t.Fatal("active benign session lost all immune nodes after sibling quarantine")
		}

		// Confirm a subsequent turn on the quarantined transcript cannot replay the poisoned KV.
		recheck, err := p.Complete(scopedCtx, badMsgs, tools, WithMaxTokens(2))
		if err != nil {
			t.Fatalf("post-quarantine Complete: %v", err)
		}
		if recheck.Usage.PromptTokensDetails != nil && recheck.Usage.PromptTokensDetails.CachedTokens > sharedGoodBadPrefix {
			t.Fatalf("post-quarantine turn replayed poisoned KV: cached=%d > shared=%d", recheck.Usage.PromptTokensDetails.CachedTokens, sharedGoodBadPrefix)
		}

		p.Close()
		if got := p.SessionImmuneNodes(); got != 0 {
			t.Fatalf("SessionImmuneNodes = %d after Close, want 0", got)
		}
	})
}

// fak-test:runtime fast est=300ms lane=default
func TestPrimeCacheFingerprintIndexCrossTurnReuse(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "")
	m := newPrimeE2EQuantModel()
	tok := loadProbeTok(t)
	const chunkSize = 32
	p := NewInKernelPlannerWithConfig(m, tok, "e2e-fingerprint", false, nil, false, InKernelPlannerConfig{
		EnableFingerprintIndex: true,
		FingerprintChunkSize:   chunkSize,
		RadixSessionID:         "sess-fp",
	})
	defer p.Close()

	if !p.FingerprintIndexEnabled() {
		t.Fatal("FingerprintIndexEnabled() = false, want true")
	}

	ctx := context.Background()
	sysText := "Stable system instructions for cross-turn context alignment."
	docA := strings.Repeat("alpha-block-context ", 8)
	docB := strings.Repeat("bravo-reusable-document ", 8)
	editInsert := "newly-edited-middle-turn-delta-that-shifts-suffix-offsets "

	encSys, err := p.EncodePrompt(ctx, []Message{{Role: RoleSystem, Content: sysText}}, nil)
	if err != nil {
		t.Fatalf("EncodePrompt sys: %v", err)
	}
	turn1Msgs := []Message{
		{Role: RoleSystem, Content: sysText},
		{Role: RoleUser, Content: docA + docB},
	}
	turn1Enc, err := p.EncodePrompt(ctx, turn1Msgs, nil)
	if err != nil {
		t.Fatalf("EncodePrompt turn1: %v", err)
	}
	// Trim trailing assistant generation marker from turn1 for clean prefix sharing.
	if len(turn1Enc.TokenIDs) <= len(encSys.TokenIDs) {
		t.Fatalf("turn1 too short: %d <= %d", len(turn1Enc.TokenIDs), len(encSys.TokenIDs))
	}

	gen, ok := p.RadixSessionGen("sess-fp")
	if !ok {
		t.Fatal("missing sess-fp generation")
	}
	rec, cov, err := p.PrimeRadixSession(ctx, "sess-fp", gen, turn1Enc.TokenIDs)
	if err != nil || !rec.Usable() || cov != len(turn1Enc.TokenIDs) {
		t.Fatalf("PrimeRadixSession turn1: rec=%+v cov=%d err=%v", rec, cov, err)
	}

	chunksReg, err := p.RegisterFingerprintTokens(turn1Enc.TokenIDs, 0)
	if err != nil || chunksReg == 0 {
		t.Fatalf("RegisterFingerprintTokens = (%d, %v), want >0 chunks", chunksReg, err)
	}
	if gotLen := p.FingerprintIndexLen(); gotLen != chunksReg {
		t.Fatalf("FingerprintIndexLen() = %d, want %d", gotLen, chunksReg)
	}

	// Build turn 2 where an edit is inserted BEFORE docB, shifting docB's token offset
	// so pure radix prefix matching stops before docB, while FingerprintIndex recovers docB.
	turn2Msgs := []Message{
		{Role: RoleSystem, Content: sysText},
		{Role: RoleUser, Content: editInsert + docB},
	}
	turn2Enc, err := p.EncodePrompt(ctx, turn2Msgs, nil)
	if err != nil {
		t.Fatalf("EncodePrompt turn2: %v", err)
	}

	radixMatched := p.CachedPrefixLen(ctx, turn2Enc.TokenIDs)
	if radixMatched >= len(turn2Enc.TokenIDs)-chunkSize {
		t.Fatalf("radix match %d unexpectedly covered shifted docB of total %d", radixMatched, len(turn2Enc.TokenIDs))
	}

	cands := p.MatchFingerprintReuse(turn2Enc.TokenIDs)
	if len(cands) == 0 {
		t.Fatalf("MatchFingerprintReuse returned 0 candidates across shifted docB turn")
	}
	foundShiftedHit := false
	for _, c := range cands {
		if c.Compatible && c.DstEnd-c.DstStart >= chunkSize && c.DstStart != c.SrcStart {
			foundShiftedHit = true
			if !strings.HasPrefix(c.ID, "fp:") {
				t.Fatalf("unexpected candidate ID %q, want fp:<offset>", c.ID)
			}
			// Verify the shifted slice in turn2 matches the registered slice in turn1 token-for-token.
			if !reflect.DeepEqual(turn2Enc.TokenIDs[c.DstStart:c.DstEnd], turn1Enc.TokenIDs[c.SrcStart:c.SrcEnd]) {
				t.Fatalf("matched candidate tokens diverged between dst[%d:%d] and src[%d:%d]",
					c.DstStart, c.DstEnd, c.SrcStart, c.SrcEnd)
			}
		}
	}
	if !foundShiftedHit {
		t.Fatalf("expected shifted ReuseCandidate (DstStart != SrcStart), got %+v", cands)
	}

	plan, err := p.ComposeFingerprintReusePlan(ctx, turn2Enc.TokenIDs, radixkv.DefaultCostModel{TokenCost: 1})
	if err != nil {
		t.Fatalf("ComposeFingerprintReusePlan: %v", err)
	}
	if plan.DirectTokens <= plan.PrefixLen {
		t.Fatalf("ReusePlan did not incorporate FingerprintIndex hits: PrefixLen=%d DirectTokens=%d Spans=%+v",
			plan.PrefixLen, plan.DirectTokens, plan.Spans)
	}
	if plan.SavingsRatio <= 0 {
		t.Fatalf("expected positive SavingsRatio, got %+v", plan)
	}

	if evicted := p.EvictFingerprintOffset(0); evicted != chunksReg {
		t.Fatalf("EvictFingerprintOffset(0) = %d, want %d", evicted, chunksReg)
	}
	if p.FingerprintIndexLen() != 0 {
		t.Fatalf("FingerprintIndexLen() = %d after Evict, want 0", p.FingerprintIndexLen())
	}
	if after := p.MatchFingerprintReuse(turn2Enc.TokenIDs); len(after) != 0 {
		t.Fatalf("MatchFingerprintReuse after Evict = %+v, want empty", after)
	}

	// Unconfigured planner fails closed cleanly.
	disabled := NewInKernelPlannerWithConfig(m, tok, "e2e-fp-off", false, nil, false, InKernelPlannerConfig{})
	defer disabled.Close()
	if _, err := disabled.RegisterFingerprintTokens([]int{1, 2, 3}, 0); !errors.Is(err, ErrFingerprintIndexDisabled) {
		t.Fatalf("disabled RegisterFingerprintTokens err = %v, want ErrFingerprintIndexDisabled", err)
	}
}

// fak-test:runtime fast est=3s lane=default
func TestPrimeCacheConcurrentCompleteAndPrimeRaceFree(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "")

	t.Run("unwrapped CPU Complete concurrently with PrimePrefixState and TriggerNextTurnPrime without waiting", func(t *testing.T) {
		m := newPrimeE2EQuantModel()
		tok := loadProbeTok(t)
		ctx := context.Background()

		p := NewInKernelPlannerWithConfig(m, tok, "e2e-concurrent-cpu", false, nil, false, InKernelPlannerConfig{
			ContextTokens:        2048,
			RadixBudgetTokens:    4096,
			NextTurnPrimeEnabled: true,
			NextTurnPrimeConfig:  NextTurnPrimeConfig{MaxConcurrent: 4, Timeout: 10 * time.Second},
			RadixSessionID:       "sess-concurrent-cpu",
		})
		defer p.Close()

		cold := NewInKernelPlannerWithConfig(m, tok, "e2e-concurrent-cold", false, nil, false, InKernelPlannerConfig{
			ContextTokens: 2048,
		})
		defer cold.Close()

		sys := Message{Role: RoleSystem, Content: "Concurrent CPU Complete vs Prime race witness prefix."}
		demandPrompts := [][]Message{
			{sys, {Role: RoleUser, Content: "Demand worker zero question on shared prefix."}},
			{sys, {Role: RoleUser, Content: "Demand worker one question on shared prefix."}},
			{sys, {Role: RoleUser, Content: "Demand worker two question on shared prefix."}},
			{sys, {Role: RoleUser, Content: "Demand worker three question on shared prefix."}},
		}
		wantOut := make([]string, len(demandPrompts))
		for i, msgs := range demandPrompts {
			comp, err := cold.Complete(ctx, msgs, nil, WithMaxTokens(4))
			if err != nil {
				t.Fatalf("cold Complete[%d]: %v", i, err)
			}
			if comp.Usage.CompletionTokens != 4 || comp.Message.Content == "" {
				t.Fatalf("cold Complete[%d] unexpected completion=%+v", i, comp.Usage)
			}
			wantOut[i] = comp.Message.Content
		}

		primePrompts := [][]Message{
			{sys},
			{sys, {Role: RoleUser, Content: "Prime branch A overlapping prefix state."}},
			{sys, {Role: RoleUser, Content: "Prime branch B overlapping prefix state."}},
		}
		runnerPrompts := [][]Message{
			{sys, {Role: RoleUser, Content: "NextTurnPrime background runner branch X."}},
			{sys, {Role: RoleUser, Content: "NextTurnPrime background runner branch Y."}},
		}
		primeIDs := make([][]int, len(primePrompts))
		for i, msgs := range primePrompts {
			enc, err := p.EncodePrompt(ctx, msgs, nil)
			if err != nil {
				t.Fatalf("EncodePrompt prime[%d]: %v", i, err)
			}
			primeIDs[i] = enc.TokenIDs
		}

		const iters = 10
		start := make(chan struct{})
		var wg sync.WaitGroup
		errCh := make(chan error, 64)

		for w := 0; w < len(demandPrompts); w++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				<-start
				msgs := demandPrompts[worker]
				for iter := 0; iter < iters; iter++ {
					// Never wait on WaitNextTurnPrime(): raw Complete must remain completely
					// isolated from concurrent context-carried cachePopulate primes.
					comp, err := p.Complete(ctx, msgs, nil, WithMaxTokens(4))
					if err != nil {
						errCh <- fmt.Errorf("demand w=%d iter=%d Complete: %w", worker, iter, err)
						return
					}
					if comp.Usage.CompletionTokens != 4 || comp.Message.Content == "" {
						errCh <- fmt.Errorf("demand w=%d iter=%d early empty/zero-gen exit: tokens=%d content=%q",
							worker, iter, comp.Usage.CompletionTokens, comp.Message.Content)
						return
					}
					if comp.Message.Content != wantOut[worker] {
						errCh <- fmt.Errorf("demand w=%d iter=%d content mismatch: got=%q want=%q",
							worker, iter, comp.Message.Content, wantOut[worker])
						return
					}
				}
			}(w)
		}

		for pw := 0; pw < 2; pw++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				<-start
				for iter := 0; iter < iters; iter++ {
					p.TriggerNextTurnPrime(ctx, runnerPrompts[(worker+iter)%len(runnerPrompts)], nil)
					ids := primeIDs[(worker+iter)%len(primeIDs)]
					if _, err := p.PrimePrefixState(ctx, ids); err != nil {
						errCh <- fmt.Errorf("PrimePrefixState pw=%d iter=%d: %w", worker, iter, err)
						return
					}
				}
			}(pw)
		}

		close(start)
		wg.Wait()
		p.WaitNextTurnPrime()
		close(errCh)

		for err := range errCh {
			t.Fatal(err)
		}
		if p.SessionImmuneNodes() == 0 {
			t.Fatal("expected active session immune nodes from concurrent NextTurnPrime triggers")
		}
	})

	t.Run("scoped cover avoids unscoped false skip and skips covered scoped prefixes cleanly", func(t *testing.T) {
		m := newPrimeE2EQuantModel()
		tok := loadProbeTok(t)
		p := NewInKernelPlannerWithConfig(m, tok, "e2e-scoped-cover", false, nil, false, InKernelPlannerConfig{
			ContextTokens:        2048,
			RadixBudgetTokens:    4096,
			NextTurnPrimeEnabled: true,
			NextTurnPrimeConfig:  NextTurnPrimeConfig{MaxConcurrent: 2, Timeout: 10 * time.Second},
			RadixSessionID:       "sess-scoped-cover",
		})
		defer p.Close()

		ctx := context.Background()
		msgs := []Message{
			{Role: RoleSystem, Content: "Scoped cover isolation prefix."},
			{Role: RoleUser, Content: "Keep unscoped vs tenant-A vs tenant-B separate."},
		}
		enc, err := p.EncodePrompt(ctx, msgs, nil)
		if err != nil {
			t.Fatalf("EncodePrompt: %v", err)
		}
		// Seed UNSCOPED tree first so p.cachedPrefixLen(enc.TokenIDs) == len(enc.TokenIDs).
		if rec, err := p.PrimePrefixState(ctx, enc.TokenIDs); err != nil || !rec.Usable() {
			t.Fatalf("unscoped PrimePrefixState: rec=%+v err=%v", rec, err)
		}

		// Trigger for tenant-A must NOT suffer a false SkippedCovered from the unscoped hit.
		tenantA := WithPrefixCacheIdentity(ctx, "tenant-a", "agent-1")
		if !p.TriggerNextTurnPrime(tenantA, msgs, nil) {
			t.Fatal("TriggerNextTurnPrime(tenantA) returned false")
		}
		p.WaitNextTurnPrime()
		st := p.NextTurnPrimeStats()
		if st.Primed != 1 || st.SkippedCovered != 0 {
			t.Fatalf("after first scoped prime stats = %+v, want Primed=1 SkippedCovered=0", st)
		}
		if cached := p.CachedPrefixLen(tenantA, enc.TokenIDs); cached != len(enc.TokenIDs) {
			t.Fatalf("tenantA CachedPrefixLen = %d, want %d", cached, len(enc.TokenIDs))
		}

		// Second trigger for the SAME tenant-A hits scoped cover cleanly without re-priming.
		if !p.TriggerNextTurnPrime(tenantA, msgs, nil) {
			t.Fatal("second TriggerNextTurnPrime(tenantA) returned false")
		}
		p.WaitNextTurnPrime()
		st = p.NextTurnPrimeStats()
		if st.Primed != 1 || st.SkippedCovered != 1 {
			t.Fatalf("after repeat scoped prime stats = %+v, want Primed=1 SkippedCovered=1", st)
		}

		// Concurrent scoped Complete + PrimePrefixState on tenant-B stays isolated and race-free.
		tenantB := WithPrefixCacheIdentity(ctx, "tenant-b", "agent-2")
		cold := NewInKernelPlannerWithConfig(m, tok, "e2e-scoped-cold", false, nil, false, InKernelPlannerConfig{
			ContextTokens: 2048,
		})
		defer cold.Close()
		coldComp, err := cold.Complete(tenantB, msgs, nil, WithMaxTokens(4))
		if err != nil {
			t.Fatalf("cold Complete: %v", err)
		}
		var wg sync.WaitGroup
		errCh := make(chan error, 16)
		start := make(chan struct{})
		for i := 0; i < 4; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				for iter := 0; iter < 6; iter++ {
					comp, err := p.Complete(tenantB, msgs, nil, WithMaxTokens(4))
					if err != nil {
						errCh <- err
						return
					}
					if comp.Usage.CompletionTokens != 4 || comp.Message.Content != coldComp.Message.Content {
						errCh <- fmt.Errorf("scoped Complete mismatch/truncation: tokens=%d got=%q want=%q",
							comp.Usage.CompletionTokens, comp.Message.Content, coldComp.Message.Content)
						return
					}
				}
			}()
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for iter := 0; iter < 6; iter++ {
				if _, err := p.PrimePrefixState(tenantB, enc.TokenIDs); err != nil {
					errCh <- err
					return
				}
				p.TriggerNextTurnPrime(tenantB, msgs, nil)
			}
		}()
		close(start)
		wg.Wait()
		p.WaitNextTurnPrime()
		close(errCh)
		for err := range errCh {
			t.Fatal(err)
		}
	})
}

// fak-test:runtime fast est=200ms lane=default
func TestPrimeCacheWeakExtrasAllowsGCWithoutClose(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "")

	wp := func() weak.Pointer[InKernelPlanner] {
		m := newPrimeE2EQuantModel()
		tok := loadProbeTok(t)
		p := NewInKernelPlannerWithConfig(m, tok, "e2e-gc-no-close", false, nil, false, InKernelPlannerConfig{
			NextTurnPrimeEnabled: true,
			NextTurnPrimeConfig:  NextTurnPrimeConfig{MaxConcurrent: 2, Timeout: 5 * time.Second},
			RadixSessionID:       "sess-gc-weak",
		})
		if p.NextTurnPrimeRunner() == nil {
			t.Fatal("expected wired NextTurnPrimeRunner")
		}
		return weak.Make(p)
	}()

	for i := 0; i < 20; i++ {
		runtime.GC()
		runtime.Gosched()
		if wp.Value() == nil {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("InKernelPlanner remained strongly pinned without Close(); weak reference stayed live across GC")
}
