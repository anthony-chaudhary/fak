package ctxmmu_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestEpisodeCompaction_TokenReduction70Percent proves that grouping turn spans into
// semantic episodes and compacting verbose tool outputs into CAS stubs reduces context
// token bloat by >70% across a simulated 40-turn task.
func TestEpisodeCompaction_TokenReduction70Percent(t *testing.T) {
	tracker := ctxmmu.NewEpisodeTracker(nil)

	// Build a 40-turn context page sequence:
	// Turn 0: System prefix & tools prefix (pinned)
	// Turns 1..10: EpisodeExplore (verbose reads, file catalogs, grep output)
	// Turns 11..20: EpisodeMutate (code edits, patches, write logs)
	// Turns 21..30: EpisodeVerify (test logs, compiler output, lint reports)
	// Turns 31..35: EpisodeRecovery (stack traces, rollback, diagnostics)
	// Turns 36..40: EpisodeVerify (clean test suite run)
	pages := make([]ctxmmu.TokenPage, 0, 160)

	// 1. Immutable Prefix
	sysPrompt := strings.Repeat("System instruction: follow repository invariants. ", 40)
	toolCatalog := strings.Repeat("Tool schema: read, write, edit, test, build, lint. ", 40)

	pages = append(pages, ctxmmu.TokenPage{
		ID:        1,
		TurnIndex: 0,
		Kind:      ctxmmu.PageKindPrefixSystem,
		Role:      "system",
		Content:   []byte(sysPrompt),
		Tokens:    ctxmmu.EstimateTokens([]byte(sysPrompt)),
		Pinned:    true,
	})
	pages = append(pages, ctxmmu.TokenPage{
		ID:        2,
		TurnIndex: 0,
		Kind:      ctxmmu.PageKindPrefixTools,
		Role:      "system",
		Content:   []byte(toolCatalog),
		Tokens:    ctxmmu.EstimateTokens([]byte(toolCatalog)),
		Pinned:    true,
	})

	pageID := uint64(3)

	// Generate 40 turns with conversational turns and verbose tool outputs
	for turn := 1; turn <= 40; turn++ {
		var toolName string
		var toolOutput string

		switch {
		case turn <= 10: // Explore
			toolName = "read"
			// 1,500 - 3,000 bytes of file chatter
			toolOutput = fmt.Sprintf("File: pkg/module_%d.go\n", turn) +
				strings.Repeat("func InspectComponent() error { return nil }\n// Line comment with details\n", 40)
		case turn <= 20: // Mutate
			toolName = "edit"
			// Diff chatter
			toolOutput = fmt.Sprintf("Diff applied to pkg/module_%d.go:\n", turn) +
				strings.Repeat("@@ -10,6 +10,12 @@\n+ func AddedOptimizedKernel() {}\n- func SlowKernel() {}\n", 30)
		case turn <= 30: // Verify
			toolName = "test"
			// Verbose test / compiler runner logs
			toolOutput = fmt.Sprintf("=== RUN TestModule_%d\n", turn) +
				strings.Repeat("--- PASS: TestModule/SubTest (0.01s)\n=== RUN TestModule/Benchmark\n", 35)
		case turn <= 35: // Recovery
			toolName = "error"
			// Compiler error log and stack trace
			toolOutput = fmt.Sprintf("panic: runtime error at turn %d\n", turn) +
				strings.Repeat("goroutine 1 [running]:\nmain.RunStep(0x1234)\n\t/work/fak/internal/ctxmmu/episodes.go:42 +0x2b\n", 30)
		default: // Verify
			toolName = "test"
			toolOutput = fmt.Sprintf("PASS\nok  fak/pkg/module_%d 0.05s\n", turn) +
				strings.Repeat("=== RUN TestCleanSuite/VerifyAll\n--- PASS: TestCleanSuite/VerifyAll (0.00s)\n", 25)
		}

		// User prompt
		userContent := []byte(fmt.Sprintf("User prompt for turn %d: proceed with task", turn))
		pages = append(pages, ctxmmu.TokenPage{
			ID:        pageID,
			TurnIndex: turn,
			Kind:      ctxmmu.PageKindUser,
			Role:      "user",
			Content:   userContent,
			Tokens:    ctxmmu.EstimateTokens(userContent),
		})
		pageID++

		// Assistant reasoning
		asstContent := []byte(fmt.Sprintf("Assistant reasoning for turn %d: invoking tool %s", turn, toolName))
		pages = append(pages, ctxmmu.TokenPage{
			ID:        pageID,
			TurnIndex: turn,
			Kind:      ctxmmu.PageKindAssistant,
			Role:      "assistant",
			Content:   asstContent,
			Tokens:    ctxmmu.EstimateTokens(asstContent),
		})
		pageID++

		// Tool call
		callContent := []byte(fmt.Sprintf(`{"tool": "%s", "args": {"turn": %d}}`, toolName, turn))
		pages = append(pages, ctxmmu.TokenPage{
			ID:        pageID,
			TurnIndex: turn,
			Kind:      ctxmmu.PageKindToolCall,
			Role:      "assistant",
			ToolName:  toolName,
			Content:   callContent,
			Tokens:    ctxmmu.EstimateTokens(callContent),
		})
		pageID++

		// Verbose tool result
		resBytes := []byte(toolOutput)
		pages = append(pages, ctxmmu.TokenPage{
			ID:        pageID,
			TurnIndex: turn,
			Kind:      ctxmmu.PageKindToolResult,
			Role:      "tool",
			ToolName:  toolName,
			Content:   resBytes,
			Tokens:    ctxmmu.EstimateTokens(resBytes),
		})
		pageID++

		// Record turn into EpisodeTracker
		_, _ = tracker.RecordTurnWithMetadata(toolName, resBytes, []string{fmt.Sprintf("module_%d.go", turn)}, nil, nil, ctxmmu.EstimateTokens(resBytes))

		// Transition at episode boundaries
		if turn == 10 {
			_, _ = tracker.Transition(ctxmmu.EpisodeMutate)
		} else if turn == 20 {
			_, _ = tracker.Transition(ctxmmu.EpisodeVerify)
		} else if turn == 30 {
			_, _ = tracker.Transition(ctxmmu.EpisodeRecovery)
		} else if turn == 35 {
			_, _ = tracker.Transition(ctxmmu.EpisodeVerify)
		}
	}

	// Compact pages using the episode tracker CAS compaction
	compactedPages, report, err := tracker.CompactPages(pages)
	if err != nil {
		t.Fatalf("CompactPages failed: %v", err)
	}

	if report.BeforeTokens <= 0 {
		t.Fatalf("invalid BeforeTokens: %d", report.BeforeTokens)
	}

	reduction := float64(report.BeforeTokens-report.AfterTokens) / float64(report.BeforeTokens)
	reductionPct := reduction * 100.0

	t.Logf("Compaction across 40 turns: BeforeTokens=%d, AfterTokens=%d, Reclaimed=%d, Reduction=%.2f%%, Tombstones=%d",
		report.BeforeTokens, report.AfterTokens, report.TokensReclaimed, reductionPct, report.TombstonesCreated)

	if reduction < 0.70 {
		t.Fatalf("token reduction %.2f%% does not exceed required 70%% threshold (before=%d, after=%d)",
			reductionPct, report.BeforeTokens, report.AfterTokens)
	}

	// Verify all tool result pages in middle turns contain valid CAS stubs
	casStubsFound := 0
	for _, p := range compactedPages {
		if p.Kind == ctxmmu.PageKindToolResult {
			if strings.HasPrefix(string(p.Content), "[CAS:sha256:") {
				casStubsFound++
			}
		}
	}

	if casStubsFound != 40 {
		t.Fatalf("expected 40 CAS stubs in compacted pages, found %d", casStubsFound)
	}
}

// TestEpisodeCompaction_ZeroPromptCacheEviction proves that episode transitions and CAS compaction
// leave the immutable prompt prefix 100% byte-identical, guaranteeing zero prompt-cache eviction.
func TestEpisodeCompaction_ZeroPromptCacheEviction(t *testing.T) {
	tracker := ctxmmu.NewEpisodeTracker(nil)

	sysPrefix := []byte("PINNED_SYSTEM_PROMPT: You are fak kernel agent. Invariants: zero data loss, strict DCO.")
	toolPrefix := []byte("PINNED_TOOL_CATALOG: read, write, edit, test, build, lint, rollback, undo.")

	pages := []ctxmmu.TokenPage{
		{
			ID:        1,
			TurnIndex: 0,
			Kind:      ctxmmu.PageKindPrefixSystem,
			Role:      "system",
			Content:   sysPrefix,
			Tokens:    ctxmmu.EstimateTokens(sysPrefix),
			Pinned:    true,
		},
		{
			ID:        2,
			TurnIndex: 0,
			Kind:      ctxmmu.PageKindPrefixTools,
			Role:      "system",
			Content:   toolPrefix,
			Tokens:    ctxmmu.EstimateTokens(toolPrefix),
			Pinned:    true,
		},
		// Middle turn
		{
			ID:        3,
			TurnIndex: 1,
			Kind:      ctxmmu.PageKindToolResult,
			Role:      "tool",
			ToolName:  "read",
			Content:   []byte(strings.Repeat("Verbose file read output line\n", 50)),
			Tokens:    300,
		},
	}

	// Make deep copy of before state
	before := make([]ctxmmu.TokenPage, len(pages))
	for i := range pages {
		before[i] = pages[i]
		before[i].Content = append([]byte(nil), pages[i].Content...)
	}

	after, report, err := tracker.CompactPages(pages)
	if err != nil {
		t.Fatalf("CompactPages failed: %v", err)
	}

	// Verify prefix warmth function
	if !report.PrefixWarm {
		t.Fatalf("expected report.PrefixWarm to be true")
	}
	if !ctxmmu.VerifyPrefixWarmth(before, after) {
		t.Fatalf("VerifyPrefixWarmth returned false across episode compaction boundary")
	}

	// Verify exact byte equality of prefix pages
	if !bytes.Equal(after[0].Content, sysPrefix) {
		t.Fatalf("system prefix mutated: got %q, want %q", after[0].Content, sysPrefix)
	}
	if !bytes.Equal(after[1].Content, toolPrefix) {
		t.Fatalf("tool prefix mutated: got %q, want %q", after[1].Content, toolPrefix)
	}
	if after[0].Tombstone.Active || after[1].Tombstone.Active {
		t.Fatalf("prefix pages must never carry active tombstones")
	}
}

// TestEpisodeCompaction_LosslessCASRetrievalDuringRecovery proves that during recovery turns,
// the agent can losslessly re-fault and hydrate original verbose tool output paged out to CAS.
func TestEpisodeCompaction_LosslessCASRetrievalDuringRecovery(t *testing.T) {
	tracker := ctxmmu.NewEpisodeTracker(nil)

	// Explore phase: read big architecture file
	exploreOutput := []byte("ARCH_DOC: Section 1. Memory hierarchy\n" +
		strings.Repeat("Layer 1: Context MMU\nLayer 2: Prompt Cache\nLayer 3: CAS Store\n", 30))
	_, err := tracker.RecordTurnWithMetadata("read", exploreOutput, []string{"ARCH.md"}, nil, nil, 500)
	if err != nil {
		t.Fatalf("RecordTurn failed: %v", err)
	}

	// Mutate phase
	_, _ = tracker.Transition(ctxmmu.EpisodeMutate)
	mutateOutput := []byte("EDIT_DIFF:\n" + strings.Repeat("@@ -1,5 +1,5 @@\n+ new implementation\n", 20))
	_, _ = tracker.RecordTurnWithMetadata("edit", mutateOutput, []string{"pkg/core.go"}, []string{"10-25"}, nil, 300)

	// Verify phase: test fails with verbose compiler panic
	_, _ = tracker.Transition(ctxmmu.EpisodeVerify)
	verifyOutput := []byte("COMPILER_PANIC: undefined symbol RunKernel at line 42\n" +
		strings.Repeat("stack trace: frame 0xdeadbeef in main.go\n", 40))
	_, _ = tracker.RecordTurnWithMetadata("test", verifyOutput, nil, nil, fmt.Errorf("exit status 2"), 600)

	// Transition to Recovery phase: all previous outputs are paged out to CAS
	_, err = tracker.Transition(ctxmmu.EpisodeRecovery)
	if err != nil {
		t.Fatalf("Transition to Recovery failed: %v", err)
	}

	// Verify that CAS store holds the blobs
	store := tracker.CASStore()
	if store.Size() < 3 {
		t.Fatalf("expected at least 3 blobs in CAS store, got %d", store.Size())
	}

	// 1. Lossless retrieval via CAS hash
	exploreSum := sha256.Sum256(exploreOutput)
	exploreRef := hex.EncodeToString(exploreSum[:])
	retrievedExplore, ok, err := store.Get(exploreRef)
	if err != nil || !ok {
		t.Fatalf("failed to retrieve explore output by hash: ok=%v, err=%v", ok, err)
	}
	if !bytes.Equal(retrievedExplore, exploreOutput) {
		t.Fatalf("lossless retrieval failed for explore output")
	}

	// 2. Lossless retrieval via full CAS stub
	verifySum := sha256.Sum256(verifyOutput)
	verifyStub := fmt.Sprintf("[CAS:sha256:%s 41 lines paged out; summary: test]", hex.EncodeToString(verifySum[:]))
	retrievedVerify, ok, err := store.Get(verifyStub)
	if err != nil || !ok {
		t.Fatalf("failed to retrieve verify output by stub: ok=%v, err=%v", ok, err)
	}
	if !bytes.Equal(retrievedVerify, verifyOutput) {
		t.Fatalf("lossless retrieval failed for verify output")
	}

	// 3. Lossless retrieval via tracker.ResolveCAS
	mutateSum := sha256Sum(mutateOutput)
	resolvedMutate, err := tracker.ResolveCAS(hex.EncodeToString(mutateSum[:]))
	if err != nil {
		t.Fatalf("tracker.ResolveCAS failed: %v", err)
	}
	if !bytes.Equal(resolvedMutate, mutateOutput) {
		t.Fatalf("lossless retrieval failed for mutate output")
	}

	// 4. Hydration helper: replace CAS stubs embedded in conversation prose
	proseWithStubs := fmt.Sprintf("Turn 1 output was %s and verify error was %s. Please diagnose.",
		fmt.Sprintf("[CAS:sha256:%s 91 lines paged out; summary: read]", hex.EncodeToString(exploreSum[:])),
		verifyStub,
	)

	hydrated := tracker.HydrateCAS(proseWithStubs)
	if !strings.Contains(hydrated, "ARCH_DOC: Section 1. Memory hierarchy") {
		t.Fatalf("HydrateCAS did not restore explore output: %s", hydrated)
	}
	if !strings.Contains(hydrated, "COMPILER_PANIC: undefined symbol RunKernel") {
		t.Fatalf("HydrateCAS did not restore verify error: %s", hydrated)
	}
	if strings.Contains(hydrated, "[CAS:sha256:") {
		t.Fatalf("HydrateCAS left unhydrated CAS stubs: %s", hydrated)
	}
}

// TestCASStore_TamperEvidentHashValidation verifies that CASStore detects corrupted or modified
// blobs and refuses them with a typed tamper detection error.
func TestCASStore_TamperEvidentHashValidation(t *testing.T) {
	store := ctxmmu.NewCASStore()

	original := []byte("critical security policy: enforce default-deny capability floor")
	stub, err := store.Put(original, "security policy")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Normal fetch succeeds
	got, ok, err := store.Get(stub)
	if err != nil || !ok {
		t.Fatalf("initial Get failed: ok=%v, err=%v", ok, err)
	}
	if !bytes.Equal(got, original) {
		t.Fatalf("content mismatch before tamper")
	}

	// Corrupt the stored blob
	tampered := []byte("tampered security policy: allow all capability floor (HACKED)")
	if err := store.TamperEntryForTest(stub, tampered); err != nil {
		t.Fatalf("TamperEntryForTest failed: %v", err)
	}

	// Attempting Get must detect the hash mismatch and refuse
	corrupted, ok, err := store.Get(stub)
	if ok {
		t.Fatalf("Get succeeded on tampered entry: %s", string(corrupted))
	}
	if err == nil {
		t.Fatalf("expected tamper detection error, got nil")
	}
	if !strings.Contains(err.Error(), "CAS tamper detected") || !strings.Contains(err.Error(), "hash mismatch") {
		t.Fatalf("unexpected error message: %v", err)
	}

	// ResolveCAS also returns the tamper error
	_, err = store.ResolveCAS(stub)
	if err == nil || !strings.Contains(err.Error(), "tamper detected") {
		t.Fatalf("ResolveCAS did not report tamper: %v", err)
	}

	// Querying a non-existent ref returns ok=false, err=nil
	dummyRef := "sha256:" + strings.Repeat("0", 64)
	missing, ok, err := store.Get(dummyRef)
	if ok || err != nil || missing != nil {
		t.Fatalf("expected missing entry for dummy ref: ok=%v, err=%v", ok, err)
	}

	// Invalid format returns error
	_, ok, err = store.Get("not-a-valid-cas-ref")
	if err == nil {
		t.Fatalf("expected error for invalid CAS ref format")
	}
}

// TestEpisodeTracker_ThreadSafety proves concurrent thread safety across all methods.
func TestEpisodeTracker_ThreadSafety(t *testing.T) {
	tracker := ctxmmu.NewEpisodeTracker(nil)
	const numGoroutines = 40
	const numOpsPerGoroutine = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(gid int) {
			defer wg.Done()
			for op := 0; op < numOpsPerGoroutine; op++ {
				payload := []byte(fmt.Sprintf("tool output from goroutine %d op %d: %s", gid, op, strings.Repeat("data ", 20)))

				// Concurrently record turns
				rec := ctxmmu.EpisodeTurnRecord{
					TurnIndex:   gid*1000 + op,
					ToolName:    "read",
					Output:      payload,
					Files:       []string{fmt.Sprintf("file_%d.go", gid)},
					TargetLines: []string{fmt.Sprintf("line_%d", op)},
					Tokens:      100,
				}
				_, _ = tracker.RecordTurn(rec)

				// Concurrently check active episode and current digest
				_ = tracker.CurrentEpisode()
				_ = tracker.CurrentDigest()

				// Occasionally transition
				if op%10 == 0 {
					var next ctxmmu.EpisodeType
					switch op % 40 {
					case 0:
						next = ctxmmu.EpisodeExplore
					case 10:
						next = ctxmmu.EpisodeMutate
					case 20:
						next = ctxmmu.EpisodeVerify
					default:
						next = ctxmmu.EpisodeRecovery
					}
					_, _ = tracker.Transition(next)
				}

				// Concurrently put and get from CASStore
				stub, err := tracker.CASStore().Put(payload, "summary")
				if err == nil {
					_, _, _ = tracker.CASStore().Get(stub)
					_ = tracker.HydrateCAS(stub)
				}
			}
		}(g)
	}

	wg.Wait()

	digests := tracker.Digests()
	t.Logf("Thread safety pass: completed with %d digests", len(digests))
}

// TestEpisodeTracker_StateTransitionsAndDigests validates that EpisodeTracker maintains
// complete, deterministic, and immutable episode summaries.
func TestEpisodeTracker_StateTransitionsAndDigests(t *testing.T) {
	tracker := ctxmmu.NewEpisodeTracker(nil)

	// Episode 1: Explore
	_, _ = tracker.RecordTurnWithMetadata("read", []byte("file1 contents"), []string{"a.go", "b.go"}, nil, nil, 150)
	_, _ = tracker.RecordTurnWithMetadata("glob", []byte("matched files"), []string{"c.go", "a.go"}, nil, nil, 80)

	d1, err := tracker.Transition(ctxmmu.EpisodeMutate)
	if err != nil {
		t.Fatalf("Transition to Mutate failed: %v", err)
	}

	if d1.Type != ctxmmu.EpisodeExplore {
		t.Errorf("d1 type = %s; want explore", d1.Type)
	}
	if len(d1.DiscoveredFiles) != 3 {
		t.Errorf("d1 discovered files = %v; want 3 files", d1.DiscoveredFiles)
	}
	if d1.ToolCallCount != 2 {
		t.Errorf("d1 tool calls = %d; want 2", d1.ToolCallCount)
	}

	// Defensive copy verification
	d1Files := d1.Files()
	d1Files[0] = "mutated.go"
	if d1.DiscoveredFiles[0] == "mutated.go" {
		t.Fatalf("d1.Files() did not return defensive copy")
	}

	// Episode 2: Mutate
	_, _ = tracker.RecordTurnWithMetadata("edit", []byte("patch 1"), []string{"a.go"}, []string{"1-10", "20-30"}, nil, 200)
	_, _ = tracker.RecordTurnWithMetadata("write", []byte("file d.go"), []string{"d.go"}, []string{"1-5"}, nil, 250)

	d2, err := tracker.Transition(ctxmmu.EpisodeVerify)
	if err != nil {
		t.Fatalf("Transition to Verify failed: %v", err)
	}

	if d2.Type != ctxmmu.EpisodeMutate {
		t.Errorf("d2 type = %s; want mutate", d2.Type)
	}
	if len(d2.TargetLines) != 3 {
		t.Errorf("d2 target lines = %v; want 3", d2.TargetLines)
	}

	// Episode 3: Verify with errors
	_, _ = tracker.RecordTurnWithMetadata("test", []byte("tests failed"), nil, nil, fmt.Errorf("FAIL: TestBroken"), 100)

	d3, err := tracker.Transition(ctxmmu.EpisodeRecovery)
	if err != nil {
		t.Fatalf("Transition to Recovery failed: %v", err)
	}

	if d3.Type != ctxmmu.EpisodeVerify {
		t.Errorf("d3 type = %s; want verify", d3.Type)
	}
	if len(d3.KeyErrors) != 1 || d3.KeyErrors[0] != "FAIL: TestBroken" {
		t.Errorf("d3 key errors = %v; want ['FAIL: TestBroken']", d3.KeyErrors)
	}

	allDigests := tracker.Digests()
	if len(allDigests) != 3 {
		t.Fatalf("expected 3 completed digests, got %d", len(allDigests))
	}
}

// TestHydrateCAS_ComprehensiveRecovery proves that during recovery turns, any CAS references
// embedded in conversation prose, user prompts, or error diagnostics can be losslessly hydrated
// into their full original text.
func TestHydrateCAS_ComprehensiveRecovery(t *testing.T) {
	store := ctxmmu.NewCASStore()

	// 1. Nil store and empty text handling
	if got := ctxmmu.HydrateCAS("hello world", nil); got != "hello world" {
		t.Fatalf("HydrateCAS with nil store: got %q, want %q", got, "hello world")
	}
	if got := ctxmmu.HydrateCAS("", store); got != "" {
		t.Fatalf("HydrateCAS with empty text: got %q, want empty", got)
	}

	// 2. Text without any CAS stubs remains untouched
	plainText := "There are no CAS references in this line of text."
	if got := ctxmmu.HydrateCAS(plainText, store); got != plainText {
		t.Fatalf("HydrateCAS altered plain text: got %q", got)
	}

	// 3. Single and multiple valid stubs
	doc1 := []byte("type Config struct {\n\tTimeout time.Duration\n}")
	doc2 := []byte("panic: test failure: assertion failed in module_test.go:42")
	doc3 := []byte("diff --git a/pkg.go b/pkg.go\n+ func New() *Server {}")

	stub1, err := store.Put(doc1, "config")
	if err != nil {
		t.Fatalf("Put doc1 failed: %v", err)
	}
	stub2, err := store.Put(doc2, "panic log")
	if err != nil {
		t.Fatalf("Put doc2 failed: %v", err)
	}
	stub3, err := store.Put(doc3, "diff")
	if err != nil {
		t.Fatalf("Put doc3 failed: %v", err)
	}

	multiStubText := fmt.Sprintf("Step 1 loaded %s, step 2 failed with %s, and patch was %s.", stub1, stub2, stub3)
	hydrated := ctxmmu.HydrateCAS(multiStubText, store)

	if !strings.Contains(hydrated, "type Config struct") {
		t.Fatalf("hydrated text missing doc1: %s", hydrated)
	}
	if !strings.Contains(hydrated, "assertion failed in module_test.go:42") {
		t.Fatalf("hydrated text missing doc2: %s", hydrated)
	}
	if !strings.Contains(hydrated, "diff --git a/pkg.go b/pkg.go") {
		t.Fatalf("hydrated text missing doc3: %s", hydrated)
	}
	if strings.Contains(hydrated, "[CAS:sha256:") {
		t.Fatalf("hydrated text still contains CAS stub: %s", hydrated)
	}

	// 4. Missing / unknown stub remains as stub
	unknownHash := strings.Repeat("f", 64)
	unknownStub := fmt.Sprintf("[CAS:sha256:%s 10 lines paged out; summary: unknown]", unknownHash)
	textWithUnknown := fmt.Sprintf("Known: %s; Unknown: %s", stub1, unknownStub)
	hydratedMixed := ctxmmu.HydrateCAS(textWithUnknown, store)

	if !strings.Contains(hydratedMixed, "type Config struct") {
		t.Fatalf("hydratedMixed missing doc1: %s", hydratedMixed)
	}
	if !strings.Contains(hydratedMixed, unknownStub) {
		t.Fatalf("hydratedMixed did not preserve unknown stub: %s", hydratedMixed)
	}

	// 5. Tampered stub remains as stub because Get fails verification
	tamperedStub, err := store.Put([]byte("critical uncorrupted code"), "original")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}
	if err := store.TamperEntryForTest(tamperedStub, []byte("tampered content")); err != nil {
		t.Fatalf("TamperEntryForTest failed: %v", err)
	}
	textWithTampered := fmt.Sprintf("Result: %s", tamperedStub)
	hydratedTampered := ctxmmu.HydrateCAS(textWithTampered, store)
	if !strings.Contains(hydratedTampered, tamperedStub) {
		t.Fatalf("tampered stub was hydrated when it should have been refused: %s", hydratedTampered)
	}

	// 6. Recovery turn simulation
	recoveryPrompt := fmt.Sprintf("RECOVERY: Previous test failed with %s on target %s. Fix the issue.", stub2, stub1)
	tracker := ctxmmu.NewEpisodeTracker(store)
	recoveryHydrated := tracker.HydrateCAS(recoveryPrompt)
	if !strings.Contains(recoveryHydrated, "assertion failed in module_test.go:42") {
		t.Fatalf("recovery hydration failed to restore failure log: %s", recoveryHydrated)
	}
	if !strings.Contains(recoveryHydrated, "Timeout time.Duration") {
		t.Fatalf("recovery hydration failed to restore config: %s", recoveryHydrated)
	}
}

// TestResolveCAS_DirectAndDefaultStore verifies package-level ResolveCAS and store-level ResolveCAS.
func TestResolveCAS_DirectAndDefaultStore(t *testing.T) {
	// 1. Package-level DefaultCASStore and ResolveCAS
	testData := []byte("default cas store payload for testing")
	stub, err := ctxmmu.DefaultCASStore().Put(testData, "test payload")
	if err != nil {
		t.Fatalf("DefaultCASStore().Put failed: %v", err)
	}

	// Resolve via stub
	res1, err := ctxmmu.ResolveCAS(stub)
	if err != nil {
		t.Fatalf("ResolveCAS(stub) failed: %v", err)
	}
	if !bytes.Equal(res1, testData) {
		t.Fatalf("ResolveCAS(stub) mismatch: got %q, want %q", res1, testData)
	}

	// Resolve via bare hash
	sum := sha256.Sum256(testData)
	hashHex := hex.EncodeToString(sum[:])
	res2, err := ctxmmu.ResolveCAS(hashHex)
	if err != nil {
		t.Fatalf("ResolveCAS(hashHex) failed: %v", err)
	}
	if !bytes.Equal(res2, testData) {
		t.Fatalf("ResolveCAS(hashHex) mismatch: got %q, want %q", res2, testData)
	}

	// 2. ResolveCAS with explicit custom store
	customStore := ctxmmu.NewCASStore()
	customData := []byte("custom store unique payload")
	customStub, err := customStore.Put(customData, "custom")
	if err != nil {
		t.Fatalf("customStore.Put failed: %v", err)
	}

	res3, err := ctxmmu.ResolveCAS(customStub, customStore)
	if err != nil {
		t.Fatalf("ResolveCAS(customStub, customStore) failed: %v", err)
	}
	if !bytes.Equal(res3, customData) {
		t.Fatalf("ResolveCAS custom mismatch: got %q, want %q", res3, customData)
	}

	// Custom stub not found in default store
	_, err = ctxmmu.ResolveCAS(customStub)
	if err == nil {
		t.Fatalf("expected error resolving custom stub in default store, got nil")
	}

	// 3. Error conditions
	// Invalid format
	if _, err := customStore.ResolveCAS("not-a-hash"); err == nil {
		t.Fatalf("expected error for invalid hash format")
	}

	// Missing ref
	missingRef := "sha256:" + strings.Repeat("a", 64)
	if _, err := customStore.ResolveCAS(missingRef); err == nil {
		t.Fatalf("expected error for missing ref")
	}

	// Nil store
	var nilStore *ctxmmu.CASStore
	if _, err := nilStore.ResolveCAS(stub); err == nil {
		t.Fatalf("expected error for nil store ResolveCAS")
	}

	// Tampered entry in store
	tamperedStub, err := customStore.Put([]byte("data to tamper"), "tamper")
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}
	_ = customStore.TamperEntryForTest(tamperedStub, []byte("corrupted data"))
	if _, err := customStore.ResolveCAS(tamperedStub); err == nil {
		t.Fatalf("expected tamper error on ResolveCAS, got nil")
	}

	// 4. Tracker ResolveCAS
	tracker := ctxmmu.NewEpisodeTracker(customStore)
	trackerData := []byte("tracker cas data")
	trackerStub, _ := customStore.Put(trackerData, "tracker")
	resolvedTracker, err := tracker.ResolveCAS(trackerStub)
	if err != nil {
		t.Fatalf("tracker.ResolveCAS failed: %v", err)
	}
	if !bytes.Equal(resolvedTracker, trackerData) {
		t.Fatalf("tracker.ResolveCAS mismatch")
	}

	// Nil tracker ResolveCAS
	var nilTracker *ctxmmu.EpisodeTracker
	if _, err := nilTracker.ResolveCAS(trackerStub); err == nil {
		t.Fatalf("expected error for nil tracker ResolveCAS")
	}
}

// TestVerifyPrefixWarmth_InvariantsAndEdgeCases verifies prefix warmth checking across all invariant conditions.
func TestVerifyPrefixWarmth_InvariantsAndEdgeCases(t *testing.T) {
	// 1. Both empty slices
	if !ctxmmu.VerifyPrefixWarmth(nil, nil) {
		t.Fatalf("VerifyPrefixWarmth(nil, nil) want true, got false")
	}

	makePrefixPages := func() []ctxmmu.TokenPage {
		return []ctxmmu.TokenPage{
			{
				ID:        1,
				TurnIndex: 0,
				Kind:      ctxmmu.PageKindPrefixSystem,
				Role:      "system",
				Content:   []byte("system prompt instructions"),
				Tokens:    10,
				Pinned:    true,
			},
			{
				ID:        2,
				TurnIndex: 0,
				Kind:      ctxmmu.PageKindPrefixTools,
				Role:      "system",
				Content:   []byte("tool catalog schemas"),
				Tokens:    20,
				Pinned:    true,
			},
		}
	}

	// 2. Identical prefix pages
	p1 := makePrefixPages()
	p2 := makePrefixPages()
	if !ctxmmu.VerifyPrefixWarmth(p1, p2) {
		t.Fatalf("VerifyPrefixWarmth on identical prefix pages want true, got false")
	}

	// 3. Middle / tail turns mutated but prefix preserved
	p3 := makePrefixPages()
	p3 = append(p3, ctxmmu.TokenPage{
		ID:        3,
		TurnIndex: 1,
		Kind:      ctxmmu.PageKindToolResult,
		Role:      "tool",
		Content:   []byte("initial tool output"),
		Tokens:    50,
	})
	p4 := makePrefixPages()
	p4 = append(p4, ctxmmu.TokenPage{
		ID:        3,
		TurnIndex: 1,
		Kind:      ctxmmu.PageKindToolResult,
		Role:      "tool",
		Content:   []byte("[CAS:sha256:1234...]"), // compacted
		Tokens:    5,
	})
	if !ctxmmu.VerifyPrefixWarmth(p3, p4) {
		t.Fatalf("VerifyPrefixWarmth want true when middle turns compacted, got false")
	}

	// 4. Truncated prefix (after has fewer prefix pages than before)
	if ctxmmu.VerifyPrefixWarmth(p1, p1[:1]) {
		t.Fatalf("VerifyPrefixWarmth want false when prefix truncated, got true")
	}

	// 5. Content mutated in prefix
	pMutContent := makePrefixPages()
	pMutContent[0].Content = []byte("mutated system prompt")
	if ctxmmu.VerifyPrefixWarmth(p1, pMutContent) {
		t.Fatalf("VerifyPrefixWarmth want false when prefix content modified, got true")
	}

	// 6. Kind mutated in prefix
	pMutKind := makePrefixPages()
	pMutKind[0].Kind = ctxmmu.PageKindUser
	if ctxmmu.VerifyPrefixWarmth(p1, pMutKind) {
		t.Fatalf("VerifyPrefixWarmth want false when prefix kind modified, got true")
	}

	// 7. Role mutated in prefix
	pMutRole := makePrefixPages()
	pMutRole[0].Role = "assistant"
	if ctxmmu.VerifyPrefixWarmth(p1, pMutRole) {
		t.Fatalf("VerifyPrefixWarmth want false when prefix role modified, got true")
	}

	// 8. Tokens mutated in prefix
	pMutTokens := makePrefixPages()
	pMutTokens[0].Tokens = 999
	if ctxmmu.VerifyPrefixWarmth(p1, pMutTokens) {
		t.Fatalf("VerifyPrefixWarmth want false when prefix tokens modified, got true")
	}

	// 9. TurnIndex mutated in prefix
	pMutTurn := makePrefixPages()
	pMutTurn[0].TurnIndex = 5
	if ctxmmu.VerifyPrefixWarmth(p1, pMutTurn) {
		t.Fatalf("VerifyPrefixWarmth want false when prefix turn index modified, got true")
	}
}

// TestMMU_EpisodeTrackerWiring proves that MMU and Compactor provide first-class helpers
// to construct and operate EpisodeTracker instances.
func TestMMU_EpisodeTrackerWiring(t *testing.T) {
	mmu := ctxmmu.New()

	// 1. EpisodeTracker from MMU
	tracker := mmu.EpisodeTracker()
	if tracker == nil {
		t.Fatalf("mmu.EpisodeTracker() returned nil")
	}
	if tracker.CurrentEpisode() != ctxmmu.EpisodeExplore {
		t.Fatalf("expected initial episode Explore, got %s", tracker.CurrentEpisode())
	}

	// 2. Tracker with custom store
	customStore := ctxmmu.NewCASStore()
	tracker2 := mmu.EpisodeTracker(customStore)
	if tracker2.CASStore() != customStore {
		t.Fatalf("tracker2 CASStore mismatch")
	}

	// 3. CompactEpisodes on MMU
	sysPrefix := []byte("system instructions")
	toolOutput := []byte(strings.Repeat("verbose tool output line\n", 40))
	pages := []ctxmmu.TokenPage{
		{
			ID:        1,
			TurnIndex: 0,
			Kind:      ctxmmu.PageKindPrefixSystem,
			Role:      "system",
			Content:   sysPrefix,
			Tokens:    ctxmmu.EstimateTokens(sysPrefix),
			Pinned:    true,
		},
		{
			ID:        2,
			TurnIndex: 1,
			Kind:      ctxmmu.PageKindToolResult,
			Role:      "tool",
			ToolName:  "test",
			Content:   toolOutput,
			Tokens:    ctxmmu.EstimateTokens(toolOutput),
		},
	}

	compacted, report, err := mmu.CompactEpisodes(pages, tracker)
	if err != nil {
		t.Fatalf("mmu.CompactEpisodes failed: %v", err)
	}
	if !report.PrefixWarm {
		t.Fatalf("mmu.CompactEpisodes report.PrefixWarm = false")
	}
	if report.TombstonesCreated != 1 {
		t.Fatalf("expected 1 tombstone, got %d", report.TombstonesCreated)
	}
	if !bytes.Equal(compacted[0].Content, sysPrefix) {
		t.Fatalf("system prefix modified during mmu.CompactEpisodes")
	}

	// 4. Compactor EpisodeTracker wiring
	compactor := mmu.Compactor()
	cTracker := compactor.EpisodeTracker()
	if cTracker == nil {
		t.Fatalf("compactor.EpisodeTracker() returned nil")
	}
	compacted2, report2, err := compactor.CompactEpisodes(pages, cTracker)
	if err != nil {
		t.Fatalf("compactor.CompactEpisodes failed: %v", err)
	}
	if !report2.PrefixWarm {
		t.Fatalf("compactor.CompactEpisodes report2.PrefixWarm = false")
	}
	if len(compacted2) != len(pages) {
		t.Fatalf("compacted pages count mismatch: got %d, want %d", len(compacted2), len(pages))
	}
}

// TestEpisodeTracker_NilSafetyAndAllTurns verifies nil resilience across all methods and
// validates defensive copying on AllTurns.
func TestEpisodeTracker_NilSafetyAndAllTurns(t *testing.T) {
	var nilTracker *ctxmmu.EpisodeTracker

	// Nil tracker method calls must not panic
	if nilTracker.CASStore() != nil {
		t.Fatalf("nilTracker.CASStore() want nil")
	}
	if nilTracker.CurrentEpisode() != ctxmmu.EpisodeExplore {
		t.Fatalf("nilTracker.CurrentEpisode() want Explore")
	}
	if nilTracker.EpisodeIndex() != 0 {
		t.Fatalf("nilTracker.EpisodeIndex() want 0")
	}
	if nilTracker.Digests() != nil {
		t.Fatalf("nilTracker.Digests() want nil")
	}
	if nilTracker.AllTurns() != nil {
		t.Fatalf("nilTracker.AllTurns() want nil")
	}
	if _, err := nilTracker.RecordTurn(ctxmmu.EpisodeTurnRecord{}); err == nil {
		t.Fatalf("nilTracker.RecordTurn want error")
	}
	if _, err := nilTracker.Transition(ctxmmu.EpisodeMutate); err == nil {
		t.Fatalf("nilTracker.Transition want error")
	}
	if got := nilTracker.HydrateCAS("test"); got != "test" {
		t.Fatalf("nilTracker.HydrateCAS want 'test'")
	}
	if _, err := nilTracker.ResolveCAS("sha256:1234"); err == nil {
		t.Fatalf("nilTracker.ResolveCAS want error")
	}
	if init, comp, red := nilTracker.TokenStats(); init != 0 || comp != 0 || red != 0 {
		t.Fatalf("nilTracker.TokenStats want 0s")
	}

	// Nil CASStore method calls must not panic
	var nilStore *ctxmmu.CASStore
	if nilStore.Size() != 0 {
		t.Fatalf("nilStore.Size() want 0")
	}
	if _, err := nilStore.Put([]byte("data"), "test"); err == nil {
		t.Fatalf("nilStore.Put want error")
	}
	if _, _, err := nilStore.Get("ref"); err == nil {
		t.Fatalf("nilStore.Get want error")
	}
	if err := nilStore.TamperEntryForTest("ref", []byte("bad")); err == nil {
		t.Fatalf("nilStore.TamperEntryForTest want error")
	}

	// Test AllTurns() defensive copy on live tracker
	tracker := ctxmmu.NewEpisodeTracker(nil)
	rec1 := ctxmmu.EpisodeTurnRecord{TurnIndex: 1, ToolName: "read", Tokens: 50}
	rec2 := ctxmmu.EpisodeTurnRecord{TurnIndex: 2, ToolName: "edit", Tokens: 60}
	_, _ = tracker.RecordTurn(rec1)
	_, _ = tracker.Transition(ctxmmu.EpisodeMutate)
	_, _ = tracker.RecordTurn(rec2)
	_, _ = tracker.Transition(ctxmmu.EpisodeVerify)

	all := tracker.AllTurns()
	if len(all) != 2 {
		t.Fatalf("expected 2 turns in AllTurns(), got %d", len(all))
	}
	// Verify defensive copy
	all[0].ToolName = "mutated_tool"
	freshAll := tracker.AllTurns()
	if freshAll[0].ToolName == "mutated_tool" {
		t.Fatalf("AllTurns() did not return a defensive copy")
	}

	// TokenStats calculation when 0 initial tokens
	emptyTracker := ctxmmu.NewEpisodeTracker(nil)
	init, comp, red := emptyTracker.TokenStats()
	if init != 0 || comp != 0 || red != 0.0 {
		t.Fatalf("emptyTracker.TokenStats() = (%d, %d, %f); want 0, 0, 0.0", init, comp, red)
	}
}

func BenchmarkEpisodeCompaction_40Turns(b *testing.B) {
	tracker := ctxmmu.NewEpisodeTracker(nil)
	pages := make([]ctxmmu.TokenPage, 40)
	payload := []byte(strings.Repeat("Verbose tool output content\n", 40))

	for i := 0; i < 40; i++ {
		pages[i] = ctxmmu.TokenPage{
			ID:        uint64(i + 1),
			TurnIndex: i + 1,
			Kind:      ctxmmu.PageKindToolResult,
			Role:      "tool",
			ToolName:  "test",
			Content:   payload,
			Tokens:    ctxmmu.EstimateTokens(payload),
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = tracker.CompactPages(pages)
	}
}

func BenchmarkCASStore_PutGet(b *testing.B) {
	store := ctxmmu.NewCASStore()
	data := []byte(strings.Repeat("Content addressed storage payload\n", 20))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stub, _ := store.Put(data, "benchmark")
		_, _, _ = store.Get(stub)
	}
}

func BenchmarkHydrateCAS(b *testing.B) {
	store := ctxmmu.NewCASStore()
	data := []byte("hydrated original data")
	stub, _ := store.Put(data, "test")
	text := fmt.Sprintf("Prefix before stub %s and suffix after stub.", stub)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ctxmmu.HydrateCAS(text, store)
	}
}

func sha256Sum(b []byte) [32]byte {
	return sha256.Sum256(b)
}
