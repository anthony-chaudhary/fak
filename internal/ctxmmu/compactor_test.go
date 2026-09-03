package ctxmmu_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"math/rand"
	"sync"
	"testing"

	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers blob CAS resolver and pager
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// makePayload creates a repeatable test payload of given byte length.
func makePayload(pattern string, length int) []byte {
	b := make([]byte, length)
	pat := []byte(pattern)
	for i := 0; i < length; i++ {
		b[i] = pat[i%len(pat)]
	}
	return b
}

func TestPrefixCacheWarmthPreserved(t *testing.T) {
	c := ctxmmu.NewCompactor(ctxmmu.CompactorConfig{
		WindowSizeK:           2,
		VerboseThresholdBytes: 256,
	})

	sysPrompt := []byte("You are fak agent kernel assistant. Always verify claims with witnesses.")
	toolSchemas := []byte(`[{"name":"bash","description":"execute command"},{"name":"read","description":"read file"}]`)

	pages := []ctxmmu.TokenPage{
		{
			ID:        1,
			TurnIndex: 0,
			Kind:      ctxmmu.PageKindPrefixSystem,
			Role:      "system",
			Content:   append([]byte(nil), sysPrompt...),
			Tokens:    ctxmmu.EstimateTokens(sysPrompt),
			Resident:  true,
			Pinned:    true,
		},
		{
			ID:        2,
			TurnIndex: 0,
			Kind:      ctxmmu.PageKindPrefixTools,
			Role:      "system",
			Content:   append([]byte(nil), toolSchemas...),
			Tokens:    ctxmmu.EstimateTokens(toolSchemas),
			Resident:  true,
			Pinned:    true,
		},
		{
			ID:        3,
			TurnIndex: 1,
			Kind:      ctxmmu.PageKindUser,
			Role:      "user",
			Content:   []byte("Inspect codebase"),
			Tokens:    10,
			Resident:  true,
		},
		{
			ID:        4,
			TurnIndex: 1,
			Kind:      ctxmmu.PageKindToolResult,
			Role:      "tool",
			ToolName:  "bash",
			Content:   makePayload("git log output 1234567890 ", 8192),
			Tokens:    2048,
			Resident:  true,
		},
		{
			ID:        5,
			TurnIndex: 2,
			Kind:      ctxmmu.PageKindUser,
			Role:      "user",
			Content:   []byte("Now check tests"),
			Tokens:    10,
			Resident:  true,
		},
		{
			ID:        6,
			TurnIndex: 3,
			Kind:      ctxmmu.PageKindUser,
			Role:      "user",
			Content:   []byte("Latest user turn"),
			Tokens:    10,
			Resident:  true,
		},
	}

	beforeSnapshot := make([]ctxmmu.TokenPage, len(pages))
	copy(beforeSnapshot, pages)

	var report ctxmmu.CompactionReport
	compacted, err := c.CompactInPlace(pages, &report)
	if err != nil {
		t.Fatalf("compaction error: %v", err)
	}

	// 1. Verify Prefix Cache Warmth witness helper returns true
	if !ctxmmu.VerifyPrefixWarmth(beforeSnapshot, compacted) {
		t.Fatalf("prefix cache warmth broken: prefix pages altered")
	}

	// 2. Direct byte-exact assertion on system prompt and tool definitions
	if !bytes.Equal(compacted[0].Content, sysPrompt) {
		t.Fatalf("system prompt mutated! Want %s, got %s", sysPrompt, compacted[0].Content)
	}
	if !bytes.Equal(compacted[1].Content, toolSchemas) {
		t.Fatalf("tool schemas mutated! Want %s, got %s", toolSchemas, compacted[1].Content)
	}

	if !report.PrefixWarm {
		t.Fatalf("report.PrefixWarm want true, got false")
	}
}

func TestActiveSlidingWindowKTurnsVerbatim(t *testing.T) {
	c := ctxmmu.NewCompactor(ctxmmu.CompactorConfig{
		WindowSizeK:           2,
		VerboseThresholdBytes: 256,
	})

	recentToolResult := makePayload("recent verbose tool output that must stay verbatim ", 4096)

	pages := []ctxmmu.TokenPage{
		// Prefix
		{ID: 1, TurnIndex: 0, Kind: ctxmmu.PageKindPrefixSystem, Role: "system", Content: []byte("System prompt"), Tokens: 20, Resident: true, Pinned: true},
		// Middle turn 1 (should be compacted)
		{ID: 2, TurnIndex: 1, Kind: ctxmmu.PageKindUser, Role: "user", Content: []byte("turn 1 user"), Tokens: 10, Resident: true},
		{ID: 3, TurnIndex: 1, Kind: ctxmmu.PageKindToolResult, Role: "tool", ToolName: "read", Content: makePayload("middle turn payload ", 2048), Tokens: 512, Resident: true},
		// Middle turn 2 (should be compacted)
		{ID: 4, TurnIndex: 2, Kind: ctxmmu.PageKindUser, Role: "user", Content: []byte("turn 2 user"), Tokens: 10, Resident: true},
		{ID: 5, TurnIndex: 2, Kind: ctxmmu.PageKindToolResult, Role: "tool", ToolName: "grep", Content: makePayload("middle turn grep ", 3000), Tokens: 750, Resident: true},
		// Active window turn 3 (WindowSizeK=2, so turns 3 and 4 are in active window)
		{ID: 6, TurnIndex: 3, Kind: ctxmmu.PageKindUser, Role: "user", Content: []byte("turn 3 user"), Tokens: 10, Resident: true},
		{ID: 7, TurnIndex: 3, Kind: ctxmmu.PageKindToolResult, Role: "tool", ToolName: "bash", Content: recentToolResult, Tokens: 1024, Resident: true},
		// Active window turn 4 (most recent turn)
		{ID: 8, TurnIndex: 4, Kind: ctxmmu.PageKindUser, Role: "user", Content: []byte("turn 4 user"), Tokens: 10, Resident: true},
		{ID: 9, TurnIndex: 4, Kind: ctxmmu.PageKindAssistant, Role: "assistant", Content: []byte("turn 4 assistant response"), Tokens: 20, Resident: true},
	}

	var report ctxmmu.CompactionReport
	compacted, err := c.CompactInPlace(pages, &report)
	if err != nil {
		t.Fatalf("CompactInPlace: %v", err)
	}

	if report.TombstonesCreated != 2 {
		t.Fatalf("want 2 tombstones created in middle turns, got %d", report.TombstonesCreated)
	}

	// Verify turn 3 tool result in active window was NOT tombstoned and is byte-identical
	var turn3ToolPage *ctxmmu.TokenPage
	for i := range compacted {
		if compacted[i].TurnIndex == 3 && compacted[i].Kind == ctxmmu.PageKindToolResult {
			turn3ToolPage = &compacted[i]
			break
		}
	}
	if turn3ToolPage == nil {
		t.Fatalf("turn 3 tool result page missing from compacted context")
	}
	if turn3ToolPage.Tombstone.Active {
		t.Fatalf("active window tool result was tombstoned! Should remain verbatim")
	}
	if !bytes.Equal(turn3ToolPage.Content, recentToolResult) {
		t.Fatalf("active window tool result bytes mutated")
	}
	if turn3ToolPage.Tokens != 1024 {
		t.Fatalf("active window tokens changed: want 1024, got %d", turn3ToolPage.Tokens)
	}
}

func TestIntermediateTurnsTombstonedAndSmallToolRetained(t *testing.T) {
	c := ctxmmu.NewCompactor(ctxmmu.CompactorConfig{
		WindowSizeK:           1, // only turn 3 is active
		VerboseThresholdBytes: 512,
		GenerateSummary:       true,
	})

	largePayload := makePayload("line 1 of verbose build log\nline 2 of verbose build log\n", 8192)
	expectedDigest := sha256.Sum256(largePayload)
	smallPayload := []byte(`{"status":"ok","count":3}`)

	pages := []ctxmmu.TokenPage{
		{ID: 1, TurnIndex: 0, Kind: ctxmmu.PageKindPrefixSystem, Role: "system", Content: []byte("System"), Tokens: 10, Resident: true, Pinned: true},
		// Middle turn 1: large tool result
		{ID: 2, TurnIndex: 1, Kind: ctxmmu.PageKindUser, Role: "user", Content: []byte("build project"), Tokens: 10, Resident: true},
		{ID: 3, TurnIndex: 1, Kind: ctxmmu.PageKindToolCall, Role: "assistant", ToolName: "build", Content: []byte(`{"target":"all"}`), Tokens: 15, Resident: true},
		{ID: 4, TurnIndex: 1, Kind: ctxmmu.PageKindToolResult, Role: "tool", ToolName: "build", Content: largePayload, Tokens: 2048, Resident: true},
		{ID: 5, TurnIndex: 1, Kind: ctxmmu.PageKindAssistant, Role: "assistant", Content: []byte("Build succeeded."), Tokens: 12, Resident: true},
		// Middle turn 2: small tool result
		{ID: 6, TurnIndex: 2, Kind: ctxmmu.PageKindUser, Role: "user", Content: []byte("check health"), Tokens: 10, Resident: true},
		{ID: 7, TurnIndex: 2, Kind: ctxmmu.PageKindToolResult, Role: "tool", ToolName: "ping", Content: smallPayload, Tokens: 8, Resident: true},
		// Active turn 3
		{ID: 8, TurnIndex: 3, Kind: ctxmmu.PageKindUser, Role: "user", Content: []byte("finalize"), Tokens: 10, Resident: true},
	}

	var report ctxmmu.CompactionReport
	compacted, err := c.CompactInPlace(pages, &report)
	if err != nil {
		t.Fatalf("CompactInPlace: %v", err)
	}

	if report.TombstonesCreated != 1 {
		t.Fatalf("want 1 tombstone created, got %d", report.TombstonesCreated)
	}

	// Page 4 (large tool result) must be tombstoned
	page4 := compacted[3]
	if page4.Kind != ctxmmu.PageKindToolResult || !page4.Tombstone.Active {
		t.Fatalf("page 4 was not tombstoned: kind=%v active=%v", page4.Kind, page4.Tombstone.Active)
	}
	if !ctxmmu.IsTombstone(page4.Content) {
		t.Fatalf("page 4 content is not recognized as tombstone: %s", page4.Content)
	}
	parsedTomb, ok := ctxmmu.ParseTombstone(page4.Content)
	if !ok {
		t.Fatalf("failed to parse tombstone JSON: %s", page4.Content)
	}
	if parsedTomb.OriginalBytes != 8192 {
		t.Errorf("tombstone OriginalBytes: want 8192, got %d", parsedTomb.OriginalBytes)
	}
	if parsedTomb.Tool != "build" {
		t.Errorf("tombstone Tool: want 'build', got %q", parsedTomb.Tool)
	}
	if parsedTomb.Summary == "" {
		t.Errorf("tombstone Summary should be non-empty when GenerateSummary=true")
	}
	if page4.Tombstone.Digest != expectedDigest {
		t.Errorf("tombstone digest mismatch")
	}

	// Page 7 (small tool result) must remain untouched
	page7 := compacted[6]
	if page7.Kind != ctxmmu.PageKindToolResult || page7.Tombstone.Active {
		t.Fatalf("small tool result should NOT be tombstoned")
	}
	if !bytes.Equal(page7.Content, smallPayload) {
		t.Fatalf("small tool result content mutated: want %s, got %s", smallPayload, page7.Content)
	}
}

func TestMultiPageToolResultReclaim(t *testing.T) {
	c := ctxmmu.NewCompactor(ctxmmu.CompactorConfig{
		WindowSizeK:           1,
		VerboseThresholdBytes: 256,
	})

	fl := c.Freelist()
	p1 := fl.Get()
	p1.ID = 10
	p1.TurnIndex = 1
	p1.Kind = ctxmmu.PageKindToolResult
	p1.ToolName = "dump"
	p1.Content = append(p1.Content[:0], makePayload("chunk1", 1024)...)
	p1.Tokens = 256
	p1.Resident = true

	p2 := fl.Get()
	p2.ID = 11
	p2.TurnIndex = 1
	p2.Kind = ctxmmu.PageKindToolResult
	p2.ToolName = "dump"
	p2.Content = append(p2.Content[:0], makePayload("chunk2", 1024)...)
	p2.Tokens = 256
	p2.Resident = true
	p2.IsContinuation = true // continuation page

	p3 := fl.Get()
	p3.ID = 12
	p3.TurnIndex = 1
	p3.Kind = ctxmmu.PageKindToolResult
	p3.ToolName = "dump"
	p3.Content = append(p3.Content[:0], makePayload("chunk3", 1024)...)
	p3.Tokens = 256
	p3.Resident = true
	p3.IsContinuation = true // continuation page

	pages := []ctxmmu.TokenPage{
		{ID: 1, TurnIndex: 0, Kind: ctxmmu.PageKindPrefixSystem, Role: "system", Content: []byte("sys"), Tokens: 5, Resident: true, Pinned: true},
		*p1,
		*p2,
		*p3,
		{ID: 20, TurnIndex: 2, Kind: ctxmmu.PageKindUser, Role: "user", Content: []byte("latest"), Tokens: 5, Resident: true},
	}

	var report ctxmmu.CompactionReport
	compacted, err := c.CompactInPlace(pages, &report)
	if err != nil {
		t.Fatalf("CompactInPlace: %v", err)
	}

	// 2 continuation pages should have been reclaimed
	if report.PagesReclaimed != 2 {
		t.Fatalf("want 2 pages reclaimed, got %d", report.PagesReclaimed)
	}

	// Compacted slice should only retain Prefix, p1 (tombstone), and Turn 2 user
	if len(compacted) != 3 {
		t.Fatalf("want 3 compacted pages, got %d", len(compacted))
	}
	if !compacted[1].Tombstone.Active {
		t.Fatalf("primary page should be active tombstone")
	}

	// Check freelist statistics
	_, reclaimed, _ := fl.Stats()
	if reclaimed < 2 {
		t.Fatalf("freelist.Stats reclaimed want >= 2, got %d", reclaimed)
	}
}

func TestTombstoneReFault(t *testing.T) {
	ctx := context.Background()
	m := ctxmmu.New()
	c := ctxmmu.NewCompactorWithMMU(ctxmmu.CompactorConfig{
		WindowSizeK:           1,
		VerboseThresholdBytes: 256,
		CASPageOut:            true,
	}, m)

	largeToolOutput := makePayload("re-faultable tool execution payload data 12345 ", 4096)
	originalSaved := append([]byte(nil), largeToolOutput...)

	pages := []ctxmmu.TokenPage{
		{ID: 1, TurnIndex: 0, Kind: ctxmmu.PageKindPrefixSystem, Role: "system", Content: []byte("sys"), Tokens: 5, Resident: true, Pinned: true},
		{ID: 2, TurnIndex: 1, Kind: ctxmmu.PageKindToolResult, Role: "tool", ToolName: "test", Content: largeToolOutput, Tokens: 1024, Resident: true},
		{ID: 3, TurnIndex: 2, Kind: ctxmmu.PageKindUser, Role: "user", Content: []byte("recent"), Tokens: 5, Resident: true},
	}

	var report ctxmmu.CompactionReport
	compacted, err := c.CompactInPlace(pages, &report)
	if err != nil {
		t.Fatalf("CompactInPlace: %v", err)
	}

	tombstonedPage := &compacted[1]
	if !tombstonedPage.Tombstone.Active {
		t.Fatalf("page should be tombstoned")
	}

	// Re-fault from CAS store
	restored, err := c.ReFault(ctx, tombstonedPage)
	if err != nil {
		t.Fatalf("ReFault: %v", err)
	}
	if !bytes.Equal(restored, originalSaved) {
		t.Fatalf("re-faulted bytes do not match original tool output")
	}
}

func TestSlidingWindow100kTokensSimulation(t *testing.T) {
	c := ctxmmu.NewCompactor(ctxmmu.CompactorConfig{
		WindowSizeK:           3,
		VerboseThresholdBytes: 512,
		GenerateSummary:       true,
	})
	sw := ctxmmu.NewSlidingWindow(c)

	// Add 100% warm prefix
	sw.AddPrefixSystem([]byte("System prompt instructions for long-running autonomous coding agent."))
	sw.AddPrefixTools([]byte(`[{"name":"bash","desc":"run bash"},{"name":"grep","desc":"grep files"},{"name":"read","desc":"read file"}]`))

	// Simulate 20 turns of interactive agent work
	for turn := 1; turn <= 20; turn++ {
		sw.AddUserTurn(turn, []byte(fmt.Sprintf("User instruction for turn %d", turn)))
		sw.AddAssistantTurn(turn, []byte(fmt.Sprintf("I will run tool commands for turn %d", turn)))
		sw.AddToolCall(turn, "bash", []byte(fmt.Sprintf(`{"command":"task-step-%d"}`, turn)))
		// Generate large tool output (~20,000 bytes = ~5,000 tokens per turn)
		toolOutput := makePayload(fmt.Sprintf("turn %d stdout chunk: [data-record-payload] ", turn), 20000)
		sw.AddToolResult(turn, "bash", toolOutput)
		sw.AddAssistantTurn(turn, []byte(fmt.Sprintf("Completed step %d successfully.", turn)))
	}

	initialTokens := sw.TotalTokens()
	if initialTokens < 100000 {
		t.Fatalf("expected 100k+ tokens, got %d", initialTokens)
	}

	scan := sw.Scan()
	if scan.InteractiveTurns != 20 {
		t.Errorf("scan.InteractiveTurns: want 20, got %d", scan.InteractiveTurns)
	}
	if scan.ActiveWindowTurns != 3 {
		t.Errorf("scan.ActiveWindowTurns: want 3, got %d", scan.ActiveWindowTurns)
	}
	if scan.ReclaimablePages != 17 { // 20 - 3 = 17 middle turn tool results
		t.Errorf("scan.ReclaimablePages: want 17, got %d", scan.ReclaimablePages)
	}

	// Compact the session window
	report, err := sw.Compact()
	if err != nil {
		t.Fatalf("sw.Compact: %v", err)
	}

	if report.TombstonesCreated != 17 {
		t.Errorf("report.TombstonesCreated: want 17, got %d", report.TombstonesCreated)
	}
	if !report.PrefixWarm {
		t.Errorf("report.PrefixWarm want true, got false")
	}

	afterTokens := sw.TotalTokens()
	if afterTokens >= initialTokens/2 {
		t.Fatalf("expected substantial token reduction: before=%d, after=%d", initialTokens, afterTokens)
	}

	// Verify active window turns (18, 19, 20) are NOT tombstoned
	pages := sw.Pages()
	for _, p := range pages {
		if p.TurnIndex >= 18 && p.Kind == ctxmmu.PageKindToolResult {
			if p.Tombstone.Active {
				t.Errorf("turn %d tool result in active window was incorrectly tombstoned", p.TurnIndex)
			}
		}
	}
}

func TestCompactorConcurrency(t *testing.T) {
	c := ctxmmu.NewCompactor(ctxmmu.CompactorConfig{
		WindowSizeK:           2,
		VerboseThresholdBytes: 256,
	})

	var wg sync.WaitGroup
	const goroutines = 16
	const iterations = 50

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(gid)))
			for it := 0; it < iterations; it++ {
				pages := make([]ctxmmu.TokenPage, 10)
				pages[0] = ctxmmu.TokenPage{ID: 1, TurnIndex: 0, Kind: ctxmmu.PageKindPrefixSystem, Content: []byte("sys"), Tokens: 5, Pinned: true}
				for turn := 1; turn <= 4; turn++ {
					pages[2*turn-1] = ctxmmu.TokenPage{ID: uint64(2 * turn), TurnIndex: turn, Kind: ctxmmu.PageKindUser, Content: []byte("user"), Tokens: 5}
					size := 128
					if rng.Intn(2) == 0 {
						size = 2048
					}
					pages[2*turn] = ctxmmu.TokenPage{
						ID:        uint64(2*turn + 1),
						TurnIndex: turn,
						Kind:      ctxmmu.PageKindToolResult,
						ToolName:  "tool",
						Content:   makePayload("random", size),
						Tokens:    size / 4,
						Resident:  true,
					}
				}

				// Concurrent Scan
				var sReport ctxmmu.ScanReport
				c.ScanInto(pages, &sReport)

				// Concurrent Compact
				var cReport ctxmmu.CompactionReport
				_, err := c.CompactInPlace(pages, &cReport)
				if err != nil {
					t.Errorf("concurrency compaction error: %v", err)
				}
			}
		}(g)
	}
	wg.Wait()
}

// ---------------------------------------------------------------------------
// ZERO-ALLOCATION BENCHMARKS
// ---------------------------------------------------------------------------

func generateBenchmarkPages(totalTurns int) []ctxmmu.TokenPage {
	pages := make([]ctxmmu.TokenPage, 0, totalTurns*4+2)
	// Prefix
	pages = append(pages, ctxmmu.TokenPage{
		ID:        1,
		TurnIndex: 0,
		Kind:      ctxmmu.PageKindPrefixSystem,
		Role:      "system",
		Content:   makePayload("system prompt guidelines for benchmark context testing ", 2048),
		Tokens:    512,
		Resident:  true,
		Pinned:    true,
	})
	pages = append(pages, ctxmmu.TokenPage{
		ID:        2,
		TurnIndex: 0,
		Kind:      ctxmmu.PageKindPrefixTools,
		Role:      "system",
		Content:   makePayload(`[{"name":"bash","desc":"run"},{"name":"grep","desc":"search"}]`, 1024),
		Tokens:    256,
		Resident:  true,
		Pinned:    true,
	})

	for turn := 1; turn <= totalTurns; turn++ {
		pages = append(pages, ctxmmu.TokenPage{
			ID:        uint64(turn*4 + 1),
			TurnIndex: turn,
			Kind:      ctxmmu.PageKindUser,
			Role:      "user",
			Content:   []byte("run command"),
			Tokens:    10,
			Resident:  true,
		})
		pages = append(pages, ctxmmu.TokenPage{
			ID:        uint64(turn*4 + 2),
			TurnIndex: turn,
			Kind:      ctxmmu.PageKindToolCall,
			Role:      "assistant",
			ToolName:  "bash",
			Content:   []byte(`{"cmd":"ls -la"}`),
			Tokens:    15,
			Resident:  true,
		})
		pages = append(pages, ctxmmu.TokenPage{
			ID:        uint64(turn*4 + 3),
			TurnIndex: turn,
			Kind:      ctxmmu.PageKindToolResult,
			Role:      "tool",
			ToolName:  "bash",
			Content:   makePayload("stdout log record line with details 1234567890 ", 8192),
			Tokens:    2048,
			Resident:  true,
		})
		pages = append(pages, ctxmmu.TokenPage{
			ID:        uint64(turn*4 + 4),
			TurnIndex: turn,
			Kind:      ctxmmu.PageKindAssistant,
			Role:      "assistant",
			Content:   []byte("Command completed successfully."),
			Tokens:    12,
			Resident:  true,
		})
	}
	return pages
}

func BenchmarkCompactorScan(b *testing.B) {
	c := ctxmmu.NewCompactor(ctxmmu.CompactorConfig{
		WindowSizeK:           4,
		VerboseThresholdBytes: 512,
	})
	pages := generateBenchmarkPages(25) // 25 turns = ~100 pages, ~50k tokens

	var report ctxmmu.ScanReport
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		c.ScanInto(pages, &report)
	}
}

func BenchmarkCompactorCompactInPlace(b *testing.B) {
	c := ctxmmu.NewCompactor(ctxmmu.CompactorConfig{
		WindowSizeK:           4,
		VerboseThresholdBytes: 512,
		GenerateSummary:       false,
	})

	template := generateBenchmarkPages(20)
	workSlice := make([]ctxmmu.TokenPage, len(template))

	var report ctxmmu.CompactionReport
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		copy(workSlice, template)
		b.StartTimer()

		_, _ = c.CompactInPlace(workSlice, &report)
	}
}

func BenchmarkTokenPageFreelistReclaim(b *testing.B) {
	fl := ctxmmu.NewTokenPageFreelist(1024)
	page := fl.Get()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		fl.Put(page)
		page = fl.Get()
	}
}

func BenchmarkFormatTombstone(b *testing.B) {
	t := ctxmmu.Tombstone{
		Active:         true,
		Digest:         sha256.Sum256([]byte("sample payload data")),
		OriginalBytes:  16384,
		OriginalTokens: 4096,
		Tool:           "bash",
		Summary:        "12 lines, 16384 bytes; head: stdout log line",
	}

	buf := make([]byte, 0, 512)
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		buf = ctxmmu.FormatTombstone(t, buf[:0])
	}
}

func BenchmarkCompactor100kTokensScan(b *testing.B) {
	c := ctxmmu.NewCompactor(ctxmmu.CompactorConfig{
		WindowSizeK:           4,
		VerboseThresholdBytes: 512,
	})
	// 50 turns with 8KB tool results = ~100k+ tokens
	pages := generateBenchmarkPages(50)

	var report ctxmmu.ScanReport
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		c.ScanInto(pages, &report)
	}
}

func BenchmarkCompactor100kTokensCompactInPlace(b *testing.B) {
	c := ctxmmu.NewCompactor(ctxmmu.CompactorConfig{
		WindowSizeK:           4,
		VerboseThresholdBytes: 512,
	})
	template := generateBenchmarkPages(50)
	workSlice := make([]ctxmmu.TokenPage, len(template))

	var report ctxmmu.CompactionReport
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		copy(workSlice, template)
		b.StartTimer()

		_, _ = c.CompactInPlace(workSlice, &report)
	}
}
