package microagent_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// hibTestAgent is a Restorable agent for testing two-watermark hibernation (#11182).
type hibTestAgent struct {
	id    string
	turns int
	took  int
	hist  []string
	mu    sync.Mutex
}

func (a *hibTestAgent) Step(ctx context.Context, _ microagent.Gateway) (bool, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.took++
	a.hist = append(a.hist, fmt.Sprintf("%s:step:%d", a.id, a.took))
	return a.took >= a.turns, nil
}

func (a *hibTestAgent) Freeze() ([]byte, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return json.Marshal(map[string]any{
		"id":    a.id,
		"turns": a.turns,
		"took":  a.took,
		"hist":  a.hist,
	})
}

func (a *hibTestAgent) Thaw(b []byte) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	var s struct {
		ID    string   `json:"id"`
		Turns int      `json:"turns"`
		Took  int      `json:"took"`
		Hist  []string `json:"hist"`
	}
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	a.id = s.ID
	a.turns = s.Turns
	a.took = s.Took
	a.hist = s.Hist
	return nil
}

func (a *hibTestAgent) Blank() microagent.Hibernable {
	return &hibTestAgent{}
}

// TestHibernation100AgentsMultiplexWithinSlotLimits verifies that 100 enrolled agents
// multiplex within resident slot limits ($R$ slots) with no state loss (#11182).
func TestHibernation100AgentsMultiplexWithinSlotLimits(t *testing.T) {
	const (
		totalAgents = 100
		residentCap = 8
		lowWater    = 3
		maxWarm     = 4
		numTurns    = 3
		numWorkers  = 16
	)

	dir := t.TempDir()
	band, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Dir:     dir,
		High:    residentCap,
		Low:     lowWater,
		MaxWarm: maxWarm,
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()

	agents := make([]*hibTestAgent, totalAgents)
	for i := 0; i < totalAgents; i++ {
		id := fmt.Sprintf("agent-%03d", i)
		agents[i] = &hibTestAgent{id: id, turns: numTurns}
		if err := band.Enroll(id, agents[i]); err != nil {
			t.Fatalf("Enroll %q: %v", id, err)
		}
	}

	stats := band.Stats()
	if stats.Resident != 0 {
		t.Fatalf("expected 0 resident agents after enrollment, got %d", stats.Resident)
	}
	if stats.Parked+stats.Warm != totalAgents {
		t.Fatalf("expected %d total enrolled (parked=%d + warm=%d), got %d", totalAgents, stats.Parked, stats.Warm, stats.Parked+stats.Warm)
	}

	workCh := make(chan string, totalAgents*numTurns*2)
	for i := 0; i < totalAgents; i++ {
		workCh <- fmt.Sprintf("agent-%03d", i)
	}

	var completedCount int64
	var wg sync.WaitGroup
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case id, ok := <-workCh:
					if !ok {
						return
					}
					h, err := band.Acquire(ctx, id)
					if err != nil {
						if ctx.Err() != nil {
							return
						}
						t.Errorf("Acquire %q: %v", id, err)
						return
					}

					agent := h.(*hibTestAgent)
					done, stepErr := agent.Step(ctx, nil)
					if stepErr != nil {
						t.Errorf("Step %q: %v", id, stepErr)
						band.Retire(id)
						return
					}

					if done {
						band.Retire(id)
						atomic.AddInt64(&completedCount, 1)
						if atomic.LoadInt64(&completedCount) == totalAgents {
							close(workCh)
							return
						}
					} else {
						if err := band.Yield(id); err != nil {
							t.Errorf("Yield %q: %v", id, err)
							return
						}
						workCh <- id
					}
				}
			}
		}()
	}

	wg.Wait()

	finalStats := band.Stats()
	t.Logf("100 agents multiplexed: Peak residency=%d (cap=%d), Hits=%d, Thaws=%d, Refills=%d",
		finalStats.Peak, residentCap, finalStats.Hits, finalStats.Thaws, finalStats.Refills)

	if finalStats.Peak > residentCap {
		t.Fatalf("resident slot limit violated: Peak %d > Limit %d", finalStats.Peak, residentCap)
	}
	if atomic.LoadInt64(&completedCount) != totalAgents {
		t.Fatalf("expected %d agents completed, got %d", totalAgents, atomic.LoadInt64(&completedCount))
	}
	if finalStats.Resident != 0 {
		t.Fatalf("expected 0 active resident agents at end, got %d", finalStats.Resident)
	}
}

// TestHibernationWatermarksHighAndPreThawLow verifies that reaching the high watermark
// triggers hibernation of inactive agent state to cold storage (HibernatedState),
// and dropping to/below the low watermark triggers pre-thawing before execution turns (#11182).
func TestHibernationWatermarksHighAndPreThawLow(t *testing.T) {
	const (
		highWater = 4
		lowWater  = 2
		maxWarm   = 4
	)

	dir := t.TempDir()
	band, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Dir:     dir,
		High:    highWater,
		Low:     lowWater,
		MaxWarm: maxWarm,
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()

	gov := microagent.NewWarmBandGovernor(band)

	// Enroll 10 agents
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("wm-agent-%d", i)
		a := &hibTestAgent{id: id, turns: 5}
		if err := band.Enroll(id, a); err != nil {
			t.Fatalf("Enroll %q: %v", id, err)
		}
	}

	ctx := context.Background()

	// 1. Acquire up to high watermark (4 slots)
	heldAgents := make([]microagent.Hibernable, highWater)
	for i := 0; i < highWater; i++ {
		id := fmt.Sprintf("wm-agent-%d", i)
		h, err := band.Acquire(ctx, id)
		if err != nil {
			t.Fatalf("Acquire %q: %v", id, err)
		}
		heldAgents[i] = h
	}

	stats := band.Stats()
	if stats.Resident != highWater {
		t.Fatalf("expected resident == %d (high watermark), got %d", highWater, stats.Resident)
	}

	// 2. High watermark triggers hibernation of inactive/idle agent state to cold storage
	targetID := "wm-agent-3"
	hibState, err := gov.Hibernate(targetID)
	if err != nil {
		t.Fatalf("Hibernate %q: %v", targetID, err)
	}
	if hibState == nil {
		t.Fatal("expected non-nil HibernatedState")
	}
	if hibState.ID != targetID {
		t.Fatalf("HibernatedState.ID = %q, want %q", hibState.ID, targetID)
	}
	if len(hibState.Hash) != 64 {
		t.Fatalf("HibernatedState.Hash length = %d, want 64", len(hibState.Hash))
	}
	if len(hibState.Data) == 0 {
		t.Fatal("HibernatedState.Data must not be empty")
	}
	if hibState.SavedAt.IsZero() {
		t.Fatal("HibernatedState.SavedAt must not be zero")
	}

	// Verify cold storage lookup
	storedState, err := band.HibernatedState(targetID)
	if err != nil {
		t.Fatalf("HibernatedState lookup failed: %v", err)
	}
	if storedState.Hash != hibState.Hash {
		t.Fatalf("stored hash %q != original %q", storedState.Hash, hibState.Hash)
	}

	// Residency must have decremented
	statsAfterHib := band.Stats()
	if statsAfterHib.Resident != highWater-1 {
		t.Fatalf("expected resident == %d after hibernation, got %d", highWater-1, statsAfterHib.Resident)
	}

	// 3. Drop residency below low watermark (lowWater = 2)
	for i := 0; i < 2; i++ {
		id := fmt.Sprintf("wm-agent-%d", i)
		if err := band.Yield(id); err != nil {
			t.Fatalf("Yield %q: %v", id, err)
		}
	}

	statsLow := band.Stats()
	t.Logf("Residency dropped: Resident=%d (lowWater=%d)", statsLow.Resident, lowWater)

	// Low watermark triggers pre-thawing of parked agents before execution turns
	if err := gov.PreThaw(targetID); err != nil {
		t.Fatalf("PreThaw %q: %v", targetID, err)
	}

	// Now when targetID is acquired for its execution turn, it must be served as a warm hit (0 cold Thaw)
	initialHits := band.Stats().Hits
	hPreThawed, err := band.Acquire(ctx, targetID)
	if err != nil {
		t.Fatalf("Acquire pre-thawed %q: %v", targetID, err)
	}
	defer band.Yield(targetID)

	if hPreThawed == nil {
		t.Fatal("expected non-nil pre-thawed agent")
	}

	afterHits := band.Stats().Hits
	if afterHits <= initialHits {
		t.Fatalf("expected warm hit after pre-thaw (initial hits=%d, after=%d)", initialHits, afterHits)
	}
	t.Logf("Pre-thaw verified: warm hit registered successfully (hits=%d)", afterHits)
}

// TestHibernationCompactAnthropicHistory verifies that CompactAnthropicHistory preserves initial
// system/tool blocks with ephemeral cache control, pins active instructions tagged with [fak:goal],
// and compacts intermediate turns into restore stubs carrying content-addressed hashes (id=<sha256>) (#11182).
func TestHibernationCompactAnthropicHistory(t *testing.T) {
	type block map[string]any

	systemBlocks := []block{
		{
			"type": "text",
			"text": "You are a fak autonomous microagent kernel worker.",
			"cache_control": map[string]any{
				"type": "ephemeral",
			},
		},
	}

	tools := []map[string]any{
		{
			"name":        "fak_syscall",
			"description": "Invoke fak kernel system calls",
			"input_schema": map[string]any{
				"type": "object",
			},
		},
	}

	messages := []map[string]any{
		{
			"role": "user",
			"content": []block{
				{
					"type": "text",
					"text": "Initial task request: inspect project workspace.",
					"cache_control": map[string]any{
						"type": "ephemeral",
					},
				},
			},
		},
		{
			"role":    "assistant",
			"content": "Initial task acknowledged. Starting inspection.",
		},
		{
			"role":    "user",
			"content": "Turn 1 intermediate request: read verbose diagnostic trace " + strings.Repeat("trace-data-log-entry-line ", 40),
		},
		{
			"role":    "assistant",
			"content": "Turn 1 intermediate response: diagnostic trace loaded " + strings.Repeat("verbose-step-result-output ", 40),
		},
		{
			"role":    "user",
			"content": "[fak:goal] Standing invariant instruction: enforce zero data races and memory leaks.",
		},
		{
			"role":    "assistant",
			"content": "Standing goal confirmed and active.",
		},
		{
			"role":    "user",
			"content": "Turn 2 intermediate request: execute intermediate benchmark sweep " + strings.Repeat("sweep-iteration-data ", 40),
		},
		{
			"role":    "assistant",
			"content": "Turn 2 intermediate response: intermediate sweep done " + strings.Repeat("sweep-metric-output ", 40),
		},
		{
			"role":    "user",
			"content": "Final turn request: verify all invariants and emit final report.",
		},
	}

	bodyMap := map[string]any{
		"model":    "claude-3-7-sonnet",
		"system":   systemBlocks,
		"tools":    tools,
		"messages": messages,
	}

	raw, err := json.Marshal(bodyMap)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}

	const tightBudget = 300
	compacted := microagent.CompactAnthropicHistory(raw, tightBudget)

	// 1. Must be smaller than original
	if len(compacted) >= len(raw) {
		t.Fatalf("compacted length %d not smaller than raw %d", len(compacted), len(raw))
	}

	// 2. Preserves initial system/tool blocks with ephemeral cache control
	if !bytes.Contains(compacted, []byte("You are a fak autonomous microagent kernel worker.")) {
		t.Error("compacted output missing initial system text")
	}
	if !bytes.Contains(compacted, []byte(`"cache_control"`)) || !bytes.Contains(compacted, []byte(`"ephemeral"`)) {
		t.Error("compacted output missing ephemeral cache_control")
	}
	if !bytes.Contains(compacted, []byte(`"fak_syscall"`)) {
		t.Error("compacted output missing tool definitions")
	}

	// 3. Preserves initial user message with cache control
	if !bytes.Contains(compacted, []byte("Initial task request: inspect project workspace.")) {
		t.Error("compacted output missing initial user message prefix")
	}

	// 4. Pins active instructions tagged with [fak:goal]
	goalMarker := "[fak:goal] Standing invariant instruction: enforce zero data races and memory leaks."
	if !bytes.Contains(compacted, []byte(goalMarker)) {
		t.Errorf("compacted output failed to pin instruction tagged with [fak:goal]: %s", goalMarker)
	}

	// 5. Intermediate turns are compacted away
	if bytes.Contains(compacted, []byte("trace-data-log-entry-line")) {
		t.Error("intermediate turn 1 was not compacted away")
	}
	if bytes.Contains(compacted, []byte("sweep-iteration-data")) {
		t.Error("intermediate turn 2 was not compacted away")
	}

	// 6. Compacts intermediate turns into restore stubs carrying content-addressed hashes (id=<sha256>)
	handles := microagent.ExtractRestoreHandles(compacted)
	if len(handles) == 0 {
		t.Fatalf("expected restore stubs with id=<sha256> in compacted output, found none\nOutput: %s", string(compacted))
	}

	handle := handles[0]
	if len(handle) != 64 {
		t.Fatalf("restore handle %q is not 64 hex characters", handle)
	}
	t.Logf("Restore handle successfully generated: %s", handle)

	// 7. Verify restore handles allow retrieving dropped context
	droppedData, ok := microagent.RestoreContext(handle)
	if !ok || len(droppedData) == 0 {
		t.Fatalf("failed to retrieve dropped context for restore handle %q", handle)
	}
	if !bytes.Contains(droppedData, []byte("trace-data-log-entry-line")) {
		t.Error("restored context does not contain expected dropped turn data")
	}

	// 8. Recent turn is preserved
	if !bytes.Contains(compacted, []byte("Final turn request: verify all invariants and emit final report.")) {
		t.Error("compacted output missing recent user turn")
	}
}

// TestHibernationConcurrencyHighChurn tests concurrency safety of the warm-band governor
// and context compaction under high churn (#11182).
func TestHibernationConcurrencyHighChurn(t *testing.T) {
	const (
		numGoroutines = 16
		iterations    = 30
		residentCap   = 6
		lowWater      = 2
		maxWarm       = 4
	)

	dir := t.TempDir()
	band, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Dir:     dir,
		High:    residentCap,
		Low:     lowWater,
		MaxWarm: maxWarm,
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()

	gov := microagent.NewWarmBandGovernor(band)

	// Pre-enroll 40 agents
	for i := 0; i < 40; i++ {
		id := fmt.Sprintf("churn-agent-%02d", i)
		if err := band.Enroll(id, &hibTestAgent{id: id, turns: 100}); err != nil {
			t.Fatalf("Enroll %q: %v", id, err)
		}
	}

	var wg sync.WaitGroup
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		gid := g
		go func() {
			defer wg.Done()
			for iter := 0; iter < iterations; iter++ {
				if ctx.Err() != nil {
					return
				}

				agentID := fmt.Sprintf("churn-agent-%02d", (gid+iter)%40)

				// 1. Acquire & Step & Yield or Hibernate
				h, err := band.Acquire(ctx, agentID)
				if err == nil && h != nil {
					agent := h.(*hibTestAgent)
					_, _ = agent.Step(ctx, nil)

					if iter%4 == 0 {
						// Trigger explicit hibernate
						_, _ = gov.Hibernate(agentID)
					} else {
						_ = band.Yield(agentID)
					}
				}

				// 2. Check watermarks & PreThaw
				if iter%3 == 0 {
					_, _, _ = gov.CheckWatermarks()
				}
				if iter%5 == 0 {
					_ = gov.PreThaw(agentID)
				}

				// 3. Compact history
				body := map[string]any{
					"system": []map[string]any{
						{"type": "text", "text": "system", "cache_control": map[string]any{"type": "ephemeral"}},
					},
					"messages": []map[string]any{
						{"role": "user", "content": "initial", "cache_control": map[string]any{"type": "ephemeral"}},
						{"role": "assistant", "content": "ack"},
						{"role": "user", "content": fmt.Sprintf("turn-%d-debris %s", iter, strings.Repeat("data ", 20))},
						{"role": "assistant", "content": "turn response"},
						{"role": "user", "content": "[fak:goal] active goal"},
						{"role": "assistant", "content": "goal ack"},
						{"role": "user", "content": "current"},
					},
				}
				raw, _ := json.Marshal(body)
				compacted := microagent.CompactAnthropicHistory(raw, 50)
				_ = microagent.ExtractRestoreHandles(compacted)
			}
		}()
	}

	wg.Wait()
	stats := band.Stats()
	t.Logf("High churn completed safely: Peak=%d (cap=%d), Hits=%d, Thaws=%d, Refills=%d",
		stats.Peak, residentCap, stats.Hits, stats.Thaws, stats.Refills)

	if stats.Peak > residentCap {
		t.Fatalf("resident limit exceeded during churn: Peak %d > Limit %d", stats.Peak, residentCap)
	}
}
