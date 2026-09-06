package sessionview

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestModelRows_InterfacesAndTags(t *testing.T) {
	now := time.Now().UTC()

	session := SessionRow{
		SessionID: "sess-1",
		TraceID:   "tr-1",
		State:     "active",
		Model:     "fak-qwen",
		CreatedAt: now,
		UpdatedAt: now,
		Labels:    map[string]string{"env": "test"},
		Metadata:  map[string]any{"user": "alice"},
	}
	llmCall := LlmCallRow{
		CallID:       "call-1",
		SessionID:    "sess-1",
		TurnID:       "turn-1",
		Model:        "fak-qwen",
		PromptTokens: 120,
		OutputTokens: 80,
		TotalTokens:  200,
		Duration:     50 * time.Millisecond,
		Timestamp:    now,
		Attributes:   map[string]any{"temperature": 0.7},
	}
	tokenUsage := TokenUsageRow{
		UsageID:      "tok-1",
		SessionID:    "sess-1",
		CallID:       "call-1",
		Model:        "fak-qwen",
		PromptTokens: 10,
		OutputTokens: 20,
		TotalTokens:  30,
		CostUSD:      0.0005,
		Timestamp:    now,
	}
	toolCall := ToolCallRow{
		ToolCallID: "tool-1",
		SessionID:  "sess-1",
		TurnID:     "turn-1",
		ToolName:   "grep",
		Arguments:  `{"pattern":"Row"}`,
		Result:     "match found",
		Duration:   10 * time.Millisecond,
		Admitted:   true,
		Timestamp:  now,
	}
	auditEvent := AuditEventRow{
		EventID:   "audit-1",
		SessionID: "sess-1",
		TraceID:   "tr-1",
		Component: "policy_guard",
		Action:    "admit",
		Severity:  "info",
		Message:   "Tool call admitted",
		Payload:   map[string]any{"tool": "grep"},
		Timestamp: now,
	}

	rows := []Row{session, llmCall, tokenUsage, toolCall, auditEvent}
	expectedKinds := []RowKind{
		RowKindSession,
		RowKindLlmCall,
		RowKindTokenUsage,
		RowKindToolCall,
		RowKindAuditEvent,
	}

	for i, r := range rows {
		if r.RowKind() != expectedKinds[i] {
			t.Errorf("row %d: expected kind %s, got %s", i, expectedKinds[i], r.RowKind())
		}
		if r.Session() != "sess-1" {
			t.Errorf("row %d: expected session 'sess-1', got %s", i, r.Session())
		}
		if r.RowID() == "" {
			t.Errorf("row %d: expected non-empty RowID", i)
		}
		if r.EventTime().IsZero() || r.OccurredAt().IsZero() {
			t.Errorf("row %d: expected non-zero EventTime/OccurredAt", i)
		}

		// Ensure JSON round-trip
		data, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("row %d: failed to marshal JSON: %v", i, err)
		}
		if len(data) == 0 {
			t.Fatalf("row %d: empty json output", i)
		}
	}

	// Test ToolCall name fallback
	t1 := ToolCallRow{ToolName: "bash"}
	if t1.Name() != "bash" {
		t.Errorf("expected ToolName 'bash', got %q", t1.Name())
	}
	t2 := ToolCallRow{Tool: "read"}
	if t2.Name() != "read" {
		t.Errorf("expected Tool fallback 'read', got %q", t2.Name())
	}
}

func TestMaterializedView_BasicsAndMonotonicCounters(t *testing.T) {
	v := New("sess-abc", 50)
	if v.Capacity() != 50 {
		t.Fatalf("expected capacity 50, got %d", v.Capacity())
	}
	if v.Session().SessionID != "sess-abc" {
		t.Fatalf("expected session sess-abc, got %s", v.Session().SessionID)
	}

	now := time.Now().UTC()

	// 1. Ingest LLM Call
	err := v.AppendLlmCall(LlmCallRow{
		CallID:       "call-1",
		SessionID:    "sess-abc",
		PromptTokens: 100,
		OutputTokens: 50,
		TotalTokens:  150,
		CachedTokens: 25,
		Timestamp:    now,
	})
	if err != nil {
		t.Fatalf("AppendLlmCall failed: %v", err)
	}

	// 2. Ingest Token Usage
	err = v.AppendTokenUsage(TokenUsageRow{
		UsageID:      "tok-1",
		SessionID:    "sess-abc",
		PromptTokens: 20,
		OutputTokens: 10,
		TotalTokens:  30,
		CachedTokens: 5,
		CostUSD:      0.002,
		Timestamp:    now.Add(1 * time.Second),
	})
	if err != nil {
		t.Fatalf("AppendTokenUsage failed: %v", err)
	}

	// 3. Ingest Tool Call
	err = v.AppendToolCall(ToolCallRow{
		ToolCallID: "tool-1",
		SessionID:  "sess-abc",
		ToolName:   "bash",
		Duration:   12 * time.Millisecond,
		Admitted:   true,
		Timestamp:  now.Add(2 * time.Second),
	})
	if err != nil {
		t.Fatalf("AppendToolCall failed: %v", err)
	}

	// 4. Ingest Audit Event with error
	err = v.AppendAuditEvent(AuditEventRow{
		EventID:   "audit-1",
		SessionID: "sess-abc",
		Component: "guard",
		Action:    "block",
		Severity:  "error",
		Message:   "Blocked dangerous tool",
		Timestamp: now.Add(3 * time.Second),
	})
	if err != nil {
		t.Fatalf("AppendAuditEvent failed: %v", err)
	}

	// 5. Ingest Session update
	err = v.AppendSession(SessionRow{
		SessionID: "sess-abc",
		State:     "completed",
		UpdatedAt: now.Add(4 * time.Second),
	})
	if err != nil {
		t.Fatalf("AppendSession failed: %v", err)
	}

	sum := v.Summary()
	if sum.TotalEvents != 5 {
		t.Errorf("expected 5 TotalEvents, got %d", sum.TotalEvents)
	}
	if sum.TotalCalls != 1 {
		t.Errorf("expected 1 TotalCalls, got %d", sum.TotalCalls)
	}
	if sum.TotalToolCalls != 1 {
		t.Errorf("expected 1 TotalToolCalls, got %d", sum.TotalToolCalls)
	}
	if sum.TotalAuditEvents != 1 {
		t.Errorf("expected 1 TotalAuditEvents, got %d", sum.TotalAuditEvents)
	}
	if sum.PromptTokens != 120 {
		t.Errorf("expected 120 PromptTokens, got %d", sum.PromptTokens)
	}
	if sum.OutputTokens != 60 {
		t.Errorf("expected 60 OutputTokens, got %d", sum.OutputTokens)
	}
	if sum.CachedTokens != 30 {
		t.Errorf("expected 30 CachedTokens, got %d", sum.CachedTokens)
	}
	if sum.TotalTokens != 180 {
		t.Errorf("expected 180 TotalTokens, got %d", sum.TotalTokens)
	}
	if sum.TotalCostUSD != 0.002 {
		t.Errorf("expected 0.002 TotalCostUSD, got %f", sum.TotalCostUSD)
	}
	if sum.TotalErrors != 1 {
		t.Errorf("expected 1 TotalErrors, got %d", sum.TotalErrors)
	}
	if sum.EvictedEvents != 0 {
		t.Errorf("expected 0 EvictedEvents, got %d", sum.EvictedEvents)
	}
	if sum.RetainedEvents != 5 {
		t.Errorf("expected 5 RetainedEvents, got %d", sum.RetainedEvents)
	}
	if sum.Drift() != 0 {
		t.Errorf("expected 0 Drift, got %d", sum.Drift())
	}
}

func TestMaterializedView_BoundedRetention_FIFOEviction(t *testing.T) {
	const capacity = 5
	v := New("sess-fifo", capacity)

	// Append 12 distinct events
	for i := 1; i <= 12; i++ {
		err := v.Append(ToolCallRow{
			ToolCallID: fmt.Sprintf("tool-%d", i),
			SessionID:  "sess-fifo",
			ToolName:   "probe",
			Timestamp:  time.Unix(int64(1700000000+i), 0).UTC(),
		})
		if err != nil {
			t.Fatalf("Append %d failed: %v", i, err)
		}
	}

	if v.RetainedCount() != capacity {
		t.Fatalf("expected %d retained items, got %d", capacity, v.RetainedCount())
	}

	sum := v.Summary()
	if sum.TotalEvents != 12 {
		t.Errorf("expected 12 TotalEvents, got %d", sum.TotalEvents)
	}
	if sum.EvictedEvents != 7 {
		t.Errorf("expected 7 EvictedEvents, got %d", sum.EvictedEvents)
	}
	if sum.RetainedEvents != 5 {
		t.Errorf("expected 5 RetainedEvents, got %d", sum.RetainedEvents)
	}
	if sum.Drift() != 0 {
		t.Errorf("expected 0 drift, got %d", sum.Drift())
	}

	snap := v.Snapshot()
	if len(snap.DetailRows) != capacity {
		t.Fatalf("expected %d detail rows in snapshot, got %d", capacity, len(snap.DetailRows))
	}
	if len(snap.ToolCalls) != capacity {
		t.Fatalf("expected %d tool calls in snapshot, got %d", capacity, len(snap.ToolCalls))
	}

	// Verify exact FIFO order: oldest retained should be tool-8, newest tool-12
	expectedIDs := []string{"tool-8", "tool-9", "tool-10", "tool-11", "tool-12"}
	for i, r := range snap.DetailRows {
		if r.RowID() != expectedIDs[i] {
			t.Errorf("index %d: expected %s, got %s", i, expectedIDs[i], r.RowID())
		}
	}
}

func TestMaterializedView_ZeroCounterDrift(t *testing.T) {
	const capacity = 10
	v := New("sess-drift", capacity)

	var expectedPromptTokens int64
	var expectedOutputTokens int64
	var expectedTotalTokens int64
	var expectedCalls int64

	const numEvents = 500
	for i := 1; i <= numEvents; i++ {
		prompt := int64(10 + (i % 7))
		output := int64(5 + (i % 5))
		total := prompt + output

		expectedPromptTokens += prompt
		expectedOutputTokens += output
		expectedTotalTokens += total
		expectedCalls++

		err := v.AppendLlmCall(LlmCallRow{
			CallID:       fmt.Sprintf("call-%d", i),
			SessionID:    "sess-drift",
			PromptTokens: prompt,
			OutputTokens: output,
			TotalTokens:  total,
			Timestamp:    time.Now().UTC(),
		})
		if err != nil {
			t.Fatalf("Append %d failed: %v", i, err)
		}

		// Verify that at each single step, drift is strictly zero
		s := v.Summary()
		if s.Drift() != 0 {
			t.Fatalf("step %d: non-zero drift detected: %d", i, s.Drift())
		}
	}

	sum := v.Summary()
	if sum.TotalEvents != numEvents {
		t.Errorf("expected %d total events, got %d", numEvents, sum.TotalEvents)
	}
	if sum.TotalCalls != expectedCalls {
		t.Errorf("expected %d total calls, got %d", expectedCalls, sum.TotalCalls)
	}
	if sum.PromptTokens != expectedPromptTokens {
		t.Errorf("expected %d prompt tokens, got %d", expectedPromptTokens, sum.PromptTokens)
	}
	if sum.OutputTokens != expectedOutputTokens {
		t.Errorf("expected %d output tokens, got %d", expectedOutputTokens, sum.OutputTokens)
	}
	if sum.TotalTokens != expectedTotalTokens {
		t.Errorf("expected %d total tokens, got %d", expectedTotalTokens, sum.TotalTokens)
	}
	if sum.EvictedEvents != int64(numEvents-capacity) {
		t.Errorf("expected %d evicted events, got %d", numEvents-capacity, sum.EvictedEvents)
	}
	if sum.RetainedEvents != capacity {
		t.Errorf("expected %d retained events, got %d", capacity, sum.RetainedEvents)
	}
	if sum.Drift() != 0 {
		t.Errorf("final drift must be 0, got %d", sum.Drift())
	}
}

func TestMaterializedView_AtomicSnapshot_ThreadSafetyAndIsolation(t *testing.T) {
	v := New("sess-snap", 5)

	v.AppendSession(SessionRow{
		SessionID: "sess-snap",
		Labels:    map[string]string{"env": "prod"},
		Metadata:  map[string]any{"k": "v1"},
	})
	v.AppendLlmCall(LlmCallRow{
		CallID:     "call-1",
		Attributes: map[string]any{"temperature": 0.5},
	})

	snap1 := v.Snapshot()
	if snap1.Session.Labels["env"] != "prod" {
		t.Errorf("expected env prod, got %v", snap1.Session.Labels["env"])
	}

	// Mutate the returned snapshot's maps and slices
	snap1.Session.Labels["env"] = "mutated"
	snap1.Session.Metadata["k"] = "mutated"
	if len(snap1.LlmCalls) > 0 {
		snap1.LlmCalls[0].Attributes["temperature"] = 99.9
	}
	snap1.DetailRows = nil

	// Ensure view state was not corrupted
	snap2 := v.Snapshot()
	if snap2.Session.Labels["env"] != "prod" {
		t.Errorf("view session labels corrupted by external mutation: got %s", snap2.Session.Labels["env"])
	}
	if snap2.Session.Metadata["k"] != "v1" {
		t.Errorf("view session metadata corrupted: got %v", snap2.Session.Metadata["k"])
	}
	if len(snap2.LlmCalls) == 0 || snap2.LlmCalls[0].Attributes["temperature"] != 0.5 {
		t.Errorf("view llm call attributes corrupted: got %v", snap2.LlmCalls)
	}
	if len(snap2.DetailRows) != 2 {
		t.Errorf("view detail rows corrupted: got %d", len(snap2.DetailRows))
	}
}

func TestMaterializedView_ViewSink_Integration(t *testing.T) {
	sliceSink := NewSliceSink()
	chanSink := NewChannelSink(10, false)
	defer chanSink.Close()

	var funcSinkCalls atomic.Int64
	funcSink := SinkFunc(func(row Row) error {
		funcSinkCalls.Add(1)
		return nil
	})

	v := New("sess-sinks", 10, sliceSink, chanSink, funcSink)

	const count = 5
	for i := 1; i <= count; i++ {
		err := v.Append(AuditEventRow{
			EventID:   fmt.Sprintf("audit-%d", i),
			SessionID: "sess-sinks",
			Action:    "probe",
		})
		if err != nil {
			t.Fatalf("Append failed: %v", err)
		}
	}

	if sliceSink.Count() != count {
		t.Errorf("SliceSink expected %d rows, got %d", count, sliceSink.Count())
	}
	if funcSinkCalls.Load() != count {
		t.Errorf("SinkFunc expected %d calls, got %d", count, funcSinkCalls.Load())
	}

	// Verify channel sink received rows
	for i := 1; i <= count; i++ {
		select {
		case r := <-chanSink.Channel():
			if r.RowID() != fmt.Sprintf("audit-%d", i) {
				t.Errorf("chanSink expected audit-%d, got %s", i, r.RowID())
			}
		case <-time.After(100 * time.Millisecond):
			t.Fatalf("timed out waiting for channel sink row %d", i)
		}
	}

	// Test UnregisterSink
	v.UnregisterSink(sliceSink)
	err := v.Append(AuditEventRow{
		EventID:   "audit-extra",
		SessionID: "sess-sinks",
	})
	if err != nil {
		t.Fatalf("Append failed: %v", err)
	}
	if sliceSink.Count() != count {
		t.Errorf("SliceSink should not have received audit-extra after unregister: got %d", sliceSink.Count())
	}

	// Test sink error propagation
	errSink := SinkFunc(func(row Row) error {
		return errors.New("sink failure simulated")
	})
	v.RegisterSink(errSink)
	err = v.Append(AuditEventRow{
		EventID:   "audit-fail",
		SessionID: "sess-sinks",
	})
	if err == nil {
		t.Errorf("expected error from failing sink, got nil")
	}
}

func TestMaterializedView_HighThroughput_ZeroDrift_Concurrent(t *testing.T) {
	const capacity = 100
	v := New("sess-concurrent", capacity)

	const goroutines = 20
	const eventsPerGoroutine = 500
	const totalExpectedEvents = int64(goroutines * eventsPerGoroutine)

	var totalExpectedPromptTokens atomic.Int64
	var totalExpectedOutputTokens atomic.Int64
	var totalExpectedTokens atomic.Int64
	var totalExpectedCalls atomic.Int64
	var totalExpectedToolCalls atomic.Int64
	var totalExpectedAuditEvents atomic.Int64

	var wg sync.WaitGroup
	startSignal := make(chan struct{})

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(gID int) {
			defer wg.Done()
			<-startSignal

			for i := 0; i < eventsPerGoroutine; i++ {
				idx := gID*eventsPerGoroutine + i
				switch idx % 3 {
				case 0:
					// LLM Call
					p := int64(20 + (idx % 11))
					o := int64(10 + (idx % 7))
					tot := p + o
					totalExpectedPromptTokens.Add(p)
					totalExpectedOutputTokens.Add(o)
					totalExpectedTokens.Add(tot)
					totalExpectedCalls.Add(1)

					_ = v.AppendLlmCall(LlmCallRow{
						CallID:       fmt.Sprintf("call-%d-%d", gID, i),
						SessionID:    "sess-concurrent",
						PromptTokens: p,
						OutputTokens: o,
						TotalTokens:  tot,
					})
				case 1:
					// Tool Call
					totalExpectedToolCalls.Add(1)
					_ = v.AppendToolCall(ToolCallRow{
						ToolCallID: fmt.Sprintf("tool-%d-%d", gID, i),
						SessionID:  "sess-concurrent",
						ToolName:   "bash",
					})
				case 2:
					// Audit Event
					totalExpectedAuditEvents.Add(1)
					_ = v.AppendAuditEvent(AuditEventRow{
						EventID:   fmt.Sprintf("audit-%d-%d", gID, i),
						SessionID: "sess-concurrent",
						Component: "kernel",
						Action:    "verify",
					})
				}
			}
		}(g)
	}

	// Concurrent snapshot reader goroutine running simultaneously
	var readerStop atomic.Bool
	var readerWg sync.WaitGroup
	readerWg.Add(1)
	go func() {
		defer readerWg.Done()
		<-startSignal
		for !readerStop.Load() {
			snap := v.Snapshot()
			if snap.Summary.Capacity != capacity {
				t.Errorf("invalid capacity in snapshot: %d", snap.Summary.Capacity)
			}
			if snap.Summary.Drift() != 0 {
				t.Errorf("drift observed during concurrent execution: %d", snap.Summary.Drift())
			}
			if len(snap.DetailRows) > capacity {
				t.Errorf("detail rows exceeded capacity: %d > %d", len(snap.DetailRows), capacity)
			}
			time.Sleep(1 * time.Millisecond)
		}
	}()

	close(startSignal)
	wg.Wait()
	readerStop.Store(true)
	readerWg.Wait()

	finalSnap := v.Snapshot()
	sum := finalSnap.Summary

	if sum.TotalEvents != totalExpectedEvents {
		t.Fatalf("expected TotalEvents %d, got %d", totalExpectedEvents, sum.TotalEvents)
	}
	if sum.TotalCalls != totalExpectedCalls.Load() {
		t.Fatalf("expected TotalCalls %d, got %d", totalExpectedCalls.Load(), sum.TotalCalls)
	}
	if sum.TotalToolCalls != totalExpectedToolCalls.Load() {
		t.Fatalf("expected TotalToolCalls %d, got %d", totalExpectedToolCalls.Load(), sum.TotalToolCalls)
	}
	if sum.TotalAuditEvents != totalExpectedAuditEvents.Load() {
		t.Fatalf("expected TotalAuditEvents %d, got %d", totalExpectedAuditEvents.Load(), sum.TotalAuditEvents)
	}
	if sum.PromptTokens != totalExpectedPromptTokens.Load() {
		t.Fatalf("expected PromptTokens %d, got %d", totalExpectedPromptTokens.Load(), sum.PromptTokens)
	}
	if sum.OutputTokens != totalExpectedOutputTokens.Load() {
		t.Fatalf("expected OutputTokens %d, got %d", totalExpectedOutputTokens.Load(), sum.OutputTokens)
	}
	if sum.TotalTokens != totalExpectedTokens.Load() {
		t.Fatalf("expected TotalTokens %d, got %d", totalExpectedTokens.Load(), sum.TotalTokens)
	}

	// Verification of zero counter drift
	expectedEvicted := totalExpectedEvents - int64(capacity)
	if sum.EvictedEvents != expectedEvicted {
		t.Errorf("expected EvictedEvents %d, got %d", expectedEvicted, sum.EvictedEvents)
	}
	if sum.RetainedEvents != int64(capacity) {
		t.Errorf("expected RetainedEvents %d, got %d", capacity, sum.RetainedEvents)
	}
	if sum.Drift() != 0 {
		t.Fatalf("zero counter drift violation: drift is %d", sum.Drift())
	}
	if len(finalSnap.DetailRows) != capacity {
		t.Errorf("expected %d detail rows in snapshot, got %d", capacity, len(finalSnap.DetailRows))
	}
}

func TestMaterializedView_EdgeCases(t *testing.T) {
	v := New("sess-edge", -10) // should fallback to DefaultCapacity
	if v.Capacity() != DefaultCapacity {
		t.Fatalf("expected fallback to %d, got %d", DefaultCapacity, v.Capacity())
	}

	// Append nil row
	err := v.Append(nil)
	if !errors.Is(err, ErrNilRow) {
		t.Errorf("expected ErrNilRow, got %v", err)
	}

	// Append pointer to nil row
	var nilPtr *LlmCallRow
	err = v.Append(nilPtr)
	if !errors.Is(err, ErrNilRow) {
		t.Errorf("expected ErrNilRow for nil pointer, got %v", err)
	}

	// Reset
	v.AppendLlmCall(LlmCallRow{CallID: "c1", PromptTokens: 10})
	if v.Summary().TotalEvents != 1 {
		t.Fatalf("expected 1 event before reset")
	}
	v.Reset()
	if v.Summary().TotalEvents != 0 {
		t.Fatalf("expected 0 events after reset")
	}
	if v.RetainedCount() != 0 {
		t.Fatalf("expected 0 retained after reset")
	}
	if v.Capacity() != DefaultCapacity {
		t.Fatalf("capacity should be preserved after reset")
	}

	// Close
	err = v.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	err = v.Append(AuditEventRow{EventID: "ev1"})
	if !errors.Is(err, ErrViewClosed) {
		t.Errorf("expected ErrViewClosed after close, got %v", err)
	}

	// Nil view safety
	var nilView *MaterializedView
	if snap := nilView.Snapshot(); !snap.IsEmpty() {
		t.Errorf("expected empty snapshot from nil view")
	}
	if sum := nilView.Summary(); sum.TotalEvents != 0 {
		t.Errorf("expected empty summary from nil view")
	}
	if r := nilView.RetainedRows(); r != nil {
		t.Errorf("expected nil retained rows from nil view")
	}
	if c := nilView.RetainedCount(); c != 0 {
		t.Errorf("expected 0 retained count from nil view")
	}
	if cap := nilView.Capacity(); cap != 0 {
		t.Errorf("expected 0 capacity from nil view")
	}
	if sum, rows := nilView.SnapshotRows(); sum.TotalEvents != 0 || rows != nil {
		t.Errorf("expected empty tuple from nil view SnapshotRows")
	}
}

func TestMaterializedView_BoundedRetention_FiftyThousandEvents(t *testing.T) {
	const capacity = 10000
	const totalEvents = 50000
	const expectedEvicted = totalEvents - capacity // 40000

	v := NewMaterializedView(capacity)
	if v.Capacity() != capacity {
		t.Fatalf("expected capacity %d, got %d", capacity, v.Capacity())
	}

	var (
		expectedTotalCalls       int64
		expectedTotalToolCalls   int64
		expectedTotalAuditEvents int64
		expectedPromptTokens     int64
		expectedOutputTokens     int64
		expectedCachedTokens     int64
		expectedTotalTokens      int64
		expectedTotalCostUSD     float64
		expectedTotalErrors      int64
	)

	now := time.Unix(1700000000, 0).UTC()
	var lastTotalEvents int64
	var lastPromptTokens int64
	var lastOutputTokens int64

	for i := 1; i <= totalEvents; i++ {
		eventTime := now.Add(time.Duration(i) * time.Millisecond)
		switch i % 5 {
		case 1:
			// LLM Call
			prompt := int64(50 + (i % 100))
			output := int64(20 + (i % 50))
			cached := int64(5 + (i % 10))
			tot := prompt + output
			expectedPromptTokens += prompt
			expectedOutputTokens += output
			expectedCachedTokens += cached
			expectedTotalTokens += tot
			expectedTotalCalls++

			var errMsg string
			if i%500 == 0 {
				errMsg = "rate limit exceeded"
				expectedTotalErrors++
			}

			err := v.IngestLlmCall(LlmCallRow{
				CallID:       fmt.Sprintf("call-%d", i),
				SessionID:    "sess-50k",
				Model:        "qwen-2.5",
				PromptTokens: prompt,
				OutputTokens: output,
				CachedTokens: cached,
				TotalTokens:  tot,
				Error:        errMsg,
				Timestamp:    eventTime,
			})
			if err != nil {
				t.Fatalf("IngestLlmCall %d failed: %v", i, err)
			}

		case 2:
			// Token Usage
			prompt := int64(10 + (i % 30))
			output := int64(5 + (i % 20))
			cached := int64(2 + (i % 5))
			tot := prompt + output
			cost := float64(tot) * 0.000002
			expectedPromptTokens += prompt
			expectedOutputTokens += output
			expectedCachedTokens += cached
			expectedTotalTokens += tot
			expectedTotalCostUSD += cost

			err := v.IngestTokenUsage(TokenUsageRow{
				UsageID:      fmt.Sprintf("usage-%d", i),
				SessionID:    "sess-50k",
				CallID:       fmt.Sprintf("call-%d", i-1),
				Model:        "qwen-2.5",
				PromptTokens: prompt,
				OutputTokens: output,
				CachedTokens: cached,
				TotalTokens:  tot,
				CostUSD:      cost,
				Timestamp:    eventTime,
			})
			if err != nil {
				t.Fatalf("IngestTokenUsage %d failed: %v", i, err)
			}

		case 3:
			// Tool Call
			expectedTotalToolCalls++
			var errMsg string
			if i%250 == 0 {
				errMsg = "permission denied"
				expectedTotalErrors++
			}

			err := v.IngestToolCall(ToolCallRow{
				ToolCallID: fmt.Sprintf("tool-%d", i),
				SessionID:  "sess-50k",
				ToolName:   "bash",
				Arguments:  `{"cmd":"echo hi"}`,
				Result:     "hi",
				Duration:   5 * time.Millisecond,
				Admitted:   true,
				Error:      errMsg,
				Timestamp:  eventTime,
			})
			if err != nil {
				t.Fatalf("IngestToolCall %d failed: %v", i, err)
			}

		case 4:
			// Audit Event
			expectedTotalAuditEvents++
			severity := "info"
			if i%1000 == 0 {
				severity = "error"
				expectedTotalErrors++
			}

			err := v.IngestAuditEvent(AuditEventRow{
				EventID:   fmt.Sprintf("audit-%d", i),
				SessionID: "sess-50k",
				Component: "guard",
				Action:    "policy_check",
				Severity:  severity,
				Message:   "policy evaluation completed",
				Timestamp: eventTime,
			})
			if err != nil {
				t.Fatalf("IngestAuditEvent %d failed: %v", i, err)
			}

		case 0:
			// Session update
			var state string
			if i%2000 == 0 {
				state = "error"
				expectedTotalErrors++
			} else {
				state = "active"
			}
			err := v.IngestSession(SessionRow{
				SessionID: "sess-50k",
				State:     state,
				UpdatedAt: eventTime,
			})
			if err != nil {
				t.Fatalf("IngestSession %d failed: %v", i, err)
			}
		}

		// Verify monotonic increase and zero drift at checkpoints
		if i%5000 == 0 {
			currSum := v.Summary()
			if currSum.TotalEvents <= lastTotalEvents {
				t.Fatalf("step %d: TotalEvents not monotonically increasing: %d <= %d", i, currSum.TotalEvents, lastTotalEvents)
			}
			if currSum.PromptTokens < lastPromptTokens {
				t.Fatalf("step %d: PromptTokens decremented: %d < %d", i, currSum.PromptTokens, lastPromptTokens)
			}
			if currSum.OutputTokens < lastOutputTokens {
				t.Fatalf("step %d: OutputTokens decremented: %d < %d", i, currSum.OutputTokens, lastOutputTokens)
			}
			if currSum.Drift() != 0 {
				t.Fatalf("step %d: non-zero drift detected: %d", i, currSum.Drift())
			}
			lastTotalEvents = currSum.TotalEvents
			lastPromptTokens = currSum.PromptTokens
			lastOutputTokens = currSum.OutputTokens
		}
	}

	// Verify Snapshot and Summary
	snap := v.Snapshot()
	sum := snap.Summary

	if sum.TotalEvents != int64(totalEvents) {
		t.Fatalf("expected TotalEvents %d, got %d", totalEvents, sum.TotalEvents)
	}
	if sum.EvictedEvents != int64(expectedEvicted) {
		t.Fatalf("expected EvictedEvents %d, got %d", expectedEvicted, sum.EvictedEvents)
	}
	if sum.RetainedEvents != int64(capacity) {
		t.Fatalf("expected RetainedEvents %d, got %d", capacity, sum.RetainedEvents)
	}
	if sum.Drift() != 0 {
		t.Fatalf("expected Drift() == 0, got %d", sum.Drift())
	}
	if len(snap.DetailRows) != capacity {
		t.Fatalf("expected %d retained detail rows in snapshot, got %d", capacity, len(snap.DetailRows))
	}
	if v.RetainedCount() != capacity {
		t.Fatalf("expected RetainedCount() %d, got %d", capacity, v.RetainedCount())
	}

	// Verify SnapshotRows tuple method
	sumRows, rows := v.SnapshotRows()
	if sumRows.Drift() != 0 {
		t.Errorf("SnapshotRows drift non-zero: %d", sumRows.Drift())
	}
	if len(rows) != capacity {
		t.Errorf("SnapshotRows expected %d rows, got %d", capacity, len(rows))
	}

	// Verify aggregate counters match un-evicted stream total exactly
	if sum.TotalCalls != expectedTotalCalls {
		t.Errorf("TotalCalls mismatch: expected %d, got %d", expectedTotalCalls, sum.TotalCalls)
	}
	if sum.TotalToolCalls != expectedTotalToolCalls {
		t.Errorf("TotalToolCalls mismatch: expected %d, got %d", expectedTotalToolCalls, sum.TotalToolCalls)
	}
	if sum.TotalAuditEvents != expectedTotalAuditEvents {
		t.Errorf("TotalAuditEvents mismatch: expected %d, got %d", expectedTotalAuditEvents, sum.TotalAuditEvents)
	}
	if sum.PromptTokens != expectedPromptTokens {
		t.Errorf("PromptTokens mismatch: expected %d, got %d", expectedPromptTokens, sum.PromptTokens)
	}
	if sum.OutputTokens != expectedOutputTokens {
		t.Errorf("OutputTokens mismatch: expected %d, got %d", expectedOutputTokens, sum.OutputTokens)
	}
	if sum.CachedTokens != expectedCachedTokens {
		t.Errorf("CachedTokens mismatch: expected %d, got %d", expectedCachedTokens, sum.CachedTokens)
	}
	if sum.TotalTokens != expectedTotalTokens {
		t.Errorf("TotalTokens mismatch: expected %d, got %d", expectedTotalTokens, sum.TotalTokens)
	}
	if diff := sum.TotalCostUSD - expectedTotalCostUSD; diff < -1e-6 || diff > 1e-6 {
		t.Errorf("TotalCostUSD mismatch: expected %f, got %f", expectedTotalCostUSD, sum.TotalCostUSD)
	}
	if sum.TotalErrors != expectedTotalErrors {
		t.Errorf("TotalErrors mismatch: expected %d, got %d", expectedTotalErrors, sum.TotalErrors)
	}

	// Verify session identity was captured
	if snap.Session.SessionID != "sess-50k" {
		t.Errorf("expected session sess-50k, got %s", snap.Session.SessionID)
	}

	// Verify FIFO order: retained rows should be events (totalEvents - capacity + 1) to totalEvents
	firstRetainedExpectedID := fmt.Sprintf("%d", totalEvents-capacity+1)
	if !containsID(snap.DetailRows[0].RowID(), firstRetainedExpectedID) {
		t.Errorf("first retained row ID mismatch: expected to contain %s, got %s", firstRetainedExpectedID, snap.DetailRows[0].RowID())
	}
	lastRetainedExpectedID := fmt.Sprintf("%d", totalEvents)
	if snap.DetailRows[capacity-1].RowID() != "sess-50k" && !containsID(snap.DetailRows[capacity-1].RowID(), lastRetainedExpectedID) {
		t.Errorf("last retained row ID mismatch: expected to contain %s, got %s", lastRetainedExpectedID, snap.DetailRows[capacity-1].RowID())
	}
}

func containsID(rowID, target string) bool {
	return len(rowID) >= len(target) && rowID[len(rowID)-len(target):] == target
}

func TestMaterializedView_NewMaterializedView_ConstructorAndSinks(t *testing.T) {
	sliceSink := NewSliceSink()
	v := NewMaterializedView(20, sliceSink)
	if v.Capacity() != 20 {
		t.Fatalf("expected capacity 20, got %d", v.Capacity())
	}

	err := v.Ingest(ToolCallRow{
		ToolCallID: "tool-first",
		SessionID:  "sess-auto-populated",
		ToolName:   "grep",
	})
	if err != nil {
		t.Fatalf("Ingest failed: %v", err)
	}

	// Session ID should be auto-populated
	if v.Session().SessionID != "sess-auto-populated" {
		t.Errorf("expected auto-populated session sess-auto-populated, got %s", v.Session().SessionID)
	}
	if v.Summary().SessionID != "sess-auto-populated" {
		t.Errorf("expected auto-populated summary session sess-auto-populated, got %s", v.Summary().SessionID)
	}

	// Sink received the row
	if sliceSink.Count() != 1 {
		t.Errorf("expected sliceSink count 1, got %d", sliceSink.Count())
	}

	// SnapshotRows
	sum, rows := v.SnapshotRows()
	if sum.TotalEvents != 1 || sum.RetainedEvents != 1 || sum.EvictedEvents != 0 || sum.Drift() != 0 {
		t.Errorf("unexpected summary from SnapshotRows: %+v", sum)
	}
	if len(rows) != 1 || rows[0].RowID() != "tool-first" {
		t.Errorf("unexpected rows from SnapshotRows: %+v", rows)
	}
}

func BenchmarkMaterializedView_Ingest_FiftyThousandEvents(b *testing.B) {
	row := LlmCallRow{
		CallID:       "bench-call",
		SessionID:    "bench-sess",
		Model:        "fak-model",
		PromptTokens: 100,
		OutputTokens: 50,
		TotalTokens:  150,
		CachedTokens: 25,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := NewMaterializedView(10000)
		for j := 0; j < 50000; j++ {
			_ = v.Ingest(row)
		}
	}
}
