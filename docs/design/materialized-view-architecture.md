# Bounded Materialized View Architecture for Session Observability

## 1. Overview and Problem Statement

Autonomous agent workflows in `fak` produce high-velocity event streams spanning LLM completions, tool invocations, token debits, policy checks, and session lifecycle transitions. Consuming these events poses two conflicting requirements:
1. **Low-latency, bounded memory consumption**: The runtime cannot allow in-memory event retention to grow without bound during long-running or multi-turn sessions. Unbounded event lists cause heap bloat, GC pressure, and catastrophic OOMs under high throughput.
2. **Accurate lifetime accounting and auditability**: Downstream systems, operators, and billing observers require point-in-time accurate aggregate counters (e.g. cumulative token expenditure, total call counts, error rates) that remain strictly monotonic and drift-free, even after transient detail rows are evicted.

The `internal/sessionview` package resolves this tension by implementing a **bounded materialized read-model** with dual semantics:
- **Bounded detail retention**: Recent event rows are stored in a fixed-size, zero-reallocation circular ring buffer with deterministic FIFO eviction.
- **Monotonic cumulative counters**: Summary statistics are accumulated lifetime counters that never decrease or drift when detail rows are evicted.
- **Atomic snapshot isolation**: Point-in-time `Snapshot()` queries capture consistent state across summary metrics and current retained rows without lock contention on sink delivery.
- **Direct row-level sink ingestion**: Pluggable `ViewSink` interfaces allow external consumers (metrics collectors, structured loggers, streaming buses) to intercept materialized rows directly as they arrive.

---

## 2. Architecture & Data Flow

```
                     +---------------------------------------+
                     |             Incoming Row              |
                     | (LlmCall, ToolCall, TokenUsage, etc.) |
                     +-------------------+-------------------+
                                         |
                                         v
                         +-------------------------------+
                         |     MaterializedView.Append   |
                         +---------------+---------------+
                                         |
                 [Acquire v.mu.Lock()]   |
                                         v
          +-------------------------------------------------------------+
          | 1. Update Monotonic Summary Counters                        |
          |    (TotalEvents++, TotalTokens+=N, TotalCalls++, etc.)      |
          |                                                             |
          | 2. Push to Bounded Circular Ring Buffer (Capacity = K)      |
          |    - If count < K: retain row                               |
          |    - If count == K: evict oldest FIFO row, EvictedEvents++  |
          |                                                             |
          | 3. Copy Registered Sinks Slice                              |
          +------------------------------+------------------------------+
                                         |
                 [Release v.mu.Lock()]   |
                                         v
                         +---------------+---------------+
                         |    Sink Dispatch (sinkMu)     |
                         |   sink.ConsumeRow(clonedRow)  |
                         +---------------+---------------+
                                         |
                     +-------------------+-------------------+
                     |                   |                   |
                     v                   v                   v
              [SliceSink]          [ChannelSink]       [Custom Logger/Metrics]
```

---

## 3. Core Row Models

All rows in `internal/sessionview` implement the universal `Row` interface:

```go
type Row interface {
    RowKind() RowKind
    RowID() string
    Session() string
    EventTime() time.Time
    OccurredAt() time.Time
}
```

### Row Model Catalog

| Type | Discriminator (`RowKind`) | Core Payload & Identity | Summary Impact |
| :--- | :--- | :--- | :--- |
| `SessionRow` | `RowKindSession` | `SessionID`, `TraceID`, `State`, `Model`, `Labels`, `Metadata` | Updates view session metadata; error states increment `TotalErrors` |
| `LlmCallRow` | `RowKindLlmCall` | `CallID`, `TurnID`, `Model`, `PromptTokens`, `OutputTokens`, `TotalTokens`, `CachedTokens`, `Duration`, `FinishReason`, `Error` | Increments `TotalCalls`, `TotalTokens`, `PromptTokens`, `OutputTokens`, `CachedTokens`; errors increment `TotalErrors` |
| `TokenUsageRow` | `RowKindTokenUsage` | `UsageID`, `CallID`, `PromptTokens`, `OutputTokens`, `CachedTokens`, `TotalTokens`, `CostUSD` | Increments `PromptTokens`, `OutputTokens`, `CachedTokens`, `TotalTokens`, `TotalCostUSD` |
| `ToolCallRow` | `RowKindToolCall` | `ToolCallID`, `TurnID`, `ToolName`, `Arguments`, `Result`, `Duration`, `Admitted`, `Error` | Increments `TotalToolCalls`; execution errors increment `TotalErrors` |
| `AuditEventRow` | `RowKindAuditEvent` | `EventID`, `TraceID`, `Component`, `Action`, `Severity`, `Message`, `Payload` | Increments `TotalAuditEvents`; severity `"error"` increments `TotalErrors` |
| `SnapshotSummary` | N/A (Summary Model) | Aggregate counters across all historical events | Monotonic summary record exportable via `Snapshot()` or `Summary()` |

---

## 4. Architectural Invariants

The `MaterializedView` enforces five invariants:

### Invariant 1: Strict Monotonicity
All cumulative summary counters (`TotalTokens`, `TotalCalls`, `TotalToolCalls`, `TotalAuditEvents`, `TotalEvents`, `PromptTokens`, `OutputTokens`, `CachedTokens`, `TotalCostUSD`, `TotalErrors`, and `EvictedEvents`) are non-decreasing over the lifetime of the view:
$$\forall t_2 \ge t_1, \quad C(t_2) \ge C(t_1)$$
Eviction of detail rows from the ring buffer never decrements or alters these counters.

### Invariant 2: Zero Counter Drift
At all times $t$ (under single-threaded and high-throughput concurrent workloads), the total number of events processed is exactly equal to the number of evicted events plus the number of currently retained detail rows:
$$\text{TotalEvents} = \text{EvictedEvents} + \text{RetainedEvents}$$
The package provides a formal verification helper `s.Drift()`, defined as:
```go
func (s SnapshotSummary) Drift() int64 {
    return s.TotalEvents - (s.EvictedEvents + s.RetainedEvents)
}
```
A non-zero drift value indicates state corruption or lost update anomalies. The test suite verifies `Drift() == 0` continuously under high-throughput concurrent streams.

### Invariant 3: Strictly Bounded In-Memory Retention
The internal detail buffer uses a fixed-capacity circular ring buffer initialized once at construction (`newRingBuffer(capacity)`).
- Push operations execute in $O(1)$ time with zero slice reallocations.
- Oldest rows are evicted in strict FIFO order when capacity is reached.
- Once capacity is reached, memory occupancy of the buffer remains strictly constant $O(\text{Capacity})$.
- Overwritten slots replace old references, enabling garbage collection of transient payload strings and maps.

### Invariant 4: Atomic Point-in-Time Snapshot Isolation
The `Snapshot()` method acquires a read lock (`mu.RLock()`) and copies:
1. The scalar `SnapshotSummary` struct.
2. A cloned `SessionRow` with deep-copied label and metadata maps.
3. Deep-copied slices of currently retained detail rows (`DetailRows`, `LlmCalls`, `TokenUsages`, `ToolCalls`, `AuditEvents`).
Mutations to returned snapshot objects by callers cannot corrupt or mutate internal view state.

### Invariant 5: Decoupled Sink Dispatch
External `ViewSink` consumers receive rows outside the primary view lock (`mu`).
- Summary counter updates and buffer enqueueing occur under `v.mu.Lock()`.
- Sinks are invoked under a dedicated `v.sinkMu.Lock()` to ensure serialized in-order delivery without blocking concurrent readers (`v.Snapshot()`, `v.Summary()`) or starving fast writers.
- Sinks that fail return errors aggregated via `errors.Join`.

---

## 5. View Construction & Ingestion API

The engine provides flexible constructors and strongly-typed ingestion methods:

```go
// Create view with configurable retention capacity and optional sinks
v := sessionview.NewMaterializedView(10000, sinks...)

// Or explicitly bind a session ID at construction
v := sessionview.New("sess-123", 10000, sinks...)
```

### Ingestion Methods
```go
// Universal row ingestion
err := v.Ingest(row) // or v.Append(row)

// Typed ingestion helpers
err := v.IngestSession(sessionRow)
err := v.IngestLlmCall(llmCallRow)
err := v.IngestTokenUsage(tokenUsageRow)
err := v.IngestToolCall(toolCallRow)
err := v.IngestAuditEvent(auditEventRow)
```

If a view is created without an explicit session ID (via `NewMaterializedView`), the session ID is automatically discovered and bound from the first ingested row with a non-empty `Session()`.

### Snapshot Export
```go
// Point-in-time consistent snapshot with typed collections
snap := v.Snapshot() // returns ViewSnapshot (alias for Snapshot)

// Direct summary and retained rows tuple
summary, retainedRows := v.SnapshotRows()
```

---

## 6. ViewSink Interface & Integration

Consumers implement `ViewSink` to receive materialized rows as they arrive:

```go
type ViewSink interface {
    ConsumeRow(row Row) error
}
```

Built-in implementations include:
- `SinkFunc`: Functional adapter (`type SinkFunc func(row Row) error`).
- `ChannelSink`: Non-blocking or bounded channel queue for asynchronous streaming to remote observers or background workers.
- `SliceSink`: Thread-safe in-memory slice accumulator for deterministic unit testing and audit assertions.

---

## 7. Verification and Benchmarks

The package is verified via extensive unit tests, concurrent stress tests, large-stream retention proofs, and microbenchmarks:

```
BenchmarkMaterializedView_Append_Sequential-32              9911497       223.8 ns/op
BenchmarkMaterializedView_Append_Parallel-32                6556101       201.1 ns/op
BenchmarkMaterializedView_Snapshot-32                         13093    102048.0 ns/op
BenchmarkMaterializedView_ConcurrentAppendAndSnapshot-32     72734     18200.0 ns/op
BenchmarkRingBuffer_PushAndEvict-32                       513072883         2.33 ns/op
BenchmarkMaterializedView_Ingest_CappedRingEviction-32      6268546       203.3 ns/op
BenchmarkMaterializedView_Ingest_FiftyThousandEvents-32         219   5883575.0 ns/op
```

- **Ring Buffer Throughput**: ~2.33 ns per push/evict operation (>513M ops/sec).
- **Ingestion & Eviction Overhead**: ~203 ns/op (>6.2M events/sec) under continuous eviction with full monotonic summary maintenance.
- **Large-Stream Bounded Retention Verification**: `TestMaterializedView_BoundedRetention_FiftyThousandEvents` ingests 50,000 synthetic events into a view capped at 10,000 rows. The test validates:
  - Exactly 10,000 retained rows, 40,000 evicted events, 50,000 total events.
  - Zero counter drift: `Drift() == 0` at every checkpoint.
  - Aggregate metrics (`TotalTokens`, `PromptTokens`, `OutputTokens`, `CachedTokens`, `TotalCalls`, `TotalToolCalls`, `TotalAuditEvents`, `TotalCostUSD`, `TotalErrors`) match the un-evicted stream total exactly.
  - Detail rows follow strict FIFO eviction order (retaining events 40,001 through 50,000).
- **Parallel Append Throughput**: >6.5 million operations/sec (~201 ns/op) under 32 concurrent worker routines.
- **Zero Counter Drift**: Proven under high concurrency and mixed event workloads with simultaneous asynchronous `Snapshot()` executions.
