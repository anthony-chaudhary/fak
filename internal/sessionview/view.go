package sessionview

import (
	"errors"
	"sync"
	"time"
)

var (
	// ErrNilRow is returned when an attempt is made to append a nil row.
	ErrNilRow = errors.New("cannot append nil row")
	// ErrViewClosed indicates the view is closed and no longer accepts writes.
	ErrViewClosed = errors.New("materialized view is closed")
)

// DefaultCapacity is the default detail row retention capacity if not specified.
const DefaultCapacity = 1000

// Config configures a MaterializedView instance.
type Config struct {
	SessionID string
	Capacity  int
	Sinks     []ViewSink
}

// Snapshot exports consistent point-in-time session summaries and current retained detail rows.
type Snapshot struct {
	Summary     SnapshotSummary `json:"summary"`
	Session     SessionRow      `json:"session"`
	DetailRows  []Row           `json:"detail_rows"`
	LlmCalls    []LlmCallRow    `json:"llm_calls"`
	TokenUsages []TokenUsageRow `json:"token_usages"`
	ToolCalls   []ToolCallRow   `json:"tool_calls"`
	AuditEvents []AuditEventRow `json:"audit_events"`
}

// ViewSnapshot is an alias for Snapshot, representing point-in-time session state.
type ViewSnapshot = Snapshot

// RetainedCount returns the number of currently retained detail rows in this snapshot.
func (s Snapshot) RetainedCount() int {
	return len(s.DetailRows)
}

// IsEmpty reports whether the snapshot has zero events and zero retained rows.
func (s Snapshot) IsEmpty() bool {
	return len(s.DetailRows) == 0 && s.Summary.TotalEvents == 0
}

// ringBuffer is a bounded, zero-reallocation FIFO buffer of Row elements.
type ringBuffer struct {
	capacity int
	items    []Row
	head     int // index of oldest element
	count    int // current count of stored items
}

func newRingBuffer(capacity int) *ringBuffer {
	if capacity <= 0 {
		capacity = DefaultCapacity
	}
	return &ringBuffer{
		capacity: capacity,
		items:    make([]Row, capacity),
		head:     0,
		count:    0,
	}
}

func (r *ringBuffer) push(item Row) (evicted Row, wasEvicted bool) {
	if r.count < r.capacity {
		idx := (r.head + r.count) % r.capacity
		r.items[idx] = item
		r.count++
		return nil, false
	}
	evicted = r.items[r.head]
	r.items[r.head] = item
	r.head = (r.head + 1) % r.capacity
	return evicted, true
}

func (r *ringBuffer) all() []Row {
	out := make([]Row, r.count)
	for i := 0; i < r.count; i++ {
		out[i] = r.items[(r.head+i)%r.capacity]
	}
	return out
}

func (r *ringBuffer) len() int {
	return r.count
}

func (r *ringBuffer) cap() int {
	return r.capacity
}

func (r *ringBuffer) reset() {
	for i := range r.items {
		r.items[i] = nil
	}
	r.head = 0
	r.count = 0
}

// MaterializedView maintains bounded in-memory detail rows (FIFO) while accumulating
// strictly monotonic summary counters across the session lifecycle.
type MaterializedView struct {
	mu      sync.RWMutex
	sinkMu  sync.Mutex
	session SessionRow
	summary SnapshotSummary
	buffer  *ringBuffer
	sinks   []ViewSink
	closed  bool
}

// New creates a new MaterializedView with the specified sessionID, retention capacity, and optional sinks.
func New(sessionID string, capacity int, sinks ...ViewSink) *MaterializedView {
	if capacity <= 0 {
		capacity = DefaultCapacity
	}
	now := time.Now().UTC()
	v := &MaterializedView{
		session: SessionRow{
			SessionID: sessionID,
			State:     "active",
			CreatedAt: now,
			UpdatedAt: now,
		},
		summary: SnapshotSummary{
			SessionID: sessionID,
			Capacity:  capacity,
		},
		buffer: newRingBuffer(capacity),
		sinks:  make([]ViewSink, 0, len(sinks)),
	}
	for _, s := range sinks {
		if s != nil {
			v.sinks = append(v.sinks, s)
		}
	}
	return v
}

// NewMaterializedView creates a new MaterializedView with configurable retention capacity and optional sinks.
func NewMaterializedView(capacity int, sinks ...ViewSink) *MaterializedView {
	return New("", capacity, sinks...)
}

// NewWithSession creates a new MaterializedView with the specified sessionID, retention capacity, and optional sinks.
func NewWithSession(sessionID string, capacity int, sinks ...ViewSink) *MaterializedView {
	return New(sessionID, capacity, sinks...)
}

// NewWithConfig creates a MaterializedView from a Config struct.
func NewWithConfig(cfg Config) *MaterializedView {
	return New(cfg.SessionID, cfg.Capacity, cfg.Sinks...)
}

// Append adds a row to the view: updates monotonic counters, stores in bounded FIFO,
// and notifies registered ViewSinks.
func (v *MaterializedView) Append(row Row) error {
	if row == nil {
		return ErrNilRow
	}

	v.mu.Lock()
	if v.closed {
		v.mu.Unlock()
		return ErrViewClosed
	}

	cloned := v.processRowLocked(row)
	if cloned == nil {
		v.mu.Unlock()
		return ErrNilRow
	}

	_, wasEvicted := v.buffer.push(cloned)
	if wasEvicted {
		v.summary.EvictedEvents++
	}
	v.summary.RetainedEvents = int64(v.buffer.len())

	var sinksCopy []ViewSink
	if len(v.sinks) > 0 {
		sinksCopy = make([]ViewSink, len(v.sinks))
		copy(sinksCopy, v.sinks)
	}
	v.mu.Unlock()

	// Dispatch to sinks in sequence order outside the primary read/write lock.
	if len(sinksCopy) > 0 {
		v.sinkMu.Lock()
		defer v.sinkMu.Unlock()
		var errs []error
		for _, sink := range sinksCopy {
			if err := sink.ConsumeRow(cloned); err != nil {
				errs = append(errs, err)
			}
		}
		if len(errs) > 0 {
			return errors.Join(errs...)
		}
	}

	return nil
}

// Ingest is an alias for Append.
func (v *MaterializedView) Ingest(row Row) error {
	return v.Append(row)
}

// AppendSession records or updates the session row.
func (v *MaterializedView) AppendSession(row SessionRow) error {
	return v.Append(row)
}

// AppendLlmCall records an LLM invocation turn row.
func (v *MaterializedView) AppendLlmCall(row LlmCallRow) error {
	return v.Append(row)
}

// AppendTokenUsage records a token debit usage row.
func (v *MaterializedView) AppendTokenUsage(row TokenUsageRow) error {
	return v.Append(row)
}

// AppendToolCall records a tool execution row.
func (v *MaterializedView) AppendToolCall(row ToolCallRow) error {
	return v.Append(row)
}

// AppendAuditEvent records a security/governance audit event row.
func (v *MaterializedView) AppendAuditEvent(row AuditEventRow) error {
	return v.Append(row)
}

// IngestSession is an alias for AppendSession.
func (v *MaterializedView) IngestSession(row SessionRow) error { return v.Append(row) }

// IngestLlmCall is an alias for AppendLlmCall.
func (v *MaterializedView) IngestLlmCall(row LlmCallRow) error { return v.Append(row) }

// IngestTokenUsage is an alias for AppendTokenUsage.
func (v *MaterializedView) IngestTokenUsage(row TokenUsageRow) error { return v.Append(row) }

// IngestToolCall is an alias for AppendToolCall.
func (v *MaterializedView) IngestToolCall(row ToolCallRow) error { return v.Append(row) }

// IngestAuditEvent is an alias for AppendAuditEvent.
func (v *MaterializedView) IngestAuditEvent(row AuditEventRow) error { return v.Append(row) }

// Snapshot atomically exports a point-in-time consistent session summary and retained detail rows.
func (v *MaterializedView) Snapshot() ViewSnapshot {
	if v == nil {
		return Snapshot{}
	}

	v.mu.RLock()
	defer v.mu.RUnlock()

	sum := v.summary
	sum.SnapshotAt = time.Now().UTC()
	sum.RetainedEvents = int64(v.buffer.len())

	session := v.session.clone()
	retained := v.buffer.all()

	detailRows := make([]Row, len(retained))
	llmCalls := make([]LlmCallRow, 0)
	tokenUsages := make([]TokenUsageRow, 0)
	toolCalls := make([]ToolCallRow, 0)
	auditEvents := make([]AuditEventRow, 0)

	for i, r := range retained {
		switch row := r.(type) {
		case LlmCallRow:
			c := row.clone()
			detailRows[i] = c
			llmCalls = append(llmCalls, c)
		case TokenUsageRow:
			c := row.clone()
			detailRows[i] = c
			tokenUsages = append(tokenUsages, c)
		case ToolCallRow:
			c := row.clone()
			detailRows[i] = c
			toolCalls = append(toolCalls, c)
		case AuditEventRow:
			c := row.clone()
			detailRows[i] = c
			auditEvents = append(auditEvents, c)
		case SessionRow:
			c := row.clone()
			detailRows[i] = c
		default:
			detailRows[i] = r
		}
	}

	return Snapshot{
		Summary:     sum,
		Session:     session,
		DetailRows:  detailRows,
		LlmCalls:    llmCalls,
		TokenUsages: tokenUsages,
		ToolCalls:   toolCalls,
		AuditEvents: auditEvents,
	}
}

// SnapshotRows returns the summary and retained rows directly as a tuple in a thread-safe manner.
func (v *MaterializedView) SnapshotRows() (SnapshotSummary, []Row) {
	if v == nil {
		return SnapshotSummary{}, nil
	}
	v.mu.RLock()
	defer v.mu.RUnlock()

	sum := v.summary
	sum.SnapshotAt = time.Now().UTC()
	sum.RetainedEvents = int64(v.buffer.len())
	return sum, v.buffer.all()
}

// Summary atomically returns a copy of the current monotonic summary counters.
func (v *MaterializedView) Summary() SnapshotSummary {
	if v == nil {
		return SnapshotSummary{}
	}
	v.mu.RLock()
	defer v.mu.RUnlock()

	sum := v.summary
	sum.SnapshotAt = time.Now().UTC()
	sum.RetainedEvents = int64(v.buffer.len())
	return sum
}

// Session atomically returns a copy of current session metadata.
func (v *MaterializedView) Session() SessionRow {
	if v == nil {
		return SessionRow{}
	}
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.session.clone()
}

// RetainedRows returns a point-in-time snapshot of the currently retained FIFO detail rows.
func (v *MaterializedView) RetainedRows() []Row {
	if v == nil {
		return nil
	}
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.buffer.all()
}

// RetainedCount returns the number of detail rows currently held in the bounded buffer.
func (v *MaterializedView) RetainedCount() int {
	if v == nil {
		return 0
	}
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.buffer.len()
}

// Capacity returns the maximum retention capacity of the view.
func (v *MaterializedView) Capacity() int {
	if v == nil {
		return 0
	}
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.summary.Capacity
}

// RegisterSink attaches a ViewSink to receive subsequent rows.
func (v *MaterializedView) RegisterSink(sink ViewSink) {
	if sink == nil {
		return
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	v.sinks = append(v.sinks, sink)
}

// RegisterSinkFunc attaches a function as a ViewSink.
func (v *MaterializedView) RegisterSinkFunc(fn func(row Row) error) {
	if fn == nil {
		return
	}
	v.RegisterSink(SinkFunc(fn))
}

// UnregisterSink detaches a previously registered ViewSink.
func (v *MaterializedView) UnregisterSink(sink ViewSink) {
	if sink == nil {
		return
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	filtered := v.sinks[:0]
	for _, s := range v.sinks {
		if s != sink {
			filtered = append(filtered, s)
		}
	}
	v.sinks = filtered
}

// UpdateSession mutates the session state or labels directly.
func (v *MaterializedView) UpdateSession(state string, labels map[string]string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if state != "" {
		v.session.State = state
	}
	if labels != nil {
		if v.session.Labels == nil {
			v.session.Labels = make(map[string]string)
		}
		for k, val := range labels {
			v.session.Labels[k] = val
		}
	}
	v.session.UpdatedAt = time.Now().UTC()
}

// Reset clears the retained FIFO detail buffer and resets summary counters to zero.
func (v *MaterializedView) Reset() {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.buffer.reset()
	sessionID := v.summary.SessionID
	cap := v.summary.Capacity
	v.summary = SnapshotSummary{
		SessionID: sessionID,
		Capacity:  cap,
	}
	v.session.UpdatedAt = time.Now().UTC()
}

// Close terminates the view, refusing subsequent Appends.
func (v *MaterializedView) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.closed = true
	return nil
}

func (v *MaterializedView) processRowLocked(row Row) Row {
	// Dereference pointer types first to avoid double counting
	switch r := row.(type) {
	case *SessionRow:
		if r == nil {
			return nil
		}
		row = *r
	case *LlmCallRow:
		if r == nil {
			return nil
		}
		row = *r
	case *TokenUsageRow:
		if r == nil {
			return nil
		}
		row = *r
	case *ToolCallRow:
		if r == nil {
			return nil
		}
		row = *r
	case *AuditEventRow:
		if r == nil {
			return nil
		}
		row = *r
	}

	now := time.Now().UTC()
	rowTime := row.EventTime()
	if rowTime.IsZero() {
		rowTime = now
	}

	if v.summary.FirstEventAt.IsZero() || rowTime.Before(v.summary.FirstEventAt) {
		v.summary.FirstEventAt = rowTime
	}
	if rowTime.After(v.summary.LastEventAt) {
		v.summary.LastEventAt = rowTime
	}

	v.summary.TotalEvents++

	if v.summary.SessionID == "" && row.Session() != "" {
		v.summary.SessionID = row.Session()
		if v.session.SessionID == "" {
			v.session.SessionID = row.Session()
		}
	}

	switch r := row.(type) {
	case SessionRow:
		cloned := r.clone()
		if cloned.CreatedAt.IsZero() {
			cloned.CreatedAt = now
		}
		if cloned.UpdatedAt.IsZero() {
			cloned.UpdatedAt = now
		}
		v.session = cloned
		if v.summary.SessionID == "" {
			v.summary.SessionID = cloned.SessionID
		}
		if cloned.State == "failed" || cloned.State == "error" {
			v.summary.TotalErrors++
		}
		return v.session

	case LlmCallRow:
		cloned := r.clone()
		if cloned.Timestamp.IsZero() {
			cloned.Timestamp = rowTime
		}
		v.summary.TotalCalls++
		v.summary.PromptTokens += cloned.PromptTokens
		v.summary.OutputTokens += cloned.OutputTokens
		v.summary.CachedTokens += cloned.CachedTokens

		toks := cloned.TotalTokens
		if toks == 0 && (cloned.PromptTokens > 0 || cloned.OutputTokens > 0) {
			toks = cloned.PromptTokens + cloned.OutputTokens
			cloned.TotalTokens = toks
		}
		v.summary.TotalTokens += toks

		if cloned.Error != "" {
			v.summary.TotalErrors++
		}
		v.session.UpdatedAt = rowTime
		return cloned

	case TokenUsageRow:
		cloned := r.clone()
		if cloned.Timestamp.IsZero() {
			cloned.Timestamp = rowTime
		}
		v.summary.PromptTokens += cloned.PromptTokens
		v.summary.OutputTokens += cloned.OutputTokens
		v.summary.CachedTokens += cloned.CachedTokens

		toks := cloned.TotalTokens
		if toks == 0 && (cloned.PromptTokens > 0 || cloned.OutputTokens > 0) {
			toks = cloned.PromptTokens + cloned.OutputTokens
			cloned.TotalTokens = toks
		}
		v.summary.TotalTokens += toks
		if cloned.CostUSD > 0 {
			v.summary.TotalCostUSD += cloned.CostUSD
		}
		v.session.UpdatedAt = rowTime
		return cloned

	case ToolCallRow:
		cloned := r.clone()
		if cloned.Timestamp.IsZero() {
			cloned.Timestamp = rowTime
		}
		v.summary.TotalToolCalls++
		if cloned.Error != "" {
			v.summary.TotalErrors++
		}
		v.session.UpdatedAt = rowTime
		return cloned

	case AuditEventRow:
		cloned := r.clone()
		if cloned.Timestamp.IsZero() {
			cloned.Timestamp = rowTime
		}
		v.summary.TotalAuditEvents++
		if cloned.Severity == "error" {
			v.summary.TotalErrors++
		}
		v.session.UpdatedAt = rowTime
		return cloned

	default:
		v.session.UpdatedAt = rowTime
		return row
	}
}
