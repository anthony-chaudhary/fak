package agent

import (
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

type RuntimeObserver struct {
	mu                 sync.Mutex
	writer             io.Writer
	transport          trajectory.RuntimeTransport
	sessionID, traceID string
	source             trajectory.RuntimeSource
	seenResults        map[string]struct{}
	sequence           uint64
	err                error
	now                func() time.Time
}

func NewRuntimeObserver(w io.Writer, transport trajectory.RuntimeTransport, sessionID, traceID string, source trajectory.RuntimeSource) (*RuntimeObserver, error) {
	if source.Rung == "" {
		source.Rung = "owned-agent-loop"
	}
	if w == nil || sessionID == "" || traceID == "" || source.Component == "" || source.Instance == "" || source.Runtime == "" {
		return nil, fmt.Errorf("runtime observer writer, identity, and source are required")
	}
	return &RuntimeObserver{writer: w, transport: transport, sessionID: sessionID, traceID: traceID, source: source, seenResults: map[string]struct{}{}, now: time.Now}, nil
}

func (o *RuntimeObserver) Observe(progress ProgressEvent) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.err != nil {
		return
	}
	kind, payload, err := runtimeProgressPayload(progress)
	if err != nil {
		o.err = err
		return
	}
	if kind == trajectory.RuntimeToolResult {
		if _, ok := o.seenResults[progress.CallID]; ok {
			o.err = fmt.Errorf("duplicate result admission for %q", progress.CallID)
			return
		}
		o.seenResults[progress.CallID] = struct{}{}
	}
	o.sequence++
	turnID := fmt.Sprintf("%s:turn:%d", o.traceID, progress.Turn)
	eventID := fmt.Sprintf("%s:event:%d", o.traceID, o.sequence)
	event, err := trajectory.NewRuntimeEvent(eventID, o.sessionID, turnID, o.traceID, o.sequence, o.now().UTC(), kind, o.source, payload)
	if err != nil {
		o.err = err
		return
	}
	o.err = trajectory.WriteRuntimeEvent(o.writer, event, o.transport)
}

func (o *RuntimeObserver) Err() error { o.mu.Lock(); defer o.mu.Unlock(); return o.err }

func runtimeProgressPayload(p ProgressEvent) (trajectory.RuntimeEventKind, json.RawMessage, error) {
	var kind trajectory.RuntimeEventKind
	var body any
	switch p.Kind {
	case ProgressTurnStarted:
		kind = trajectory.RuntimeTurnStarted
		body = map[string]any{"turn": p.Turn}
	case ProgressToolStarted:
		kind = trajectory.RuntimeToolProposed
		body = map[string]any{"turn": p.Turn, "call_id": p.CallID, "tool": p.Tool}
	case ProgressCallAdjudicated:
		kind = trajectory.RuntimeVerdict
		body = map[string]any{"turn": p.Turn, "call_id": p.CallID, "tool": p.Tool, "verdict": p.Verdict, "reason": p.Reason}
	case ProgressResultAdmitted:
		kind = trajectory.RuntimeToolResult
		body = map[string]any{"turn": p.Turn, "call_id": p.CallID, "tool": p.Tool, "taint": p.Taint, "summary": p.Summary}
	case ProgressTurnDone:
		kind = trajectory.RuntimeTerminalWitness
		body = map[string]any{"turn": p.Turn, "status": "completed"}
	default:
		return "", nil, fmt.Errorf("unknown progress kind %q", p.Kind)
	}
	b, err := json.Marshal(body)
	return kind, b, err
}
