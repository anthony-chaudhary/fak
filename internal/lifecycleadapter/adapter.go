// Package lifecycleadapter negotiates and invokes heterogeneous process-forest lifecycle adapters.
package lifecycleadapter

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

const ProtocolVersion = "fak-lifecycle-adapter/1"

type Operation string

const (
	Prepare    Operation = "prepare"
	Pause      Operation = "pause"
	Checkpoint Operation = "checkpoint"
	Restore    Operation = "restore"
	Resume     Operation = "resume"
	Readiness  Operation = "readiness"
)

type CapabilityDocument struct {
	Protocol              string      `json:"protocol"`
	AdapterKind           string      `json:"adapter_kind"`
	Operations            []Operation `json:"operations"`
	ApplicationCheckpoint bool        `json:"application_checkpoint"`
}
type ResultState string

const (
	ResultCompleted   ResultState = "completed"
	ResultUnsupported ResultState = "unsupported"
	ResultRefused     ResultState = "refused"
	ResultFailed      ResultState = "failed"
)

type Result struct {
	State         ResultState `json:"state"`
	Reason        string      `json:"reason,omitempty"`
	ReadbackRef   string      `json:"readback_ref,omitempty"`
	CheckpointRef string      `json:"checkpoint_ref,omitempty"`
}
type Request struct {
	TransactionID string
	ForestID      string
	MemberID      string
	Generation    uint64
	Operation     Operation
	Deadline      time.Time
}
type Adapter interface {
	Capabilities() CapabilityDocument
	Invoke(context.Context, Request) Result
}
type Negotiation struct {
	TransactionID string             `json:"transaction_id"`
	ForestID      string             `json:"forest_id"`
	MemberID      string             `json:"member_id"`
	Generation    uint64             `json:"generation"`
	Document      CapabilityDocument `json:"capability_document"`
	Operation     Operation          `json:"operation"`
	Supported     bool               `json:"supported"`
	Reason        string             `json:"reason,omitempty"`
}

func Negotiate(req Request, a Adapter) (Negotiation, error) {
	n := Negotiation{TransactionID: req.TransactionID, ForestID: req.ForestID, MemberID: req.MemberID, Generation: req.Generation, Operation: req.Operation}
	if a == nil {
		return n, errors.New("adapter unavailable")
	}
	n.Document = a.Capabilities()
	if n.Document.Protocol != ProtocolVersion {
		n.Reason = "adapter protocol version unsupported"
		return n, errors.New(n.Reason)
	}
	for _, op := range n.Document.Operations {
		if op == req.Operation {
			n.Supported = true
			return n, nil
		}
	}
	n.Reason = "operation unsupported by adapter"
	return n, errors.New(n.Reason)
}
func Execute(ctx context.Context, req Request, a Adapter) (Negotiation, Result) {
	n, err := Negotiate(req, a)
	if err != nil {
		return n, Result{State: ResultUnsupported, Reason: n.Reason}
	}
	if req.Deadline.IsZero() {
		return n, Result{State: ResultRefused, Reason: "deadline required"}
	}
	runCtx, cancel := context.WithDeadline(ctx, req.Deadline)
	defer cancel()
	result := a.Invoke(runCtx, req)
	if runCtx.Err() != nil && result.State == "" {
		return n, Result{State: ResultFailed, Reason: "adapter deadline exceeded"}
	}
	if result.State == "" {
		return n, Result{State: ResultFailed, Reason: "adapter returned no typed state"}
	}
	return n, result
}

type Builtin struct {
	protocol              string
	kind                  string
	operations            []Operation
	applicationCheckpoint bool
	invoke                func(context.Context, Request) Result
}

func (b Builtin) Capabilities() CapabilityDocument {
	ops := append([]Operation(nil), b.operations...)
	sort.Slice(ops, func(i, j int) bool { return ops[i] < ops[j] })
	return CapabilityDocument{Protocol: func() string {
		if b.protocol != "" {
			return b.protocol
		}
		return ProtocolVersion
	}(), AdapterKind: b.kind, Operations: ops, ApplicationCheckpoint: b.applicationCheckpoint}
}
func (b Builtin) Invoke(ctx context.Context, r Request) Result {
	if b.invoke != nil {
		return b.invoke(ctx, r)
	}
	select {
	case <-ctx.Done():
		return Result{State: ResultFailed, Reason: "adapter deadline exceeded"}
	default:
		return Result{State: ResultCompleted, ReadbackRef: fmt.Sprintf("%s:%s", b.kind, r.Operation)}
	}
}
func NativeFAK() Adapter {
	return Builtin{kind: "fak-harness", operations: []Operation{Prepare, Pause, Checkpoint, Restore, Resume, Readiness}, applicationCheckpoint: true}
}

//enumlint:exempt Codex has no application checkpoint/restore capability; Capabilities reports that boundary explicitly.
func Codex() Adapter {
	return Builtin{kind: "codex", operations: []Operation{Prepare, Pause, Resume, Readiness}, applicationCheckpoint: false}
}

//enumlint:exempt Claude has no application checkpoint/restore capability; Capabilities reports that boundary explicitly.
func Claude() Adapter {
	return Builtin{kind: "claude", operations: []Operation{Prepare, Pause, Resume, Readiness}, applicationCheckpoint: false}
}
func Custom(doc CapabilityDocument, invoke func(context.Context, Request) Result) Adapter {
	doc.AdapterKind = strings.TrimSpace(doc.AdapterKind)
	return Builtin{protocol: doc.Protocol, kind: doc.AdapterKind, operations: doc.Operations, applicationCheckpoint: doc.ApplicationCheckpoint, invoke: invoke}
}
func Unknown(kind string) Adapter { return Builtin{kind: strings.TrimSpace(kind)} }
