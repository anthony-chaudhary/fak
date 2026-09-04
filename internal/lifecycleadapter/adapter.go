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

// ProtocolVersion defines the canonical wire protocol identifier for lifecycle adapter negotiation.
const ProtocolVersion = "fak-lifecycle-adapter/1"

// Operation identifies a lifecycle transition requested on a process forest member.
type Operation string

const (
	// Prepare signals the adapter to initialize resources prior to execution.
	Prepare Operation = "prepare"
	// Pause requests temporary execution suspension for a member process.
	Pause Operation = "pause"
	// Checkpoint persists running state to durable storage for subsequent restoration.
	Checkpoint Operation = "checkpoint"
	// Restore recovers member execution state from a persisted snapshot.
	Restore Operation = "restore"
	// Resume unpauses a suspended process member to continue execution.
	Resume Operation = "resume"
	// Readiness probes whether the member process is initialized and healthy.
	Readiness Operation = "readiness"
)

// CapabilityDocument describes supported lifecycle operations and features advertised by an adapter.
type CapabilityDocument struct {
	Protocol              string      `json:"protocol"`
	AdapterKind           string      `json:"adapter_kind"`
	Operations            []Operation `json:"operations"`
	ApplicationCheckpoint bool        `json:"application_checkpoint"`
}

// ResultState classifies the terminal execution outcome of an adapter operation.
type ResultState string

const (
	// ResultCompleted indicates the requested operation succeeded.
	ResultCompleted ResultState = "completed"
	// ResultUnsupported indicates the adapter lacks capability for the requested operation.
	ResultUnsupported ResultState = "unsupported"
	// ResultRefused indicates the operation was rejected due to unmet admission prerequisites.
	ResultRefused ResultState = "refused"
	// ResultFailed indicates the operation encountered an execution or deadline error.
	ResultFailed ResultState = "failed"
)

// Result conveys the outcome, readback reference, and optional checkpoint metadata from an adapter invocation.
type Result struct {
	State         ResultState `json:"state"`
	Reason        string      `json:"reason,omitempty"`
	ReadbackRef   string      `json:"readback_ref,omitempty"`
	CheckpointRef string      `json:"checkpoint_ref,omitempty"`
}

// Request parameters specify the member, generation, operation, and deadline for a lifecycle invocation.
type Request struct {
	TransactionID string
	ForestID      string
	MemberID      string
	Generation    uint64
	Operation     Operation
	Deadline      time.Time
}

// Adapter abstracts heterogeneous process-forest lifecycle management implementations.
type Adapter interface {
	Capabilities() CapabilityDocument
	Invoke(context.Context, Request) Result
}

// Negotiation records the capability match evaluation between a request and an adapter.
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

// Negotiate verifies protocol compatibility and operation support against adapter capabilities.
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

// Execute validates negotiation and invokes the adapter under the request deadline.
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

// Builtin provides a baseline in-memory Adapter implementation for standard harness and agent environments.
type Builtin struct {
	protocol              string
	kind                  string
	operations            []Operation
	applicationCheckpoint bool
	invoke                func(context.Context, Request) Result
}

// Capabilities returns the capability document advertised by the builtin adapter.
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

// Invoke executes the lifecycle request or runs the configured invocation handler.
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

// NativeFAK instantiates an adapter supporting all native harness operations including application checkpoints.
func NativeFAK() Adapter {
	return Builtin{kind: "fak-harness", operations: []Operation{Prepare, Pause, Checkpoint, Restore, Resume, Readiness}, applicationCheckpoint: true}
}

// Codex returns an adapter configured for OpenAI Codex process lifecycle capabilities.
//
//enumlint:exempt Codex has no application checkpoint/restore capability; Capabilities reports that boundary explicitly.
func Codex() Adapter {
	return Builtin{kind: "codex", operations: []Operation{Prepare, Pause, Resume, Readiness}, applicationCheckpoint: false}
}

// Claude returns an adapter configured for Anthropic Claude process lifecycle capabilities.
//
//enumlint:exempt Claude has no application checkpoint/restore capability; Capabilities reports that boundary explicitly.
func Claude() Adapter {
	return Builtin{kind: "claude", operations: []Operation{Prepare, Pause, Resume, Readiness}, applicationCheckpoint: false}
}

// Custom constructs an adapter from an explicit capability document and invocation handler.
func Custom(doc CapabilityDocument, invoke func(context.Context, Request) Result) Adapter {
	doc.AdapterKind = strings.TrimSpace(doc.AdapterKind)
	return Builtin{protocol: doc.Protocol, kind: doc.AdapterKind, operations: doc.Operations, applicationCheckpoint: doc.ApplicationCheckpoint, invoke: invoke}
}

// Unknown returns an unconfigured fallback adapter that refuses unsupported operations.
func Unknown(kind string) Adapter { return Builtin{kind: strings.TrimSpace(kind)} }
