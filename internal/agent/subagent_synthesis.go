package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

// CompositeReport stores the in-process zero-copy aggregated subagent references.
type CompositeReport struct {
	ID        uint64            `json:"id"`
	Refs      []*abi.Ref        `json:"refs"`
	TotalLen  int64             `json:"total_len"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
}

var (
	compositeMu      sync.RWMutex
	compositeSeq     uint64
	compositeReports = make(map[uint64]*CompositeReport)
)

// PublishSubagentArtifact publishes an artifact reference produced by a subagent directly to the blackboard,
// bypassing serialized stdout or JSON receipts.
func PublishSubagentArtifact(bb *ctxmmu.Blackboard, topic string, ref *abi.Ref, epoch uint64, subagentID string, metadata map[string]string) (string, error) {
	if bb == nil {
		return "", errors.New("agent: blackboard cannot be nil")
	}
	if ref == nil {
		return "", errors.New("agent: subagent ref cannot be nil")
	}

	meta := make(map[string]string, len(metadata)+1)
	for k, v := range metadata {
		meta[k] = v
	}
	if subagentID != "" {
		meta["subagent_id"] = subagentID
	}
	return bb.Publish(topic, ref, epoch, meta)
}

// PublishSubagentPayload wraps raw bytes into an immutable inline abi.Ref and publishes it to the blackboard.
func PublishSubagentPayload(bb *ctxmmu.Blackboard, topic string, payload []byte, epoch uint64, subagentID string, metadata map[string]string) (string, *abi.Ref, error) {
	ref := &abi.Ref{
		Kind:   abi.RefInline,
		Inline: payload,
		Len:    int64(len(payload)),
		Taint:  abi.TaintTrusted,
		Scope:  abi.ScopeAgent,
	}
	id, err := PublishSubagentArtifact(bb, topic, ref, epoch, subagentID, metadata)
	if err != nil {
		return "", nil, err
	}
	return id, ref, nil
}

// ReadSubagentArtifacts reads subagent artifact references directly from the blackboard for the given topic,
// enabling coordinator agents to bypass parsing string receipts or JSON over stdout.
func ReadSubagentArtifacts(bb *ctxmmu.Blackboard, topic string) ([]*abi.Ref, error) {
	if bb == nil {
		return nil, errors.New("agent: blackboard cannot be nil")
	}
	entries := bb.Subscribe(topic)
	refs := make([]*abi.Ref, 0, len(entries))
	for _, e := range entries {
		if e != nil && e.Ref != nil {
			refs = append(refs, e.Ref)
		}
	}
	return refs, nil
}

// ReadSubagentEntries retrieves full blackboard entries for the given topic with metadata and epoch information.
func ReadSubagentEntries(bb *ctxmmu.Blackboard, topic string) ([]*ctxmmu.BlackboardEntry, error) {
	if bb == nil {
		return nil, errors.New("agent: blackboard cannot be nil")
	}
	return bb.Subscribe(topic), nil
}

// ReadSubagentArtifactByID retrieves a single subagent artifact reference by entry ID.
func ReadSubagentArtifactByID(bb *ctxmmu.Blackboard, id string) (*abi.Ref, error) {
	if bb == nil {
		return nil, errors.New("agent: blackboard cannot be nil")
	}
	entry, ok := bb.Lookup(id)
	if !ok || entry == nil {
		return nil, fmt.Errorf("agent: subagent artifact %s not found on blackboard", id)
	}
	return entry.Ref, nil
}

// AggregateSubagentRefs performs zero-copy aggregation of multiple subagent references (*abi.Ref)
// into a single combined synthesized report reference. It unifies lengths, applies lattice taint
// escalation, and scopes conservatively.
func AggregateSubagentRefs(refs []*abi.Ref, metadata map[string]string) (*abi.Ref, error) {
	if len(refs) == 0 {
		return nil, errors.New("agent: cannot aggregate empty subagent refs")
	}

	var totalLen int64
	maxTaint := abi.TaintTrusted
	minScope := abi.ScopeTenant

	refsCopy := make([]*abi.Ref, 0, len(refs))
	for _, r := range refs {
		if r == nil {
			continue
		}
		refsCopy = append(refsCopy, r)
		rlen := r.Len
		if rlen <= 0 && len(r.Inline) > 0 {
			rlen = int64(len(r.Inline))
		}
		totalLen += rlen

		// Taint escalation: Trusted < Tainted < Quarantined
		if r.Taint > maxTaint {
			maxTaint = r.Taint
		}

		// Scope contraction: ScopeAgent (0) < ScopeFleet (1) < ScopeTenant (2)
		if r.Scope < minScope {
			minScope = r.Scope
		}
	}

	compositeMu.Lock()
	compositeSeq++
	handle := compositeSeq

	var metaCopy map[string]string
	if len(metadata) > 0 {
		metaCopy = make(map[string]string, len(metadata))
		for k, v := range metadata {
			metaCopy[k] = v
		}
	}

	report := &CompositeReport{
		ID:        handle,
		Refs:      refsCopy,
		TotalLen:  totalLen,
		Metadata:  metaCopy,
		CreatedAt: time.Now().UTC(),
	}
	compositeReports[handle] = report
	compositeMu.Unlock()

	digest := computeCompositeDigest(refsCopy)

	return &abi.Ref{
		Kind:   abi.RefRegion,
		Handle: handle,
		Len:    totalLen,
		Taint:  maxTaint,
		Scope:  minScope,
		Digest: digest,
	}, nil
}

// SynthesizeReportRef aggregates multiple subagent artifact references into a single report reference.
func SynthesizeReportRef(refs []*abi.Ref) (*abi.Ref, error) {
	return AggregateSubagentRefs(refs, nil)
}

// ResolveSynthesizedRefs unpacks the underlying slice of subagent *abi.Ref from a synthesized report
// reference with zero memory copies.
func ResolveSynthesizedRefs(ref *abi.Ref) ([]*abi.Ref, bool) {
	if ref == nil || ref.Kind != abi.RefRegion || ref.Handle == 0 {
		return nil, false
	}
	compositeMu.RLock()
	defer compositeMu.RUnlock()
	rep, ok := compositeReports[ref.Handle]
	if !ok {
		return nil, false
	}
	return rep.Refs, true
}

// MaterializeReportBytes materializes the full concatenated content of all subagent references in the synthesized report.
func MaterializeReportBytes(ctx context.Context, ref *abi.Ref) ([]byte, error) {
	if ref == nil {
		return nil, errors.New("agent: cannot materialize nil ref")
	}
	if ref.Kind == abi.RefInline {
		return ref.Inline, nil
	}
	if ref.Kind == abi.RefRegion && ref.Handle != 0 {
		refs, ok := ResolveSynthesizedRefs(ref)
		if !ok {
			return nil, fmt.Errorf("agent: synthesized report handle %d not found", ref.Handle)
		}
		buf := make([]byte, 0, ref.Len)
		for _, r := range refs {
			b := refutil.Bytes(ctx, *r)
			buf = append(buf, b...)
		}
		return buf, nil
	}
	b := refutil.Bytes(ctx, *ref)
	return b, nil
}

// SynthesizeTopic reads all subagent artifact references published to sourceTopic, aggregates them zero-copy
// into a synthesized report reference, and publishes the synthesized report to targetTopic.
func SynthesizeTopic(bb *ctxmmu.Blackboard, sourceTopic, targetTopic string, epoch uint64, metadata map[string]string) (string, *abi.Ref, error) {
	refs, err := ReadSubagentArtifacts(bb, sourceTopic)
	if err != nil {
		return "", nil, err
	}
	if len(refs) == 0 {
		return "", nil, fmt.Errorf("agent: no subagent artifacts found under topic %q", sourceTopic)
	}
	reportRef, err := AggregateSubagentRefs(refs, metadata)
	if err != nil {
		return "", nil, err
	}
	var id string
	if targetTopic != "" {
		id, err = bb.Publish(targetTopic, reportRef, epoch, metadata)
		if err != nil {
			return "", nil, err
		}
	}
	return id, reportRef, nil
}

// CoordinatorSynthesizer coordinates subagent synthesis using an in-process blackboard.
type CoordinatorSynthesizer struct {
	bb *ctxmmu.Blackboard
}

// NewCoordinatorSynthesizer constructs a coordinator synthesizer over the given blackboard.
func NewCoordinatorSynthesizer(bb *ctxmmu.Blackboard) *CoordinatorSynthesizer {
	if bb == nil {
		bb = ctxmmu.NewBlackboard()
	}
	return &CoordinatorSynthesizer{bb: bb}
}

// Blackboard returns the underlying blackboard MMU.
func (c *CoordinatorSynthesizer) Blackboard() *ctxmmu.Blackboard {
	return c.bb
}

// Collect reads subagent artifact references published under topic.
func (c *CoordinatorSynthesizer) Collect(topic string) ([]*abi.Ref, error) {
	return ReadSubagentArtifacts(c.bb, topic)
}

// CollectEntries retrieves full blackboard entries for the topic.
func (c *CoordinatorSynthesizer) CollectEntries(topic string) ([]*ctxmmu.BlackboardEntry, error) {
	return ReadSubagentEntries(c.bb, topic)
}

// Synthesize aggregates subagent artifacts from sourceTopic and publishes the synthesized report to targetTopic.
func (c *CoordinatorSynthesizer) Synthesize(sourceTopic, targetTopic string, epoch uint64, metadata map[string]string) (string, *abi.Ref, error) {
	return SynthesizeTopic(c.bb, sourceTopic, targetTopic, epoch, metadata)
}

// SynthesizeDirect performs zero-copy aggregation directly on an existing slice of subagent references.
func (c *CoordinatorSynthesizer) SynthesizeDirect(refs []*abi.Ref, metadata map[string]string) (*abi.Ref, error) {
	return AggregateSubagentRefs(refs, metadata)
}

func computeCompositeDigest(refs []*abi.Ref) string {
	h := sha256.New()
	for _, r := range refs {
		if r == nil {
			continue
		}
		if r.Digest != "" {
			h.Write([]byte(r.Digest))
		} else if len(r.Inline) > 0 {
			h.Write(r.Inline)
		} else {
			fmt.Fprintf(h, "ref:%d:%d", r.Kind, r.Handle)
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}
