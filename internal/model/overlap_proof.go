package model

import "fmt"

// OverlapOperation is a measured copy, collective, or compute interval with explicit dependencies.
type OverlapOperation struct {
	ID        string   `json:"id"`
	Kind      string   `json:"kind"`
	StartNS   int64    `json:"start_ns"`
	EndNS     int64    `json:"end_ns"`
	Bytes     int64    `json:"bytes"`
	DependsOn []string `json:"depends_on,omitempty"`
}

type OverlapReceipt struct {
	Schema                  string `json:"schema"`
	Engine                  string `json:"engine"`
	AcceptedTokens          int    `json:"accepted_tokens"`
	SerialNanoseconds       int64  `json:"serial_nanoseconds"`
	CriticalPathNanoseconds int64  `json:"critical_path_nanoseconds"`
	OverlapNanoseconds      int64  `json:"overlap_nanoseconds"`
	BytesOverlapped         int64  `json:"bytes_overlapped"`
	DependencyProof         string `json:"dependency_proof"`
	StopRule                string `json:"stop_rule"`
	Rollback                string `json:"rollback"`
}

// ProveOperationOverlap validates interval ordering and every declared dependency,
// then reports only witnessed overlap. Missing dependencies fail closed.
func ProveOperationOverlap(ops []OverlapOperation, acceptedTokens int) (OverlapReceipt, error) {
	if acceptedTokens < 0 {
		return OverlapReceipt{}, fmt.Errorf("model: negative accepted tokens")
	}
	r := OverlapReceipt{Schema: "fak-overlap-receipt/1", Engine: "fak-native", AcceptedTokens: acceptedTokens, DependencyProof: "all declared predecessors complete before dependent start", StopRule: "reject dependency violation or nonpositive overlap", Rollback: "serialize copies, collectives, and compute"}
	ends := map[string]int64{}
	minStart, maxEnd := int64(0), int64(0)
	for i, op := range ops {
		if op.ID == "" || op.StartNS < 0 || op.EndNS < op.StartNS || op.Bytes < 0 {
			return OverlapReceipt{}, fmt.Errorf("model: invalid operation")
		}
		if _, ok := ends[op.ID]; ok {
			return OverlapReceipt{}, fmt.Errorf("model: duplicate operation %q", op.ID)
		}
		for _, dep := range op.DependsOn {
			end, ok := ends[dep]
			if !ok || end > op.StartNS {
				return OverlapReceipt{}, fmt.Errorf("model: dependency %q not complete for %q", dep, op.ID)
			}
		}
		ends[op.ID] = op.EndNS
		r.SerialNanoseconds += op.EndNS - op.StartNS
		if i == 0 || op.StartNS < minStart {
			minStart = op.StartNS
		}
		if op.EndNS > maxEnd {
			maxEnd = op.EndNS
		}
	}
	if len(ops) > 0 {
		r.CriticalPathNanoseconds = maxEnd - minStart
	}
	r.OverlapNanoseconds = r.SerialNanoseconds - r.CriticalPathNanoseconds
	if r.OverlapNanoseconds < 0 {
		r.OverlapNanoseconds = 0
	}
	for i, a := range ops {
		for _, b := range ops[i+1:] {
			if a.StartNS < b.EndNS && b.StartNS < a.EndNS {
				if a.Bytes > 0 {
					r.BytesOverlapped += a.Bytes
				}
				if b.Bytes > 0 {
					r.BytesOverlapped += b.Bytes
				}
				break
			}
		}
	}
	return r, nil
}
