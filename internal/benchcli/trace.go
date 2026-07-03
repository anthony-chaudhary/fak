package benchcli

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"sort"
)

// TraceCall is the common workload prefix shared by benchmark trace schemas.
type TraceCall struct {
	Tool string            `json:"tool"`
	Args json.RawMessage   `json:"args"`
	Meta map[string]string `json:"meta,omitempty"`
}

// Trace is the common slice envelope used by benchmark trace schemas.
type Trace[C any] struct {
	SliceID string `json:"slice_id"`
	Calls   []C    `json:"calls"`
}

// LoadJSONFile decodes a JSON file into a benchmark trace/report shape.
func LoadJSONFile[T any](path string) (*T, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var t T
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, err
	}
	return &t, nil
}

// WorkloadHash returns the stable hash used by trace A/B identical-workload guards.
func WorkloadHash(sliceID string, calls []TraceCall) string {
	return WorkloadHashBy(sliceID, calls, func(c TraceCall) TraceCall { return c })
}

// WorkloadHashBy hashes a benchmark trace through a caller-supplied schema adapter.
func WorkloadHashBy[C any](sliceID string, calls []C, asTraceCall func(C) TraceCall) string {
	h := sha256.New()
	h.Write([]byte(sliceID))
	for _, call := range calls {
		c := asTraceCall(call)
		h.Write([]byte(c.Tool))
		h.Write([]byte{0})
		h.Write(c.Args)
		h.Write([]byte{0})
		keys := make([]string, 0, len(c.Meta))
		for k := range c.Meta {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			h.Write([]byte(k + "=" + c.Meta[k] + ";"))
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}
