package promptmmu

import "encoding/json"

const (
	SerializationStable        = "stable"
	SerializationChanged       = "changed"
	SerializationInvalidJSON   = "invalid-json"
	SerializationMarshalFailed = "marshal-failed"
)

// ChangedRange is the byte interval where a serializer first diverged from the
// original prompt bytes. OriginalEnd and CandidateEnd are exclusive offsets in
// their respective byte slices.
type ChangedRange struct {
	Start        int `json:"start"`
	OriginalEnd  int `json:"original_end"`
	CandidateEnd int `json:"candidate_end"`
}

// SerializationAudit reports whether a candidate prompt serialization is byte-
// identical to the original and, when it is not, where the byte drift begins.
type SerializationAudit struct {
	Status string       `json:"status"`
	Range  ChangedRange `json:"range,omitempty,omitzero"`
}

// IsZero reports whether the range is absent, for json omitzero.
func (r ChangedRange) IsZero() bool {
	return r.Start == 0 && r.OriginalEnd == 0 && r.CandidateEnd == 0
}

// AuditSerialization compares original prompt bytes with a candidate serialization.
// It never interprets JSON; callers can use it for any wire body whose provider cache
// depends on byte-stable prefixes.
func AuditSerialization(original, candidate []byte) SerializationAudit {
	if string(original) == string(candidate) {
		return SerializationAudit{Status: SerializationStable}
	}
	return SerializationAudit{Status: SerializationChanged, Range: firstChangedRange(original, candidate)}
}

// AuditJSONRemarshal decodes raw JSON into interface{} and re-marshals it with Go's
// canonical encoder, then reports whether that serialization would mutate bytes.
// This is an audit/report helper, not a rewrite path.
func AuditJSONRemarshal(raw []byte) SerializationAudit {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return SerializationAudit{Status: SerializationInvalidJSON}
	}
	candidate, err := json.Marshal(v)
	if err != nil {
		return SerializationAudit{Status: SerializationMarshalFailed}
	}
	return AuditSerialization(raw, candidate)
}

func firstChangedRange(a, b []byte) ChangedRange {
	start := 0
	for start < len(a) && start < len(b) && a[start] == b[start] {
		start++
	}
	aEnd := len(a)
	bEnd := len(b)
	for aEnd > start && bEnd > start && a[aEnd-1] == b[bEnd-1] {
		aEnd--
		bEnd--
	}
	return ChangedRange{Start: start, OriginalEnd: aEnd, CandidateEnd: bEnd}
}
