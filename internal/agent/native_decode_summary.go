package agent

// Native decode paths, mirroring enginestep's closed path vocabulary so a
// per-request row and the process-wide fak_engine_* families name the same thing.
const (
	NativeDecodePathSerial      = "serial"
	NativeDecodePathBatched     = "batched"
	NativeDecodePathSpeculative = "speculative"
)

// SpeculativeDecodeTally is one request's verified draft-verify rounds: how many
// rounds ran, how many draft tokens they proposed, and how many the target accepted.
type SpeculativeDecodeTally struct {
	Rounds         int `json:"rounds"`
	DraftTokens    int `json:"draft_tokens"`
	AcceptedTokens int `json:"accepted_tokens"`
}

// NativeDecodeSummary is the request-local view of the native engine's decode
// cycle. It is built from the request's own generate result, never from the
// shared enginestep recorder, so concurrent requests cannot bleed into each other.
type NativeDecodeSummary struct {
	// Path is the dominant decode path: speculative when any verify round ran,
	// batched when the request decoded inside a multi-lane cohort, else serial.
	Path string `json:"path"`
	// CohortSize is the coalesced cohort this request decoded in (0 when it
	// decoded alone).
	CohortSize int `json:"cohort_size,omitempty"`
	// Speculative is set only when at least one verify round ran.
	Speculative *SpeculativeDecodeTally `json:"speculative,omitempty"`
}

func newNativeDecodeSummary(res inKernelGenerateResult) *NativeDecodeSummary {
	sum := &NativeDecodeSummary{Path: NativeDecodePathSerial}
	if res.batchReceipt.CohortID != 0 {
		sum.CohortSize = res.batchReceipt.CohortSize
		if sum.CohortSize > 1 {
			sum.Path = NativeDecodePathBatched
		}
	}
	if res.spec.Rounds > 0 {
		spec := res.spec
		sum.Speculative = &spec
		sum.Path = NativeDecodePathSpeculative
	}
	return sum
}
