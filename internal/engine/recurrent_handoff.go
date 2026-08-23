package engine

import "errors"

// RecurrentStateCheckpoint is an immutable snapshot at a shared-prefix boundary.
// The opaque state includes every recurrent tensor and short-convolution window
// needed to continue the prefix without replaying it.
type RecurrentStateCheckpoint struct {
	modelID   string
	prefixKey string
	tokens    uint64
	state     []byte
}

// NewRecurrentStateCheckpoint captures recurrent state for a prefix boundary.
// Input bytes are copied so later backend writes cannot mutate the checkpoint.
func NewRecurrentStateCheckpoint(modelID, prefixKey string, tokens uint64, state []byte) (RecurrentStateCheckpoint, error) {
	if modelID == "" || prefixKey == "" || tokens == 0 || len(state) == 0 {
		return RecurrentStateCheckpoint{}, errors.New("recurrent checkpoint requires model, prefix, boundary, and state")
	}
	return RecurrentStateCheckpoint{
		modelID:   modelID,
		prefixKey: prefixKey,
		tokens:    tokens,
		state:     append([]byte(nil), state...),
	}, nil
}

// RecurrentHandoffRequest binds recurrent-state reuse to the same prefix clone
// admitted for the full-attention layers of a hybrid request.
type RecurrentHandoffRequest struct {
	ModelID      string
	PrefixKey    string
	PrefixTokens uint64
	PrefixClone  CacheCapability
	Checkpoint   RecurrentStateCheckpoint
}

// RecurrentHandoff is a request-owned fork. State returns a fresh copy because
// recurrent decode mutates its accumulator in place.
type RecurrentHandoff struct {
	modelID   string
	prefixKey string
	tokens    uint64
	state     []byte
}

func (h RecurrentHandoff) ModelID() string      { return h.modelID }
func (h RecurrentHandoff) PrefixKey() string    { return h.prefixKey }
func (h RecurrentHandoff) PrefixTokens() uint64 { return h.tokens }
func (h RecurrentHandoff) State() []byte        { return append([]byte(nil), h.state...) }

// RecurrentHandoffRefusal is the closed fail-closed result vocabulary.
type RecurrentHandoffRefusal string

const (
	RecurrentHandoffAdmitted           RecurrentHandoffRefusal = ""
	RecurrentHandoffPrefixRefused      RecurrentHandoffRefusal = "prefix-clone-refused"
	RecurrentHandoffRequestMalformed   RecurrentHandoffRefusal = "request-malformed"
	RecurrentHandoffCheckpointMismatch RecurrentHandoffRefusal = "checkpoint-mismatch"
)

func (r RecurrentHandoffRefusal) Refused() bool { return r != RecurrentHandoffAdmitted }

// AdmitRecurrentHandoff admits KV prefix cloning and recurrent-state handoff as
// one operation. Hybrid reuse is all-or-cold: no recurrent state is exposed when
// the measured, cold-path-witnessed KV prefix clone is refused or identities differ.
func AdmitRecurrentHandoff(req RecurrentHandoffRequest) (RecurrentHandoff, RecurrentHandoffRefusal) {
	if AdmitActiveCache(req.PrefixClone, CachePrefixClone).Refused() {
		return RecurrentHandoff{}, RecurrentHandoffPrefixRefused
	}
	if req.ModelID == "" || req.PrefixKey == "" || req.PrefixTokens == 0 {
		return RecurrentHandoff{}, RecurrentHandoffRequestMalformed
	}
	cp := req.Checkpoint
	if cp.modelID == "" || cp.prefixKey == "" || cp.tokens == 0 || len(cp.state) == 0 ||
		cp.modelID != req.ModelID || cp.prefixKey != req.PrefixKey || cp.tokens != req.PrefixTokens {
		return RecurrentHandoff{}, RecurrentHandoffCheckpointMismatch
	}
	return RecurrentHandoff{
		modelID: cp.modelID, prefixKey: cp.prefixKey, tokens: cp.tokens,
		state: append([]byte(nil), cp.state...),
	}, RecurrentHandoffAdmitted
}
