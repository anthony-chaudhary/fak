package compute

import (
	"fmt"
	"math"
)

// OnlineSoftmaxSinkState maintains streaming online softmax statistics (Milakov-Gimelshein / FlashAttention)
// with optional attention sink folding.
type OnlineSoftmaxSinkState struct {
	MaxVal      float32 `json:"max_val"`
	Denominator float32 `json:"denominator"`
	HasSink     bool    `json:"has_sink"`
	SinkLogit   float32 `json:"sink_logit"`
	Count       int     `json:"count"`
}

// NewOnlineSoftmaxSink constructs a new online softmax accumulator. If hasSink is true,
// the sink logit participates in the running max and denominator.
func NewOnlineSoftmaxSink(hasSink bool, sinkLogit float32) OnlineSoftmaxSinkState {
	if hasSink {
		return OnlineSoftmaxSinkState{
			MaxVal:      sinkLogit,
			Denominator: 1.0,
			HasSink:     true,
			SinkLogit:   sinkLogit,
			Count:       0,
		}
	}
	return OnlineSoftmaxSinkState{
		MaxVal:      float32(math.Inf(-1)),
		Denominator: 0.0,
		HasSink:     false,
		SinkLogit:   0.0,
		Count:       0,
	}
}

// Update incorporates a single logit into the running state.
func (s *OnlineSoftmaxSinkState) Update(logit float32) {
	s.Count++
	if logit > s.MaxVal {
		s.Denominator = s.Denominator*float32(math.Exp(float64(s.MaxVal-logit))) + 1.0
		s.MaxVal = logit
	} else {
		s.Denominator += float32(math.Exp(float64(logit - s.MaxVal)))
	}
}

// UpdateSlice incorporates a batch of logits into the running state.
func (s *OnlineSoftmaxSinkState) UpdateSlice(logits []float32) {
	for _, l := range logits {
		s.Update(l)
	}
}

// Normalize normalizes logits in-place using the current state.
func (s *OnlineSoftmaxSinkState) Normalize(logits []float32) {
	if s.Denominator <= 0 || math.IsNaN(float64(s.Denominator)) || math.IsInf(float64(s.Denominator), 0) {
		return
	}
	inv := float32(1.0 / float64(s.Denominator))
	for i, l := range logits {
		logits[i] = float32(math.Exp(float64(l-s.MaxVal))) * inv
	}
}

// OnlineSoftmaxSinkReceipt records metrics of the sink-aware online softmax computation.
type OnlineSoftmaxSinkReceipt struct {
	Count            int     `json:"count"`
	HasSink          bool    `json:"has_sink"`
	SinkLogit        float32 `json:"sink_logit,omitempty"`
	MaxVal           float32 `json:"max_val"`
	Denominator      float32 `json:"denominator"`
	SinkProbability  float32 `json:"sink_probability,omitempty"`
	SumProbabilities float32 `json:"sum_probabilities"`
}

// OnlineSoftmaxSinkInPlace executes online softmax with optional sink in-place over logits.
func OnlineSoftmaxSinkInPlace(logits []float32, hasSink bool, sinkLogit float32) (OnlineSoftmaxSinkReceipt, error) {
	var receipt OnlineSoftmaxSinkReceipt
	if len(logits) == 0 {
		return receipt, fmt.Errorf("logits slice must not be empty")
	}

	state := NewOnlineSoftmaxSink(hasSink, sinkLogit)
	state.UpdateSlice(logits)

	if math.IsNaN(float64(state.Denominator)) || math.IsInf(float64(state.Denominator), 0) {
		return receipt, fmt.Errorf("non-finite denominator encountered in softmax: %v", state.Denominator)
	}

	state.Normalize(logits)

	var sumProb float32
	for _, p := range logits {
		sumProb += p
	}

	var sinkProb float32
	if hasSink && state.Denominator > 0 {
		sinkProb = float32(math.Exp(float64(sinkLogit-state.MaxVal))) / state.Denominator
	}

	receipt = OnlineSoftmaxSinkReceipt{
		Count:            len(logits),
		HasSink:          hasSink,
		SinkLogit:        sinkLogit,
		MaxVal:           state.MaxVal,
		Denominator:      state.Denominator,
		SinkProbability:  sinkProb,
		SumProbabilities: sumProb,
	}

	return receipt, nil
}
