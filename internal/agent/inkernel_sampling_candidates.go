package agent

import (
	"math"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// compactSamplingState belongs to one serial decode lane. The initial prompt
// distribution remains full. This opt-in changes transport, never the RNG or
// sampler implementation, and is not performance-qualified by source presence.
type compactSamplingState struct {
	packet compute.SamplingCandidatePacket
	ready  bool
}

func (ln *decodeLane) deviceSamplingCandidates() bool {
	return os.Getenv("FAK_CUDA_SAMPLING_COMPACT") == "1" &&
		ln.s != nil && ln.engramAttempt == nil && ln.structured == nil &&
		ln.temp > 0 && !math.IsInf(ln.temp, 0) &&
		ln.topK >= 1 && ln.topK <= compute.SamplingCandidateMaxK && ln.topK < ln.vocabSize() &&
		len(ln.logitBias) == 0 && ln.freqPenalty == 0 && ln.presPenalty == 0 &&
		(ln.measurement == nil || (ln.measurement.inferenceDisabled && ln.measurement.topLogprobsK == 0))
}

func (ln *decodeLane) stepSamplingCandidates(id int) {
	ln.vocab = ln.vocabSize()
	ln.greedyReady = false
	ln.logits, ln.compact.ready = ln.s.StepSamplingCandidates(id, ln.temp, ln.topK, ln.forwarded != nil, &ln.compact.packet)
	if ln.compact.ready {
		ln.vocab = ln.compact.packet.Vocab
		ln.logits = ln.compact.packet.Raw[:ln.compact.packet.Count]
	}
}
