package model

// v41_activation_trace.go -- the bounded V4.1 activation-checkpoint producer
// (#13325).
//
// The native V4.1 forward exposes named stages and typed layer errors
// (v41_forward.go), and computetrace already owns an opt-in, bounded global
// recorder (Enable/Enabled/Record). What was missing was a producer binding the
// REAL forward to that recorder at a small, fixed set of numerical boundaries, so
// an integration failure can be localized to its first divergent stage without
// another full-model hardware run.
//
// The producer emits at exactly four existing per-layer boundaries:
//
//   - q_latent              qLat after the optional q RMSNorm, before wq_b
//   - kv_latent             kv after the optional kv RMSNorm, before RoPE
//   - attention_projected   attnOut[t] after the grouped output projection
//   - moe_sum               moe after v41SharedExpertAdd, before delta/residual
//   - residual_streams      the post-FFN hc-stream residual at the end of each
//                           layer, flattened [hc*H] per token
//
// It never changes the forward arithmetic: every emission is a read of an
// already-computed slice, and while tracing is disabled the tracer is nil and the
// call sites are skipped entirely (zero added allocations, byte-identical logits).
// Samples are deterministic logical indices into the stage output, capped at
// v41ActivationTraceMaxSamples; the recorder's own per-event/total caps remain the
// hard bound, and a trace that drops samples is reported INCOMPARABLE by
// CompareActivationTraces rather than as agreement.
//
// Provenance: each event carries the pinned model@revision, a config-geometry
// digest, the input token-ID digest, and a manifest (weight-set) digest. The
// reduced fixture carries tensors in the f32 manifest, so the manifest identity is
// the trustworthy weight identity available here; a full checkpoint identity is
// left to the loader's LoadProvenance rather than invented.

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

const (
	// v41ActivationTraceOperation is the Event.Operation for every V4.1 activation
	// checkpoint.
	v41ActivationTraceOperation = "v41.activation"

	// v41ActivationTraceMaxSamples bounds the logical sample points per stage
	// output. It is deliberately small: a checkpoint is a diagnostic sample, never
	// a tensor dump, and Pairing it with the recorder's own caps keeps a trace
	// bounded on a memory-constrained host.
	v41ActivationTraceMaxSamples = 8

	v41TraceStageQLatent     = "q_latent"
	v41TraceStageKVLatent    = "kv_latent"
	v41TraceStageAttnOut     = "attention_projected"
	v41TraceStageMoESum      = "moe_sum"
	v41TraceStageResidual    = "residual_streams"
	v41TraceCaptureBackend   = "host"
	v41TraceCaptureDeviceTag = "host"
)

// v41ActivationTraceState is the forward-scoped producer. A nil *state is the
// disabled tracer: every emit is skipped, so the forward is byte-for-byte
// unchanged and adds no allocations.
type v41ActivationTraceState struct {
	inputDigest  string
	weightDigest string
	recorder     *computetrace.Recorder
	captureID    string
}

// v41ActivationTracer returns the forward's tracer, or nil when tracing is not
// enabled. It is resolved ONCE per forward so an enabled forward pays the digest
// cost once, and a disabled forward pays nothing at all (the digest helpers are
// not reached).
func (m *Model) v41ActivationTracer(ids []int) *v41ActivationTraceState {
	if m == nil || !computetrace.Enabled() {
		return nil
	}
	return &v41ActivationTraceState{
		inputDigest:  m.v41ActivationInputIdentity(ids),
		weightDigest: m.v41ActivationWeightIdentity(),
	}
}

// record emits one bounded checkpoint at a stage boundary. It reports false (and
// records nothing) for a nil tracer or an empty stage output, so an absence is
// visible to the caller rather than a silent empty event.
func (s *v41ActivationTraceState) record(layer, token int, stage string, values []float32) bool {
	if s == nil || len(values) == 0 {
		return false
	}
	width := len(values)
	idx := computetrace.ActivationSampleIndices(width, v41ActivationTraceMaxSamples)
	vals := make([]float32, len(idx))
	for i, j := range idx {
		vals[i] = values[j]
	}
	s.emit(layer, token, stage, width, idx, vals)
	return true
}

// recordStreams emits the residual_streams checkpoint for one token: the hc
// streams read as one flattened [hc*H] row, sampled by the same index rule
// without materializing the flattened copy.
func (s *v41ActivationTraceState) recordStreams(layer, token int, streams [][]float32) bool {
	if s == nil || len(streams) == 0 {
		return false
	}
	H := len(streams[0])
	for _, stream := range streams {
		if len(stream) != H {
			return false
		}
	}
	width := H * len(streams)
	if width == 0 {
		return false
	}
	idx := computetrace.ActivationSampleIndices(width, v41ActivationTraceMaxSamples)
	vals := make([]float32, len(idx))
	for i, j := range idx {
		vals[i] = streams[j/H][j%H]
	}
	s.emit(layer, token, v41TraceStageResidual, width, idx, vals)
	return true
}

// recordLayerStreams emits residual_streams for every token at the end of layer.
// A device-resident panel is read back for the sample only while tracing.
func (s *v41ActivationTraceState) recordLayerStreams(layer int, streams [][][]float32, resident *v41MHCResident) error {
	if s == nil {
		return nil
	}
	if resident != nil {
		host, err := resident.hostStreams(layer)
		if err != nil {
			return err
		}
		streams = host
	}
	for t := range streams {
		s.recordStreams(layer, t, streams[t])
	}
	return nil
}

func (s *v41ActivationTraceState) recordStepStreams(layer, token int, streams [][]float32, resident *v41MHCResident) error {
	if s == nil {
		return nil
	}
	if resident != nil {
		host, err := resident.hostStreams(layer)
		if err != nil {
			return err
		}
		if len(host) != 1 {
			return fmt.Errorf("model: incremental activation readback has %d rows, want one", len(host))
		}
		streams = host[0]
	}
	s.recordStreams(layer, token, streams)
	return nil
}

func (s *v41ActivationTraceState) emit(layer, token int, stage string, width int, idx []int, vals []float32) {
	event := computetrace.Event{
		RunID:            s.captureID,
		RequestID:        s.captureID,
		Operation:        v41ActivationTraceOperation,
		Phase:            stage,
		Backend:          v41TraceCaptureBackend,
		Device:           v41TraceCaptureDeviceTag,
		Layer:            layer,
		Token:            token,
		InputDigest:      s.inputDigest,
		WeightDigest:     s.weightDigest,
		Shapes:           [][]int{{width}},
		SampleIndices:    idx,
		SampleValues:     vals,
		ProvenanceDigest: computetrace.Digest("v41-activation", DeepSeekV41FlashModelID+"@"+DeepSeekV41FlashRevision, stage),
		Status:           "ok",
	}
	if s.recorder != nil {
		s.recorder.Record(event)
		return
	}
	computetrace.Record(event)
}

// v41ActivationInputIdentity binds the pinned source revision, the flat decoder
// geometry, and the input token IDs into one digest, so a comparison refuses an
// apples-to-oranges pair whose input or geometry differs.
func (m *Model) v41ActivationInputIdentity(ids []int) string {
	var b strings.Builder
	b.WriteString("tokens=")
	for i, id := range ids {
		if i != 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(id))
	}
	return computetrace.Digest(
		"v41-input",
		DeepSeekV41FlashModelID+"@"+DeepSeekV41FlashRevision,
		m.v41ActivationConfigIdentity(),
		b.String(),
	)
}

// v41ActivationConfigIdentity is the flat decoder geometry the forward actually
// reads. It is derived from Config, never from the retained metadata pointer, so a
// narrowed fixture and a published config hash differently.
func (m *Model) v41ActivationConfigIdentity() string {
	c := m.Cfg
	return strings.Join([]string{
		"layers=" + strconv.Itoa(c.NumLayers),
		"hidden=" + strconv.Itoa(c.HiddenSize),
		"heads=" + strconv.Itoa(c.NumHeads),
		"head_dim=" + strconv.Itoa(c.HeadDim),
		"experts=" + strconv.Itoa(c.NumExperts),
		"topk=" + strconv.Itoa(c.NumExpertsPerTok),
		"qlora=" + strconv.Itoa(c.QLoraRank),
		"olorank=" + strconv.Itoa(c.OLoraRank),
		"ogroups=" + strconv.Itoa(c.OGroups),
		"moe_inter=" + strconv.Itoa(c.MoEIntermediateSize),
	}, ";")
}

// v41ActivationWeightIdentity is a deterministic digest of the loaded weight set:
// every manifest tensor's canonical name and shape, sorted. It is the trustworthy
// identity available to the reduced fixture, whose tensors live in the f32
// manifest; a full checkpoint's byte identity stays with LoadProvenance.
func (m *Model) v41ActivationWeightIdentity() string {
	names := make([]string, 0, len(m.manifest))
	for name := range m.manifest {
		names = append(names, name)
	}
	sort.Strings(names)
	var b strings.Builder
	b.WriteString("manifest=")
	for i, name := range names {
		if i != 0 {
			b.WriteByte(';')
		}
		b.WriteString(name)
		b.WriteByte(':')
		shape := m.manifest[name].Shape
		for j, d := range shape {
			if j != 0 {
				b.WriteByte('x')
			}
			b.WriteString(strconv.Itoa(d))
		}
	}
	return computetrace.Digest("v41-weights", DeepSeekV41FlashModelID+"@"+DeepSeekV41FlashRevision, b.String())
}
