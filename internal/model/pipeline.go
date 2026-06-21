package model

import (
	"encoding/binary"
	"fmt"
	"math"
)

// pipeline.go — the cross-worker handoff contract for pipeline-parallel serving.
//
// ForwardBand (partition.go) proves a contiguous layer band runs in-process and
// is bit-exact vs the monolithic Forward. This file models the boundary BETWEEN
// workers: each PipelineStage owns a band [Lo,Hi), is loaded STANDALONE via
// WithLayerWindow (so only its weights are resident), and hands the next stage a
// SERIALIZED hidden state. The Marshal->[]byte->Unmarshal round-trip stands in
// for the NCCL/wire send, so the transport backend is a swap underneath this
// contract — the correctness it guarantees does not change when the bytes travel
// a real network instead of a slice.
//
// The correctness gate (pipeline_test.go) is the one ForwardBand cannot give:
// an N-stage pipeline where EACH stage is a SEPARATELY-loaded windowed model
// produces bit-identical logits to one monolithic Forward. That is the real
// cross-device-transport proof (GLM-5.2-NATIVE-ENGINE-GAP first-land #2's
// remaining half), runnable on one box. It holds because Forward/ForwardBand
// read every weight by per-layer NAME (residentKernel + layerName) and key RoPE
// on the ABSOLUTE layer index, so a stage holding only [Lo,Hi) runs the identical
// instruction stream over the identical bytes; and the payload is float32, which
// round-trips bit-exact as IEEE-754 little-endian.
//
// Design decision (DSA state never crosses a boundary): only the hidden state is
// on the wire. Every interior stage boundary must fall on a FULL-indexer layer
// start, so a GLM IndexShare group lives entirely inside one stage and no top-k
// state is ever marshalled. ForwardBand already rejects a band that begins on a
// shared-indexer layer; the partition validator below lifts that same rule to
// plan time, before any worker loads weights.

// StageSpec describes one worker's responsibility: the half-open transformer-layer
// band [Lo,Hi) it owns, plus its role. First means it embeds the token ids (it is
// the head of the pipeline); Last means it runs the final norm + LM head and
// returns logits (it is the tail).
type StageSpec struct {
	Lo, Hi int
	First  bool
	Last   bool
}

// PipelineStage is a runnable worker: a standalone Model plus the band it owns.
// Model is expected to have been loaded with WithLayerWindow(Spec.Lo, Spec.Hi)
// (a full-window model also works — the band selects what runs), so a real
// deployment holds only this stage's weights resident.
type PipelineStage struct {
	Spec  StageSpec
	Model *Model
}

// PartitionPlan is an ordered, validated tiling of [0,NumLayers) into stages.
type PartitionPlan struct {
	NumLayers int
	Stages    []StageSpec
}

// NewPartitionPlan builds and validates a contiguous tiling from interior cut
// points. cuts are the boundaries between bands, e.g. cuts {k} over N layers gives
// stages [0,k) and [k,N); cuts {a,b} gives [0,a),[a,b),[b,N). The first stage is
// marked First (it embeds), the last Last (it runs the head). It fails closed via
// Validate on any gap, overlap, out-of-range, empty band, role error, or — for a
// GLM-MoE-DSA cfg — a cut that lands on a shared-indexer layer.
func NewPartitionPlan(cfg Config, cuts []int) (PartitionPlan, error) {
	n := cfg.NumLayers
	bounds := make([]int, 0, len(cuts)+2)
	bounds = append(bounds, 0)
	bounds = append(bounds, cuts...)
	bounds = append(bounds, n)
	// bounds must be strictly increasing for the bands to be non-empty and ordered;
	// Validate re-checks this, but a clear error here names the offending cut.
	stages := make([]StageSpec, 0, len(bounds)-1)
	for i := 0; i+1 < len(bounds); i++ {
		stages = append(stages, StageSpec{
			Lo:    bounds[i],
			Hi:    bounds[i+1],
			First: i == 0,
			Last:  i+2 == len(bounds),
		})
	}
	p := PartitionPlan{NumLayers: n, Stages: stages}
	if err := p.Validate(cfg); err != nil {
		return PartitionPlan{}, err
	}
	return p, nil
}

// Validate re-checks a plan against cfg and fails closed on any malformed tiling.
// The rules mirror what ForwardBand enforces per-band, surfaced at plan time so a
// bad partition is rejected before any worker loads weights:
//   - the bands tile [0,NumLayers) contiguously: first Lo==0, last Hi==NumLayers,
//     each Lo == previous Hi (no gap, no overlap);
//   - every band is non-empty and in range (0<=Lo, Lo<Hi<=NumLayers);
//   - exactly one First (the first stage) and exactly one Last (the last stage);
//   - for a GLM-MoE-DSA cfg, every interior boundary is a full-indexer layer start
//     (so no IndexShare group is split across the wire).
func (p PartitionPlan) Validate(cfg Config) error {
	if len(p.Stages) == 0 {
		return fmt.Errorf("model: partition plan has no stages")
	}
	if p.NumLayers <= 0 {
		return fmt.Errorf("model: partition plan NumLayers = %d, want > 0", p.NumLayers)
	}
	var firstCount, lastCount int
	for i, s := range p.Stages {
		if s.Lo < 0 || s.Hi <= s.Lo || s.Hi > p.NumLayers {
			return fmt.Errorf("model: stage %d band [%d,%d) invalid for %d layers", i, s.Lo, s.Hi, p.NumLayers)
		}
		if i == 0 {
			if s.Lo != 0 {
				return fmt.Errorf("model: first stage band starts at %d, want 0", s.Lo)
			}
		} else if s.Lo != p.Stages[i-1].Hi {
			return fmt.Errorf("model: stage %d starts at %d but stage %d ends at %d (gap or overlap)", i, s.Lo, i-1, p.Stages[i-1].Hi)
		}
		if s.First {
			firstCount++
			if i != 0 {
				return fmt.Errorf("model: stage %d is marked First but is not the first stage", i)
			}
		}
		if s.Last {
			lastCount++
			if i != len(p.Stages)-1 {
				return fmt.Errorf("model: stage %d is marked Last but is not the last stage", i)
			}
		}
		// A GLM DSA shared-indexer layer reuses the previous full layer's index, so a
		// stage that BEGINS on it never computed that index. Reject the boundary here.
		if i > 0 && cfg.isGLMMoeDsa() && glmDsaIndexerIsShared(cfg, s.Lo) {
			return fmt.Errorf("model: stage %d boundary at GLM shared-indexer layer %d (boundaries must fall on full-indexer layers)", i, s.Lo)
		}
	}
	if p.Stages[len(p.Stages)-1].Hi != p.NumLayers {
		return fmt.Errorf("model: last stage ends at %d, want %d (incomplete coverage)", p.Stages[len(p.Stages)-1].Hi, p.NumLayers)
	}
	if firstCount != 1 {
		return fmt.Errorf("model: partition plan has %d stages marked First, want exactly 1", firstCount)
	}
	if lastCount != 1 {
		return fmt.Errorf("model: partition plan has %d stages marked Last, want exactly 1", lastCount)
	}
	return nil
}

// hiddenHeaderBytes is the fixed little-endian handoff header: seq, hidden, nextLo
// as int32 each.
const hiddenHeaderBytes = 12

// MarshalHidden encodes the per-position hidden state x (seq vectors of length
// hidden) plus the absolute layer index the receiver resumes at, for the wire.
// The layout is [seq:int32][hidden:int32][nextLo:int32] then seq*hidden float32
// words, each via math.Float32bits (the identity bit pattern — preserves signed
// zero and NaN payloads), so the round-trip cannot perturb a single bit.
func MarshalHidden(x [][]float32, nextLo int) ([]byte, error) {
	seq := len(x)
	hidden := 0
	if seq > 0 {
		hidden = len(x[0])
	}
	for t := range x {
		if len(x[t]) != hidden {
			return nil, fmt.Errorf("model: MarshalHidden row %d len = %d, want uniform %d", t, len(x[t]), hidden)
		}
	}
	buf := make([]byte, hiddenHeaderBytes+seq*hidden*4)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(seq))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(hidden))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(nextLo))
	off := hiddenHeaderBytes
	for t := 0; t < seq; t++ {
		for i := 0; i < hidden; i++ {
			binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(x[t][i]))
			off += 4
		}
	}
	return buf, nil
}

// UnmarshalHidden decodes a payload produced by MarshalHidden back to the
// per-position hidden state and the resume layer. It fails closed on a truncated
// or mis-sized buffer rather than running garbage through the next stage.
func UnmarshalHidden(b []byte) ([][]float32, int, error) {
	if len(b) < hiddenHeaderBytes {
		return nil, 0, fmt.Errorf("model: UnmarshalHidden buffer %d bytes, want >= %d header", len(b), hiddenHeaderBytes)
	}
	seq := int(int32(binary.LittleEndian.Uint32(b[0:4])))
	hidden := int(int32(binary.LittleEndian.Uint32(b[4:8])))
	nextLo := int(int32(binary.LittleEndian.Uint32(b[8:12])))
	if seq < 0 || hidden < 0 {
		return nil, 0, fmt.Errorf("model: UnmarshalHidden negative seq=%d hidden=%d", seq, hidden)
	}
	want := hiddenHeaderBytes + seq*hidden*4
	if len(b) != want {
		return nil, 0, fmt.Errorf("model: UnmarshalHidden buffer %d bytes, want %d for seq=%d hidden=%d", len(b), want, seq, hidden)
	}
	x := make([][]float32, seq)
	off := hiddenHeaderBytes
	for t := 0; t < seq; t++ {
		row := make([]float32, hidden)
		for i := 0; i < hidden; i++ {
			row[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[off : off+4]))
			off += 4
		}
		x[t] = row
	}
	return x, nextLo, nil
}

// StageTransport is the worker-to-worker boundary: it carries a stage's output hidden
// state to the next stage and returns what the next stage receives. This is the seam
// the "real NCCL/RPC wire backend" plugs into — RunPipeline / StepPipelineDecode call
// Send at every stage boundary instead of inlining the codec, so swapping single-box
// bytes for a network send is a new StageTransport implementation, not a change to the
// generation loops. The contract (and its bit-exactness gates) is unchanged underneath.
//
// Send must round-trip the hidden state losslessly (so the bit-exact gates hold) and
// preserve nextLo (the absolute layer the receiver resumes at, used as the boundary
// integrity check). dstStage is the receiving stage index, for diagnostics only.
type StageTransport interface {
	Send(hidden [][]float32, nextLo, dstStage int) ([][]float32, int, error)
}

// LocalTransport is the default single-box StageTransport: the MarshalHidden ->
// UnmarshalHidden round-trip that stands in for the wire send. It is bit-exact by
// construction (IEEE-754 round-trip) and is what the in-process pipeline uses. A real
// fleet swaps this for an implementation that ships the bytes over NCCL/gRPC/etc.; the
// bytes on the wire are exactly MarshalHidden's output, so the two are interchangeable.
type LocalTransport struct{}

// Send marshals the hidden state and immediately unmarshals it (the single-box stand-in
// for send-then-receive), validating the boundary the same way the old inline code did.
func (LocalTransport) Send(hidden [][]float32, nextLo, dstStage int) ([][]float32, int, error) {
	payload, err := MarshalHidden(hidden, nextLo)
	if err != nil {
		return nil, 0, fmt.Errorf("model: marshal handoff into stage %d: %w", dstStage, err)
	}
	recv, gotLo, err := UnmarshalHidden(payload)
	if err != nil {
		return nil, 0, fmt.Errorf("model: unmarshal handoff into stage %d: %w", dstStage, err)
	}
	return recv, gotLo, nil
}

// handoff sends hidden across transport to dstStage and verifies the resume layer
// equals wantLo (the receiving band's start) — the boundary integrity check shared by
// both pipeline loops. A nil transport defaults to LocalTransport.
func handoff(transport StageTransport, hidden [][]float32, wantLo, dstStage int) ([][]float32, error) {
	if transport == nil {
		transport = LocalTransport{}
	}
	recv, gotLo, err := transport.Send(hidden, wantLo, dstStage)
	if err != nil {
		return nil, err
	}
	if gotLo != wantLo {
		return nil, fmt.Errorf("model: handoff resume layer %d != stage %d band start %d", gotLo, dstStage, wantLo)
	}
	return recv, nil
}

// RunPipeline drives an N-stage pipeline over ids and returns the last stage's
// per-position logits. It is the single-box end-to-end harness: the first stage
// embeds and runs its band; at each boundary the hidden state is MarshalHidden ->
// UnmarshalHidden (the stand-in for a worker-to-worker send); each subsequent
// stage runs its band; the last stage runs the head. On a real fleet each stage
// call becomes a worker RPC and the []byte is the wire payload.
//
// It fails closed: the stages must form a valid PartitionPlan (re-validated here,
// since a caller can hand-build PipelineStages), and each stage's model config
// must agree on NumLayers so a stage loaded against the wrong checkpoint is
// rejected rather than silently mixing geometries.
//
// RunPipeline uses the single-box LocalTransport; RunPipelineWith takes an explicit
// StageTransport (the seam a real NCCL/RPC wire backend plugs into).
func RunPipeline(ids []int, stages []PipelineStage) ([][]float32, error) {
	return RunPipelineWith(ids, stages, LocalTransport{})
}

// RunPipelineWith is RunPipeline parameterized by the worker-to-worker StageTransport.
// A nil transport defaults to LocalTransport. The generation logic is identical; only
// the boundary send is swappable, so a network transport changes nothing the bit-exact
// gates depend on.
func RunPipelineWith(ids []int, stages []PipelineStage, transport StageTransport) ([][]float32, error) {
	if len(stages) == 0 {
		return nil, fmt.Errorf("model: RunPipeline has no stages")
	}
	cfg := stages[0].Model.Cfg
	specs := make([]StageSpec, len(stages))
	for i, st := range stages {
		if st.Model == nil {
			return nil, fmt.Errorf("model: RunPipeline stage %d has nil model", i)
		}
		if st.Model.Cfg.NumLayers != cfg.NumLayers {
			return nil, fmt.Errorf("model: RunPipeline stage %d NumLayers = %d, stage 0 = %d (mismatched checkpoints)", i, st.Model.Cfg.NumLayers, cfg.NumLayers)
		}
		specs[i] = st.Spec
	}
	plan := PartitionPlan{NumLayers: cfg.NumLayers, Stages: specs}
	if err := plan.Validate(cfg); err != nil {
		return nil, err
	}

	var x [][]float32
	for i, st := range stages {
		if i == 0 {
			x = st.Model.embedBand(ids)
		} else {
			// Receive the previous stage's hidden state across the transport. The resume
			// layer it reports must equal this stage's band start (handoff checks it).
			recv, err := handoff(transport, x, st.Spec.Lo, i)
			if err != nil {
				return nil, err
			}
			x = recv
		}
		hidden, logits, err := st.Model.ForwardBand(x, st.Spec.Lo, st.Spec.Hi, st.Spec.Last)
		if err != nil {
			return nil, fmt.Errorf("model: stage %d ForwardBand [%d,%d): %w", i, st.Spec.Lo, st.Spec.Hi, err)
		}
		if st.Spec.Last {
			return logits, nil
		}
		x = hidden
	}
	return nil, fmt.Errorf("model: RunPipeline reached end without a Last stage")
}

// PipelineGenerate greedily decodes up to n tokens after prompt, routing EVERY
// forward through the partitioned stages via RunPipeline. It is the pipeline-
// parallel analog of Session.Generate (kv.go): with temperature-0 (argmax)
// sampling it is bit-identical to monolithic generation, because RunPipeline's
// logits equal Forward's. It re-forwards the growing sequence each step — O(n²),
// no cross-stage incremental KV decode yet (that is GLM-5.2-NATIVE-ENGINE-GAP
// #4/#5). It is the runnable proof that fak's native engine generates across
// genuinely partitioned stages, not a throughput path.
//
// It stops early on an EOS token (read from the first stage's config, which all
// stages share) and otherwise returns exactly n generated ids. The prompt must be
// non-empty so there is a last-position logit to sample.
func PipelineGenerate(prompt []int, n int, stages []PipelineStage) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("model: PipelineGenerate needs a non-empty prompt")
	}
	if len(stages) == 0 {
		return nil, fmt.Errorf("model: PipelineGenerate has no stages")
	}
	cfg := stages[0].Model.Cfg
	ids := append([]int(nil), prompt...)
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		logits, err := RunPipeline(ids, stages)
		if err != nil {
			return nil, fmt.Errorf("model: PipelineGenerate step %d: %w", i, err)
		}
		if len(logits) == 0 {
			return nil, fmt.Errorf("model: PipelineGenerate step %d produced no logits", i)
		}
		next := argmaxF32(logits[len(logits)-1])
		out = append(out, next)
		if cfg.IsEOS(next) {
			break
		}
		ids = append(ids, next)
	}
	return out, nil
}

// PipelineStageDecoder is a stateful pipeline worker for INCREMENTAL decode: a
// long-lived per-stage Session whose DSA KV cache holds ONLY this stage's band
// [Spec.Lo,Spec.Hi), resident across tokens. It is the incremental analog of
// PipelineStage (used by the O(n²) re-forward RunPipeline): each decode call advances
// one new position, mutating only this stage's band slots. The Session is the unit (not
// a bare cache) so the LAST stage's head is LITERALLY Session.glmDsaHead — the exact
// oracle Session.Generate uses (f32 head, or headQ under quant) — with no second head
// path to drift. Out-of-band cache layer slices stay nil, so resident KV is bounded to
// the band (the point of pipeline parallelism).
type PipelineStageDecoder struct {
	Spec    StageSpec
	Session *Session
}

// NewPipelineStageDecoder builds a stateful decoder over a windowed (or full) model.
// The model is expected to have been loaded WithLayerWindow(Spec.Lo,Spec.Hi). It
// allocates the band's glmDsaKVCache (NumLayers slots; only band layers ever fill).
func NewPipelineStageDecoder(spec StageSpec, m *Model) *PipelineStageDecoder {
	s := m.NewSession()
	s.requireGLMDsaSession()
	return &PipelineStageDecoder{Spec: spec, Session: s}
}

// decode advances this stage's band by one position. First embeds id; later stages
// consume x. Returns the hidden to hand on (non-last) or the post-finalNorm hidden
// (last). pos is the driver-owned absolute position.
func (d *PipelineStageDecoder) decode(id int, x []float32, pos int) ([]float32, error) {
	return d.Session.decodeBandGLMDsa(id, x, d.Spec.Lo, d.Spec.Hi, pos, d.Spec.First, d.Spec.Last)
}

// validateDecoderPlan re-checks a hand-built decoder set the same way RunPipeline checks
// PipelineStages: nil-session/model rejected, every stage's NumLayers must agree (a stage
// loaded against the wrong checkpoint is rejected, not silently mixed), and the specs must
// form a valid PartitionPlan (so a band starting on a shared-indexer layer is refused at
// plan time, before any decode). Returns the shared cfg.
func validateDecoderPlan(stages []*PipelineStageDecoder) (Config, error) {
	if len(stages) == 0 {
		return Config{}, fmt.Errorf("model: pipeline decode has no stages")
	}
	if stages[0] == nil || stages[0].Session == nil || stages[0].Session.M == nil {
		return Config{}, fmt.Errorf("model: pipeline decode stage 0 is nil")
	}
	cfg := stages[0].Session.M.Cfg
	specs := make([]StageSpec, len(stages))
	for i, st := range stages {
		if st == nil || st.Session == nil || st.Session.M == nil {
			return Config{}, fmt.Errorf("model: pipeline decode stage %d is nil", i)
		}
		if st.Session.M.Cfg.NumLayers != cfg.NumLayers {
			return Config{}, fmt.Errorf("model: pipeline decode stage %d NumLayers = %d, stage 0 = %d (mismatched checkpoints)", i, st.Session.M.Cfg.NumLayers, cfg.NumLayers)
		}
		specs[i] = st.Spec
	}
	plan := PartitionPlan{NumLayers: cfg.NumLayers, Stages: specs}
	if err := plan.Validate(cfg); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// ResetStageDecoders restores every stage to a fresh, position-zero KV cache. A
// PipelineStageDecoder carries decode state across calls; reusing a decoded-on set for a
// new sequence without resetting would silently CONCATENATE the two sequences. Call this
// before decoding a new prompt on stages that have already been used.
func ResetStageDecoders(stages []*PipelineStageDecoder) {
	for _, st := range stages {
		if st == nil || st.Session == nil || st.Session.M == nil {
			continue
		}
		st.Session.Cache = st.Session.M.NewSession().Cache
		st.Session.requireGLMDsaSession()
		st.Session.glmDsaSharedTopK = nil
	}
}

// assertBandAtPos fails closed if stage st's band has not run exactly pos times (its
// first band layer should hold pos K rows before decoding position pos). It reads the
// stage's own band cache rather than Cache.Len(), since interior stages never advance pos.
func assertBandAtPos(st *PipelineStageDecoder, pos int) error {
	c := st.Session.Cache.glm
	stride := glmDsaAttentionKStride(st.Session.M.Cfg)
	if stride <= 0 {
		return fmt.Errorf("model: bad GLM DSA K stride %d", stride)
	}
	have := len(c.K[st.Spec.Lo]) / stride
	if have != pos {
		return fmt.Errorf("band at position %d, want %d (cache drift)", have, pos)
	}
	return nil
}

// StepPipelineDecode advances every stage by ONE position for token id at absolute
// position pos, threading the single-position hidden across the stage transport, and
// returns the last stage's logits via that stage's own Session.glmDsaHead. Each stage
// mutates only its band's resident KV — real per-token incremental decode, not
// RunPipeline's re-forward. A stage that has drifted out of position lockstep is rejected
// fail-closed before its DSA append would mis-assert the (pos+1)-th row.
//
// StepPipelineDecode uses the single-box LocalTransport; StepPipelineDecodeWith takes an
// explicit StageTransport (the NCCL/RPC wire seam).
func StepPipelineDecode(id, pos int, stages []*PipelineStageDecoder) ([]float32, error) {
	return StepPipelineDecodeWith(id, pos, stages, LocalTransport{})
}

// StepPipelineDecodeWith is StepPipelineDecode parameterized by the StageTransport. A nil
// transport defaults to LocalTransport. Only the boundary send is swappable.
func StepPipelineDecodeWith(id, pos int, stages []*PipelineStageDecoder, transport StageTransport) ([]float32, error) {
	var x []float32
	for i, st := range stages {
		if err := assertBandAtPos(st, pos); err != nil {
			return nil, fmt.Errorf("model: StepPipelineDecode stage %d: %w", i, err)
		}
		var xrow []float32
		if !st.Spec.First {
			recv, err := handoff(transport, [][]float32{x}, st.Spec.Lo, i)
			if err != nil {
				return nil, err
			}
			xrow = recv[0]
		}
		h, err := st.decode(id, xrow, pos)
		if err != nil {
			return nil, err
		}
		if st.Spec.Last {
			return st.Session.glmDsaHead(h), nil
		}
		x = h
	}
	return nil, fmt.Errorf("model: StepPipelineDecode reached end without a Last stage")
}

// PipelineGenerateIncremental greedily decodes up to n tokens, keeping each stage's band
// KV resident and advancing ONE position per stage per token — true incremental decode,
// vs PipelineGenerate's O(n²) stateless re-forward (GLM-5.2-NATIVE-ENGINE-GAP #4/#5). At
// temperature 0 it is bit-identical to Session.Generate: each stage runs the identical
// per-position DSA instruction stream the monolith runs for layers [Lo,Hi) against an
// identically-grown band cache, the hidden handoff is the bit-exact codec, and the last
// stage's head IS Session.glmDsaHead. It resets the stages first (so reuse is safe), then
// prefills the prompt in lockstep at pos 0..len-1 and decodes from pos len.
func PipelineGenerateIncremental(prompt []int, n int, stages []*PipelineStageDecoder) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("model: PipelineGenerateIncremental needs a non-empty prompt")
	}
	cfg, err := validateDecoderPlan(stages)
	if err != nil {
		return nil, err
	}
	ResetStageDecoders(stages)
	var logits []float32
	for pos, id := range prompt {
		l, err := StepPipelineDecode(id, pos, stages)
		if err != nil {
			return nil, fmt.Errorf("model: PipelineGenerateIncremental prefill pos %d: %w", pos, err)
		}
		logits = l
	}
	out := make([]int, 0, n)
	pos := len(prompt)
	for i := 0; i < n; i++ {
		next := argmaxF32(logits)
		out = append(out, next)
		if cfg.IsEOS(next) {
			break
		}
		l, err := StepPipelineDecode(next, pos, stages)
		if err != nil {
			return nil, fmt.Errorf("model: PipelineGenerateIncremental decode %d: %w", i, err)
		}
		logits = l
		pos++
	}
	return out, nil
}
