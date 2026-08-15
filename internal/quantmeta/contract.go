// Package quantmeta is fak's neutral quantization capability descriptor (#6222,
// child of the broad-interoperability epic #6221).
//
// What it is: a typed, versioned vocabulary for SAYING what a quantized model
// artifact is -- the weight and activation formats, the KV-cache treatment, the
// scale and zero-point layout, the grouping, any codebook, any sparsity, and the
// training provenance -- plus an adjudicator that returns an EXPLICIT typed
// result when it cannot describe something.
//
// What it is deliberately NOT:
//
//   - It is not a ranking. No method, format or runtime is scored, preferred or
//     blessed here. #6221's standing guardrail is that fak selects no universal
//     quantization winner, so Adjudicate answers only "can this be described
//     without guessing?", never "is this good?".
//   - It is not a new artifact format. fak does not ask anyone to convert. This
//     package describes artifacts that already exist in the ecosystem's own
//     formats, and a descriptor that passes through comes out unchanged.
//   - It is not a claim of support. A descriptor being parseable says nothing
//     about whether any runtime can execute it; that is the runtime's answer,
//     which is why OutcomeDelegate exists.
//
// The losslessness fence. Every spec here preserves the fields it does not
// recognize (the Extra maps) and re-emits them unchanged. This is what keeps fak
// from quietly becoming the authority on which fields are legal: a producer's
// private key survives a round trip intact, so passing a descriptor through fak
// is never lossy and never editorial. The known-key set of every spec is derived
// by REFLECTION over its own json tags rather than a hand-maintained list,
// because a hand-maintained list silently drifts out of sync with the struct and
// turns unknown-field preservation into unknown-field deletion.
package quantmeta

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
)

// SchemaVersion is the only descriptor schema this package claims to
// understand. A descriptor carrying any other value abstains -- it is never
// silently read as if it were this version.
const SchemaVersion = "quantmeta/v1"

// Format is a numeric/encoding format identifier. The values below are names the
// ecosystem already uses; they are an OPEN vocabulary listed so descriptors
// agree on spelling, and they carry no ordering. A Format outside this set is
// unknown, not invalid.
type Format string

// Granularity is the axis along which a quantization parameter varies.
type Granularity string

// TrainingStage records WHEN quantization entered the model's life. It is
// provenance, not quality: "trained-natively" is not better than
// "post-training", it is a different thing to say about the artifact.
type TrainingStage string

const (
	FormatF32     Format = "f32"
	FormatF16     Format = "f16"
	FormatBF16    Format = "bf16"
	FormatFP8E4M3 Format = "fp8-e4m3"
	FormatFP8E5M2 Format = "fp8-e5m2"
	FormatFP4E2M1 Format = "fp4-e2m1"
	FormatMXFP4   Format = "mxfp4"
	FormatMXFP6   Format = "mxfp6"
	FormatMXFP8   Format = "mxfp8"
	FormatNF4     Format = "nf4"
	FormatInt8    Format = "int8"
	FormatInt6    Format = "int6"
	FormatInt4    Format = "int4"
	FormatInt3    Format = "int3"
	FormatInt2    Format = "int2"
	FormatTernary Format = "ternary"
	FormatBinary  Format = "binary"
	// FormatCodebook marks a non-uniform / learned mapping whose meaning lives
	// in the codebook, not in the format name.
	FormatCodebook Format = "codebook"

	GranularityPerTensor  Granularity = "per-tensor"
	GranularityPerChannel Granularity = "per-channel"
	GranularityPerToken   Granularity = "per-token"
	GranularityPerGroup   Granularity = "per-group"
	GranularityPerBlock   Granularity = "per-block"
	GranularityPerHead    Granularity = "per-head"

	TrainingStagePostTraining      TrainingStage = "post-training"
	TrainingStageQuantizationAware TrainingStage = "quantization-aware"
	TrainingStageTrainedNatively   TrainingStage = "trained-natively"
	TrainingStageUnknown           TrainingStage = "unknown"
)

// Known reports whether f is a format identifier this package has a spelling
// for. An unknown format is a reason to ABSTAIN, never to refuse: the artifact
// may be entirely valid and merely newer than this build.
func (f Format) Known() bool { return knownFormats[f] }

var knownFormats = map[Format]bool{
	FormatF32: true, FormatF16: true, FormatBF16: true,
	FormatFP8E4M3: true, FormatFP8E5M2: true, FormatFP4E2M1: true,
	FormatMXFP4: true, FormatMXFP6: true, FormatMXFP8: true, FormatNF4: true,
	FormatInt8: true, FormatInt6: true, FormatInt4: true, FormatInt3: true,
	FormatInt2: true, FormatTernary: true, FormatBinary: true,
	FormatCodebook: true,
}

// ArtifactSpec names the container the weights ship in. It is provenance about
// packaging only -- fak neither prefers a container nor asks for conversion.
type ArtifactSpec struct {
	ContainerID      string                     `json:"container_id,omitempty"`
	ContainerVersion string                     `json:"container_version,omitempty"`
	Extra            map[string]json.RawMessage `json:"-"`
}

// ScaleSpec describes the scale factors: their own numeric format and the axis
// they vary along.
type ScaleSpec struct {
	Format      Format                     `json:"format,omitempty"`
	Granularity Granularity                `json:"granularity,omitempty"`
	GroupSize   int                        `json:"group_size,omitempty"`
	BlockSize   int                        `json:"block_size,omitempty"`
	Extra       map[string]json.RawMessage `json:"-"`
}

// ZeroPointSpec describes the asymmetric offset. Present is not omitempty so
// that "there is no zero point" stays distinct from "nobody said".
type ZeroPointSpec struct {
	Present     bool                       `json:"present"`
	Format      Format                     `json:"format,omitempty"`
	Granularity Granularity                `json:"granularity,omitempty"`
	Extra       map[string]json.RawMessage `json:"-"`
}

// CodebookSpec describes a non-uniform / learned mapping. Its realization is
// owned by whoever produced it, so a descriptor carrying one delegates.
type CodebookSpec struct {
	Kind           string                     `json:"kind,omitempty"`
	Entries        int                        `json:"entries,omitempty"`
	VectorDim      int                        `json:"vector_dim,omitempty"`
	IndexBits      int                        `json:"index_bits,omitempty"`
	ResidualStages int                        `json:"residual_stages,omitempty"`
	Extra          map[string]json.RawMessage `json:"-"`
}

// TensorSpec is the descriptor for one quantized tensor role: weights,
// activations, or one half of the KV cache.
type TensorSpec struct {
	Format      Format                     `json:"format,omitempty"`
	Bits        float64                    `json:"bits_per_value,omitempty"`
	Granularity Granularity                `json:"granularity,omitempty"`
	GroupSize   int                        `json:"group_size,omitempty"`
	Symmetric   *bool                      `json:"symmetric,omitempty"`
	Scale       *ScaleSpec                 `json:"scale,omitempty"`
	ZeroPoint   *ZeroPointSpec             `json:"zero_point,omitempty"`
	Codebook    *CodebookSpec              `json:"codebook,omitempty"`
	Rotation    string                     `json:"rotation,omitempty"`
	Extra       map[string]json.RawMessage `json:"-"`
}

// Weight and Activation are role aliases; both are ordinary tensor descriptors.
type Weight = TensorSpec
type Activation = TensorSpec

// KVSpec describes KV-cache quantization, including the parts deliberately kept
// at full precision (attention sinks and a recent window), which is where most
// long-context KV methods differ from one another.
type KVSpec struct {
	Key                       *TensorSpec                `json:"key,omitempty"`
	Value                     *TensorSpec                `json:"value,omitempty"`
	SinkTokensFullPrecision   int                        `json:"sink_tokens_full_precision,omitempty"`
	RecentWindowFullPrecision int                        `json:"recent_window_full_precision,omitempty"`
	Extra                     map[string]json.RawMessage `json:"-"`
}

// SparsitySpec describes structured or unstructured sparsity applied alongside
// quantization.
type SparsitySpec struct {
	Kind    string                     `json:"kind,omitempty"`
	Pattern string                     `json:"pattern,omitempty"`
	N       int                        `json:"n,omitempty"`
	M       int                        `json:"m,omitempty"`
	Scope   string                     `json:"scope,omitempty"`
	Ratio   float64                    `json:"ratio,omitempty"`
	Extra   map[string]json.RawMessage `json:"-"`
}

// ProvenanceSpec records who produced the artifact and how. MethodID is an
// opaque identifier for citation and routing -- it is never compared, ordered or
// scored anywhere in this package.
type ProvenanceSpec struct {
	MethodID      string        `json:"method_id,omitempty"`
	MethodVersion string        `json:"method_version,omitempty"`
	TrainingStage TrainingStage `json:"training_stage,omitempty"`
	// CalibrationDisclosed is a pointer so an explicit false (the producer says
	// the calibration set is undisclosed) survives distinct from absence.
	CalibrationDisclosed *bool                      `json:"calibration_disclosed,omitempty"`
	ProducerID           string                     `json:"producer_id,omitempty"`
	ProducerVersion      string                     `json:"producer_version,omitempty"`
	SourceModel          string                     `json:"source_model,omitempty"`
	Extra                map[string]json.RawMessage `json:"-"`
}

// MeasuredEnvelope is the ONLY thing that licenses a measured hardware claim: a
// real device, a real runtime, and when it was measured. Adjudicate withholds
// ClaimMeasuredEnvelope unless all three are present, so a performance claim can
// never be inferred from a descriptor alone.
type MeasuredEnvelope struct {
	DeviceID       string                     `json:"device_id,omitempty"`
	RuntimeID      string                     `json:"runtime_id,omitempty"`
	RuntimeVersion string                     `json:"runtime_version,omitempty"`
	MeasuredOn     string                     `json:"measured_on,omitempty"`
	Extra          map[string]json.RawMessage `json:"-"`
}

// Descriptor is the top-level neutral quantization capability descriptor.
type Descriptor struct {
	Schema     string                     `json:"schema"`
	Artifact   *ArtifactSpec              `json:"artifact,omitempty"`
	Weight     *Weight                    `json:"weight,omitempty"`
	Activation *Activation                `json:"activation,omitempty"`
	KV         *KVSpec                    `json:"kv,omitempty"`
	Sparsity   *SparsitySpec              `json:"sparsity,omitempty"`
	Provenance ProvenanceSpec             `json:"provenance"`
	Envelope   *MeasuredEnvelope          `json:"envelope,omitempty"`
	Extra      map[string]json.RawMessage `json:"-"`
}

// Parse reads a descriptor, preserving every field it does not recognize.
func Parse(data []byte) (Descriptor, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	var probe json.RawMessage
	if err := dec.Decode(&probe); err != nil {
		return Descriptor{}, fmt.Errorf("quantmeta: %w", err)
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(probe, &obj); err != nil {
		return Descriptor{}, fmt.Errorf("quantmeta: descriptor must be a JSON object: %w", err)
	}
	if obj == nil {
		return Descriptor{}, fmt.Errorf("quantmeta: descriptor must be a JSON object, got null")
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err == nil {
		return Descriptor{}, fmt.Errorf("quantmeta: trailing JSON after the descriptor")
	}
	var d Descriptor
	if err := json.Unmarshal(probe, &d); err != nil {
		return Descriptor{}, fmt.Errorf("quantmeta: %w", err)
	}
	return d, nil
}

// Encode renders the canonical form: keys sorted at every level, so two equal
// descriptors always produce identical bytes.
func Encode(d Descriptor) ([]byte, error) { return json.Marshal(d) }

func (a *ArtifactSpec) UnmarshalJSON(b []byte) error {
	type wire ArtifactSpec
	return decodeWithExtra(b, (*wire)(a), &a.Extra)
}

func (a ArtifactSpec) MarshalJSON() ([]byte, error) {
	type wire ArtifactSpec
	return encodeWithExtra(wire(a), a.Extra)
}

func (s *ScaleSpec) UnmarshalJSON(b []byte) error {
	type wire ScaleSpec
	return decodeWithExtra(b, (*wire)(s), &s.Extra)
}

func (s ScaleSpec) MarshalJSON() ([]byte, error) {
	type wire ScaleSpec
	return encodeWithExtra(wire(s), s.Extra)
}

func (z *ZeroPointSpec) UnmarshalJSON(b []byte) error {
	type wire ZeroPointSpec
	return decodeWithExtra(b, (*wire)(z), &z.Extra)
}

func (z ZeroPointSpec) MarshalJSON() ([]byte, error) {
	type wire ZeroPointSpec
	return encodeWithExtra(wire(z), z.Extra)
}

func (c *CodebookSpec) UnmarshalJSON(b []byte) error {
	type wire CodebookSpec
	return decodeWithExtra(b, (*wire)(c), &c.Extra)
}

func (c CodebookSpec) MarshalJSON() ([]byte, error) {
	type wire CodebookSpec
	return encodeWithExtra(wire(c), c.Extra)
}

func (t *TensorSpec) UnmarshalJSON(b []byte) error {
	type wire TensorSpec
	return decodeWithExtra(b, (*wire)(t), &t.Extra)
}

func (t TensorSpec) MarshalJSON() ([]byte, error) {
	type wire TensorSpec
	return encodeWithExtra(wire(t), t.Extra)
}

func (k *KVSpec) UnmarshalJSON(b []byte) error {
	type wire KVSpec
	return decodeWithExtra(b, (*wire)(k), &k.Extra)
}

func (k KVSpec) MarshalJSON() ([]byte, error) {
	type wire KVSpec
	return encodeWithExtra(wire(k), k.Extra)
}

func (s *SparsitySpec) UnmarshalJSON(b []byte) error {
	type wire SparsitySpec
	return decodeWithExtra(b, (*wire)(s), &s.Extra)
}

func (s SparsitySpec) MarshalJSON() ([]byte, error) {
	type wire SparsitySpec
	return encodeWithExtra(wire(s), s.Extra)
}

func (p *ProvenanceSpec) UnmarshalJSON(b []byte) error {
	type wire ProvenanceSpec
	return decodeWithExtra(b, (*wire)(p), &p.Extra)
}

func (p ProvenanceSpec) MarshalJSON() ([]byte, error) {
	type wire ProvenanceSpec
	return encodeWithExtra(wire(p), p.Extra)
}

func (e *MeasuredEnvelope) UnmarshalJSON(b []byte) error {
	type wire MeasuredEnvelope
	return decodeWithExtra(b, (*wire)(e), &e.Extra)
}

func (e MeasuredEnvelope) MarshalJSON() ([]byte, error) {
	type wire MeasuredEnvelope
	return encodeWithExtra(wire(e), e.Extra)
}

func (d *Descriptor) UnmarshalJSON(b []byte) error {
	type wire Descriptor
	return decodeWithExtra(b, (*wire)(d), &d.Extra)
}

func (d Descriptor) MarshalJSON() ([]byte, error) {
	type wire Descriptor
	return encodeWithExtra(wire(d), d.Extra)
}

// decodeWithExtra unmarshals into v and routes every key v does not declare into
// extra, preserved as raw JSON.
func decodeWithExtra(data []byte, v any, extra *map[string]json.RawMessage) error {
	if err := json.Unmarshal(data, v); err != nil {
		return err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	known := knownJSONKeys(reflect.TypeOf(v).Elem())
	var out map[string]json.RawMessage
	for k, val := range raw {
		if known[k] {
			continue
		}
		if out == nil {
			out = map[string]json.RawMessage{}
		}
		out[k] = append(json.RawMessage(nil), val...)
	}
	*extra = out
	return nil
}

// encodeWithExtra marshals v and merges the preserved unknown keys back in,
// refusing any that would shadow a declared field (which would emit a document
// with duplicate keys).
func encodeWithExtra(v any, extra map[string]json.RawMessage) ([]byte, error) {
	base, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(base, &m); err != nil {
		return nil, err
	}
	if m == nil {
		m = map[string]json.RawMessage{}
	}
	known := knownJSONKeys(reflect.TypeOf(v))
	for k, val := range extra {
		if known[k] {
			return nil, fmt.Errorf("quantmeta: extra field %q collides with a declared field", k)
		}
		if !json.Valid(val) {
			return nil, fmt.Errorf("quantmeta: extra field %q is not valid JSON", k)
		}
		m[k] = val
	}
	return marshalMap(m)
}

var knownKeyCache sync.Map // reflect.Type -> map[string]bool

// knownJSONKeys derives a struct's declared JSON key set from its own tags, so
// the preserve/declare split can never drift from the struct definition.
func knownJSONKeys(t reflect.Type) map[string]bool {
	if v, ok := knownKeyCache.Load(t); ok {
		return v.(map[string]bool)
	}
	m := map[string]bool{}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.PkgPath != "" {
			continue
		}
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name, _, _ := strings.Cut(tag, ",")
		if name == "" {
			name = f.Name
		}
		m[name] = true
	}
	knownKeyCache.Store(t, m)
	return m
}

// marshalMap writes an object with its keys in sorted order.
func marshalMap(m map[string]json.RawMessage) ([]byte, error) {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b bytes.Buffer
	b.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		kb, err := json.Marshal(k)
		if err != nil {
			return nil, err
		}
		b.Write(kb)
		b.WriteByte(':')
		b.Write(m[k])
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}
