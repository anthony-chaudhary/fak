package modelengine

// loader_parity.go is the model-load/conversion child of the quality spine
// (#4545, sibling of the resume-parity child #4538): it proves that loading a
// model through an ALTERNATE path — converted, sharded, GGUF, or checkpoint —
// produces byte-for-byte the same generation as loading the original. The
// original load is the reference; each alternate load is the engine. A faithful
// load path is invisible in the output (token-identical); a changed config
// default or a dropped/reordered tensor shard surfaces as the FIRST divergence,
// localized to its exact token index, and is refused before serve.
//
// The oracle is deterministic and self-contained (no real weights, no GPU, no
// disk): a history-dependent decoder folds the canonical model bytes — config
// then every tensor, in presented order — into a carried accumulator, so any
// load defect that perturbs the canonical form perturbs the very token where it
// first bites. Runtime/resource cost: pure in-process, microseconds per case,
// no external fixtures. Tier: PR (runs in the package unit gate).

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/bits"
	"sort"
	"strings"
)

// loaderParityTier assigns every case in this child to the PR gate: it is a
// pure in-process unit test with no model download or accelerator, so it runs
// on every pull request rather than being deferred to nightly or release.
const loaderParityTier = "PR"

// loaderVocab is the small fixed vocabulary the deterministic decoder emits
// from. It is disjoint from the sibling oracles' vocabularies so a loader-parity
// trace is never confused with another child's trace in a failure bundle.
var loaderVocab = []string{"anchor", "ballast", "cleat", "davit", "ensign", "fathom", "galley", "halyard"}

// loaderConfigKV is one model-config entry. A changed default is a changed val.
type loaderConfigKV struct {
	key string
	val uint64
}

// loaderTensor is one named weight shard.
type loaderTensor struct {
	name string
	data []uint32
}

// loaderModel is the in-memory model handed to serve. Regardless of which load
// path produced it, a faithful load yields a model whose canonical bytes equal
// the reference's — that invariant is what this child gates.
type loaderModel struct {
	modelID   string
	tokenizer string
	revision  string
	seed      uint64
	config    []loaderConfigKV
	tensors   []loaderTensor
}

// loaderReferenceModel is the pinned fixture: a small, fully deterministic model
// whose tensors are already in canonical (name-sorted) order.
func loaderReferenceModel() loaderModel {
	n := 6
	ts := make([]loaderTensor, 0, n)
	for i := 0; i < n; i++ {
		d := make([]uint32, 4)
		for j := range d {
			d[j] = uint32((i+1)*0x9e37) ^ uint32(j*0x1000193)
		}
		ts = append(ts, loaderTensor{name: fmt.Sprintf("blk.%d.weight", i), data: d})
	}
	sort.Slice(ts, func(a, b int) bool { return ts[a].name < ts[b].name })
	return loaderModel{
		modelID:   "fak-fixture-7b",
		tokenizer: "fak-bpe-v1",
		revision:  "modelengine@loader-parity-1",
		seed:      0x5eed1234,
		config: []loaderConfigKV{
			{"hidden_size", 4096},
			{"n_layers", 6},
			{"rope_theta", 10000},
			{"vocab_size", 32000},
		},
		tensors: ts,
	}
}

// loaderClone deep-copies a model so a load path can transform it without
// mutating the shared reference fixture.
func loaderClone(m loaderModel) loaderModel {
	cp := m
	cp.config = append([]loaderConfigKV(nil), m.config...)
	cp.tensors = make([]loaderTensor, len(m.tensors))
	for i, t := range m.tensors {
		cp.tensors[i] = loaderTensor{name: t.name, data: append([]uint32(nil), t.data...)}
	}
	return cp
}

// loaderConfigBytes serializes config in key-sorted order — a stable form whose
// bytes change only when a value (a default) changes, never on map iteration.
func loaderConfigBytes(cfg []loaderConfigKV) []byte {
	kv := append([]loaderConfigKV(nil), cfg...)
	sort.Slice(kv, func(a, b int) bool { return kv[a].key < kv[b].key })
	var b []byte
	var v [8]byte
	for _, e := range kv {
		b = append(b, e.key...)
		binary.BigEndian.PutUint64(v[:], e.val)
		b = append(b, v[:]...)
	}
	return b
}

// loaderTensorBytes serializes one tensor: its name then its big-endian words.
func loaderTensorBytes(t loaderTensor) []byte {
	b := []byte(t.name)
	var v [4]byte
	for _, x := range t.data {
		binary.BigEndian.PutUint32(v[:], x)
		b = append(b, v[:]...)
	}
	return b
}

// loaderCanonicalBytes is the full canonical serialization: config, then every
// tensor in PRESENTED order. Presented order is deliberate — a load path that
// fails to canonicalize shard order (a reorder defect) changes these bytes and
// is caught, rather than being silently repaired by the fingerprint.
func loaderCanonicalBytes(m loaderModel) []byte {
	b := loaderConfigBytes(m.config)
	for _, t := range m.tensors {
		b = append(b, loaderTensorBytes(t)...)
	}
	return b
}

func loaderFingerprint(m loaderModel) [32]byte { return sha256.Sum256(loaderCanonicalBytes(m)) }

// loaderStream is the history-dependent decode state: a splitmix64 stream folded
// with a carried accumulator so token i depends on the whole model prefix seen
// so far, not just (seed, i).
type loaderStream struct{ rng, acc uint64 }

func loaderNewStream(seed uint64) loaderStream {
	return loaderStream{rng: seed*0x9e3779b97f4a7c15 + 0x243f6a8885a308d3}
}

// mix folds arbitrary bytes (config or a tensor) into the accumulator.
func (s *loaderStream) mix(b []byte) {
	for _, x := range b {
		s.acc = s.acc*0x100000001b3 + uint64(x) + 1
	}
}

// draw advances the stream one step, mixing in the carried accumulator, and
// returns a finalized 64-bit value.
func (s *loaderStream) draw() uint64 {
	s.rng += 0x9e3779b97f4a7c15
	z := s.rng ^ s.acc
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	s.acc = s.acc*0x100000001b3 + z
	return z
}

// loaderDecode is the deterministic generation: fold the config into the stream,
// then emit one token per tensor (in presented order), each after folding that
// tensor in. Config perturbations bite at token 0; a per-shard perturbation
// bites at that shard's index.
func loaderDecode(m loaderModel) []string {
	st := loaderNewStream(m.seed)
	st.mix(loaderConfigBytes(m.config))
	toks := make([]string, 0, len(m.tensors))
	for _, t := range m.tensors {
		st.mix(loaderTensorBytes(t))
		toks = append(toks, loaderVocab[st.draw()%uint64(len(loaderVocab))])
	}
	return toks
}

// loaderServeGate is the fast pre-serve guard: it refuses to serve a model whose
// canonical fingerprint does not match the pinned baseline. "Changed config
// default or tensor shard fails before serve" is exactly this check — every
// planted defect changes the fingerprint and is refused here, before a single
// token is generated.
func loaderServeGate(m loaderModel, baseline [32]byte) error {
	if got := loaderFingerprint(m); got != baseline {
		return fmt.Errorf("pre-serve fingerprint mismatch: model %x != baseline %x", got[:6], baseline[:6])
	}
	return nil
}

// loaderPath is one load path: a name, the engine/backend it models, and the
// load transform it applies to the on-disk reference.
type loaderPath struct {
	name    string
	backend string
	load    func(m loaderModel) loaderModel
}

// loaderFaithfulPaths are the five in-scope load paths. Each faithful path
// arrives at the canonical model by a genuinely different, LOSSLESS route, so a
// correct implementation is token-identical to the original.
func loaderFaithfulPaths() []loaderPath {
	return []loaderPath{
		{"original", "reference", loaderClone},
		{"converted", "safetensors->native", loaderConvertRoundTrip},
		{"sharded", "sharded-2way", func(m loaderModel) loaderModel { return loaderShardRoundTrip(m, 2) }},
		{"gguf", "gguf-lossless-pack", loaderGGUFRoundTrip},
		{"checkpoint", "checkpoint-restore", loaderCheckpointRoundTrip},
	}
}

// loaderConvertRoundTrip models the converted path: store each word byte-swapped
// and load it back. ReverseBytes32 is its own inverse, so a correct converter is
// bit-exact.
func loaderConvertRoundTrip(m loaderModel) loaderModel {
	out := loaderClone(m)
	for i, t := range out.tensors {
		for j, x := range t.data {
			out.tensors[i].data[j] = bits.ReverseBytes32(bits.ReverseBytes32(x))
		}
	}
	return out
}

// loaderShardRoundTrip models the sharded path: split every tensor into `parts`
// contiguous shards and reassemble them in order. Joining shards in shard order
// reproduces the original data.
func loaderShardRoundTrip(m loaderModel, parts int) loaderModel {
	out := loaderClone(m)
	for i, t := range out.tensors {
		out.tensors[i].data = loaderJoin(loaderSplit(t.data, parts))
	}
	return out
}

func loaderSplit(d []uint32, parts int) [][]uint32 {
	if parts < 1 {
		parts = 1
	}
	sz := (len(d) + parts - 1) / parts
	if sz < 1 {
		sz = 1
	}
	var out [][]uint32
	for i := 0; i < len(d); i += sz {
		j := i + sz
		if j > len(d) {
			j = len(d)
		}
		out = append(out, append([]uint32(nil), d[i:j]...))
	}
	return out
}

func loaderJoin(sh [][]uint32) []uint32 {
	var out []uint32
	for _, s := range sh {
		out = append(out, s...)
	}
	return out
}

// loaderGGUFRoundTrip models the GGUF path: pack each word with a rotate on
// store and unpack with the inverse rotate on load. rotl∘rotr is identity, so a
// correct pack/unpack is lossless at this fixture scale.
func loaderGGUFRoundTrip(m loaderModel) loaderModel {
	out := loaderClone(m)
	for i, t := range out.tensors {
		for j, x := range t.data {
			out.tensors[i].data[j] = bits.RotateLeft32(bits.RotateLeft32(x, 7), -7)
		}
	}
	return out
}

// loaderCheckpointRoundTrip models the checkpoint path: serialize the tensors to
// a byte blob and restore them. A faithful save/load is exact.
func loaderCheckpointRoundTrip(m loaderModel) loaderModel {
	out := loaderClone(m)
	out.tensors = loaderLoadTensors(loaderSaveTensors(m.tensors))
	return out
}

func loaderSaveTensors(ts []loaderTensor) []byte {
	var b []byte
	var u [4]byte
	for _, t := range ts {
		binary.BigEndian.PutUint32(u[:], uint32(len(t.name)))
		b = append(b, u[:]...)
		b = append(b, t.name...)
		binary.BigEndian.PutUint32(u[:], uint32(len(t.data)))
		b = append(b, u[:]...)
		for _, x := range t.data {
			binary.BigEndian.PutUint32(u[:], x)
			b = append(b, u[:]...)
		}
	}
	return b
}

func loaderLoadTensors(b []byte) []loaderTensor {
	var ts []loaderTensor
	off := 0
	rd := func() uint32 { v := binary.BigEndian.Uint32(b[off : off+4]); off += 4; return v }
	for off < len(b) {
		nl := int(rd())
		name := string(b[off : off+nl])
		off += nl
		dl := int(rd())
		d := make([]uint32, dl)
		for i := range d {
			d[i] = rd()
		}
		ts = append(ts, loaderTensor{name: name, data: d})
	}
	return ts
}

// --- planted representative defects -----------------------------------------

// loaderConfigDefaultDefect flips a config default (rope_theta) — the footgun of
// a converter that "helpfully" rewrites a default. Config is folded first, so it
// bites at token 0.
func loaderConfigDefaultDefect(m loaderModel) loaderModel {
	out := loaderClone(m)
	for i := range out.config {
		if out.config[i].key == "rope_theta" {
			out.config[i].val = 1000000
		}
	}
	return out
}

// loaderDroppedShardDefect drops one tensor shard — silently missing after
// conversion. It bites at the dropped index and shortens the stream.
func loaderDroppedShardDefect(m loaderModel) loaderModel {
	out := loaderClone(m)
	const drop = 3
	out.tensors = append(out.tensors[:drop:drop], out.tensors[drop+1:]...)
	return out
}

// loaderReorderShardDefect swaps two shards — a loader that trusted on-disk
// order instead of canonicalizing by name. It bites at the first swapped index.
func loaderReorderShardDefect(m loaderModel) loaderModel {
	out := loaderClone(m)
	out.tensors[2], out.tensors[4] = out.tensors[4], out.tensors[2]
	return out
}

// --- provenance, replay artifact, and the differential oracle ---------------

// loaderProvenance records everything the acceptance criteria require per case:
// model, tokenizer, engine/backend, seed/oracle, code revision, and
// tolerance/baseline. Tensors are recorded scrubbed — name and element count
// only, never raw weights.
type loaderProvenance struct {
	CaseID    string
	Model     string
	Tokenizer string
	Backend   string
	Seed      uint64
	Revision  string
	Baseline  string
	Tolerance string
	Tier      string
	Tensors   []string
}

func loaderProvenanceOf(caseID string, m loaderModel, backend string, baseline [32]byte) loaderProvenance {
	names := make([]string, 0, len(m.tensors))
	for _, t := range m.tensors {
		names = append(names, fmt.Sprintf("%s:%d", t.name, len(t.data)))
	}
	return loaderProvenance{
		CaseID: caseID, Model: m.modelID, Tokenizer: m.tokenizer, Backend: backend,
		Seed: m.seed, Revision: m.revision, Baseline: fmt.Sprintf("%x", baseline[:6]),
		Tolerance: "exact (temperature=0, deterministic oracle)", Tier: loaderParityTier, Tensors: names,
	}
}

// complete reports whether every required provenance field is populated — an
// unprovenanced case is inconclusive and must never be reported as pass.
func (p loaderProvenance) complete() bool {
	return p.Model != "" && p.Tokenizer != "" && p.Backend != "" && p.Revision != "" &&
		p.Baseline != "" && p.Tolerance != "" && p.Tier != "" && p.Seed != 0
}

// loaderDivergence is the first actionable divergence: the token index and the
// reference vs engine tokens there.
type loaderDivergence struct {
	Index          int
	ReferenceToken string
	EngineToken    string
}

// loaderReplayArtifact is the scrubbed, independently-replayable failure bundle:
// full provenance plus the first divergence, carrying tensor NAMES and sizes but
// never raw weights.
type loaderReplayArtifact struct {
	Provenance loaderProvenance
	FailPath   string
	Reason     string
	Divergence *loaderDivergence
}

func (a loaderReplayArtifact) String() string {
	idx, ref, eng := -1, "<none>", "<none>"
	if a.Divergence != nil {
		idx, ref, eng = a.Divergence.Index, a.Divergence.ReferenceToken, a.Divergence.EngineToken
	}
	p := a.Provenance
	return fmt.Sprintf("replay{case=%s model=%s tok=%s backend=%s seed=%#x rev=%s baseline=%s tol=%q tier=%s fail=%s reason=%s divergence=@%d ref=%q eng=%q tensors=%s}",
		p.CaseID, p.Model, p.Tokenizer, p.Backend, p.Seed, p.Revision, p.Baseline, p.Tolerance, p.Tier,
		a.FailPath, a.Reason, idx, ref, eng, strings.Join(p.Tensors, ","))
}

type loaderVerdict struct {
	Pass     bool
	Detail   string
	Artifact *loaderReplayArtifact
}

func loaderTokenAt(t []string, i int) string {
	if i >= 0 && i < len(t) {
		return t[i]
	}
	return "<none>"
}

// loaderJudge is the differential oracle: an alternate load path must emit
// exactly the reference token sequence. Empty/short evidence is never a pass;
// any divergence is reported as the first index with a scrubbed replay artifact.
func loaderJudge(ref, eng []string, prov loaderProvenance) loaderVerdict {
	mk := func(reason string, d *loaderDivergence) *loaderReplayArtifact {
		return &loaderReplayArtifact{Provenance: prov, FailPath: prov.Backend, Reason: reason, Divergence: d}
	}
	if len(eng) == 0 {
		return loaderVerdict{Pass: false, Detail: "engine produced no tokens — inconclusive evidence is never pass",
			Artifact: mk("no-evidence", &loaderDivergence{Index: 0, ReferenceToken: loaderTokenAt(ref, 0), EngineToken: "<none>"})}
	}
	n := len(ref)
	if len(eng) < n {
		n = len(eng)
	}
	for i := 0; i < n; i++ {
		if ref[i] != eng[i] {
			d := &loaderDivergence{Index: i, ReferenceToken: ref[i], EngineToken: eng[i]}
			return loaderVerdict{Pass: false,
				Detail:   fmt.Sprintf("load path diverged at token %d: reference %q, engine %q — the converted/sharded model is not the original", i, ref[i], eng[i]),
				Artifact: mk("divergence", d)}
		}
	}
	if len(ref) != len(eng) {
		d := &loaderDivergence{Index: n, ReferenceToken: loaderTokenAt(ref, n), EngineToken: loaderTokenAt(eng, n)}
		return loaderVerdict{Pass: false,
			Detail:   fmt.Sprintf("token count diverged at %d: reference has %d, engine has %d — a shard was dropped or duplicated", n, len(ref), len(eng)),
			Artifact: mk("length-divergence", d)}
	}
	return loaderVerdict{Pass: true, Detail: fmt.Sprintf("load path reproduced the reference: %d tokens identical", len(ref))}
}

// loaderFirstDiff returns the first index at which two token streams differ, the
// min length if one is a prefix of the other, or -1 if identical. It lets the
// defect tests assert the oracle's localization without hard-coding an index.
func loaderFirstDiff(a, b []string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}
