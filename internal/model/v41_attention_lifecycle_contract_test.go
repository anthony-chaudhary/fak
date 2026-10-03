package model_test

// v41_attention_lifecycle_contract_test.go — external conformance corpus for the
// two DeepSeek-V4.1 attention-state owners shipped in this module:
//
//   * package model — internal/model/v41_attention_state.go (the root owner)
//   * package v41   — internal/model/v41/v41_attention_state.go (the reference copy)
//
// The two owners keep INDEPENDENT arithmetic, and this file keeps it that way: it
// asserts only the LIFECYCLE promises they share, through small local adapters, so
// an overlapping rule cannot drift silently while the independent math stays
// independent. No production code is added and no equation is merged.
//
// Out of scope by design: incremental Step ordering / append mutation
// (fak#13305-#13313), storage-resolver admission (fak#13321), latent-norm
// arithmetic (fak#13322), activation checkpoints (fak#13325), the root owner's
// compressor-source seams, and every intentional geometry/admission difference
// between the owners.

import (
	"math"
	"testing"

	model "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/model/v41"
)

const (
	lifecycleDim      = 4
	lifecycleRatio    = 2
	lifecycleRatioCap = 4
	// lifecycleWindow is the pinned reference ModelArgs sliding-window default
	// (v41WindowSize in both owners); WindowKV always returns this many slots.
	lifecycleWindow = 128
)

// lifecycleRow is the corpus's deterministic projected KV row for an absolute
// position. Every expected value below is derived from this arithmetic alone; no
// owner's output is ever used as the oracle for the other.
func lifecycleRow(dim, pos int) []float32 {
	row := make([]float32, dim)
	for d := range row {
		row[d] = float32(pos*10 + d + 1)
	}
	return row
}

type lifecycleRef struct {
	layer       int
	ratio       int
	kvSource    bool
	indexSource bool
}

type lifecycleUpdate struct {
	ref      lifecycleRef
	latent   []float32
	indexKey []float32
}

// lifecycleOwner is the neutral adapter the corpus drives. Both owners expose
// exactly these reader/writer promises; anything they do not share (the root
// owner's compressor-source seams, the config geometry guards) is absent by
// design, so the corpus never imposes one owner's extra API on the other.
type lifecycleOwner interface {
	name() string
	prefill(kv [][]float32, updates []lifecycleUpdate) error
	step(kv []float32, updates []lifecycleUpdate) error
	reset()
	windowKV() [][]float32
	kvRows(layer int) ([][]float32, bool)
	indexKeys(layer int) ([][]float32, bool)
	publishCandidates(ratio int, mask []bool) error
	candidates() ([]bool, int, bool)
	publishTopK(ratio int, rows [][]int32) error
	topK() ([][]int32, int, bool)
}

// --- root owner (package model) -------------------------------------------

type rootLifecycleOwner struct{ s *model.V41AttentionState }

func newRootLifecycleOwner(dim, cap int) (lifecycleOwner, error) {
	s, err := model.NewV41AttentionState(dim, cap)
	if err != nil {
		return nil, err
	}
	return &rootLifecycleOwner{s: s}, nil
}

func (o *rootLifecycleOwner) name() string { return "root" }

func rootLifecycleUpdates(us []lifecycleUpdate) []model.V41AttentionStateUpdate {
	if us == nil {
		return nil
	}
	out := make([]model.V41AttentionStateUpdate, len(us))
	for i, u := range us {
		out[i] = model.V41AttentionStateUpdate{
			Ref:      model.V41AttentionStateRef{LayerID: u.ref.layer, Ratio: u.ref.ratio, IsKVSource: u.ref.kvSource, IsIndexSource: u.ref.indexSource},
			Latent:   u.latent,
			IndexKey: u.indexKey,
		}
	}
	return out
}

func (o *rootLifecycleOwner) prefill(kv [][]float32, us []lifecycleUpdate) error {
	return o.s.Prefill(kv, rootLifecycleUpdates(us))
}

func (o *rootLifecycleOwner) step(kv []float32, us []lifecycleUpdate) error {
	return o.s.Step(kv, rootLifecycleUpdates(us))
}

func (o *rootLifecycleOwner) reset() { o.s.Reset() }

func (o *rootLifecycleOwner) windowKV() [][]float32 { return o.s.WindowKV() }

func (o *rootLifecycleOwner) kvRows(layer int) ([][]float32, bool) { return o.s.KVSourceRows(layer) }

func (o *rootLifecycleOwner) indexKeys(layer int) ([][]float32, bool) { return o.s.IndexKeys(layer) }

func (o *rootLifecycleOwner) publishCandidates(ratio int, mask []bool) error {
	return o.s.PublishCandidates(ratio, mask)
}

func (o *rootLifecycleOwner) candidates() ([]bool, int, bool) { return o.s.Candidates() }

func (o *rootLifecycleOwner) publishTopK(ratio int, rows [][]int32) error {
	return o.s.PublishTopK(ratio, rows)
}

func (o *rootLifecycleOwner) topK() ([][]int32, int, bool) { return o.s.TopK() }

// --- reference owner (package v41) ----------------------------------------

type v41LifecycleOwner struct{ s *v41.V41AttentionState }

func newV41LifecycleOwner(dim, cap int) (lifecycleOwner, error) {
	s, err := v41.NewV41AttentionState(dim, cap)
	if err != nil {
		return nil, err
	}
	return &v41LifecycleOwner{s: s}, nil
}

func (o *v41LifecycleOwner) name() string { return "reference" }

func v41LifecycleUpdates(us []lifecycleUpdate) []v41.V41AttentionStateUpdate {
	if us == nil {
		return nil
	}
	out := make([]v41.V41AttentionStateUpdate, len(us))
	for i, u := range us {
		out[i] = v41.V41AttentionStateUpdate{
			Ref:      v41.V41AttentionStateRef{LayerID: u.ref.layer, Ratio: u.ref.ratio, IsKVSource: u.ref.kvSource, IsIndexSource: u.ref.indexSource},
			Latent:   u.latent,
			IndexKey: u.indexKey,
		}
	}
	return out
}

func (o *v41LifecycleOwner) prefill(kv [][]float32, us []lifecycleUpdate) error {
	return o.s.Prefill(kv, v41LifecycleUpdates(us))
}

func (o *v41LifecycleOwner) step(kv []float32, us []lifecycleUpdate) error {
	return o.s.Step(kv, v41LifecycleUpdates(us))
}

func (o *v41LifecycleOwner) reset() { o.s.Reset() }

func (o *v41LifecycleOwner) windowKV() [][]float32 { return o.s.WindowKV() }

func (o *v41LifecycleOwner) kvRows(layer int) ([][]float32, bool) { return o.s.KVSourceRows(layer) }

func (o *v41LifecycleOwner) indexKeys(layer int) ([][]float32, bool) { return o.s.IndexKeys(layer) }

func (o *v41LifecycleOwner) publishCandidates(ratio int, mask []bool) error {
	return o.s.PublishCandidates(ratio, mask)
}

func (o *v41LifecycleOwner) candidates() ([]bool, int, bool) { return o.s.Candidates() }

func (o *v41LifecycleOwner) publishTopK(ratio int, rows [][]int32) error {
	return o.s.PublishTopK(ratio, rows)
}

func (o *v41LifecycleOwner) topK() ([][]int32, int, bool) { return o.s.TopK() }

// --- shared corpus --------------------------------------------------------

// TestV41AttentionLifecycleContract runs one documented lifecycle corpus against
// both owners. Failure of either arm fails the test; there is no arm that skips.
//
// fak-test:runtime fast est=30ms lane=default
func TestV41AttentionLifecycleContract(t *testing.T) {
	owners := []struct {
		name string
		open func(int, int) (lifecycleOwner, error)
	}{
		{"root", newRootLifecycleOwner},
		{"reference", newV41LifecycleOwner},
	}
	for _, o := range owners {
		o := o
		t.Run(o.name, func(t *testing.T) {
			// Each lifecycle corpus starts from a freshly constructed owner, so
			// no contract can observe another contract's mutations.
			fresh := func(t *testing.T) lifecycleOwner {
				t.Helper()
				owner, err := o.open(lifecycleDim, lifecycleRatioCap)
				if err != nil {
					t.Fatalf("open %s owner: %v", o.name, err)
				}
				return owner
			}
			t.Run("reset", func(t *testing.T) { lifecycleResetContract(t, fresh(t)) })
			t.Run("snapshot", func(t *testing.T) { lifecycleSnapshotContract(t, fresh(t)) })
			t.Run("rejection", func(t *testing.T) { lifecycleRejectionContract(t, fresh(t)) })
			t.Run("publication", func(t *testing.T) { lifecyclePublicationContract(t, fresh(t)) })
		})
	}
}

// seedSource prefills one KV+index source layer over a 4-position chunk, which
// completes floor(4/2) = 2 groups, and returns the chunk for expected-value use.
func seedSource(t *testing.T, o lifecycleOwner) [][]float32 {
	t.Helper()
	const positions = 4
	chunk := make([][]float32, positions)
	for i := range chunk {
		chunk[i] = lifecycleRow(lifecycleDim, i)
	}
	ref := lifecycleRef{layer: 2, ratio: lifecycleRatio, kvSource: true, indexSource: true}
	var ups []lifecycleUpdate
	for g := 1; g*lifecycleRatio <= positions; g++ {
		ups = append(ups, lifecycleUpdate{
			ref:      ref,
			latent:   lifecycleRow(lifecycleDim, g*lifecycleRatio-1),
			indexKey: lifecycleRow(lifecycleDim, g*lifecycleRatio-1),
		})
	}
	if err := o.prefill(chunk, ups); err != nil {
		t.Fatalf("%s: seed prefill: %v", o.name(), err)
	}
	return chunk
}

// assertSeedWindow derives the post-Prefill logical window from position
// arithmetic alone: the newest len(chunk) rows sit in the tail, oldest first,
// and every slot before them is untouched (zero) in a fresh state.
func assertSeedWindow(t *testing.T, o lifecycleOwner, w, chunk [][]float32) {
	t.Helper()
	if len(w) != lifecycleWindow {
		t.Fatalf("%s: WindowKV len=%d want %d", o.name(), len(w), lifecycleWindow)
	}
	lead := lifecycleWindow - len(chunk)
	for i := 0; i < lead; i++ {
		for d := range w[i] {
			if w[i][d] != 0 {
				t.Fatalf("%s: unwritten window slot out[%d][%d]=%g want 0", o.name(), i, d, w[i][d])
			}
		}
	}
	for j := range chunk {
		got := w[lead+j]
		if len(got) != lifecycleDim {
			t.Fatalf("%s: window row out[%d] width=%d want %d", o.name(), lead+j, len(got), lifecycleDim)
		}
		for d := range got {
			if got[d] != chunk[j][d] {
				t.Fatalf("%s: window out[%d][%d]=%g want %g", o.name(), lead+j, d, got[d], chunk[j][d])
			}
		}
	}
}

func assertWindowUnchanged(t *testing.T, o lifecycleOwner, want, got [][]float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: window length changed %d -> %d", o.name(), len(want), len(got))
	}
	for i := range want {
		if len(want[i]) != len(got[i]) {
			t.Fatalf("%s: window[%d] width changed %d -> %d", o.name(), i, len(want[i]), len(got[i]))
		}
		for d := range want[i] {
			if want[i][d] != got[i][d] {
				t.Fatalf("%s: window[%d][%d] changed %g -> %g", o.name(), i, d, want[i][d], got[i][d])
			}
		}
	}
}

func sameBools(a, b []bool) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// lifecycleResetContract: Reset clears every row and publication, and the state
// accepts a fresh Prefill of the same geometry.
func lifecycleResetContract(t *testing.T, o lifecycleOwner) {
	t.Helper()
	chunk := seedSource(t, o)
	if err := o.publishCandidates(lifecycleRatio, []bool{true, false, true}); err != nil {
		t.Fatalf("%s: publish candidates: %v", o.name(), err)
	}
	if err := o.publishTopK(lifecycleRatio, [][]int32{{3, 1}}); err != nil {
		t.Fatalf("%s: publish top-k: %v", o.name(), err)
	}
	if _, ok := o.kvRows(2); !ok {
		t.Fatalf("%s: reset contract needs a seeded KV row", o.name())
	}

	o.reset()

	if _, ok := o.kvRows(2); ok {
		t.Fatalf("%s: Reset left KV rows", o.name())
	}
	if _, ok := o.indexKeys(2); ok {
		t.Fatalf("%s: Reset left index keys", o.name())
	}
	if _, _, ok := o.candidates(); ok {
		t.Fatalf("%s: Reset left candidates", o.name())
	}
	if _, _, ok := o.topK(); ok {
		t.Fatalf("%s: Reset left top-k", o.name())
	}
	w := o.windowKV()
	if len(w) != lifecycleWindow {
		t.Fatalf("%s: post-reset WindowKV len=%d want %d", o.name(), len(w), lifecycleWindow)
	}
	for i := range w {
		for d := range w[i] {
			if w[i][d] != 0 {
				t.Fatalf("%s: Reset left window[%d][%d]=%g", o.name(), i, d, w[i][d])
			}
		}
	}
	if err := o.prefill(chunk, nil); err != nil {
		t.Fatalf("%s: post-reset prefill of the same geometry: %v", o.name(), err)
	}
	assertSeedWindow(t, o, o.windowKV(), chunk)
}

// lifecycleSnapshotContract: every reader returns an owned copy, and no reader
// aliases caller-supplied input.
func lifecycleSnapshotContract(t *testing.T, o lifecycleOwner) {
	t.Helper()
	chunk := seedSource(t, o)
	if err := o.publishCandidates(lifecycleRatio, []bool{true, false, true}); err != nil {
		t.Fatalf("%s: publish candidates: %v", o.name(), err)
	}
	if err := o.publishTopK(lifecycleRatio, [][]int32{{3, 1}, {4, 2}}); err != nil {
		t.Fatalf("%s: publish top-k: %v", o.name(), err)
	}

	w := o.windowKV()
	rows, ok := o.kvRows(2)
	if !ok || len(rows) != 2 {
		t.Fatalf("%s: KV rows=(len %d, ok %v) want 2 rows", o.name(), len(rows), ok)
	}
	keys, ok := o.indexKeys(2)
	if !ok || len(keys) != 2 {
		t.Fatalf("%s: index keys=(len %d, ok %v) want 2 rows", o.name(), len(keys), ok)
	}
	cand, _, cok := o.candidates()
	if !cok || len(cand) != 3 || !cand[0] {
		t.Fatalf("%s: candidates=(%v,%v) want a 3-wide mask with [0]=true", o.name(), cand, cok)
	}
	tk, _, tok := o.topK()
	if !tok || len(tk) != 2 || tk[0][0] != 3 {
		t.Fatalf("%s: top-k=(%v,%v) want [[3,1],[4,2]]", o.name(), tk, tok)
	}

	// Two reader calls must not hand back the same backing array.
	if w2 := o.windowKV(); &w2[0][0] == &w[0][0] {
		t.Fatalf("%s: two WindowKV calls alias one backing array", o.name())
	}

	// Mutating a returned snapshot must not reach internal state.
	w[len(w)-1][0] = -1
	rows[0][0] = -1
	keys[0][0] = -1
	cand[0] = false
	tk[0][0] = 99

	if got := o.windowKV(); got[len(got)-1][0] == -1 {
		t.Fatalf("%s: WindowKV reader aliased internal state", o.name())
	}
	if got, _ := o.kvRows(2); got[0][0] == -1 {
		t.Fatalf("%s: KVSourceRows reader aliased internal state", o.name())
	}
	if got, _ := o.indexKeys(2); got[0][0] == -1 {
		t.Fatalf("%s: IndexKeys reader aliased internal state", o.name())
	}
	if got, _, _ := o.candidates(); len(got) > 0 && !got[0] {
		t.Fatalf("%s: Candidates reader aliased internal state", o.name())
	}
	if got, _, _ := o.topK(); got[0][0] == 99 {
		t.Fatalf("%s: TopK reader aliased internal state", o.name())
	}

	// Mutating caller-supplied input after a call must not reach internal state.
	mask := []bool{true, false, true}
	if err := o.publishCandidates(lifecycleRatio, mask); err != nil {
		t.Fatalf("%s: republish candidates: %v", o.name(), err)
	}
	mask[0] = false
	if got, _, _ := o.candidates(); !got[0] {
		t.Fatalf("%s: Candidates reader aliased caller input", o.name())
	}
	callerRows := [][]int32{{3, 1}}
	if err := o.publishTopK(lifecycleRatio, callerRows); err != nil {
		t.Fatalf("%s: republish top-k: %v", o.name(), err)
	}
	callerRows[0][0] = 99
	if got, _, _ := o.topK(); got[0][0] != 3 {
		t.Fatalf("%s: TopK reader aliased caller input", o.name())
	}
	chunk[0][0] = -1
	if got := o.windowKV(); got[lifecycleWindow-len(chunk)][0] == -1 {
		t.Fatalf("%s: Prefill retained the caller's KV buffer", o.name())
	}
}

// lifecycleRejectionContract: a rejected update must commit nothing, and a
// valid update must still be accepted afterward (rejecting everything cannot
// pass this arm).
func lifecycleRejectionContract(t *testing.T, o lifecycleOwner) {
	t.Helper()
	seedSource(t, o)
	before := o.windowKV()
	beforeCand, _, _ := o.candidates()

	malformed := []struct {
		name string
		// layer the bad update would publish under, if it were accepted.
		layer int
		up    lifecycleUpdate
	}{
		{"latent on a non-source", 5, lifecycleUpdate{ref: lifecycleRef{layer: 5, ratio: lifecycleRatio}, latent: lifecycleRow(lifecycleDim, 4)}},
		{"ratio above the cap", 6, lifecycleUpdate{ref: lifecycleRef{layer: 6, ratio: lifecycleRatioCap + 1, kvSource: true}, latent: lifecycleRow(lifecycleDim, 4)}},
		{"index key on a non-index source", 7, lifecycleUpdate{ref: lifecycleRef{layer: 7, ratio: lifecycleRatio, kvSource: true}, latent: lifecycleRow(lifecycleDim, 4), indexKey: lifecycleRow(lifecycleDim, 4)}},
		{"short latent", 8, lifecycleUpdate{ref: lifecycleRef{layer: 8, ratio: lifecycleRatio, kvSource: true}, latent: []float32{1, 2}}},
		{"non-finite latent", 9, lifecycleUpdate{ref: lifecycleRef{layer: 9, ratio: lifecycleRatio, kvSource: true}, latent: []float32{1, float32(math.Inf(1)), 3, 4}}},
	}
	for _, tc := range malformed {
		if err := o.step(lifecycleRow(lifecycleDim, 4), []lifecycleUpdate{tc.up}); err == nil {
			t.Fatalf("%s/%s: accepted a malformed update", o.name(), tc.name)
		}
		assertWindowUnchanged(t, o, before, o.windowKV())
		if _, ok := o.kvRows(tc.layer); ok {
			t.Fatalf("%s/%s: a failed update published a KV row", o.name(), tc.name)
		}
		if _, ok := o.indexKeys(tc.layer); ok {
			t.Fatalf("%s/%s: a failed update published an index key", o.name(), tc.name)
		}
		afterCand, _, _ := o.candidates()
		if !sameBools(beforeCand, afterCand) {
			t.Fatalf("%s/%s: a failed update changed candidates %v -> %v", o.name(), tc.name, beforeCand, afterCand)
		}
	}

	// Descending layer order within one call is malformed as a whole.
	desc := []lifecycleUpdate{
		{ref: lifecycleRef{layer: 9, ratio: lifecycleRatio, kvSource: true}, latent: lifecycleRow(lifecycleDim, 4)},
		{ref: lifecycleRef{layer: 7, ratio: lifecycleRatio, kvSource: true}, latent: lifecycleRow(lifecycleDim, 4)},
	}
	if err := o.step(lifecycleRow(lifecycleDim, 4), desc); err == nil {
		t.Fatalf("%s: accepted an out-of-order update batch", o.name())
	}
	assertWindowUnchanged(t, o, before, o.windowKV())

	// A Prefill on a non-empty state is refused atomically.
	fresh := make([][]float32, 2)
	for i := range fresh {
		fresh[i] = lifecycleRow(lifecycleDim, i)
	}
	if err := o.prefill(fresh, nil); err == nil {
		t.Fatalf("%s: accepted a Prefill on a non-empty state", o.name())
	}
	assertWindowUnchanged(t, o, before, o.windowKV())

	// Positive control: a well-formed step is still accepted.
	if err := o.step(lifecycleRow(lifecycleDim, 4), nil); err != nil {
		t.Fatalf("%s: a well-formed step was refused: %v", o.name(), err)
	}
}

// lifecyclePublicationContract: a source layer publishes its rows in group order
// under its own layer id, an unpublished layer reports nothing, and the
// selection readers return the publishing ratio.
func lifecyclePublicationContract(t *testing.T, o lifecycleOwner) {
	t.Helper()
	seedSource(t, o)

	rows, ok := o.kvRows(2)
	if !ok || len(rows) != 2 {
		t.Fatalf("%s: source rows=(len %d, ok %v) want 2 group rows", o.name(), len(rows), ok)
	}
	keys, ok := o.indexKeys(2)
	if !ok || len(keys) != 2 {
		t.Fatalf("%s: source index keys=(len %d, ok %v) want 2 group rows", o.name(), len(keys), ok)
	}
	for g := 0; g < 2; g++ {
		want := lifecycleRow(lifecycleDim, (g+1)*lifecycleRatio-1)
		for d := range want {
			if rows[g][d] != want[d] {
				t.Fatalf("%s: KV row %d dim %d=%g want %g", o.name(), g, d, rows[g][d], want[d])
			}
			if keys[g][d] != want[d] {
				t.Fatalf("%s: index key %d dim %d=%g want %g", o.name(), g, d, keys[g][d], want[d])
			}
		}
	}
	if _, ok := o.kvRows(99); ok {
		t.Fatalf("%s: an unpublished layer reported KV rows", o.name())
	}
	if _, ok := o.indexKeys(99); ok {
		t.Fatalf("%s: an unpublished layer reported index keys", o.name())
	}

	if err := o.publishCandidates(3, []bool{true, false}); err != nil {
		t.Fatalf("%s: publish candidates: %v", o.name(), err)
	}
	if _, ratio, ok := o.candidates(); !ok || ratio != 3 {
		t.Fatalf("%s: Candidates ratio=(%d, ok %v) want the publisher ratio 3", o.name(), ratio, ok)
	}
	if err := o.publishTopK(lifecycleRatio, [][]int32{{1, 2}}); err != nil {
		t.Fatalf("%s: publish top-k: %v", o.name(), err)
	}
	if _, ratio, ok := o.topK(); !ok || ratio != lifecycleRatio {
		t.Fatalf("%s: TopK ratio=(%d, ok %v) want the publisher ratio %d", o.name(), ratio, ok, lifecycleRatio)
	}
}
