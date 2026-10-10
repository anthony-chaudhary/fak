package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// The real portable restore loop uses synchronous CPU storage with injected
// ownership failures here. This is not a native-driver lifetime witness.
func TestKVHostFallbackRestoreFailureReleasesOwnership(t *testing.T) {
	t.Parallel()
	state := KVHostSnapshot{
		Config: KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 1},
		Pos:    []int{0, 1},
		K:      [][]float32{{1, 2}},
		KRaw:   [][]float32{{3, 4}},
		V:      [][]float32{{5, 6}},
	}
	for _, tc := range []struct {
		name        string
		failUpload  int
		failAppend  int
		wrongLength bool
	}{
		{name: "second upload", failUpload: 2},
		{name: "third upload", failUpload: 3},
		{name: "upload after committed row", failUpload: 4},
		{name: "append after mutation", failAppend: 2},
		{name: "length mismatch", wrongLength: true},
		{name: "success transfers owner"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fault := errors.New("injected restore failure")
			be := &kvHostFailureBackend{
				Backend: cpu(), fault: fault, failUpload: tc.failUpload,
				failAppend: tc.failAppend, wrongLength: tc.wrongLength,
				uploaded: map[Buffer]int{},
			}
			var out KVStore
			var err error
			var caught any
			func() {
				defer func() { caught = recover() }()
				out, err = RestoreKVFromHost(be, state)
			}()
			wantPanic := tc.failUpload != 0 || tc.failAppend != 0
			if wantPanic && caught != fault || !wantPanic && caught != nil {
				t.Fatalf("panic = %v, want original fault only on injected panic", caught)
			}
			if (err != nil) != tc.wrongLength {
				t.Fatalf("error = %v, want length error = %t", err, tc.wrongLength)
			}
			for buffer, freed := range be.uploaded {
				if freed != 1 {
					t.Fatalf("returned upload %p freed %d times, want once", buffer, freed)
				}
			}
			if wantPanic || tc.wrongLength {
				if out != nil || be.kv.freed != 1 {
					t.Fatalf("failed restore published=%t KV frees=%d, want false/1", out != nil, be.kv.freed)
				}
			} else {
				if out != be.kv || be.kv.freed != 0 || out.Len() != len(state.Pos) {
					t.Fatal("successful restore did not transfer a live complete owner")
				}
				out.Free()
				if be.kv.freed != 1 {
					t.Fatal("successful owner was not released exactly once by caller")
				}
			}
		})
	}
}

// Embedding Backend deliberately hides cpuBackend's optional bulk restorer.
type kvHostFailureBackend struct {
	Backend
	fault       error
	failUpload  int
	failAppend  int
	wrongLength bool
	uploads     int
	uploaded    map[Buffer]int
	kv          *kvHostFailureStore
}

func (b *kvHostFailureBackend) NewKV(cfg KVConfig) KVStore {
	b.kv = &kvHostFailureStore{KVStore: b.Backend.NewKV(cfg), owner: b}
	return b.kv
}

func (b *kvHostFailureBackend) Upload(t Tensor, as Dtype) Tensor {
	b.uploads++
	if b.uploads == b.failUpload {
		panic(b.fault)
	}
	out := b.Backend.Upload(t, as)
	b.uploaded[out.Buf()] = 0
	return out
}

func (b *kvHostFailureBackend) Free(t Tensor) {
	b.uploaded[t.Buf()]++
	b.Backend.Free(t)
}

type kvHostFailureStore struct {
	KVStore
	owner          *kvHostFailureBackend
	appends, freed int
}

func (k *kvHostFailureStore) AppendKV(layer int, raw, rope, value Tensor, pos int) {
	k.KVStore.AppendKV(layer, raw, rope, value, pos)
	k.appends++
	if k.appends == k.owner.failAppend {
		panic(k.owner.fault)
	}
}

func (k *kvHostFailureStore) Len() int {
	if k.owner.wrongLength {
		return 0
	}
	return k.KVStore.Len()
}

func (k *kvHostFailureStore) Free() {
	k.freed++
	k.KVStore.Free()
}
