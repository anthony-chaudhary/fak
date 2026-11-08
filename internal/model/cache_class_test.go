package model

import (
	"errors"
	"reflect"
	"testing"
)

// cache_class_test.go — pins the forward-requested cache-class registration schema
// (arch.go): a declared-but-unregistered class is refused with the
// ErrUnregisteredCacheClass sentinel before load, and the DeepSeek-V4.1-Flash recipe
// enumeration registers exactly its four classes with the pinned metadata. Software
// evidence only: no hardware or whole-model claim.

func TestCacheClassRegistrationRefusesUnregisteredBeforeLoad(t *testing.T) {
	m, err := NewCacheClassManifest(
		CacheClassSpec{Name: "kv", Kind: CacheClassTokenKV},
		CacheClassSpec{Name: "indexer_kv", Kind: CacheClassIndexerKV, DType: "mxfp4"},
	)
	if err != nil {
		t.Fatalf("NewCacheClassManifest: %v", err)
	}

	// Every requested class registered: the gate passes.
	if err := m.CheckForwardRequests([]string{"kv", "indexer_kv"}); err != nil {
		t.Fatalf("CheckForwardRequests on fully-registered listing = %v, want nil", err)
	}
	// An empty request list is vacuously satisfied.
	if err := m.CheckForwardRequests(nil); err != nil {
		t.Fatalf("CheckForwardRequests(nil) = %v, want nil", err)
	}

	// A declared-but-unregistered class is refused with the sentinel, naming the
	// offender, BEFORE any load happens.
	err = m.CheckForwardRequests([]string{"kv", "compressor_state"})
	if !errors.Is(err, ErrUnregisteredCacheClass) {
		t.Fatalf("CheckForwardRequests with unregistered class: err = %v, want errors.Is(ErrUnregisteredCacheClass)", err)
	}
	if err == nil || err.Error() == "" {
		t.Fatalf("refusal must carry detail, got %v", err)
	}

	// LayersFor on an unknown class refuses with the same sentinel.
	if _, err := m.LayersFor("compressor_state", 8); !errors.Is(err, ErrUnregisteredCacheClass) {
		t.Fatalf("LayersFor on unregistered class: err = %v, want errors.Is(ErrUnregisteredCacheClass)", err)
	}
}

func TestCacheClassRegistrationDuplicateAndInvalidSpecs(t *testing.T) {
	m, err := NewCacheClassManifest(CacheClassSpec{Name: "kv", Kind: CacheClassTokenKV})
	if err != nil {
		t.Fatalf("NewCacheClassManifest: %v", err)
	}
	// One class per name: a second spec under the same name is refused, not merged.
	if err := m.Register(CacheClassSpec{Name: "kv", Kind: CacheClassRecurrentState}); !errors.Is(err, ErrDuplicateCacheClass) {
		t.Fatalf("duplicate Register: err = %v, want errors.Is(ErrDuplicateCacheClass)", err)
	}
	if !m.Has("kv") {
		t.Fatal("original registration lost after duplicate refusal")
	}

	for name, bad := range map[string]CacheClassSpec{
		"empty-name":      {Kind: CacheClassTokenKV},
		"unknown-kind":    {Name: "x", Kind: CacheClassKind(99)},
		"negative-window": {Name: "w", Kind: CacheClassTokenKV, Window: -1},
		"zero-ratio":      {Name: "r", Kind: CacheClassCompressedLatent, Ratios: []int{0}},
		"negative-layer":  {Name: "l", Kind: CacheClassTokenKV, Layers: []int{-3}},
	} {
		if err := validateCacheClass(bad); err == nil {
			t.Errorf("validateCacheClass(%s): wanted refusal, got nil", name)
		}
	}
}

func TestCacheClassRegistrationV41FlashEnumeration(t *testing.T) {
	const numLayers = 32
	m, err := DeepSeekV41FlashCacheClasses(numLayers)
	if err != nil {
		t.Fatalf("DeepSeekV41FlashCacheClasses(%d): %v", numLayers, err)
	}

	// Exactly the recipe's four classes, in one-pass order.
	wantNames := []string{"kv", "compressed_latent", "indexer_kv", "compressor_state"}
	if got := m.Names(); !reflect.DeepEqual(got, wantNames) {
		t.Fatalf("Names() = %v, want %v", got, wantNames)
	}

	// Per-layer 128-token sliding window over every layer.
	kv, ok := m.Spec("kv")
	if !ok {
		t.Fatal("kv class missing from manifest")
	}
	if kv.Kind != CacheClassTokenKV || kv.Window != 128 || len(kv.Layers) != 0 {
		t.Fatalf("kv spec = %+v, want token-kv with 128-token window on all layers", kv)
	}
	all, err := m.LayersFor("kv", numLayers)
	if err != nil {
		t.Fatalf("LayersFor(kv): %v", err)
	}
	if len(all) != numLayers || all[0] != 0 || all[numLayers-1] != numLayers-1 {
		t.Fatalf("LayersFor(kv) = %v, want all %d layers", all, numLayers)
	}

	// FP4 compressed latents at ratios 1 and 2, pinned to layers 2/8/14/20.
	cl, ok := m.Spec("compressed_latent")
	if !ok {
		t.Fatal("compressed_latent class missing from manifest")
	}
	if cl.Kind != CacheClassCompressedLatent || cl.DType != "fp4" || !reflect.DeepEqual(cl.Ratios, []int{1, 2}) {
		t.Fatalf("compressed_latent spec = %+v, want fp4 kind compressed-latent ratios [1 2]", cl)
	}
	layers, err := m.LayersFor("compressed_latent", numLayers)
	if err != nil {
		t.Fatalf("LayersFor(compressed_latent): %v", err)
	}
	if !reflect.DeepEqual(layers, []int{2, 8, 14, 20}) {
		t.Fatalf("LayersFor(compressed_latent) = %v, want [2 8 14 20]", layers)
	}

	// mxfp4 sparse-indexer KV on every layer; compressor state is recurrent.
	idx, _ := m.Spec("indexer_kv")
	if idx.Kind != CacheClassIndexerKV || idx.DType != "mxfp4" {
		t.Fatalf("indexer_kv spec = %+v, want indexer-kv mxfp4", idx)
	}
	st, _ := m.Spec("compressor_state")
	if st.Kind != CacheClassRecurrentState {
		t.Fatalf("compressor_state spec = %+v, want recurrent-state", st)
	}

	// The full recipe listing passes the pre-load gate in one pass.
	if err := m.CheckForwardRequests(wantNames); err != nil {
		t.Fatalf("CheckForwardRequests(full recipe listing) = %v, want nil", err)
	}
}

func TestCacheClassRegistrationV41FlashRefusesSmallCheckpoint(t *testing.T) {
	// Layer 20 is declared for the compressed latents: a checkpoint without it must
	// be refused, not silently lose the class.
	if _, err := DeepSeekV41FlashCacheClasses(5); err == nil {
		t.Fatal("DeepSeekV41FlashCacheClasses(5): wanted refusal for out-of-range layer 20, got nil")
	}
	if _, err := DeepSeekV41FlashCacheClasses(0); err == nil {
		t.Fatal("DeepSeekV41FlashCacheClasses(0): wanted refusal, got nil")
	}
	// Boundary: exactly 21 layers admits the highest declared layer.
	if _, err := DeepSeekV41FlashCacheClasses(21); err != nil {
		t.Fatalf("DeepSeekV41FlashCacheClasses(21): %v, want success at the boundary", err)
	}
}
