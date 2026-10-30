package model

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default

// The pinned 1017 profile must validate as shipped: exact target identity, the
// complete ordered 48-shard manifest, the unchanged config/tokenizer/index
// pins, the Engram extents on shards 47/48, the bundled DSpark intake read from
// the pinned config, and the explicit unavailable draft fixture pinning the
// inspected storage whose gap was witnessed.
func TestV41Flash1017ArtifactProfilePinned(t *testing.T) {
	t.Parallel()
	p := &DeepSeekV41Flash1017Artifact
	if err := p.Validate(); err != nil {
		t.Fatalf("pinned 1017 profile must validate: %v", err)
	}
	if p.TargetRepository != DeepSeekV41Flash1017ModelID || p.TargetRevision != DeepSeekV41Flash1017Revision {
		t.Fatalf("target identity drifted: %s@%s", p.TargetRepository, p.TargetRevision)
	}
	if len(p.WeightShards) != 48 {
		t.Fatalf("manifest carries %d shards, want 48", len(p.WeightShards))
	}
	if !reflect.DeepEqual(p.EngramShardNames, []string{"model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"}) {
		t.Fatalf("Engram extents drifted: %v", p.EngramShardNames)
	}
	if !reflect.DeepEqual(p.EngramLayerIDs, []int{1, 14}) || !reflect.DeepEqual(p.EngramNumEmbeddings, []int{384006168, 384016682}) {
		t.Fatalf("Engram encoded extents drifted: layers=%v embeddings=%v", p.EngramLayerIDs, p.EngramNumEmbeddings)
	}
	d := p.DSpark
	if d.NextNPredictLayers != 3 || d.BlockSize != 5 || !reflect.DeepEqual(d.TargetLayerIDs, []int{37, 38, 39}) ||
		d.MarkovRank != 256 || d.NumRoutedExperts != 128 || d.NumExpertsPerTok != 3 {
		t.Fatalf("DSpark intake drifted: %+v", d)
	}
	sel := p.Draft
	if sel.Status != V41DraftStatusUnavailable || sel.Repository != "mrfakename/DeepSeek-V4.1-Flash-1017-GGUF" ||
		sel.Revision != "1df54e7d1b1e442051ad393ce871870a929f243e" || sel.Reason == "" {
		t.Fatalf("draft selection must be the explicit unavailable fixture pinning the inspected storage: %+v", sel)
	}
}

// Manifest completeness: the exact pinned set admits in any presentation order;
// a missing shard, an extra shard or a size drift each fail closed with the
// typed refusal.
func TestV41AdmitArtifactManifestCompleteness(t *testing.T) {
	t.Parallel()
	p := &DeepSeekV41Flash1017Artifact
	shuffled := append([]V41FilePin(nil), p.WeightShards...)
	for i := 1; i < len(shuffled); i++ {
		shuffled[i], shuffled[0] = shuffled[0], shuffled[i]
	}
	if err := AdmitV41ArtifactAgainstProfile(p, shuffled); err != nil {
		t.Fatalf("exact manifest must admit regardless of presentation order: %v", err)
	}
	missing := append([]V41FilePin(nil), p.WeightShards[:46]...)
	if err := AdmitV41ArtifactAgainstProfile(p, missing); err == nil || !errors.Is(err, ErrV41ArtifactProfile) {
		t.Fatalf("missing shard must refuse with %v, got %v", ErrV41ArtifactProfile, err)
	}
	extra := append(append([]V41FilePin(nil), p.WeightShards...), p.WeightShards[0])
	if err := AdmitV41ArtifactAgainstProfile(p, extra); err == nil || !errors.Is(err, ErrV41ArtifactProfile) {
		t.Fatalf("extra shard must refuse with %v, got %v", ErrV41ArtifactProfile, err)
	}
	drift := append([]V41FilePin(nil), p.WeightShards...)
	drift[0].Size++
	if err := AdmitV41ArtifactAgainstProfile(p, drift); err == nil || !errors.Is(err, ErrV41ArtifactProfile) {
		t.Fatalf("size drift must refuse with %v, got %v", ErrV41ArtifactProfile, err)
	}
}

// Same-name/same-size pre-1017 shards under the 1017 profile must fail closed:
// the content identity differs even though the shape matches. This is the
// silent-pairing guard for the Engram payload and for ordinary dense shards.
func TestV41AdmitArtifactRejectsOldReleaseShardsUnder1017(t *testing.T) {
	t.Parallel()
	p := &DeepSeekV41Flash1017Artifact
	old := append([]V41FilePin(nil), p.WeightShards...)
	// dba1be0a Engram shards 47/48: identical names and sizes, different bytes.
	old[46] = V41FilePin{Name: old[46].Name, ID: "824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f", Size: old[46].Size, LFS: true}
	old[47] = V41FilePin{Name: old[47].Name, ID: "976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed", Size: old[47].Size, LFS: true}
	err := AdmitV41ArtifactAgainstProfile(p, old)
	if err == nil || !errors.Is(err, ErrV41ArtifactProfile) {
		t.Fatalf("old-release Engram shards must refuse under the 1017 profile, got %v", err)
	}
	if !strings.Contains(err.Error(), "model-00047-of-00048.safetensors") {
		t.Fatalf("refusal must name the offending Engram shard: %v", err)
	}
	foreign := append([]V41FilePin(nil), p.WeightShards...)
	// dba1be0a dense shard 1: same name and size, different bytes.
	foreign[0] = V41FilePin{Name: foreign[0].Name, ID: "886aebdafa08cc27bbae2165ed35bdfe0de9370bf88c1411283c155c6ae4ff89", Size: foreign[0].Size, LFS: true}
	if err := AdmitV41ArtifactAgainstProfile(p, foreign); err == nil || !errors.Is(err, ErrV41ArtifactProfile) {
		t.Fatalf("old-release dense shard must refuse under the 1017 profile, got %v", err)
	}
}

// Cache/trace provenance must distinguish releases: the provenance tag and the
// full-content digest both separate the 1017 profile from the pre-1017 release,
// stay stable across calls, and the pre-1017 profile remains valid against its
// own manifest (the old profile tests keep their meaning).
func TestV41ReleaseProvenanceDistinguishesReleases(t *testing.T) {
	t.Parallel()
	newer := &DeepSeekV41Flash1017Artifact
	older := *newer
	older.TargetRepository = DeepSeekV41FlashModelID
	older.TargetRevision = DeepSeekV41FlashRevision
	older.WeightShards = append([]V41FilePin(nil), newer.WeightShards...)
	older.WeightShards[46].ID = "824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f"
	older.WeightShards[47].ID = "976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed"
	if err := older.Validate(); err != nil {
		t.Fatalf("pre-1017 profile must validate in its own right: %v", err)
	}
	if newer.ReleaseProvenance() == older.ReleaseProvenance() {
		t.Fatalf("provenance tag aliases across releases: %s", newer.ReleaseProvenance())
	}
	nd, err := newer.ProvenanceDigest()
	if err != nil {
		t.Fatal(err)
	}
	od, err := older.ProvenanceDigest()
	if err != nil {
		t.Fatal(err)
	}
	if nd == od {
		t.Fatalf("provenance digest aliases across releases: %s", nd)
	}
	again, err := newer.ProvenanceDigest()
	if err != nil || again != nd {
		t.Fatalf("provenance digest unstable: %q vs %q (err=%v)", again, nd, err)
	}
	if err := AdmitV41ArtifactAgainstProfile(&older, older.WeightShards); err != nil {
		t.Fatalf("pre-1017 manifest must still admit against its own profile: %v", err)
	}
}
