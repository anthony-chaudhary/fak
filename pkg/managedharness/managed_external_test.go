package managedharness_test

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	mh "github.com/anthony-chaudhary/fak/pkg/managedharness"
)

func TestRetentionSetsIncludeEveryRootKind(t *testing.T) {
	inst := mh.Installation{
		Effective:     "generation-current",
		LastKnownGood: "generation-last-good",
		Generations: []mh.GenerationID{
			"generation-unpinned",
			"generation-checkpoint",
			"generation-session",
			"generation-staged",
			"generation-last-good",
			"generation-current",
		},
		Pins: []mh.GenerationPin{
			{Kind: mh.PinStagedCandidate, Generation: "generation-staged"},
			{Kind: mh.PinOpenSession, Reference: "session-1", Generation: "generation-session"},
			{Kind: mh.PinCheckpoint, Reference: "checkpoint-1", Generation: "generation-checkpoint"},
		},
	}
	wantRetained := []mh.GenerationID{
		"generation-checkpoint",
		"generation-current",
		"generation-last-good",
		"generation-session",
		"generation-staged",
	}
	if got := mh.RetainedGenerations(inst); !reflect.DeepEqual(got, wantRetained) {
		t.Fatalf("retained generations = %v, want %v", got, wantRetained)
	}
	if got, want := mh.ReclaimableGenerations(inst), []mh.GenerationID{"generation-unpinned"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("reclaimable generations = %v, want %v", got, want)
	}
}

func TestGarbageCollectPreservesEveryRootAndReportsBytes(t *testing.T) {
	root := t.TempDir()
	s, err := mh.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	bundles, generations := installGenerations(t, s, 6)
	pins := []mh.GenerationPin{
		{Kind: mh.PinStagedCandidate, Generation: generations[3]},
		{Kind: mh.PinOpenSession, Reference: "session-old", Generation: generations[2]},
		{Kind: mh.PinCheckpoint, Reference: "checkpoint-1", Generation: generations[1]},
	}
	for _, pin := range pins {
		if receipt, err := s.Pin("local", pin); err != nil || receipt.Status != "pinned" {
			t.Fatalf("pin %+v: %+v %v", pin, receipt, err)
		}
	}
	state, err := s.Inspect("local")
	if err != nil {
		t.Fatal(err)
	}
	wantRetained := mh.RetainedGenerations(state)
	wantReclaimed := mh.ReclaimableGenerations(state)
	if !reflect.DeepEqual(wantReclaimed, []mh.GenerationID{generations[0]}) {
		t.Fatalf("fixture reclaimable generations = %v, want only %s", wantReclaimed, generations[0])
	}
	var wantRetainedBytes int64
	for _, generation := range wantRetained {
		info, err := os.Stat(generationArtifact(root, "local", generation))
		if err != nil {
			t.Fatal(err)
		}
		wantRetainedBytes += info.Size()
	}
	reclaimedInfo, err := os.Stat(generationArtifact(root, "local", generations[0]))
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := s.GarbageCollect("local")
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Status != "completed" || !reflect.DeepEqual(receipt.Retained, wantRetained) || !reflect.DeepEqual(receipt.Reclaimed, wantReclaimed) {
		t.Fatalf("GC receipt identities: %+v", receipt)
	}
	if receipt.RetainedBytes != wantRetainedBytes || receipt.ReclaimedBytes != reclaimedInfo.Size() {
		t.Fatalf("GC receipt bytes: %+v, want retained=%d reclaimed=%d", receipt, wantRetainedBytes, reclaimedInfo.Size())
	}
	if _, err := os.Stat(generationArtifact(root, "local", generations[0])); !os.IsNotExist(err) {
		t.Fatalf("reclaimed generation artifact still exists: %v", err)
	}
	for _, generation := range wantRetained {
		if _, err := os.Stat(generationArtifact(root, "local", generation)); err != nil {
			t.Fatalf("retained generation %s disappeared: %v", generation, err)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "releases", string(bundles[0].Release.ID)+".json")); err != nil {
		t.Fatalf("shared release artifact was deleted: %v", err)
	}
	after, err := s.Inspect("local")
	if err != nil || after.LastKnownGood != generations[4] || len(mh.ReclaimableGenerations(after)) != 0 {
		t.Fatalf("post-GC rollback state: %+v %v", after, err)
	}
	bad := release(t, product("managed", "local", "v1", []string{"offline-work"}, []string{"kernel"}), "bad health")
	if _, err := s.Publish(bad); err != nil {
		t.Fatal(err)
	}
	rolledBack, err := s.Update("local", bad.Release.ID, func(mh.Bundle) error { return errors.New("health failed") })
	if err != nil || rolledBack.Status != "rolled_back" || rolledBack.After != after.Effective {
		t.Fatalf("rollback after GC: %+v %v", rolledBack, err)
	}
}

func TestSessionReleaseMakesGenerationReclaimable(t *testing.T) {
	root := t.TempDir()
	s, err := mh.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	_, generations := installGenerations(t, s, 3)
	if _, err := s.Pin("local", mh.GenerationPin{Kind: mh.PinOpenSession, Reference: "session-1", Generation: generations[0]}); err != nil {
		t.Fatal(err)
	}
	first, err := s.GarbageCollect("local")
	if err != nil || len(first.Reclaimed) != 0 {
		t.Fatalf("open session was not retained: %+v %v", first, err)
	}
	closed, err := s.Unpin("local", mh.PinOpenSession, "session-1")
	if err != nil || closed.Status != "released" || closed.Before != generations[0] {
		t.Fatalf("close session: %+v %v", closed, err)
	}
	second, err := s.GarbageCollect("local")
	if err != nil || !reflect.DeepEqual(second.Reclaimed, []mh.GenerationID{generations[0]}) {
		t.Fatalf("closed session generation not reclaimed: %+v %v", second, err)
	}
	state, err := s.Inspect("local")
	if err != nil || state.LastKnownGood != generations[1] {
		t.Fatalf("rollback root changed after session close: %+v %v", state, err)
	}
}

func TestGarbageCollectDeletionFailureKeepsGenerationRecord(t *testing.T) {
	root := t.TempDir()
	s, err := mh.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	_, generations := installGenerations(t, s, 3)
	blocked := generationArtifact(root, "local", generations[0])
	if err := os.Remove(blocked); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(blocked, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(blocked, "blocker"), []byte("not owned by GC"), 0600); err != nil {
		t.Fatal(err)
	}
	receipt, err := s.GarbageCollect("local")
	if err == nil || receipt.Status != "partial" || !reflect.DeepEqual(receipt.Failed, []mh.GenerationID{generations[0]}) {
		t.Fatalf("failed delete receipt: %+v %v", receipt, err)
	}
	state, inspectErr := s.Inspect("local")
	if inspectErr != nil || !hasGeneration(state.Generations, generations[0]) {
		t.Fatalf("failed deletion dropped generation record: %+v %v", state, inspectErr)
	}
	for _, generation := range generations[1:] {
		if _, err := os.Stat(generationArtifact(root, "local", generation)); err != nil {
			t.Fatalf("live generation %s disappeared after failed deletion: %v", generation, err)
		}
	}
}

func TestConcurrentPinAndGarbageCollectAreLinearizable(t *testing.T) {
	root := t.TempDir()
	s, err := mh.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	peer, err := mh.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	_, generations := installGenerations(t, s, 3)
	target := generations[0]
	start := make(chan struct{})
	pinResult := make(chan error, 1)
	type gcResult struct {
		receipt mh.GCReceipt
		err     error
	}
	gcDone := make(chan gcResult, 1)
	go func() {
		<-start
		_, err := peer.Pin("local", mh.GenerationPin{Kind: mh.PinCheckpoint, Reference: "concurrent", Generation: target})
		pinResult <- err
	}()
	go func() {
		<-start
		receipt, err := s.GarbageCollect("local")
		gcDone <- gcResult{receipt: receipt, err: err}
	}()
	close(start)
	pinErr := <-pinResult
	gc := <-gcDone
	if gc.err != nil {
		t.Fatalf("concurrent GC: %+v %v", gc.receipt, gc.err)
	}
	_, artifactErr := os.Stat(generationArtifact(root, "local", target))
	if pinErr == nil {
		if artifactErr != nil || hasGeneration(gc.receipt.Reclaimed, target) {
			t.Fatalf("successful concurrent pin disappeared: receipt=%+v artifact=%v", gc.receipt, artifactErr)
		}
		state, err := s.Inspect("local")
		if err != nil || !hasGeneration(mh.RetainedGenerations(state), target) {
			t.Fatalf("successful concurrent pin absent from state: %+v %v", state, err)
		}
		return
	}
	if !os.IsNotExist(artifactErr) || !hasGeneration(gc.receipt.Reclaimed, target) {
		t.Fatalf("failed pin did not linearize after GC: pin=%v receipt=%+v artifact=%v", pinErr, gc.receipt, artifactErr)
	}
}

func installGenerations(t *testing.T, s *mh.Store, count int) ([]mh.Bundle, []mh.GenerationID) {
	t.Helper()
	p := product("managed", "local", "v1", []string{"offline-work"}, []string{"kernel"})
	bundles := make([]mh.Bundle, 0, count)
	generations := make([]mh.GenerationID, 0, count)
	for n := 0; n < count; n++ {
		bundle := release(t, p, fmt.Sprintf("generation %d", n+1))
		if _, err := s.Publish(bundle); err != nil {
			t.Fatal(err)
		}
		var receipt mh.Receipt
		var err error
		if n == 0 {
			receipt, err = s.Install("local", bundle.Release.ID, nil)
		} else {
			receipt, err = s.Update("local", bundle.Release.ID, nil)
		}
		if err != nil || receipt.Status != "activated" {
			t.Fatalf("activate generation %d: %+v %v", n+1, receipt, err)
		}
		bundles = append(bundles, bundle)
		generations = append(generations, receipt.After)
	}
	return bundles, generations
}

func generationArtifact(root string, installation mh.InstallationID, generation mh.GenerationID) string {
	return filepath.Join(root, "installations", string(installation), "generations", string(generation)+".json")
}

func hasGeneration(generations []mh.GenerationID, target mh.GenerationID) bool {
	for _, generation := range generations {
		if generation == target {
			return true
		}
	}
	return false
}

func product(id, variant, compat string, caps, layers []string) mh.Product {
	return mh.Product{ID: mh.ProductID(id), Variant: variant, Compatibility: compat, Capabilities: caps, Layers: layers}
}
func release(t *testing.T, p mh.Product, reply string) mh.Bundle {
	t.Helper()
	b, err := mh.BuildRelease(p, map[string]any{"offline_reply": reply}, mh.Provenance{Source: "fixture", Revision: "r1", Builder: "test"})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestLocalLifecycleTwoVariants(t *testing.T) {
	s, err := mh.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	base := []string{"kernel", "offline-work"}
	selfV1 := release(t, product("fak-self", "self", "v1", []string{"private-context", "offline-work"}, append(base, "private-layer")), "self v1")
	publicV1 := release(t, product("public-safe-project", "public", "v1", []string{"offline-work"}, append(base, "public-layer")), "public v1")
	for _, b := range []mh.Bundle{selfV1, publicV1} {
		if _, err := s.Publish(b); err != nil {
			t.Fatal(err)
		}
	}
	health := func(mh.Bundle) error { return nil }
	if r, err := s.Install("self-local", selfV1.Release.ID, health); err != nil || r.Status != "activated" {
		t.Fatalf("self install: %+v %v", r, err)
	}
	if r, err := s.Install("public-local", publicV1.Release.ID, health); err != nil || r.Status != "activated" {
		t.Fatalf("public install: %+v %v", r, err)
	}
	work := func(b mh.Bundle) (string, error) { return string(b.Payload), nil }
	for _, id := range []mh.InstallationID{"self-local", "public-local"} {
		r, err := s.Run(id, work)
		if err != nil || r.Status != "completed" || !strings.Contains(r.Output, "offline_reply") {
			t.Fatalf("work %s: %+v %v", id, r, err)
		}
	}
	pub, _ := s.Run("public-local", work)
	if contains(pub.Capabilities, "private-context") {
		t.Fatalf("private capability leaked: %+v", pub)
	}

	selfV2 := release(t, product("fak-self", "self", "v1", []string{"private-context", "offline-work"}, append(base, "private-layer")), "self v2")
	if _, err := s.Publish(selfV2); err != nil {
		t.Fatal(err)
	}
	updated, err := s.Update("self-local", selfV2.Release.ID, health)
	if err != nil || updated.Status != "activated" || updated.LastKnownGood == "" {
		t.Fatalf("update: %+v %v", updated, err)
	}

	badHealth := release(t, product("fak-self", "self", "v1", []string{"offline-work"}, base), "unhealthy")
	s.Publish(badHealth)
	rolled, err := s.Update("self-local", badHealth.Release.ID, func(mh.Bundle) error { return errors.New("selfcheck failed") })
	if err != nil || rolled.Status != "rolled_back" || rolled.After != updated.After {
		t.Fatalf("rollback: %+v %v", rolled, err)
	}
	state, _ := s.Inspect("self-local")
	if state.Effective != updated.After || state.Desired != selfV2.Release.ID {
		t.Fatalf("failed update mutated state: %+v", state)
	}

	incompatible := release(t, product("fak-self", "self", "v2", []string{"offline-work"}, base), "future")
	s.Publish(incompatible)
	refused, err := s.Update("self-local", incompatible.Release.ID, health)
	if err != nil || refused.Status != "refused" || refused.After != updated.After {
		t.Fatalf("refusal: %+v %v", refused, err)
	}
}

func TestDeterministicReleaseAndSecretRefusal(t *testing.T) {
	p := product("p", "v", "v1", []string{"b", "a", "a"}, []string{"z", "a"})
	a := release(t, p, "ok")
	b := release(t, p, "ok")
	if a.Release.Digest != b.Release.Digest || a.Release.ID != b.Release.ID {
		t.Fatal("release serialization is nondeterministic")
	}
	secret, err := mh.BuildRelease(p, map[string]any{"token": "do-not-package"}, mh.Provenance{Source: "fixture", Revision: "r1"})
	if err != nil {
		t.Fatal(err)
	}
	s, _ := mh.Open(t.TempDir())
	if _, err = s.Publish(secret); err == nil {
		t.Fatal("release containing installation secret accepted")
	}
}

func contains(in []string, w string) bool {
	for _, v := range in {
		if v == w {
			return true
		}
	}
	return false
}

func BenchmarkGenerationHashing(b *testing.B) {
	prod := mh.Product{
		ID:            "bench-product",
		Variant:       "standard",
		Compatibility: "v1",
		Capabilities:  []string{"inference", "memory", "tool-use"},
		Layers:        []string{"base", "security", "cache"},
	}
	payload := map[string]any{
		"model":       "qwen3.8",
		"context_len": 32768,
		"quant":       "q4_k_m",
		"parameters": map[string]any{
			"temperature": 0.7,
			"top_p":       0.95,
		},
	}
	prov := mh.Provenance{
		Source:   "git@github.com:anthony-chaudhary/fak.git",
		Revision: "r100+gabcdef12",
		Builder:  "go-test",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundle, err := mh.BuildRelease(prod, payload, prov)
		if err != nil {
			b.Fatalf("BuildRelease failed: %v", err)
		}
		if bundle.Release.Digest == "" {
			b.Fatal("empty release digest")
		}
	}
}

func BenchmarkProductLifecycle(b *testing.B) {
	root := b.TempDir()
	s, err := mh.Open(root)
	if err != nil {
		b.Fatal(err)
	}
	prod := mh.Product{
		ID:            "bench-lifecycle",
		Variant:       "server",
		Compatibility: "v1",
		Capabilities:  []string{"offline-work"},
		Layers:        []string{"kernel"},
	}
	rel1, err := mh.BuildRelease(prod, map[string]any{"version": 1}, mh.Provenance{Source: "bench", Revision: "r1", Builder: "bench"})
	if err != nil {
		b.Fatal(err)
	}
	rel2, err := mh.BuildRelease(prod, map[string]any{"version": 2}, mh.Provenance{Source: "bench", Revision: "r2", Builder: "bench"})
	if err != nil {
		b.Fatal(err)
	}
	if _, err := s.Publish(rel1); err != nil {
		b.Fatal(err)
	}
	if _, err := s.Publish(rel2); err != nil {
		b.Fatal(err)
	}

	work := func(bundle mh.Bundle) (string, error) {
		return string(bundle.Payload), nil
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := mh.InstallationID(fmt.Sprintf("inst-%d", i))
		instReceipt, err := s.Install(id, rel1.Release.ID, nil)
		if err != nil || instReceipt.Status != "activated" {
			b.Fatalf("Install failed: receipt=%+v, err=%v", instReceipt, err)
		}
		updReceipt, err := s.Update(id, rel2.Release.ID, nil)
		if err != nil || updReceipt.Status != "activated" {
			b.Fatalf("Update failed: receipt=%+v, err=%v", updReceipt, err)
		}
		runReceipt, err := s.Run(id, work)
		if err != nil || runReceipt.Status != "completed" {
			b.Fatalf("Run failed: receipt=%+v, err=%v", runReceipt, err)
		}
		state, err := s.Inspect(id)
		if err != nil || state.Effective != updReceipt.After {
			b.Fatalf("Inspect failed: state=%+v, err=%v", state, err)
		}
	}
}

func BenchmarkRetainedGenerations(b *testing.B) {
	inst := mh.Installation{
		ID:            "bench-inst",
		Effective:     "gen-10",
		LastKnownGood: "gen-08",
		Generations: []mh.GenerationID{
			"gen-01", "gen-02", "gen-03", "gen-04", "gen-05",
			"gen-06", "gen-07", "gen-08", "gen-09", "gen-10",
		},
		Pins: []mh.GenerationPin{
			{Kind: mh.PinStagedCandidate, Generation: "gen-09"},
			{Kind: mh.PinOpenSession, Reference: "sess-1", Generation: "gen-05"},
			{Kind: mh.PinCheckpoint, Reference: "ckpt-1", Generation: "gen-03"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		retained := mh.RetainedGenerations(inst)
		if len(retained) == 0 {
			b.Fatal("unexpected empty retained generations")
		}
	}
}

func BenchmarkReclaimableGenerations(b *testing.B) {
	inst := mh.Installation{
		ID:            "bench-inst",
		Effective:     "gen-10",
		LastKnownGood: "gen-08",
		Generations: []mh.GenerationID{
			"gen-01", "gen-02", "gen-03", "gen-04", "gen-05",
			"gen-06", "gen-07", "gen-08", "gen-09", "gen-10",
		},
		Pins: []mh.GenerationPin{
			{Kind: mh.PinStagedCandidate, Generation: "gen-09"},
			{Kind: mh.PinOpenSession, Reference: "sess-1", Generation: "gen-05"},
			{Kind: mh.PinCheckpoint, Reference: "ckpt-1", Generation: "gen-03"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reclaimable := mh.ReclaimableGenerations(inst)
		if len(reclaimable) == 0 {
			b.Fatal("unexpected empty reclaimable generations")
		}
	}
}
