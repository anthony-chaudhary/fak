package model

import (
	"bytes"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func denseSpanCfg() Config {
	return Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 97, RMSNormEps: 1e-5, RopeTheta: 10000,
		RopeThetaPerLayer: []float64{10000, 1000000}, BlockTopology: SandwichNorm,
	}
}

func TestSerializeSpanRoundTripsMiddleSpanAndContinuationBitExact(t *testing.T) {
	cfg := denseSpanCfg()
	m := NewSynthetic(cfg)
	control, victim := m.NewSession(), m.NewSession()
	ids := []int{3, 17, 5, 23, 41, 2, 19}
	control.Prefill(ids)
	victim.Prefill(ids)
	want := control.Cache.Clone()

	blob, err := victim.Cache.SerializeSpan(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if removed := victim.Cache.Evict(2, 3); removed != 3 {
		t.Fatalf("removed=%d", removed)
	}
	if restored, err := victim.Cache.RestoreSpan(blob); err != nil || restored != 3 {
		t.Fatalf("restored=%d err=%v", restored, err)
	}
	assertSpanCacheEqual(t, victim.Cache, want)
	for _, token := range []int{29, 31, 37} {
		if got, expected := victim.Step(token), control.Step(token); !reflect.DeepEqual(got, expected) {
			t.Fatalf("continuation logits differ for token %d", token)
		}
	}
}

func TestRestoreSpanRefusesStaleCorruptForeignWithoutMutation(t *testing.T) {
	cfg := denseSpanCfg()
	m := NewSynthetic(cfg)
	ids := []int{3, 17, 5, 23, 41, 2, 19}
	makeEvicted := func() (*Session, []byte) {
		s := m.NewSession()
		s.Prefill(ids)
		blob, err := s.Cache.SerializeSpan(2, 3)
		if err != nil {
			t.Fatal(err)
		}
		s.Cache.Evict(2, 3)
		return s, blob
	}

	stale, blob := makeEvicted()
	stale.Step(11)
	before := stale.Cache.Clone()
	if _, err := stale.Cache.RestoreSpan(blob); err == nil {
		t.Fatal("restore after Step succeeded")
	}
	assertSpanCacheEqual(t, stale.Cache, before)

	cases := map[string]func([]byte) []byte{
		"corrupt":   func(b []byte) []byte { b[len(b)/2] ^= 1; return b },
		"truncated": func(b []byte) []byte { return b[:len(b)-1] },
		"trailing":  func(b []byte) []byte { return append(b, 0) },
		"legacy":    func([]byte) []byte { return []byte{'L', '3', 'K', 'V', 0, 0, 0, 0} },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			s, clean := makeEvicted()
			before := s.Cache.Clone()
			if _, err := s.Cache.RestoreSpan(mutate(append([]byte(nil), clean...))); err == nil {
				t.Fatal("invalid image restored")
			}
			assertSpanCacheEqual(t, s.Cache, before)
		})
	}

	wrong, clean := makeEvicted()
	wrong.Cache.cfg.RopeTheta++
	before = wrong.Cache.Clone()
	if _, err := wrong.Cache.RestoreSpan(clean); err == nil {
		t.Fatal("wrong config restored")
	}
	assertSpanCacheEqual(t, wrong.Cache, before)
}

func TestKVBackendExposesSpanStageAndRestoreCapabilities(t *testing.T) {
	s := NewSynthetic(denseSpanCfg()).NewSession()
	s.Prefill([]int{3, 17, 5, 23, 41})
	kb, ok := KVBackend(s)
	if !ok {
		t.Fatal("KVBackend ok=false")
	}
	st, ok := kb.(interface {
		StageSpanBytes(int, int) ([]byte, error)
	})
	if !ok {
		t.Fatal("missing StageSpanBytes")
	}
	rst, ok := kb.(interface{ RestoreSpanBytes([]byte) (int, error) })
	if !ok {
		t.Fatal("missing RestoreSpanBytes")
	}
	blob, err := st.StageSpanBytes(1, 2)
	if err != nil || len(blob) == 0 {
		t.Fatalf("stage bytes=%d err=%v", len(blob), err)
	}
	s.Cache.Evict(1, 2)
	if restored, err := rst.RestoreSpanBytes(blob); err != nil || restored != 2 {
		t.Fatalf("restore=%d err=%v", restored, err)
	}
}

func TestKVBackendSpanBytesRefuseHALDeviceOwnershipWithoutMutation(t *testing.T) {
	cfg := denseSpanCfg()
	m := NewSynthetic(cfg)
	ids := []int{3, 17, 5, 23, 41}
	host := m.NewSession()
	host.Prefill(ids)
	blob, err := host.Cache.SerializeSpan(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	deviceCfg := cfg
	deviceCfg.BlockTopology = PreNorm
	deviceCfg.RopeThetaPerLayer = nil
	device, err := NewSynthetic(deviceCfg).NewBackendSessionChecked(compute.Default())
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	device.Prefill(ids)
	hostBefore, err := spanCacheDigest(device.Cache)
	if err != nil {
		t.Fatal(err)
	}
	deviceBefore, err := compute.SnapshotKVToHost(device.halKV)
	if err != nil {
		t.Fatal(err)
	}
	hostLen, deviceLen := device.Cache.Len(), device.halKV.Len()
	kb, ok := KVBackend(device)
	if !ok {
		t.Fatal("KVBackend ok=false")
	}
	stage := kb.(interface {
		StageSpanBytes(int, int) ([]byte, error)
	})
	restore := kb.(interface{ RestoreSpanBytes([]byte) (int, error) })
	if got, err := stage.StageSpanBytes(0, 1); err == nil || got != nil {
		t.Fatalf("HAL StageSpanBytes bytes=%d err=%v, want nil/refusal", len(got), err)
	}
	if got, err := restore.RestoreSpanBytes(blob); err == nil || got != 0 {
		t.Fatalf("HAL RestoreSpanBytes positions=%d err=%v, want zero/refusal", got, err)
	}
	deviceAfter, err := compute.SnapshotKVToHost(device.halKV)
	if err != nil {
		t.Fatal(err)
	}
	if device.Cache.Len() != hostLen || device.halKV.Len() != deviceLen {
		t.Fatalf("HAL refusal changed cache lengths: host=%d/%d device=%d/%d", device.Cache.Len(), hostLen, device.halKV.Len(), deviceLen)
	}
	hostAfter, err := spanCacheDigest(device.Cache)
	if err != nil {
		t.Fatal(err)
	}
	if hostAfter != hostBefore {
		t.Fatal("HAL refusal changed host cache state")
	}
	if !reflect.DeepEqual(deviceAfter, deviceBefore) {
		t.Fatal("HAL refusal changed device cache state")
	}

	var nilBackend kvBackend
	if _, err := nilBackend.StageSpanBytes(0, 1); err == nil {
		t.Fatal("nil StageSpanBytes succeeded")
	}
	if _, err := nilBackend.RestoreSpanBytes(blob); err == nil {
		t.Fatal("nil RestoreSpanBytes succeeded")
	}
}

func TestKVBackendSpanBytesHonorCacheGeometryLock(t *testing.T) {
	s := NewSynthetic(denseSpanCfg()).NewSession()
	s.Prefill([]int{3, 17, 5, 23, 41})
	kb, _ := KVBackend(s)
	stage := kb.(interface {
		StageSpanBytes(int, int) ([]byte, error)
	})
	restore := kb.(interface{ RestoreSpanBytes([]byte) (int, error) })

	s.cacheGeometryMu.Lock()
	stageStarted := make(chan struct{})
	stageDone := make(chan struct {
		blob []byte
		err  error
	}, 1)
	go func() {
		close(stageStarted)
		blob, err := stage.StageSpanBytes(1, 2)
		stageDone <- struct {
			blob []byte
			err  error
		}{blob, err}
	}()
	<-stageStarted
	select {
	case got := <-stageDone:
		s.cacheGeometryMu.Unlock()
		t.Fatalf("stage escaped geometry write lock: bytes=%d err=%v", len(got.blob), got.err)
	case <-time.After(100 * time.Millisecond):
	}
	s.cacheGeometryMu.Unlock()
	staged := <-stageDone
	if staged.err != nil || len(staged.blob) == 0 {
		t.Fatalf("stage after unlock bytes=%d err=%v", len(staged.blob), staged.err)
	}
	s.Cache.Evict(1, 2)

	s.cacheGeometryMu.RLock()
	restoreStarted := make(chan struct{})
	restoreDone := make(chan struct {
		positions int
		err       error
	}, 1)
	go func() {
		close(restoreStarted)
		positions, err := restore.RestoreSpanBytes(staged.blob)
		restoreDone <- struct {
			positions int
			err       error
		}{positions, err}
	}()
	<-restoreStarted
	select {
	case got := <-restoreDone:
		s.cacheGeometryMu.RUnlock()
		t.Fatalf("restore escaped geometry read lock: positions=%d err=%v", got.positions, got.err)
	case <-time.After(100 * time.Millisecond):
	}
	s.cacheGeometryMu.RUnlock()
	restored := <-restoreDone
	if restored.err != nil || restored.positions != 2 {
		t.Fatalf("restore after unlock positions=%d err=%v", restored.positions, restored.err)
	}
}

func TestSerializeSpanRejectsBadRangeAndHybrid(t *testing.T) {
	s := NewSynthetic(denseSpanCfg()).NewSession()
	s.Prefill([]int{1, 2, 3})
	for _, tc := range [][2]int{{2, 5}, {-1, 1}, {0, 0}} {
		if _, err := s.Cache.SerializeSpan(tc[0], tc[1]); err == nil {
			t.Fatalf("SerializeSpan%v succeeded", tc)
		}
	}
	hybrid := NewSynthetic(qwen35HybridTestCfg()).NewSession()
	hybrid.Prefill([]int{1, 2, 3})
	if _, err := hybrid.Cache.SerializeSpan(0, 1); err == nil {
		t.Fatal("hybrid span serialized")
	}
}

func assertSpanCacheEqual(t *testing.T, got, want *KVCache) {
	t.Helper()
	if !reflect.DeepEqual(got.pos, want.pos) || !reflect.DeepEqual(got.lineage, want.lineage) ||
		!reflect.DeepEqual(got.K, want.K) || !reflect.DeepEqual(got.Kraw, want.Kraw) || !reflect.DeepEqual(got.V, want.V) {
		t.Fatal("cache changed or failed exact restoration")
	}
}

func TestSpanSerializationDeterministic(t *testing.T) {
	s := NewSynthetic(denseSpanCfg()).NewSession()
	s.Prefill([]int{3, 17, 5, 23, 41})
	a, _ := s.Cache.SerializeSpan(1, 2)
	b, _ := s.Cache.SerializeSpan(1, 2)
	if !bytes.Equal(a, b) {
		t.Fatal("serialization is not deterministic")
	}
}
