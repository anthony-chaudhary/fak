package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41CompressorNormAssertPublications(t *testing.T, got, want *v41ForwardState) {
	t.Helper()
	if !reflect.DeepEqual(got.history, want.history) {
		t.Fatal("source/reader history differs")
	}
	for layer := range got.layers {
		a, b := got.layerState(layer), want.layerState(layer)
		rows, rowOK := a.KVSourceRows(layer)
		refRows, refRowOK := b.KVSourceRows(layer)
		if rowOK != refRowOK || !reflect.DeepEqual(rows, refRows) {
			t.Fatal("exact source KV cache publication changed")
		}
		if !reflect.DeepEqual(a.kvPublications, b.kvPublications) || !reflect.DeepEqual(a.indexPublications, b.indexPublications) {
			t.Fatal("layer-owned publication identity or exact latent values changed")
		}
		if !reflect.DeepEqual(a.partialPositions, b.partialPositions) || len(a.partialInputs) != len(b.partialInputs) || a.nextCompressRow != b.nextCompressRow || a.nextWindowPos != b.nextWindowPos {
			t.Fatal("retained carrier ownership/cursors differ")
		}
		for row := range a.partialInputs {
			v41CompressorTestFiniteParity(t, a.partialInputs[row], b.partialInputs[row], "retained compressor carrier")
		}
	}
	if !reflect.DeepEqual(got.attn.kvPublications, want.attn.kvPublications) || !reflect.DeepEqual(got.attn.indexPublications, want.attn.indexPublications) {
		t.Fatal("shared registry changed source/group identities or exact latents")
	}
	keys, keyOK := got.attn.IndexKeys(0)
	refKeys, refKeyOK := want.attn.IndexKeys(0)
	if keyOK != refKeyOK || !reflect.DeepEqual(keys, refKeys) {
		t.Fatal("exact normalized index-key publication changed")
	}
	rows, ratio, ok := got.attn.TopK()
	refRows, refRatio, refOK := want.attn.TopK()
	if ok != refOK || ratio != refRatio || !reflect.DeepEqual(rows, refRows) {
		t.Fatal("source/reader top-k identity changed")
	}
	mask, maskRatio, maskOK := got.attn.Candidates()
	refMask, refMaskRatio, refMaskOK := want.attn.Candidates()
	if maskOK != refMaskOK || maskRatio != refMaskRatio || !reflect.DeepEqual(mask, refMask) {
		t.Fatal("source/reader candidate identity changed")
	}
	if !reflect.DeepEqual(got.attn.kvPublishedEnd, want.attn.kvPublishedEnd) || !reflect.DeepEqual(got.attn.indexPublishedEnd, want.attn.indexPublishedEnd) {
		t.Fatal("publication source/group ranges changed")
	}
}

// Real Session entry points with CPU-recording arithmetic. Both exact latent
// publications and reader selections are checked independently of final logits.
// Neither the ratio-two shared reader nor the ratio-zero layer owns a compressor.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41CompressorNormSession(t *testing.T) {
	t.Parallel()
	for _, readerRatio := range []int{0, 2} {
		for _, prefixLength := range []int{1, 2, 3} {
			t.Run(itoa(readerRatio)+"/prefix-"+itoa(prefixLength), func(t *testing.T) {
				m, reference := v41CompressorTestFixture(t), v41CompressorTestFixture(t)
				m.Cfg.DeepSeekV41.CompressRatios[1] = readerRatio
				reference.Cfg.DeepSeekV41.CompressRatios[1] = readerRatio
				s, b := v41CompressorNormTestSession(t, m)
				t.Cleanup(func() {
					if err := reference.CloseWeights(); err != nil {
						t.Error(err)
					}
				})
				host := reference.NewSession()
				t.Cleanup(host.Close)
				if s.v41State().compressorNorm == nil {
					t.Fatal("qualified session did not bind compressor normalization")
				}
				// Malformed IDs still refuse; continuation below must use the
				// reduced fixture's vocabulary rather than token positions.
				if err := recoverError(func() { s.Prefill([]int{m.Cfg.VocabSize}) }); !errors.Is(err, ErrV41ForwardStage) || len(s.v41State().history) != 0 || len(b.records) != 0 {
					t.Fatal("out-of-range token did not refuse before compressor work")
				}
				producers := 1
				check := func(tokens int) {
					t.Helper()
					wantCalls := producers * (tokens / 2)
					if len(b.records) != wantCalls || b.readbacks != wantCalls || b.uploadBytes != wantCalls*4*v41CompressorWidth(m.Cfg) || b.readBytes != b.uploadBytes || len(b.outputs) != 0 {
						t.Fatalf("completed-group normalization/transfer count changed: calls=%d reads=%d want=%d", len(b.records), b.readbacks, wantCalls)
					}
					counts := make([]int, 2)
					for _, record := range b.records {
						counts[record.layer]++
					}
					if counts[0] != tokens/2 || counts[1] != 0 {
						t.Fatal("reader reuse normalized or projected a private compressed row")
					}
					v41CompressorNormAssertPublications(t, s.v41Forward, host.v41Forward)
				}
				prefix := []int{1, 2, 3}[:prefixLength]
				v41CompressorTestFiniteParity(t, s.Prefill(prefix), host.Prefill(prefix), "compressor normalization cold prefix")
				check(prefixLength)
				v41CompressorTestFiniteParity(t, s.Step(4), host.Step(4), "compressor normalization decode")
				check(prefixLength + 1)
				v41CompressorTestFiniteParity(t, s.Prefill([]int{5, 6}), host.Prefill([]int{5, 6}), "compressor normalization suffix prefill")
				check(prefixLength + 3)

				before := captureV41ForwardSnapshot(s.v41Forward)
				dataOnly := before.clone().restore()
				if dataOnly.compressorNorm != nil || dataOnly.callbackOwner != nil || !reflect.DeepEqual(captureV41ForwardSnapshot(dataOnly), before) {
					t.Fatal("snapshot persisted a callback or changed continuation values")
				}
				if len(dataOnly.layers[0].partialInputs) > 0 && &dataOnly.layers[0].partialInputs[0][0] == &s.v41Forward.layers[0].partialInputs[0][0] {
					t.Fatal("snapshot retained a source-owned incomplete carrier")
				}
				if len(dataOnly.history) > 0 {
					dataOnly.history[0] = 99
				}
				for _, row := range dataOnly.attn.kvPublications {
					if len(row) > 0 {
						row[0] += 1
					}
					break
				}
				if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
					t.Fatal("snapshot clone aliases source history or latent publication")
				}
				snap, err := s.PrefixSnapshot()
				if err != nil {
					t.Fatal(err)
				}
				defer snap.Close()
				target := v41DenseTestSession(t, m, b)
				if err := snap.Restore(target); err != nil {
					t.Fatal(err)
				}
				if target.v41Forward.callbackOwner != target || target.v41Forward.compressorNorm == nil {
					t.Fatal("snapshot restore did not rebind normalization to target session")
				}
				cached := map[int]compute.Buffer{}
				for _, record := range b.records {
					cached[record.layer] = record.weight
				}
				s.Close()
				for _, weight := range cached {
					if b.frees[weight] != 0 {
						t.Fatal("source Close freed a model-owned norm gain")
					}
				}
				b.owner, s = target, target
				v41CompressorTestFiniteParity(t, target.Prefill([]int{7, 0}), host.Prefill([]int{7, 0}), "restored normalization owner after source close")
				check(prefixLength + 5)
				for _, record := range b.records {
					if record.weight != cached[record.layer] {
						t.Fatal("restored owner restaged immutable compressor gain")
					}
				}
				calls := len(b.records)
				target.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
				var refused *BackendForwardOperationError
				if err := recoverError(func() { target.Step(1) }); !errors.As(err, &refused) || refused.Path != "device-only" || len(b.records) != calls {
					t.Fatal("whole-model DeviceOnly guard invoked compressor normalization")
				}
			})
		}
	}
}

// Direct step exposes the borrowed scratch so its callback lifetime is observable;
// the session tests above exercise public entry points and transaction ownership.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41CompressorNormEphemeralScratch(t *testing.T) {
	t.Parallel()
	for _, fail := range []bool{false, true} {
		t.Run(itoa(boolToIntV41Expert(fail)), func(t *testing.T) {
			m := v41CompressorProducerTestFixture(t)
			s, b := v41CompressorNormTestSession(t, m)
			s.Prefill([]int{1, 2, 3})
			if fail {
				b.faultLayer, b.faultSite = 1, "rmsnorm"
				b.fault = &compute.BackendError{Backend: "norm-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			}
			scratch := &v41ProjScratch{}
			value := v41CompressorTestRecover(func() {
				_, stats, err := m.forwardV41Step(4, s.v41State(), scratch)
				if err != nil || (!fail && !stats.Committed) {
					t.Fatalf("direct step failed: %v", err)
				}
			})
			if (value != nil) != fail || scratch.compressorNorm != nil {
				t.Fatal("incremental scratch retained callback after success/failure")
			}
		})
	}
}
