package model

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type quantReadAheadBackend struct {
	*v41RoutedLayerTestBackend
	quant         compute.Dtype
	packedUploads int
}

func (b *quantReadAheadBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == b.quant || b.v41RoutedLayerTestBackend.SupportsDeviceWeightDtype(dt)
}

func (b *quantReadAheadBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if len(x.Shape) == 2 && x.Shape[0] == 256 && x.Shape[1] == 256 {
		if x.Dtype != b.quant || dt != b.quant {
			panic("routed checkpoint weight widened before upload")
		}
		b.packedUploads++
	}
	return b.Backend.Upload(x, dt)
}

// This is a CPU-recorded ordinary routed-layer witness, not device qualification.
// It compares the exact same packed request with advice disabled, then verifies
// that selected advice cannot cover the unselected expert between two ranges.
// fak-test:runtime medium est=5s lane=default
func TestV41MappedQuantReadAhead(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41RoutedFusedEnv, "0")
	t.Setenv(v41RoutedPrefetchEnv, "0")
	t.Setenv(v41CrossLayerPrefetchEnv, "0")
	for _, tc := range []struct {
		name  string
		kind  kQuantKind
		quant ExpertCheckpointQuant
		dtype compute.Dtype
	}{
		{"iq2_xxs", kindIQ2XXS, ExpertCheckpointIQ2XXS, compute.IQ2_XXS},
		{"iq2_xs", kindIQ2XS, ExpertCheckpointIQ2XS, compute.IQ2_XS},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := v41IncrementalExpertFixture(t, true, false)
			m.Cfg.SwigluLimit = .5
			tier := NewExpertCheckpointTier(0)
			var events []string
			type span struct {
				shard     int
				off, size int64
			}
			var hinted []span
			var source, snapshots [][]byte
			var tripleBytes int64
			boom := errors.New("optional advice refused")
			for p, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
				stride := 256 * tc.kind.blockBytes()
				raw := make([]byte, 0, 4*stride)
				for e := 0; e < 4; e++ {
					raw = append(raw, makeIQ12Raw(tc.kind, 256, 256, uint64(101+3*e+p))...)
					leaf := []string{"w1.weight", "w3.weight", "w2.weight"}[p]
					name := layerName(0, "ffn.experts."+itoa(e)+"."+leaf)
					delete(m.kqw, name)
					delete(m.q4kw, name)
					delete(m.manifest, name)
				}
				source = append(source, raw)
				snapshots = append(snapshots, append([]byte(nil), raw...))
				reader := &mappedExpertReader{data: raw}
				reader.rangeFn = func(off, size int64) ([]byte, error) {
					events = append(events, "read")
					return raw[off : off+size : off+size], nil
				}
				r := &checkpointReadAheadWitness{mappedExpertReader: reader, ahead: func(off, size int64) error {
					events = append(events, "hint")
					hinted = append(hinted, span{p, off, size})
					if p == 0 && off == 0 {
						return boom
					}
					return nil
				}}
				if err := tier.AddShard(r, int64(len(raw)), []FusedExpertTensor{{
					Name: proj, Layer: 0, Proj: proj, Arch: "deepseek41",
					Quant: tc.quant, Experts: 4, Rows: 256, Cols: 256,
				}}); err != nil {
					t.Fatal(err)
				}
				tripleBytes += int64(stride)
			}
			m.SetExpertCheckpoint(tier)
			// Both real sessions pin the shared mapped source before either closes.
			var sessions []*Session
			var backends []*quantReadAheadBackend
			for i := 0; i < 2; i++ {
				b := &quantReadAheadBackend{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), quant: tc.dtype}
				s, err := m.NewBackendSessionChecked(b)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(s.Close)
				s.ExpertRingBytes = 3 * tripleBytes
				sessions = append(sessions, s)
				backends = append(backends, b)
			}
			sessions[0].ExpertPrefetch = ExpertPrefetchOnDemand
			picks := []routePick{{expert: 3, weight: .25}, {expert: 1, weight: .5}, {expert: 0, weight: .125}, {expert: 1, weight: .0625}}
			_, x := v41RoutedLayerFaultPicks(m)
			for i := range x {
				x[i] *= .001
			}
			run := func(s *Session) []float32 {
				t.Helper()
				layer := s.v41RoutedLayerFunc()
				if layer == nil {
					t.Fatal("ordinary routed layer did not bind")
				}
				out, outcome, err := layer(0, picks, x)
				if err != nil || outcome != v41RoutedExpertHandled || s.BackendSessionClosed() {
					t.Fatalf("ordinary route: outcome=%v err=%v", outcome, err)
				}
				return out
			}
			want := run(sessions[0])
			if len(hinted) != 0 {
				t.Fatal("on-demand control issued advice")
			}
			events = nil
			got := run(sessions[1])
			if !reflect.DeepEqual(got, want) {
				t.Fatal("advice changed routed output")
			}
			wantHints := 0
			if expertMappedReadaheadQualification {
				wantHints = 6
			}
			if len(hinted) != wantHints {
				t.Fatalf("hint calls=%d want=%d", len(hinted), wantHints)
			}
			if wantHints > 0 {
				var expected []span
				for p := 0; p < 3; p++ {
					stride := int64(len(source[p]) / 4)
					expected = append(expected, span{p, 0, 2 * stride}, span{p, 3 * stride, stride})
				}
				if !reflect.DeepEqual(hinted, expected) {
					t.Fatalf("payload union crossed a hole or shard: got=%v want=%v", hinted, expected)
				}
				if len(events) < wantHints {
					t.Fatal("advice did not precede demand")
				}
				for i, event := range events[:wantHints] {
					if event != "hint" {
						t.Fatalf("advice did not precede demand: event[%d]=%q", i, event)
					}
				}
				st := tier.Stats()
				if st.Readahead.Hints != 5 || st.Readahead.Failures != 1 || st.Readahead.Bytes != 7*tripleBytes/3 || st.Failures != 0 {
					t.Fatalf("advice accounting was confused with demand: %+v", st)
				}
			}
			for p := range source {
				if !bytes.Equal(source[p], snapshots[p]) {
					t.Fatal("mapped packed source mutated")
				}
			}
			for i, s := range sessions {
				if backends[i].packedUploads != 9 || backends[i].matmuls != 12 || len(backends[i].pending) != 0 {
					t.Fatalf("packed ordinary dispatch/retirement differs: uploads=%d matmuls=%d pending=%d", backends[i].packedUploads, backends[i].matmuls, len(backends[i].pending))
				}
				st := s.ExpertRing()
				if st.PageIns != 9 || st.ResidentBytes != 3*tripleBytes || st.PeakBytes > s.ExpertRingBytes ||
					len(s.expertRing.holds) != 0 || len(s.expertRing.heldPins) != 0 || s.ringDepth != 0 {
					t.Fatalf("ring ownership/accounting differs: %+v", st)
				}
			}
			before := tier.Stats().Readahead
			if !reflect.DeepEqual(run(sessions[1]), want) || tier.Stats().Readahead != before {
				t.Fatal("resident replay issued advice or changed output")
			}
			if st := tier.Stats(); st.ResidentBytes != 0 || st.Prefetched != 0 || len(tier.loading) != 0 || len(tier.ready) != 0 {
				t.Fatalf("mapped advice created host residency or background work: %+v", st)
			}
			// Repeated close remains the existing idempotent owner boundary.
			for _, s := range sessions {
				s.Close()
				s.Close()
			}
			for _, b := range backends {
				if len(b.pending) != 0 {
					t.Fatal(fmt.Errorf("deferred records survived close: %d", len(b.pending)))
				}
			}
		})
	}
}
