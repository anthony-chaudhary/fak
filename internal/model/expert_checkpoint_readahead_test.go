package model

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type checkpointReadAheadWitness struct {
	*mappedExpertReader
	ahead func(int64, int64) error
}

func (r *checkpointReadAheadWitness) ReadAheadRange(off, size int64) error {
	return r.ahead(off, size)
}

type v41ReadAheadBackend struct {
	*v41RoutedLayerTestBackend
	declineQ3 bool
}

func (b *v41ReadAheadBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !(b.declineQ3 && dt == compute.Q3_K) && b.v41RoutedLayerTestBackend.SupportsDeviceWeightDtype(dt)
}

// fak-test:runtime fast est=100ms lane=default
func TestExpertCheckpointReadAheadEligibilityAndFailureAccounting(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name            string
		quant           ExpertCheckpointQuant
		budget          int64
		page, overlay   bool
		supported, hint bool
	}{
		{name: "q2", quant: ExpertCheckpointQ2K, supported: true, hint: true},
		{name: "q3", quant: ExpertCheckpointQ3K, supported: true, hint: true},
		{name: "q4", quant: ExpertCheckpointQ4K, supported: true},
		{name: "retained", quant: ExpertCheckpointQ2K, budget: 4096, supported: true},
		{name: "pagecache", quant: ExpertCheckpointQ3K, page: true, supported: true},
		{name: "overlay", quant: ExpertCheckpointQ2K, overlay: true, supported: true},
		{name: "unavailable", quant: ExpertCheckpointQ2K},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tier, reader, name, stride := mappedCheckpointFixture(t, tc.quant, tc.budget, tc.page, tc.overlay)
			calls := 0
			boom := errors.New("advice refused")
			if tc.supported {
				tier.shards[0].r = &checkpointReadAheadWitness{mappedExpertReader: reader, ahead: func(off, size int64) error {
					calls++
					if off != 0 || size != int64(stride) {
						t.Fatalf("range=(%d,%d), want one indexed expert", off, size)
					}
					return boom
				}}
			}
			tier.readAhead(name)
			st := tier.Stats()
			if (calls == 1) != tc.hint || (st.Readahead.Failures == 1) != tc.hint ||
				st.Reads != 0 || st.BytesRead != 0 || st.Failures != 0 || st.Prefetched != 0 || st.ResidentBytes != 0 || len(tier.loading) != 0 || len(tier.ready) != 0 {
				t.Fatalf("hint changed demand/cache accounting: calls=%d stats=%+v", calls, st)
			}
			if tc.hint && !strings.Contains(st.Readahead.LastError, boom.Error()) {
				t.Fatal("advice failure was not exposed")
			}
			if !tc.supported && st.Readahead.Unavailable != 1 {
				t.Fatal("unavailable capability was not exposed")
			}
			w, err := tier.fault(name)
			if err != nil || !bytes.Equal(mappedCheckpointRaw(w), reader.data[:stride]) {
				t.Fatalf("advice changed subsequent demand bytes: %v", err)
			}
		})
	}
}

// The actual batched-layer callback must advise only selected, nonresident
// payloads before its first demand read. The CPU recorder proves ordering,
// admission and failure semantics; Linux syscall acceptance is tested by the
// loader package, and physical I/O overlap remains a native benchmark question.
// fak-test:runtime slow est=30s lane=default
func TestV41RoutedLayerReadAheadSelectedPayloads(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	var oracle []float32
	for _, mode := range []string{"cold", "resident", "on_demand", "advice_failure", "duplicate", "ring_refusal", "dtype_refusal", "bias_refusal", "demand_failure"} {
		t.Run(mode, func(t *testing.T) {
			m := v41IncrementalExpertFixture(t, true, false)
			tier := NewExpertCheckpointTier(0)
			names := func(e int) [3]string {
				stem := "ffn.experts." + itoa(e)
				return [3]string{layerName(0, stem+".w1.weight"), layerName(0, stem+".w3.weight"), layerName(0, stem+".w2.weight")}
			}
			var events, expected []string
			var tripleBytes int64
			boom := errors.New("injected mapped I/O failure")
			for p, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
				w := m.kqw[names(0)[p]]
				stride := int64(len(w.raw))
				var blob []byte
				for e := 0; e < 3; e++ {
					blob = append(blob, m.kqw[names(e)[p]].raw...)
					delete(m.kqw, names(e)[p])
				}
				reader := &mappedExpertReader{data: blob}
				reader.rangeFn = func(off, size int64) ([]byte, error) {
					events = append(events, fmt.Sprintf("read:%s:%d:%d", proj, off, size))
					if mode == "demand_failure" {
						return nil, boom
					}
					return blob[off : off+size : off+size], nil
				}
				r := &checkpointReadAheadWitness{mappedExpertReader: reader, ahead: func(off, size int64) error {
					events = append(events, fmt.Sprintf("hint:%s:%d:%d", proj, off, size))
					if size != stride || (off != 0 && off != 2*stride) {
						t.Fatalf("hint read unselected expert or union: %s off=%d size=%d", proj, off, size)
					}
					if mode == "advice_failure" {
						return boom
					}
					return nil
				}}
				quant := ExpertCheckpointQ2K
				if p == 2 {
					quant = ExpertCheckpointQ3K
				}
				if err := tier.AddShard(r, int64(len(blob)), []FusedExpertTensor{{Name: proj, Layer: 0, Proj: proj, Arch: "deepseek41", Quant: quant, Experts: 3, Rows: w.out, Cols: w.in}}); err != nil {
					t.Fatal(err)
				}
				tripleBytes += stride
			}
			m.SetExpertCheckpoint(tier)
			b := &v41ReadAheadBackend{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), declineQ3: mode == "dtype_refusal"}
			s, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			s.ExpertRingBytes = 2 * tripleBytes
			picks := []routePick{{expert: 0, weight: .25}, {expert: 2, weight: .5}}
			if mode == "duplicate" {
				picks[1].expert = 0
			}
			if mode == "ring_refusal" {
				s.ExpertRingBytes = tripleBytes
			}
			if mode == "on_demand" {
				s.ExpertPrefetch = ExpertPrefetchOnDemand
			}
			if mode == "bias_refusal" {
				// Admission needs only presence; no bias payload is read.
				m.manifest[layerName(0, "ffn.experts.2.w3.bias")] = tensorMeta{}
			}
			if mode == "resident" {
				for _, name := range names(2) {
					_, release, ok := s.expertInputDeviceWeight(name)
					if !ok {
						t.Fatal("resident setup declined")
					}
					release()
				}
				events = nil
			}
			wantHints := 6
			switch mode {
			case "resident", "duplicate":
				wantHints = 3
			case "on_demand", "ring_refusal", "dtype_refusal", "bias_refusal":
				wantHints = 0
			}
			for _, pick := range picks {
				if len(expected) == wantHints {
					break
				}
				for p, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
					entry := tier.index[names(pick.expert)[p]]
					expected = append(expected, fmt.Sprintf("hint:%s:%d:%d", proj, int64(pick.expert)*entry.stride, entry.stride))
				}
			}
			_, x := v41RoutedLayerFaultPicks(m)
			layer := s.v41RoutedLayerFunc()
			if layer == nil {
				t.Fatal("layer did not bind")
			}
			var result []float32
			var outcome v41RoutedExpertOutcome
			var panicValue any
			func() {
				defer func() { panicValue = recover() }()
				result, outcome, err = layer(0, picks, x)
			}()
			var panicErr error
			if panicValue != nil {
				panicErr = fmt.Errorf("%v", panicValue)
			}
			if mode == "demand_failure" {
				// Existing checkpoint staging re-panics with a diagnostic string.
				// Preserve that contract while also checking the tier's original
				// typed I/O cause, rather than claiming panic error identity.
				if !strings.Contains(fmt.Sprint(panicValue), boom.Error()) || !errors.Is(tier.lastErr, boom) {
					t.Fatalf("real demand error lost: returned=%v panic=%v", err, panicErr)
				}
				if !s.BackendSessionClosed() || tier.Stats().Failures != 1 {
					t.Fatal("real demand failure was not fail-closed")
				}
			} else if mode == "ring_refusal" || mode == "dtype_refusal" || mode == "bias_refusal" {
				if outcome != v41RoutedExpertDeclined || err != nil || panicErr != nil {
					t.Fatalf("admission refusal changed: outcome=%v err=%v panic=%v", outcome, err, panicErr)
				}
			} else {
				if outcome != v41RoutedExpertHandled || err != nil || panicErr != nil || s.BackendSessionClosed() {
					t.Fatalf("hint changed inference: outcome=%v err=%v panic=%v", outcome, err, panicErr)
				}
				if mode == "cold" {
					oracle = append([]float32(nil), result...)
				} else if mode != "duplicate" && !reflect.DeepEqual(result, oracle) {
					t.Fatal("hint changed routed arithmetic")
				}
			}
			if len(events) < wantHints || (wantHints > 0 && !reflect.DeepEqual(events[:wantHints], expected)) {
				t.Fatalf("hints did not precede demand reads: events=%v want prefix=%v", events, expected)
			}
			st := tier.Stats()
			if st.Readahead.Hints+st.Readahead.Failures != wantHints || st.Prefetched != 0 || len(tier.ready) != 0 || len(tier.loading) != 0 || st.ResidentBytes != 0 {
				t.Fatalf("unexpected hint lifecycle/accounting: %+v", st)
			}
			if mode == "advice_failure" && (st.Readahead.Failures != 6 || !strings.Contains(st.Readahead.LastError, boom.Error()) || st.Failures != 0) {
				t.Fatal("advice failures hidden or confused with demand failures")
			}
			if s.expertRing != nil && (len(s.expertRing.holds) != 0 || len(s.expertRing.heldPins) != 0 || s.ringDepth != 0) {
				t.Fatal("readahead left demand holds behind")
			}
			if mode == "cold" {
				before := st.Readahead
				if _, outcome, err := layer(0, picks, x); outcome != v41RoutedExpertHandled || err != nil || tier.Stats().Readahead != before {
					t.Fatal("fully resident replay issued another payload hint")
				}
			}
		})
	}
}
