package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The recorder's score reads only the leading rows of its key buffer, so it may
// advertise spare key capacity. Current Vulkan binds an exact key extent.
type v41SpareKeyScoreBackend struct{ *v41ScoreTestBackend }

func (b *v41SpareKeyScoreBackend) V41IndexerScoreSpareKeyCapacity() bool { return true }

func v41KeyCacheFixture(t *testing.T) *Model {
	m := v41IndexerTestFixture(t)
	m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
	return m
}

func v41KeyCacheSession(t *testing.T, m *Model, spare bool) (*Session, *v41ScoreTestBackend) {
	t.Helper()
	b := newV41ScoreTestBackend()
	b.rowLog = true
	var backend compute.Backend = b
	if spare {
		wrapped := &v41SpareKeyScoreBackend{b}
		b.identity, backend = wrapped, wrapped
	}
	s := v41DenseTestSession(t, m, backend)
	v41ScoreOnly(s)
	s.v41State().gatherOps.indexerKeyBudgetBytes = 1 << 20
	return s, b
}

func v41KeyCacheRows(s *Session) map[int]v41IndexerKeyEntry {
	out := map[int]v41IndexerKeyEntry{}
	if c := s.v41State().gatherOps.indexerKeys; c != nil {
		for layer, entry := range c.entries {
			out[layer] = entry
		}
	}
	return out
}

// CPU recorder witness of the resident index-key log; not a device timing or
// physical qualification.
// fak-test:runtime medium est=2s lane=default
func TestV41IndexerKeyCache(t *testing.T) {
	t.Run("appends match host scorer with amortized copies", func(t *testing.T) {
		m := v41KeyCacheFixture(t)
		hostBackend := newV41ScoreTestBackend()
		hostBackend.supported = false
		host := v41DenseTestSession(t, m, hostBackend)
		v41ScoreOnly(host)
		if host.v41State().indexerScore() != nil {
			t.Fatal("host control selected the device scorer")
		}
		type lane struct {
			s                     *Session
			b                     *v41ScoreTestBackend
			spare                 bool
			scored, residentCalls int
		}
		lanes := []*lane{{spare: false}, {spare: true}}
		for _, l := range lanes {
			l.s, l.b = v41KeyCacheSession(t, m, l.spare)
			st := l.s.v41State()
			bound := st.indexerScore()
			st.setIndexerScore(func(layer int, q, keys, weights []float32, heads, width, rows int) ([]float32, error) {
				got, err := bound(layer, q, keys, weights, heads, width, rows)
				if err != nil {
					return got, err
				}
				flat := keys
				if keys == nil {
					l.residentCalls++
					plan, perr := m.v41AttentionPlan(layer)
					if perr != nil {
						t.Errorf("plan: %v", perr)
						return got, err
					}
					owner, source := st.attn, plan.KVSourceLayer
					if v41OwnsCompressedRows(plan) {
						owner, source = st.layerState(layer), layer
					}
					published, _ := owner.borrowIndexKeys(source)
					if len(published) != rows {
						t.Errorf("layer %d scored %d rows of %d published", layer, rows, len(published))
						return got, err
					}
					for _, row := range published {
						flat = append(flat, row...)
					}
				}
				want, werr := V41IndexerScore(q, flat, weights, heads, width, rows)
				if werr != nil || len(want) != len(got) {
					t.Errorf("host score: %v", werr)
					return got, err
				}
				for i := range want {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Errorf("layer %d row %d score bits %08x want %08x", layer, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
					}
				}
				topK := min(m.Cfg.IndexTopK, rows)
				gotRows, gerr := V41SelectIndexRows(got, rows, topK, 0, nil)
				wantRows, herr := V41SelectIndexRows(want, rows, topK, 0, nil)
				if (gerr != nil) != (herr != nil) || !reflect.DeepEqual(gotRows, wantRows) {
					t.Errorf("layer %d selection %v want %v", layer, gotRows, wantRows)
				}
				l.scored++
				return got, err
			})
		}
		want := host.Prefill([]int{1, 2, 3})
		for _, l := range lanes {
			v41ScoreTestBits(t, l.s.Prefill([]int{1, 2, 3}), want)
		}
		step := func(token int) {
			want := host.Step(token)
			for _, l := range lanes {
				v41ScoreTestBits(t, l.s.Step(token), want)
				if !reflect.DeepEqual(captureV41ForwardSnapshot(l.s.v41Forward), captureV41ForwardSnapshot(host.v41Forward)) {
					t.Fatalf("spare=%v token=%d retained state diverged from host scorer", l.spare, token)
				}
			}
		}
		step(4) // seed both layers
		rowBytes := int64(m.Cfg.IndexHeadDim * 4)
		type totals struct{ uploaded, copied, appended, exactCopies, validated, hits int64 }
		sums := make([]totals, len(lanes))
		for n := 5; n <= 36; n++ {
			token := n % m.Cfg.VocabSize
			type mark struct {
				upload, copy       int64
				validated, lookups int
				rows               map[int]v41IndexerKeyEntry
			}
			before := make([]mark, len(lanes))
			for i, l := range lanes {
				c := l.s.v41State().gatherOps.indexerKeys
				before[i] = mark{l.b.keyUploadBytes, l.b.keyCopyBytes, c.validatedRows, c.descriptorLookups, v41KeyCacheRows(l.s)}
			}
			step(token)
			for i, l := range lanes {
				after := v41KeyCacheRows(l.s)
				var appended, exact, walked, hits int64
				for layer, entry := range after {
					prev := before[i].rows[layer]
					if grew := entry.rows - prev.rows; grew > 0 {
						appended += int64(grew)
						exact += int64(entry.rows)
					}
					if entry.capacity < entry.rows || (l.spare && entry.capacity > max(2*entry.rows, 2)) || (!l.spare && entry.capacity != entry.rows) {
						t.Fatalf("spare=%v layer %d capacity %d for %d rows", l.spare, layer, entry.capacity, entry.rows)
					}
					switch {
					case entry.pos == prev.pos:
					case prev.rows > 0 && prev.pos+1 == entry.pos:
						walked += int64(entry.rows-prev.rows) + 1
						hits++
					default:
						walked += int64(entry.rows)
					}
				}
				c := l.s.v41State().gatherOps.indexerKeys
				validated := int64(c.validatedRows - before[i].validated)
				lookups := int64(c.descriptorLookups - before[i].lookups)
				uploaded := l.b.keyUploadBytes - before[i].upload
				if uploaded != appended*rowBytes || validated != appended || len(l.b.copyPending) != 0 {
					t.Fatalf("spare=%v token=%d H2D=%d validated=%d want %d appended rows (prefix revalidated or resident path declined)", l.spare, token, uploaded, validated, appended)
				}
				if lookups != walked {
					t.Fatalf("spare=%v token=%d walked %d descriptors, want %d (appended plus one anchor per hit; prefix re-walked)", l.spare, token, lookups, walked)
				}
				sums[i].hits += hits
				sums[i].uploaded += uploaded
				sums[i].copied += l.b.keyCopyBytes - before[i].copy
				sums[i].appended += appended
				sums[i].exactCopies += exact
				sums[i].validated += validated
			}
		}
		for i, l := range lanes {
			if l.scored == 0 || l.residentCalls == 0 || sums[i].hits == 0 {
				t.Fatalf("spare=%v scorer witness vacuous", l.spare)
			}
			final := int64(0)
			for _, entry := range v41KeyCacheRows(l.s) {
				final += int64(entry.rows)
			}
			copied := sums[i].copied / rowBytes
			t.Logf("spare=%v appended=%d copied=%d exact-extent=%d final=%d", l.spare, sums[i].appended, copied, sums[i].exactCopies, final)
			if sums[i].appended < 16 {
				t.Fatalf("only %d appended rows; amortization witness vacuous", sums[i].appended)
			}
			if !l.spare {
				if copied != sums[i].exactCopies {
					t.Fatalf("exact-extent log copied %d rows, want %d (prefix plus append per step)", copied, sums[i].exactCopies)
				}
				continue
			}
			// Doubling bounds total device copies by appended rows plus twice the
			// final rows, i.e. O(1) per append, against the exact log's O(N).
			if copied > sums[i].appended+2*final || 2*copied >= sums[i].exactCopies {
				t.Fatalf("capacity log copied %d rows for %d appends (final %d, exact-extent %d)", copied, sums[i].appended, final, sums[i].exactCopies)
			}
		}
	})
	for _, spare := range []bool{false, true} {
		name := map[bool]string{false: "exact extent", true: "spare capacity"}[spare]
		t.Run(name+"/fresh non-finite key refuses", func(t *testing.T) {
			m := v41KeyCacheFixture(t)
			s, b := v41KeyCacheSession(t, m, spare)
			s.Prefill([]int{1, 2, 3})
			s.Step(4)
			s.Step(5)
			st := s.v41State()
			owner := st.layerState(0)
			row := make([]float32, m.Cfg.IndexHeadDim)
			row[3] = float32(math.NaN())
			owner.publish([]V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 0, IsIndexSource: true}, IndexKey: row}})
			keys, _ := owner.borrowIndexKeys(0)
			lookups := st.gatherOps.indexerKeys.descriptorLookups
			score := st.gatherOps.residentIndexerScore(m, 0, len(st.history), keys, &v41IndexerKeyView{owner: owner, source: 0}, st.indexerScore())
			if walked := st.gatherOps.indexerKeys.descriptorLookups - lookups; score == nil || len(keys) != 3 || walked != 2 {
				t.Fatalf("anchor hit did not admit the append with an anchor-onward walk (rows=%d walked=%d)", len(keys), walked)
			}
			validated, uploads := st.gatherOps.indexerKeys.validatedRows, b.keyUploadBytes
			q := make([]float32, m.Cfg.IndexHeadDim)
			q[0] = 1
			got, err := score(0, q, nil, []float32{1}, 1, len(q), len(keys))
			var closed *BackendForwardOperationError
			if got != nil || !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &closed) || closed.Stage != "payload" || !s.BackendSessionClosed() ||
				b.keyUploadBytes != uploads || st.gatherOps.indexerKeys.validatedRows-validated != 1 {
				t.Fatalf("fresh non-finite key was not refused before upload: %v", err)
			}
		})
	}
	t.Run("rollback anchor mismatch revalidates every row", func(t *testing.T) {
		m := v41KeyCacheFixture(t)
		s, b := v41KeyCacheSession(t, m, true)
		s.Prefill([]int{1, 2, 3})
		s.Step(4)
		s.Step(5)
		st := s.v41State()
		fault := errors.New("after index scoring")
		st.setFinalNorm(func([]float32) ([]float32, error) { return nil, fault })
		if err := recoverError(func() { _, _, _ = m.forwardV41Step(6, st, nil) }); !errors.Is(err, fault) {
			t.Fatalf("late fault did not unwind: %v", err)
		}
		st.setFinalNorm(nil)
		c := st.gatherOps.indexerKeys
		validated, lookups, uploads := c.validatedRows, c.descriptorLookups, b.keyUploadBytes
		s.Step(7)
		total := 0
		for _, entry := range v41KeyCacheRows(s) {
			total += entry.rows
		}
		if total != 6 || c.validatedRows-validated != total || c.descriptorLookups-lookups != total || b.keyUploadBytes-uploads != int64(total*m.Cfg.IndexHeadDim*4) {
			t.Fatalf("rolled-back anchor reused the cached prefix: rows=%d validated=%d walked=%d", total, c.validatedRows-validated, c.descriptorLookups-lookups)
		}
	})
	t.Run("reset anchor mismatch revalidates the prefix", func(t *testing.T) {
		m := v41KeyCacheFixture(t)
		s, b := v41KeyCacheSession(t, m, true)
		s.Prefill([]int{1, 2, 3})
		s.Step(4)
		s.Step(5)
		st := s.v41State()
		owner := st.layerState(0)
		previous, _ := owner.borrowIndexKeys(0)
		republish := cloneV41Rows(previous)
		republish[0][0] = float32(math.Inf(1))
		republish = append(republish, make([]float32, m.Cfg.IndexHeadDim))
		owner.Reset()
		for _, row := range republish {
			owner.publish([]V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 0, IsIndexSource: true}, IndexKey: row}})
		}
		keys, _ := owner.borrowIndexKeys(0)
		c := st.gatherOps.indexerKeys
		lookups := c.descriptorLookups
		score := st.gatherOps.residentIndexerScore(m, 0, len(st.history), keys, &v41IndexerKeyView{owner: owner, source: 0}, st.indexerScore())
		if walked := c.descriptorLookups - lookups; score == nil || walked != len(keys) {
			t.Fatalf("same-owner reset did not admit a fully re-walked reseed (walked=%d of %d)", walked, len(keys))
		}
		validated, uploads := c.validatedRows, b.keyUploadBytes
		q := make([]float32, m.Cfg.IndexHeadDim)
		q[0] = 1
		got, err := score(0, q, nil, []float32{1}, 1, len(q), len(keys))
		// Row 0 is the poisoned prefix row: refusal there with exactly one row
		// checked proves validation restarted at row 0 rather than the stale anchor.
		if got != nil || !errors.Is(err, ErrV41ForwardStage) || !s.BackendSessionClosed() || b.keyUploadBytes != uploads ||
			c.validatedRows-validated != 1 {
			t.Fatalf("reset prefix was trusted instead of revalidated: %v (validated=%d)", err, c.validatedRows-validated)
		}
	})
}
