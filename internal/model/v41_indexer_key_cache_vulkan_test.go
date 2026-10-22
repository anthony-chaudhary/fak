//go:build vulkan && (windows || linux) && cgo

package model

import (
	"math"
	"testing"
)

// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no witness. On
// the registered Vulkan backend the resident indexer key log keeps spare row
// capacity: each resident score equals the host scorer and the exact-upload
// Vulkan lane bit for bit, logits stay bit-identical, and device row copies
// stay O(1) per appended row across capacity doublings.
// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime integration est=20s lane=optin
func TestV41IndexerKeyCacheVulkan(t *testing.T) {
	be := v41VulkanSeamBackend(t)
	if !v41IndexerScoreSpareKeyCapacity(be) {
		t.Fatal("Vulkan does not advertise spare indexer key capacity")
	}
	m := v41KeyCacheFixture(t)
	type lane struct {
		s             *Session
		scores        [][]float32
		residentCalls int
	}
	lanes := []*lane{{}, {}}
	for i, l := range lanes {
		l.s = v41DenseTestSession(t, m, be)
		v41ScoreOnly(l.s)
		st := l.s.v41State()
		bound := st.indexerScore()
		if st.gatherOps == nil || bound == nil {
			t.Fatal("Vulkan left the gather ops or indexer score seam unbound")
		}
		// Lane 0 uploads exact-size keys per call; lane 1 scores the resident log.
		st.gatherOps.indexerKeyBudgetBytes = int64(i) << 20
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
			for r := range want {
				if math.Float32bits(got[r]) != math.Float32bits(want[r]) {
					t.Errorf("lane %d layer %d row %d score bits %08x want %08x", i, layer, r, math.Float32bits(got[r]), math.Float32bits(want[r]))
				}
			}
			l.scores = append(l.scores, append([]float32(nil), got...))
			return got, err
		})
	}
	exact, resident := lanes[0], lanes[1]
	v41ScoreTestBits(t, resident.s.Prefill([]int{1, 2, 3}), exact.s.Prefill([]int{1, 2, 3}))
	v41ScoreTestBits(t, resident.s.Step(4), exact.s.Step(4))
	c := resident.s.v41State().gatherOps.indexerKeys
	if c == nil || exact.s.v41State().gatherOps.indexerKeys != nil {
		t.Fatal("lanes did not split into resident and exact-upload key paths")
	}
	var appended, copied, exactCopies, hits, doublings, spareSteps int
	for n := 5; n <= 36; n++ {
		token := n % m.Cfg.VocabSize
		before := v41KeyCacheRows(resident.s)
		validated, lookups, copies := c.validatedRows, c.descriptorLookups, c.copiedRows
		v41ScoreTestBits(t, resident.s.Step(token), exact.s.Step(token))
		var grew, walked int
		spare := false
		for layer, entry := range v41KeyCacheRows(resident.s) {
			prev := before[layer]
			if d := entry.rows - prev.rows; d > 0 {
				grew += d
				exactCopies += entry.rows
			}
			if entry.capacity < entry.rows || entry.capacity > max(2*entry.rows, 2) {
				t.Fatalf("token=%d layer %d capacity %d for %d rows", token, layer, entry.capacity, entry.rows)
			}
			if prev.rows > 0 && entry.capacity != prev.capacity {
				doublings++
			}
			spare = spare || entry.capacity > entry.rows
			switch {
			case entry.pos == prev.pos:
			case prev.rows > 0 && prev.pos+1 == entry.pos:
				walked += entry.rows - prev.rows + 1
				hits++
			default:
				walked += entry.rows
			}
		}
		if c.validatedRows-validated != grew || c.descriptorLookups-lookups != walked {
			t.Fatalf("token=%d validated=%d walked=%d, want %d appended and %d descriptors (prefix revalidated or resident path declined)",
				token, c.validatedRows-validated, c.descriptorLookups-lookups, grew, walked)
		}
		if spare {
			spareSteps++
		}
		appended += grew
		copied += c.copiedRows - copies
	}
	final := 0
	for _, entry := range v41KeyCacheRows(resident.s) {
		final += entry.rows
	}
	t.Logf("vulkan appended=%d copied=%d exact-extent=%d final=%d doublings=%d spare-steps=%d", appended, copied, exactCopies, final, doublings, spareSteps)
	if resident.residentCalls == 0 || exact.residentCalls != 0 || hits == 0 || appended < 16 || doublings == 0 || spareSteps == 0 {
		t.Fatalf("spare-capacity witness vacuous: resident calls=%d/%d hits=%d appended=%d doublings=%d spare steps=%d",
			resident.residentCalls, exact.residentCalls, hits, appended, doublings, spareSteps)
	}
	if copied > appended+2*final || 2*copied >= exactCopies {
		t.Fatalf("capacity log copied %d rows for %d appends (final %d, exact-extent %d)", copied, appended, final, exactCopies)
	}
	if len(resident.scores) != len(exact.scores) {
		t.Fatalf("lanes scored %d and %d indexer calls", len(resident.scores), len(exact.scores))
	}
	for k := range exact.scores {
		v41ScoreTestBits(t, resident.scores[k], exact.scores[k])
	}
	for i, l := range lanes {
		if l.s.BackendSessionClosed() {
			t.Fatalf("lane %d closed its backend session", i)
		}
	}
}
