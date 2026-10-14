package model

import "math"

// v41IndexKeyMemoRow pairs one completed group's exact pre-rotary compressed
// source row with the rotated index key published for it. A full-history
// re-fold reuses the key only when the recomputed source row is bit-identical,
// so the reuse can never change a published key (#13793).
type v41IndexKeyMemoRow struct {
	source []float32
	key    []float32
}

// v41IndexKeysReusing returns one index key per compressed row. Groups whose
// source row matches the previous fold's memo reuse its already-rotated key
// (reused[g] is true, so the caller must not rotate it again); every other row
// is projected through v41IndexKeysWithOperations exactly as before. memo
// carries each group's source and, for reused groups, its key; the caller sets
// the fresh keys after rotary.
func (m *Model) v41IndexKeysReusing(l int, rows [][]float32, st *v41ForwardState, scratch *v41ProjScratch) (keys [][]float32, reused []bool, memo []v41IndexKeyMemoRow, err error) {
	var prior []v41IndexKeyMemoRow
	if st != nil {
		prior = st.indexKeyPrior[l]
	}
	keys = make([][]float32, len(rows))
	reused = make([]bool, len(rows))
	memo = make([]v41IndexKeyMemoRow, len(rows))
	pending := make([][]float32, 0, len(rows))
	fresh := make([]int, 0, len(rows))
	for g, row := range rows {
		if g < len(prior) && v41IndexKeySourceSame(prior[g].source, row) {
			keys[g] = append([]float32(nil), prior[g].key...)
			reused[g] = true
			memo[g] = prior[g]
			continue
		}
		memo[g].source = append([]float32(nil), row...)
		pending = append(pending, row)
		fresh = append(fresh, g)
	}
	projected, err := m.v41IndexKeysPublished(l, pending, scratch.denseProjection(), scratch.indexKeyNorm, scratch.gatherOps.bf16PublishFunc())
	if err != nil {
		return nil, nil, nil, err
	}
	for i, g := range fresh {
		keys[g] = projected[i]
	}
	return keys, reused, memo, nil
}

func v41IndexKeySourceSame(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}
