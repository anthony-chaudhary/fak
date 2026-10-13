package model

// v41ColdAttentionResidentOwner admits only a whole cold layer whose visible
// windows fit the existing decode mirror. A cold combined request exposes the
// entire compressed bank, including causally hidden or never-selected rows:
// retaining it must not increase KV H2D rows even when no compressed ID is used.
//
// With S queries and maximum window W, the old path uploads at least
// S + W*(W-1)/2 + (S-W)*(W-1) window rows. The mirror uploads S window rows and
// the whole G-row bank once. Compare G to the repeated-window savings without
// reading IDs, changing masks, or overflowing native arithmetic. This is only
// an upload-volume admission; gathers and device copies can still cost more.
// The selected callback independently declines unsupported row-log capability
// before its first upload. No backend operation is performed here.
func v41ColdAttentionResidentOwner(owner *V41AttentionState, seq, window, compressedRows int, attend v41SharedAttentionFunc) *V41AttentionState {
	if owner == nil || attend == nil || v41ResidentKVDisabled || seq <= 1 || compressedRows < 0 ||
		owner.windowSize < 1 || window == 0 || window < -1 {
		return nil
	}
	width := seq
	if window > 0 {
		width = min(width, window)
	}
	if width-1 > owner.windowSize {
		return nil
	}
	a, b := width, width-1
	if a%2 == 0 {
		a /= 2
	} else {
		b /= 2
	}
	warm, ok := checkedMulInt(a, b)
	if !ok || compressedRows <= warm {
		return owner
	}
	need, perQuery := compressedRows-warm, width-1
	if perQuery == 0 || seq-width < (need-1)/perQuery+1 {
		return nil
	}
	return owner
}

// The owner is the successfully seeded layer state, never a temporary identity.
// The original row streams remain authoritative and are validated by the
// existing composer/adapter before the mirror receives any of their data.
func v41ColdAttentionResidentView(owner *V41AttentionState, layer, pos, windowRows, compressedRows int) *v41ResidentKVView {
	if owner == nil {
		return nil
	}
	return &v41ResidentKVView{
		owner: owner, pos: pos, ringRows: owner.windowSize, firstKey: pos - windowRows + 1,
		windowRows: windowRows, compressedRows: compressedRows, sinkKey: layerName(layer, "attn.sink"),
	}
}
