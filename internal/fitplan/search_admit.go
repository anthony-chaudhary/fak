package fitplan

import "strings"

// searchAxes holds the enumerated axes that survived admission: the deduped KV
// tiers, context ladder, and slot ladder Search enumerates over.
type searchAxes struct {
	tiers     []KVTier
	ctxs      []int
	slotsList []int
}

// admitSearch runs the pre-enumeration gates in fixed order: hard memory facts
// first (they hold regardless of calibration quality), then calibration
// rankability, probeability, and the axis ladders. It fills rep's typed
// refusal/unknown status when a gate fails and returns false when Search must
// stop before enumerating rows.
func admitSearch(p *planner, rep *SearchReport, in SearchInput, isl, target int) (searchAxes, bool) {
	var ax searchAxes
	if searchTiered(p, rep, in, isl) {
		return ax, false
	}
	if p.hostShort {
		refuseSearch(rep, ReasonWeightsExceedPool, "", 0, 0, p.hostWeights-p.in.Hardware.HostPoolUsableBytes, true)
		return ax, false
	}
	if !in.Calibration.rankable() {
		refuseSearch(rep, ReasonNoRankableCalibration, "", 0, 0, 0, false)
		return ax, false
	}
	if p.gate != "" {
		rep.Status = StatusUnknown
		rep.Omitted = []Omission{p.gate}
		return ax, false
	}
	if p.avail <= 0 {
		rep.Status = StatusUnknown
		rep.Omitted = append(rep.Omitted, OmitPoolCapacityUnknown)
		return ax, false
	}

	ax.tiers = dedupeTiers(in.Tiers)
	if len(ax.tiers) == 0 {
		rep.Status = StatusUnknown
		rep.Omitted = append(rep.Omitted, OmitKVGeometryUnknown)
		return ax, false
	}

	ax.ctxs = contextLadder(in.ResolvedContext, in.Model.MaxContextTokens, in.Pins.ContextTokens)
	if len(ax.ctxs) == 0 {
		rep.Status = StatusUnknown
		rep.Omitted = append(rep.Omitted, OmitContextUnknown)
		return ax, false
	}
	if in.Pins.ContextTokens > 0 {
		ax.ctxs = []int{in.Pins.ContextTokens} // a pin fixes the axis exactly
	}
	ax.slotsList = slotLadder(target, in.Pins.Slots)
	if in.Pins.Slots > 0 {
		ax.slotsList = []int{in.Pins.Slots}
	}
	if pin := strings.ToLower(strings.TrimSpace(in.Pins.KVPrecision)); pin != "" {
		var filtered []KVTier
		for _, t := range ax.tiers {
			if strings.ToLower(t.Precision) == pin {
				filtered = append(filtered, t)
			}
		}
		if len(filtered) == 0 {
			refuseSearch(rep, ReasonPinInfeasible, pin, 0, 0, 0, false)
			rep.Refusal.Pin = "kv_precision=" + pin
			return ax, false
		}
		ax.tiers = filtered
	}
	return ax, true
}
