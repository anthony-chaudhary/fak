package logvault

import "sort"

// SourceFootprint is one source's observability rollup, folded PURELY from the
// vault manifest. Every field is a WITNESSED value fak computed from its own
// hash-chained record — the sizes were stat'd and the hashes computed at capture
// time and recorded in the chain — never a self-reported live counter. This is
// the "is my backup current?" answer surfaced by `fak logvault du` and the
// audit-usage vault section (#2455).
type SourceFootprint struct {
	Source              string // registry source id
	Files               int    // distinct files currently tracked in the vault (skip-error rows do not count)
	ManifestRows        int    // total manifest rows naming this source (capture ops + skip-errors)
	Bytes               int64  // current vault footprint: sum of the latest witnessed SizeAfter per tracked file (current mirrors, excluding .history/)
	Errors              int    // skip-error rows for this source (advisory: a file that could not be read this or a prior cycle)
	LastCaptureUnixNano int64  // newest TSUnixNano over this source's SUCCESSFUL capture rows (0 = never successfully captured)
}

// Footprint folds manifest rows into a per-source observability rollup, sorted
// by source id for a stable render. It is PURE: no I/O and no clock read — the
// caller supplies the already-read rows (ReadManifestRows) and applies its own
// clock for the capture-age subtraction, matching the audit-usage honesty fence
// (every disk/clock read stays in the CLI shell).
//
// LastCaptureUnixNano tracks only SUCCESSFUL ops, so a source whose most recent
// cycle only recorded skip-errors keeps the timestamp of its last good capture —
// the honest "when was the last SUCCESSFUL capture" answer the acceptance asks
// for. Bytes and Files reflect the current tracked mirrors (skip-errors advance
// neither); Errors counts every skip-error row so a silent read failure surfaces.
func Footprint(rows []ManifestRow) []SourceFootprint {
	type acc struct {
		fp   SourceFootprint
		size map[string]int64 // rel-path -> latest witnessed SizeAfter
	}
	bySrc := make(map[string]*acc)
	for _, r := range rows {
		a := bySrc[r.Source]
		if a == nil {
			a = &acc{fp: SourceFootprint{Source: r.Source}, size: make(map[string]int64)}
			bySrc[r.Source] = a
		}
		a.fp.ManifestRows++
		if r.Op == OpSkip || r.SHA256 == "" {
			a.fp.Errors++
			continue
		}
		a.size[r.RelPath] = r.SizeAfter
		if r.TSUnixNano > a.fp.LastCaptureUnixNano {
			a.fp.LastCaptureUnixNano = r.TSUnixNano
		}
	}
	out := make([]SourceFootprint, 0, len(bySrc))
	for _, a := range bySrc {
		a.fp.Files = len(a.size)
		for _, sz := range a.size {
			a.fp.Bytes += sz
		}
		out = append(out, a.fp)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Source < out[j].Source })
	return out
}

// NewestCaptureUnixNano returns the newest SUCCESSFUL capture time across every
// source in the footprint (0 when nothing has ever been captured) — the single
// "last successful capture" anchor a scalar surface (a /metrics gauge) reports
// for the whole vault. Skip-error-only sources contribute 0 and never move it.
func NewestCaptureUnixNano(fps []SourceFootprint) int64 {
	var newest int64
	for _, fp := range fps {
		if fp.LastCaptureUnixNano > newest {
			newest = fp.LastCaptureUnixNano
		}
	}
	return newest
}

// TotalBytes sums the current tracked footprint across every source — the vault's
// logical current-mirror size (excluding retired .history/ versions).
func TotalBytes(fps []SourceFootprint) int64 {
	var total int64
	for _, fp := range fps {
		total += fp.Bytes
	}
	return total
}
