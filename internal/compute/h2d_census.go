package compute

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

// FAK_H2D_CENSUS=1 turns on measured-mode counting of host->device copies at
// the per-decode-token state seams inventoried by `fak-dev h2d-inventory`
// (the shared activation-upload funnel in internal/model and the paged
// block-table sync here). It is OFF by default and zero-cost when unset: every
// call site degrades to one predictable branch on h2dCensusOn, which is read
// from the environment exactly once at package init. When on, cumulative
// totals flush to stderr every h2dCensusFlushEvery copies so a hardware
// receipt's captured stderr carries MEASURED copies/bytes beside the static
// estimate from `fak-dev h2d-inventory` — the two are meant to be compared,
// not trusted separately.
var h2dCensusOn = os.Getenv("FAK_H2D_CENSUS") == "1"

const h2dCensusFlushEvery = 1024

type h2dCensusSite struct {
	copies int64
	bytes  int64
}

var (
	h2dCensusCopies atomic.Int64
	h2dCensusBytes  atomic.Int64
	h2dCensusMu     sync.Mutex
	h2dCensusSites  = map[string]*h2dCensusSite{}
)

// H2DCensusRecord counts one host->device copy of nbytes bytes at the named
// state seam. It is a no-op unless FAK_H2D_CENSUS=1 was set before process
// start, so the decode hot path pays a single branch when the census is off.
func H2DCensusRecord(site string, nbytes int64) {
	if !h2dCensusOn {
		return
	}
	total := h2dCensusCopies.Add(1)
	h2dCensusBytes.Add(nbytes)
	h2dCensusMu.Lock()
	st := h2dCensusSites[site]
	if st == nil {
		st = &h2dCensusSite{}
		h2dCensusSites[site] = st
	}
	st.copies++
	st.bytes += nbytes
	var line string
	if total%h2dCensusFlushEvery == 0 {
		line = renderH2DCensusLineLocked(total)
	}
	h2dCensusMu.Unlock()
	if line != "" {
		fmt.Fprintln(os.Stderr, line)
	}
}

// renderH2DCensusLineLocked snapshots the per-site counts; caller holds
// h2dCensusMu.
func renderH2DCensusLineLocked(total int64) string {
	type kv struct {
		site string
		st   *h2dCensusSite
	}
	rows := make([]kv, 0, len(h2dCensusSites))
	for s, st := range h2dCensusSites {
		rows = append(rows, kv{s, st})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].st.bytes != rows[j].st.bytes {
			return rows[i].st.bytes > rows[j].st.bytes
		}
		return rows[i].site < rows[j].site
	})
	top := rows
	if len(top) > 8 {
		top = top[:8]
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("fak-h2d-census: copies=%d bytes=%d sites=%d top=", total, h2dCensusBytes.Load(), len(rows)))
	for i, r := range top {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%s=c%d/b%d", r.site, r.st.copies, r.st.bytes)
	}
	return b.String()
}

// H2DCensusEnabled reports whether measured-mode counting was on at process
// start. A receipt producer uses it to tell "census off, field unobservable"
// apart from "census on, zero bytes copied" — the counters stay indistinguishable
// from an empty measurement while the census is off.
func H2DCensusEnabled() bool { return h2dCensusOn }

// H2DCensusSnapshot reads the cumulative counters lock-free, so a decode-window
// sample can diff them around its span like any other process-wide counter.
func H2DCensusSnapshot() (copies, bytes int64) {
	return h2dCensusCopies.Load(), h2dCensusBytes.Load()
}

// H2DCensusTotals exposes the current measured totals for tests and receipts.
func H2DCensusTotals() (copies, bytes int64, sites map[string]struct {
	Copies int64 `json:"copies"`
	Bytes  int64 `json:"bytes"`
}) {
	out := map[string]struct {
		Copies int64 `json:"copies"`
		Bytes  int64 `json:"bytes"`
	}{}
	h2dCensusMu.Lock()
	defer h2dCensusMu.Unlock()
	for s, st := range h2dCensusSites {
		out[s] = struct {
			Copies int64 `json:"copies"`
			Bytes  int64 `json:"bytes"`
		}{st.copies, st.bytes}
	}
	return h2dCensusCopies.Load(), h2dCensusBytes.Load(), out
}
