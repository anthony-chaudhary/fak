package model

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// FAK_GLUE_CENSUS=1 turns on measured-mode counting of synchronous host<->device
// round-trips at the GLM-DSA device seams (backendKernel.mul / sparseAttend /
// indexSelect). It is OFF by default and zero-cost when unset: every call site
// below degrades to one predictable branch on glueCensusOn, which is read from
// the environment exactly once at package init. When on, cumulative totals
// flush to stderr every glueCensusFlushEvery round-trips so a hardware receipt's
// captured stderr carries MEASURED values beside the static estimate from
// `fak-dev glue-census` — the two are meant to be compared, not trusted
// separately.
var glueCensusOn = os.Getenv("FAK_GLUE_CENSUS") == "1"

const glueCensusFlushEvery = 1024

var (
	glueCensusTotal atomic.Int64
	glueCensusMu    sync.Mutex
	glueCensusSites = map[string]int64{} // normalized site -> count
)

// glueCensusCountIfDevice records one synchronous host<->device round-trip ONLY
// for a real device backend. The cpu-ref backend's Upload/Read are identity/no-op
// (no PCIe traffic), so its Caps().DeviceMemory=false keeps host forwards out of
// the count. Call it after the device op has completed (the fenced readback
// returned), never before.
func glueCensusCountIfDevice(be compute.Backend, site string) {
	if !glueCensusOn || be == nil {
		return
	}
	if !be.Caps().DeviceMemory {
		return
	}
	glueCensusCount(site)
}

func glueCensusCount(site string) {
	if !glueCensusOn {
		return
	}
	total := glueCensusTotal.Add(1)
	glueCensusMu.Lock()
	key := normalizeGlueCensusSite(site)
	glueCensusSites[key]++
	var line string
	if total%glueCensusFlushEvery == 0 {
		line = renderGlueCensusLineLocked(total)
	}
	glueCensusMu.Unlock()
	if line != "" {
		fmt.Fprintln(os.Stderr, line)
	}
}

// normalizeGlueCensusSite strips the "model.layers.<n>." prefix so per-layer
// projections aggregate into their projection class (self_attn.q_a_proj.weight,
// mlp.gate.weight, ...) instead of ~92 near-duplicate keys.
func normalizeGlueCensusSite(name string) string {
	parts := strings.Split(name, ".")
	if len(parts) > 3 && parts[0] == "model" && parts[1] == "layers" {
		return strings.Join(parts[3:], ".")
	}
	return name
}

// renderGlueCensusLineLocked snapshots the per-site counts; caller holds
// glueCensusMu.
func renderGlueCensusLineLocked(total int64) string {
	type kv struct {
		site string
		n    int64
	}
	rows := make([]kv, 0, len(glueCensusSites))
	for s, n := range glueCensusSites {
		rows = append(rows, kv{s, n})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].n != rows[j].n {
			return rows[i].n > rows[j].n
		}
		return rows[i].site < rows[j].site
	})
	top := rows
	if len(top) > 8 {
		top = top[:8]
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("fak-glue-census: round_trips=%d sites=%d top=", total, len(rows)))
	for i, r := range top {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%s=%d", r.site, r.n)
	}
	return b.String()
}

// glueCensusTotals exposes the current measured totals for tests and receipts.
func glueCensusTotals() (total int64, sites map[string]int64) {
	total = glueCensusTotal.Load()
	glueCensusMu.Lock()
	defer glueCensusMu.Unlock()
	out := make(map[string]int64, len(glueCensusSites))
	for s, n := range glueCensusSites {
		out[s] = n
	}
	return total, out
}
