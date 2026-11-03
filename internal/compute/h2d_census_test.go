package compute

import "testing"

// TestH2DCensusOffByDefault pins the zero-cost contract: with the census off
// (the default), records are dropped and totals stay empty.
func TestH2DCensusOffByDefault(t *testing.T) {
	prev := h2dCensusOn
	h2dCensusOn = false
	defer func() { h2dCensusOn = prev }()
	// Clear any counts a prior test in this binary may have left.
	h2dCensusCopies.Store(0)
	h2dCensusBytes.Store(0)
	h2dCensusMu.Lock()
	h2dCensusSites = map[string]*h2dCensusSite{}
	h2dCensusMu.Unlock()

	H2DCensusRecord("hal-token-input", 4096)
	copies, bytes, sites := H2DCensusTotals()
	if copies != 0 || bytes != 0 || len(sites) != 0 {
		t.Fatalf("census off must record nothing: copies=%d bytes=%d sites=%v", copies, bytes, sites)
	}
}

// TestH2DCensusRecordsPerSite pins the measured-mode accounting: per-site
// copy/byte totals plus the cumulative sums.
func TestH2DCensusRecordsPerSite(t *testing.T) {
	prev := h2dCensusOn
	h2dCensusOn = true
	defer func() { h2dCensusOn = prev }()
	h2dCensusCopies.Store(0)
	h2dCensusBytes.Store(0)
	h2dCensusMu.Lock()
	h2dCensusSites = map[string]*h2dCensusSite{}
	h2dCensusMu.Unlock()

	H2DCensusRecord("hal-token-input", 2048)
	H2DCensusRecord("hal-token-input", 2048)
	H2DCensusRecord("paged-kv-block-table", 16)
	copies, bytes, sites := H2DCensusTotals()
	if copies != 3 || bytes != 2048*2+16 {
		t.Fatalf("totals = %d copies / %d bytes, want 3 / %d", copies, bytes, 2048*2+16)
	}
	if st := sites["hal-token-input"]; st.Copies != 2 || st.Bytes != 4096 {
		t.Fatalf("hal-token-input site = %+v, want 2 copies / 4096 B", st)
	}
	if st := sites["paged-kv-block-table"]; st.Copies != 1 || st.Bytes != 16 {
		t.Fatalf("paged-kv-block-table site = %+v, want 1 copy / 16 B", st)
	}
}
