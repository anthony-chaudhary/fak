package benchloop

// FleetNode is a scrubbed, public registration of a sanctioned benchmark node.
// It deliberately excludes route, account, address, and device-identifier data.
type FleetNode struct {
	Name         string             `json:"name"`
	State        string             `json:"state"`
	Platform     string             `json:"platform"`
	Capabilities []string           `json:"capabilities"`
	Exclusions   []string           `json:"exclusions"`
	Baseline     *FleetNodeBaseline `json:"baseline,omitempty"`
	Witness      string             `json:"witness"`
}

type FleetNodeBaseline struct {
	Workload       string    `json:"workload"`
	Trials         int       `json:"trials"`
	PrefillTokPerS []float64 `json:"prefill_tok_s"`
	DecodeTokPerS  []float64 `json:"decode_tok_s"`
	MeanPrefill    float64   `json:"mean_prefill_tok_s"`
	MeanDecode     float64   `json:"mean_decode_tok_s"`
	PeakPowerW     []float64 `json:"peak_power_w"`
	PeakTempC      []int     `json:"peak_temp_c"`
	PeakVRAMMiB    int       `json:"peak_vram_mib"`
	SourceRevision string    `json:"source_revision"`
	Engine         string    `json:"engine"`
}

// RegisteredFleetNodes returns only evidence-backed, scrubbed registrations.
func RegisteredFleetNodes() []FleetNode {
	return []FleetNode{{
		Name: "anthony-laptop", State: "registered", Platform: "windows/amd64; CUDA SM89",
		Capabilities: []string{"offline", "cpu", "gpu", "dataset", "serving-single-user"},
		Exclusions:   []string{"public-serving", "unattended-overnight", "multi-gpu", "datacenter-network"},
		Baseline: &FleetNodeBaseline{
			Workload: "Qwen2.5-3B-Instruct Q8_0; P32/D32; fak-native CUDA; require-non-reference",
			Trials:   3, PrefillTokPerS: []float64{24.226, 17.448, 23.449}, DecodeTokPerS: []float64{38.769, 41.137, 41.411},
			MeanPrefill: 21.708, MeanDecode: 40.439, PeakPowerW: []float64{58.51, 59.50, 59.38}, PeakTempC: []int{65, 71, 66},
			PeakVRAMMiB: 4838, SourceRevision: "a0eacd4f63dc92fac68d722de269cfe59acbdfc4", Engine: "fak-in-kernel via compute HAL backend cuda",
		},
		Witness: "docs/_witnesses/issue-8486-anthony-laptop/baseline.json",
	}}
}
