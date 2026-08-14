package dispatchtick

import "strings"

const (
	ThreadSignalInventory        = "inventory"
	ThreadSignalBaselineDelta    = "baseline_delta"
	ThreadSignalWorkerAttributed = "worker_attributed"
)

// ThreadPressureInput separates absolute host inventory from marginal pressure.
// TotalThreads alone is intentionally observational: a desktop's normal population
// of mostly sleeping threads must not become a hard worker-cap cliff. A hard cap is
// derived only when a baseline or worker-attributed population is present.
type ThreadPressureInput struct {
	Cores                   int
	TotalThreads            int
	BaselineThreads         int
	WorkerAttributedThreads int
	ThreadsPerCore          int
	ThreadsPerWorker        int
}

// ThreadPressureResult is the independently testable policy result consumed by
// host-cap admission. HardCap=nil means the signal is inventory-only and abstains
// from the hard min; TotalThreads remains visible for diagnostics.
type ThreadPressureResult struct {
	Signal           string `json:"signal"`
	TotalThreads     int    `json:"total_threads"`
	ChargedThreads   int    `json:"charged_threads"`
	AvailableThreads int    `json:"available_threads"`
	HardCap          *int   `json:"hard_cap,omitempty"`
}

func EvaluateThreadPressure(in ThreadPressureInput) ThreadPressureResult {
	out := ThreadPressureResult{Signal: ThreadSignalInventory, TotalThreads: maxInt(in.TotalThreads, 0)}
	if in.Cores <= 0 || in.ThreadsPerCore <= 0 || in.ThreadsPerWorker <= 0 {
		return out
	}

	charged := 0
	switch {
	case in.WorkerAttributedThreads > 0:
		out.Signal = ThreadSignalWorkerAttributed
		charged = in.WorkerAttributedThreads
	case in.BaselineThreads > 0:
		out.Signal = ThreadSignalBaselineDelta
		charged = in.TotalThreads - in.BaselineThreads
		if charged < 0 {
			charged = 0
		}
	default:
		return out
	}

	available := in.Cores*in.ThreadsPerCore - charged
	if available < 0 {
		available = 0
	}
	cap := available / in.ThreadsPerWorker
	out.ChargedThreads = charged
	out.AvailableThreads = available
	out.HardCap = IntPtr(cap)
	return out
}

// ApplyThreadPressure folds a pressure result into an existing structural cap.
// Inventory-only results are identity transforms. Attributed/baseline-delta results
// may only contract; they never raise a CPU/RAM/configured ceiling.
func ApplyThreadPressure(cap int, pressure ThreadPressureResult) (int, string) {
	if pressure.HardCap == nil {
		return cap, ""
	}
	if *pressure.HardCap < cap {
		return *pressure.HardCap, strings.TrimSpace(pressure.Signal)
	}
	return cap, ""
}
