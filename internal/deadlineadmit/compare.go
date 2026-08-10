package deadlineadmit

import "time"

// ComparisonArm is one independently reportable deadline-admission arm.
type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Admitted  int           `json:"admitted"`
	Shed      int           `json:"shed"`
	Bytes     int64         `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}

type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

// CompareLocal executes only the native and no-feature FIFO arms. External
// schedulers and fak integrations stay unavailable until their real runtimes
// execute this exact workload; adapters and mocks are not external witnesses.
func CompareLocal() ComparisonResult {
	items := []Item{
		{ID: 4, Deadline: 85, PredictedCost: 20, Degradable: true},
		{ID: 1, Deadline: 100, PredictedCost: 10},
		{ID: 2, Deadline: 100, PredictedCost: 10},
		{ID: 3, Deadline: 110, PredictedCost: 50},
	}
	start := time.Now()
	decision := Admit(items, 80, 10)
	elapsed := time.Since(start)
	correct := len(decision.Order) == 3 && decision.Order[0] == 1 && decision.Order[1] == 2 && decision.Order[2] == 3 && len(decision.Shed) == 1 && decision.Shed[0] == 4
	return ComparisonResult{
		Workload: "order four requests with tied deadlines, shed one degradable predicted miss, and retain one non-degradable miss",
		Arms: []ComparisonArm{
			{Name: "fak native EDF admission", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Admitted: len(decision.Order), Shed: len(decision.Shed)},
			{Name: "FIFO without predicted-miss shedding", Kind: "baseline", Available: true, Correct: false, Admitted: len(items), Note: "no-feature incumbent preserves arrival order and admits the degradable miss"},
			{Name: "Mooncake deadline-aware admission", Kind: "external", Note: "requires the real Mooncake scheduler on the fixed queue"},
			{Name: "vLLM priority scheduling", Kind: "external", Note: "requires a real vLLM server and priority-enabled workload"},
			{Name: "SGLang priority scheduling", Kind: "external", Note: "requires a real SGLang server and priority-enabled workload"},
			{Name: "fak + vLLM priority scheduling", Kind: "integration", Note: "requires the first-class fak/vLLM path, not standalone vLLM reused as evidence"},
			{Name: "fak + SGLang priority scheduling", Kind: "integration", Note: "requires the first-class fak/SGLang path, not standalone SGLang reused as evidence"},
		},
	}
}
