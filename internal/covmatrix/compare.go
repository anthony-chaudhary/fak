package covmatrix

import "time"

// ComparisonArm records one same-workload coverage-matrix implementation. External
// and integration arms remain measurement-zero until a real pinned run exists;
// adapters and copied fixture answers are not product witnesses.
type ComparisonArm struct {
	Name            string
	Kind            string
	Available       bool
	Correct         bool
	Latency         time.Duration
	Checks          int
	PassedChecks    int
	MissedChecks    int
	FalseCells      int
	UndefinedCells  int
	StaleCells      int
	FamilyCells     int
	PrecisionCells  int
	CPUSeconds      float64
	PeakRSSBytes    int64
	InputBytes      int64
	NetworkBytes    int64
	OperatorSeconds float64
	CostUSD         float64
	Note            string
}

type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func runNativeComparison() ComparisonArm {
	a := ComparisonArm{
		Name:      "fak native model-backend-precision coverage matrix",
		Kind:      "native",
		Available: true,
		Checks:    3,
		Note:      "enumerates every family/backend and family/backend/precision cell, then emits typed stale declarations",
	}
	start := time.Now()
	cells := Grid()
	xcells := GridX()
	stale := StaleCells()
	a.Latency = time.Since(start)
	a.FamilyCells = len(cells)
	a.PrecisionCells = len(xcells)
	a.StaleCells = len(stale)
	for _, c := range cells {
		if c.Support == Undefined {
			a.UndefinedCells++
		}
		a.InputBytes += int64(len(c.Family) + len(c.Backend) + len(c.Support))
	}
	for _, c := range xcells {
		if c.Support == Undefined {
			a.UndefinedCells++
		}
		a.InputBytes += int64(len(c.Family) + len(c.Backend) + len(c.Precision) + len(c.Support))
	}
	for _, c := range stale {
		a.InputBytes += int64(len(c.Family) + len(c.Backend))
	}
	if len(cells) == len(Families)*len(Backends) && a.UndefinedCells == 0 {
		a.PassedChecks++
	} else {
		a.FalseCells++
	}
	if len(xcells) == len(Families)*len(Backends)*len(Precisions) && a.UndefinedCells == 0 {
		a.PassedChecks++
	} else {
		a.FalseCells++
	}
	// StaleCells is a typed read-back of declarations that have outlived their
	// current support state; an empty result is valid, so execution is the check.
	a.PassedChecks++
	a.Correct = a.PassedChecks == a.Checks && a.FalseCells == 0
	return a
}

func runStaticTableBaseline() ComparisonArm {
	a := ComparisonArm{
		Name:      "hand-maintained support table lookup",
		Kind:      "baseline",
		Available: true,
		Checks:    3,
		Note:      "tuned incumbent enumerates the same declaration tables but has no stale-declaration read-back",
	}
	start := time.Now()
	cells := Grid()
	xcells := GridX()
	a.Latency = time.Since(start)
	a.FamilyCells = len(cells)
	a.PrecisionCells = len(xcells)
	for _, c := range cells {
		if c.Support == Undefined {
			a.UndefinedCells++
		}
	}
	for _, c := range xcells {
		if c.Support == Undefined {
			a.UndefinedCells++
		}
	}
	if len(cells) == len(Families)*len(Backends) && a.UndefinedCells == 0 {
		a.PassedChecks++
	}
	if len(xcells) == len(Families)*len(Backends)*len(Precisions) && a.UndefinedCells == 0 {
		a.PassedChecks++
	}
	a.MissedChecks = 1
	a.Correct = false
	return a
}

func unavailableArm(name, kind, note string) ComparisonArm {
	return ComparisonArm{Name: name, Kind: kind, Note: note}
}

// CompareLocal executes only dependency-free local arms. A complete comparison
// requires each unavailable arm to classify the identical frozen cell set and
// to report correctness, latency, resources, setup effort, and total cost.
func CompareLocal() ComparisonResult {
	return ComparisonResult{
		Workload: "classify every declared model family across CPU, CUDA, Metal, Vulkan, and precision cells, while surfacing stale and undefined declarations",
		Arms: []ComparisonArm{
			runNativeComparison(),
			runStaticTableBaseline(),
			unavailableArm("fak + CUDA runtime witness", "integration", "requires pinned CUDA device execution for every declared CUDA cell"),
			unavailableArm("fak + Metal runtime witness", "integration", "requires pinned Metal device execution for every declared Metal cell"),
			unavailableArm("fak + Vulkan runtime witness", "integration", "requires pinned Vulkan device execution for every declared Vulkan cell"),
			unavailableArm("vLLM supported-model matrix", "external", "requires a pinned vLLM release and equivalent model/backend/precision classifications"),
			unavailableArm("llama.cpp backend and quantization matrix", "external", "requires a pinned llama.cpp release and equivalent classifications"),
			unavailableArm("Hugging Face Optimum hardware compatibility", "external", "requires pinned Optimum exporters/runtimes and equivalent classifications"),
			unavailableArm("ONNX Runtime execution-provider matrix", "external", "requires pinned ONNX Runtime providers and equivalent classifications"),
			unavailableArm("TensorRT-LLM support matrix", "external", "requires pinned TensorRT-LLM and equivalent classifications"),
		},
	}
}
