// hostsetup-bench — per-layer HOST SETUP vs KERNEL microbench for identical math
// kernels at small batch (the launch-bound decode regime).
//
// The issue class this measures: at batch 1 each decoded token walks every layer,
// and the host pays a fixed setup cost per layer (operand/dtype/layout validation,
// quant-block resolution, workspace sizing, plan freeze) BEFORE the kernel does any
// arithmetic. When the per-layer kernel is tiny (small GEMVs), that setup cost can
// dominate the step — the regime GLM52-DECODE-PATH-TO-10-TOKS localizes as ~1100
// synchronous round-trips/token. Every earlier improvement in that stack measured
// the setup side INDIRECTLY (glue-census counts sync round-trips; launch-census
// counts dispatch sites; hostshare reads a receipt-derived host_share). This bench
// measures it DIRECTLY: for N layers x {setup-only vs setup+launch} over the SAME
// prepared Q8_0 GEMV kernels, it reports setup_us_per_layer, kernel_us_per_layer,
// and host_fraction = setup/(setup+kernel), with a regression gate
// (host_fraction > threshold => LAUNCH_SETUP_DOMINANT, exit 1 under --gate).
//
// Seam: internal/compute's Planner/MatMulPlan pair. PrepareMatMul is the host-setup
// arm (it freezes the compatibility envelope and sizes the workspace without doing
// the math); MatMulPlan.Run is the kernel arm (the prepared Q8_0 matrix-vector op +
// the Read fence that makes its result usable). Both arms run the identical weight
// tensors and activation, so their difference isolates the kernel cost.
//
// Device posture: the CUDA path is wired through the same seam (backend selection
// via the compute registry) but is SKIPPED with a reported reason whenever the
// requested backend is not registered in this build or lacks the planned-runner
// seam; the measured path then runs on the cpu-ref fallback and is labeled
// [SW-VERIFIED]. A physical-silicon reading requires a -tags cuda build on a GPU
// host; this bench never fabricates one.

package devcmd

import (
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	hostSetupBenchSchema = "fak.dev.hostsetup-bench.v1"

	// hostSetupVerdictDominant fires when per-layer host setup exceeds its share of
	// the step: the step is launch/setup-bound and kernel time is not the lever.
	hostSetupVerdictDominant = "LAUNCH_SETUP_DOMINANT"
	// hostSetupVerdictBalanced: setup stays within budget; kernel time dominates.
	hostSetupVerdictBalanced = "BALANCED"

	// defaultHostSetupThreshold mirrors the hostshare/regime launch-bound threshold
	// (0.30): above it, more than a third of the per-layer step is host setup.
	defaultHostSetupThreshold = 0.30

	verificationSWVerified = "sw-verified"
	verificationDevice     = "device"
)

// HostSetupBenchShape parameterizes the microbench. Batch is fixed at 1 (decode);
// Out/In are the per-layer weight dimensions of the representative GEMV.
type HostSetupBenchShape struct {
	Name   string `json:"name,omitempty"`
	Layers int    `json:"layers"`
	Out    int    `json:"out"`
	In     int    `json:"in"`
	Block  int    `json:"block"`
}

// normalize fills defaults and validates against the planned-matmul seam's
// constraints (Q8_0 block divisible by 4, exactly dividing the input width).
func (s HostSetupBenchShape) normalize() (HostSetupBenchShape, error) {
	out := s
	if out.Layers <= 0 {
		return out, fmt.Errorf("layers must be >= 1 (got %d)", out.Layers)
	}
	if out.Layers > 4096 {
		return out, fmt.Errorf("layers=%d exceeds the 4096 sanity cap", out.Layers)
	}
	if out.Out == 0 {
		out.Out = 1024
	}
	if out.In == 0 {
		out.In = 1024
	}
	if out.Out < 0 || out.In < 0 {
		return out, fmt.Errorf("out/in must be >= 0 (got %d/%d)", out.Out, out.In)
	}
	if out.Block == 0 {
		out.Block = 32
	}
	if out.Block%4 != 0 {
		return out, fmt.Errorf("quant block=%d must be divisible by 4 (planned-matmul seam constraint)", out.Block)
	}
	if out.In%out.Block != 0 {
		return out, fmt.Errorf("in=%d must be an exact multiple of block=%d", out.In, out.Block)
	}
	if out.In < out.Block {
		return out, fmt.Errorf("in=%d smaller than block=%d", out.In, out.Block)
	}
	return out, nil
}

// hostSetupBenchPresets are named shapes. decode-small approximates one
// projection-sized GEMV per layer at a size where CPU kernel time is comparable to
// host setup; micro exists for fast CI runs and the gate test.
var hostSetupBenchPresets = map[string]HostSetupBenchShape{
	"decode-small": {Name: "decode-small", Layers: 8, Out: 1024, In: 1024, Block: 32},
	"micro":        {Name: "micro", Layers: 4, Out: 128, In: 128, Block: 32},
}

// HostSetupBenchArmStats is one timed arm, reduced to per-layer microseconds.
type HostSetupBenchArmStats struct {
	MeanUsPerLayer   float64 `json:"mean_us_per_layer"`
	MedianUsPerLayer float64 `json:"median_us_per_layer"`
}

// HostSetupBenchReps records the auto-repeat factor each arm needed for its
// samples to clear the host clock's resolution (the arms differ by orders of
// magnitude, so they are calibrated independently).
type HostSetupBenchReps struct {
	SetupOnly       int `json:"setup_only"`
	SetupPlusLaunch int `json:"setup_plus_launch"`
}

// HostSetupBenchSkip records a device backend that was wired but not measurable.
type HostSetupBenchSkip struct {
	Backend string `json:"backend"`
	Reason  string `json:"reason"`
}

// HostSetupBenchResult is the full bench payload (JSON form of the report).
type HostSetupBenchResult struct {
	Schema           string                 `json:"schema"`
	Root             string                 `json:"root"`
	HeadSHA          string                 `json:"head_sha,omitempty"`
	BackendRequested string                 `json:"backend_requested"`
	BackendMeasured  string                 `json:"backend_measured"`
	BackendTier      string                 `json:"backend_tier"`
	BackendClass     string                 `json:"backend_class"`
	Verification     string                 `json:"verification"` // sw-verified | device
	Skipped          []HostSetupBenchSkip   `json:"skipped,omitempty"`
	Shape            HostSetupBenchShape    `json:"shape"`
	Warmup           int                    `json:"warmup"`
	Iters            int                    `json:"iters"`
	RepsPerSample    HostSetupBenchReps     `json:"reps_per_sample"`
	SetupOnly        HostSetupBenchArmStats `json:"setup_only"`
	SetupPlusLaunch  HostSetupBenchArmStats `json:"setup_plus_launch"`
	KernelUsPerLayer float64                `json:"kernel_us_per_layer"`
	HostFraction     float64                `json:"host_fraction"`
	Threshold        float64                `json:"threshold"`
	Gate             bool                   `json:"gate"`
	Verdict          string                 `json:"verdict"`
}

const hostSetupBenchUsage = `usage: fak-dev hostsetup-bench [--preset NAME | --layers N --out D --in D --block B]
                                [--backend auto|NAME] [--warmup W] [--iters K]
                                [--threshold F] [--gate] [--json]

Measure per-layer HOST SETUP vs KERNEL time for identical Q8_0 GEMV kernels at
batch 1 (the launch-bound decode regime), on the compute Planner seam:

  setup-only arm      N x PrepareMatMul (envelope freeze + workspace sizing)
  setup+launch arm    N x (PrepareMatMul + MatMulPlan.Run + Read fence)

Reports setup_us_per_layer, kernel_us_per_layer (= launch arm minus setup arm),
and host_fraction = setup / (setup + kernel). Verdict: host_fraction > threshold
(default 0.30) => LAUNCH_SETUP_DOMINANT; with --gate that exits 1.

Each measured sample auto-repeats the N-layer walk as many times as needed for
the sample to span >= 2ms on this host (coarse host clocks quantize smaller
intervals to zero); reps_per_sample reports the factor actually used.

Backends: --backend auto picks cuda when registered in this build, else cpu-ref.
A requested backend that is unavailable (not built, or missing the planned-runner
seam) is SKIPPED with a reported reason and the measured path falls back to
cpu-ref, labeled [SW-VERIFIED]. Physical-silicon readings need a -tags cuda build
on a GPU host; this bench never fabricates them.

Exit codes: 0 PASS/informational, 1 gate FAIL (LAUNCH_SETUP_DOMINANT with --gate),
2 usage/seam error.`

// hostSetupLayer is one simulated model layer: a prepared-eligible Q8_0 weight
// tensor plus its caller-owned Run workspace.
type hostSetupLayer struct {
	w         compute.Tensor
	workspace []byte
}

// resolveHostSetupBackend picks the measurement backend from the request, filling
// skips for every device backend that is wired but not measurable here.
func resolveHostSetupBackend(requested string) (be compute.Backend, planner compute.Planner, skips []HostSetupBenchSkip, err error) {
	want := requested
	if want == "" || want == "auto" {
		want = "cuda"
		if _, ok := compute.Lookup("cuda"); !ok {
			want = "cpu-ref"
		}
	}
	candidate, ok := compute.Lookup(want)
	if !ok {
		reason := fmt.Sprintf("not registered in this build (%s needs its cgo/device build tags + libraries)", want)
		if want == "cuda" {
			reason = "not registered in this build (CUDA needs -tags cuda + libfakcuda.a)"
		}
		skips = append(skips, HostSetupBenchSkip{Backend: want, Reason: reason})
		candidate, ok = compute.Lookup("cpu-ref")
		if !ok {
			return nil, nil, skips, fmt.Errorf("cpu-ref fallback backend is not registered")
		}
	}
	p, hasPlanner := candidate.(compute.Planner)
	if !hasPlanner {
		if candidate.Name() != "cpu-ref" {
			skips = append(skips, HostSetupBenchSkip{
				Backend: candidate.Name(),
				Reason:  "planned-runner seam (compute.Planner) absent on this backend",
			})
			fallback, ok := compute.Lookup("cpu-ref")
			if !ok {
				return nil, nil, skips, fmt.Errorf("cpu-ref fallback backend is not registered")
			}
			p, hasPlanner = fallback.(compute.Planner)
			candidate = fallback
			if !hasPlanner {
				return nil, nil, skips, fmt.Errorf("cpu-ref lacks the planned-runner seam (internal drift)")
			}
		} else {
			return nil, nil, skips, fmt.Errorf("cpu-ref lacks the planned-runner seam (internal drift)")
		}
	}
	return candidate, p, skips, nil
}

// buildHostSetupLayers constructs the deterministic weight set: one Q8_0
// [out,in] tensor per layer plus the shared batch-1 activation. The seed is fixed
// so a given shape always benches the same math.
func buildHostSetupLayers(be compute.Backend, shape HostSetupBenchShape) ([]hostSetupLayer, compute.Tensor, error) {
	rng := rand.New(rand.NewPCG(0x5EED, 0xBEEF))
	xdata := make([]float32, shape.In)
	for i := range xdata {
		xdata[i] = rng.Float32()*2 - 1
	}
	x := compute.NewF32(be, []int{shape.In}, xdata)
	layers := make([]hostSetupLayer, 0, shape.Layers)
	for li := 0; li < shape.Layers; li++ {
		codes := make([]int8, shape.Out*shape.In)
		for i := range codes {
			codes[i] = int8(rng.IntN(256) - 128)
		}
		nblk := shape.In / shape.Block
		scale := make([]float32, shape.Out*nblk)
		for i := range scale {
			scale[i] = 0.01 + rng.Float32()*0.09
		}
		w := compute.NewQ8(be, []int{shape.Out, shape.In}, codes, scale, shape.Block)
		layers = append(layers, hostSetupLayer{w: w})
	}
	return layers, x, nil
}

// validateHostSetupSeam prepares every layer once (outside all timing) so a bad
// shape fails deterministically before any measurement, and records each layer's
// caller-owned workspace.
func validateHostSetupSeam(planner compute.Planner, layers []hostSetupLayer, x compute.Tensor) error {
	for i := range layers {
		plan, err := planner.PrepareMatMul(layers[i].w, x)
		if err != nil {
			return fmt.Errorf("layer %d prepare: %v", i, err)
		}
		layers[i].workspace = make([]byte, plan.Descriptor().WorkspaceBytes)
	}
	return nil
}

// benchHostSetupArm times one arm over samples measurements; each measurement
// spans reps sequential full N-layer walks. launch=false is the setup-only arm;
// launch=true adds the prepared kernel run and the Read fence per layer. sink
// absorbs results so no work elides.
func benchHostSetupArm(be compute.Backend, planner compute.Planner, layers []hostSetupLayer, x compute.Tensor, samples, reps int, launch bool) ([]time.Duration, error) {
	durs := make([]time.Duration, 0, samples)
	var sink float64
	for i := 0; i < samples; i++ {
		start := time.Now()
		for r := 0; r < reps; r++ {
			for li := range layers {
				plan, err := planner.PrepareMatMul(layers[li].w, x)
				if err != nil {
					return nil, fmt.Errorf("sample %d rep %d layer %d prepare: %v", i, r, li, err)
				}
				if launch {
					out, _, err := plan.Run(be, layers[li].w, x, layers[li].workspace)
					if err != nil {
						return nil, fmt.Errorf("sample %d rep %d layer %d run: %v", i, r, li, err)
					}
					y := be.Read(out)
					sink += float64(y[0])
				}
			}
		}
		durs = append(durs, time.Since(start))
	}
	_ = sink
	return durs, nil
}

const (
	// minSampleWindow is the smallest per-sample interval the bench trusts: host
	// clocks (notably coarse Windows/VM timers) quantize sub-half-millisecond
	// intervals to zero, which would silently zero out the whole report.
	minSampleWindow = 2 * time.Millisecond
	// maxRepsPerSample caps the auto-repeat growth so a pathological shape cannot
	// turn one sample into minutes of wall clock.
	maxRepsPerSample = 8192
)

// calibrateRepsPerSample picks the reps-per-sample factor so one measured sample
// of the given arm spans at least minSampleWindow on THIS host: it doubles reps
// until a single sample clears the window (or hits the cap). On hosts whose
// clocks resolve sub-millisecond work, small shapes otherwise measure as exactly
// zero. The two arms are calibrated INDEPENDENTLY because they differ by orders
// of magnitude: a factor sized for the kernel arm leaves the setup-only arm
// inside the clock's dead zone (and vice versa).
func calibrateRepsPerSample(be compute.Backend, planner compute.Planner, layers []hostSetupLayer, x compute.Tensor, launch bool) (int, error) {
	reps := 1
	for {
		durs, err := benchHostSetupArm(be, planner, layers, x, 1, reps, launch)
		if err != nil {
			return 0, err
		}
		if durs[0] >= minSampleWindow || reps >= maxRepsPerSample {
			return reps, nil
		}
		reps *= 2
	}
}

func meanDuration(durs []time.Duration) time.Duration {
	if len(durs) == 0 {
		return 0
	}
	var total time.Duration
	for _, d := range durs {
		total += d
	}
	return total / time.Duration(len(durs))
}

func medianDuration(durs []time.Duration) time.Duration {
	if len(durs) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), durs...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	mid := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return sorted[mid]
	}
	return (sorted[mid-1] + sorted[mid]) / 2
}

func usPerLayer(total time.Duration, layers int) float64 {
	if layers == 0 {
		return 0
	}
	return float64(total) / float64(time.Microsecond) / float64(layers)
}

// RunHostSetupBench implements the fak-dev hostsetup-bench verb.
// hostSetupBenchShapeFlags carries per-field numeric overrides; nil means
// "not given" (the preset/JSON value stands).
type hostSetupBenchShapeFlags struct {
	layers *int
	out    *int
	in     *int
	block  *int
}

func (f *hostSetupBenchShapeFlags) applyTo(s *HostSetupBenchShape) {
	if f.layers != nil {
		s.Layers = *f.layers
	}
	if f.out != nil {
		s.Out = *f.out
	}
	if f.in != nil {
		s.In = *f.in
	}
	if f.block != nil {
		s.Block = *f.block
	}
}

func parsePositiveInt(flag, raw string) (int, error) {
	v, err := strconv.Atoi(raw)
	if err != nil || v < 1 {
		return 0, fmt.Errorf("%s needs a positive integer (got %q)", flag, raw)
	}
	return v, nil
}

func RunHostSetupBench(stdout, stderr io.Writer, argv []string) int {
	preset := "decode-small"
	shapeJSON := ""
	shapeFlags := &hostSetupBenchShapeFlags{}
	backend := "auto"
	warmup := 4
	iters := 16
	threshold := defaultHostSetupThreshold
	gate := false
	asJSON := false
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "-h" || arg == "--help":
			fmt.Fprintln(stdout, hostSetupBenchUsage)
			return 0
		case arg == "--preset":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "hostsetup-bench: --preset needs a value")
				return 2
			}
			i++
			preset = argv[i]
		case strings.HasPrefix(arg, "--preset="):
			preset = strings.TrimPrefix(arg, "--preset=")
		case arg == "--model-shape":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "hostsetup-bench: --model-shape needs a JSON value")
				return 2
			}
			i++
			shapeJSON = argv[i]
		case strings.HasPrefix(arg, "--model-shape="):
			shapeJSON = strings.TrimPrefix(arg, "--model-shape=")
		case arg == "--layers", strings.HasPrefix(arg, "--layers="):
			var raw string
			if arg == "--layers" {
				if i+1 >= len(argv) {
					fmt.Fprintln(stderr, "hostsetup-bench: --layers needs a value")
					return 2
				}
				i++
				raw = argv[i]
			} else {
				raw = strings.TrimPrefix(arg, "--layers=")
			}
			v, err := parsePositiveInt("--layers", raw)
			if err != nil {
				fmt.Fprintln(stderr, "hostsetup-bench: "+err.Error())
				return 2
			}
			shapeFlags.layers = &v
		case arg == "--out", strings.HasPrefix(arg, "--out="):
			var raw string
			if arg == "--out" {
				if i+1 >= len(argv) {
					fmt.Fprintln(stderr, "hostsetup-bench: --out needs a value")
					return 2
				}
				i++
				raw = argv[i]
			} else {
				raw = strings.TrimPrefix(arg, "--out=")
			}
			v, err := parsePositiveInt("--out", raw)
			if err != nil {
				fmt.Fprintln(stderr, "hostsetup-bench: "+err.Error())
				return 2
			}
			shapeFlags.out = &v
		case arg == "--in", strings.HasPrefix(arg, "--in="):
			var raw string
			if arg == "--in" {
				if i+1 >= len(argv) {
					fmt.Fprintln(stderr, "hostsetup-bench: --in needs a value")
					return 2
				}
				i++
				raw = argv[i]
			} else {
				raw = strings.TrimPrefix(arg, "--in=")
			}
			v, err := parsePositiveInt("--in", raw)
			if err != nil {
				fmt.Fprintln(stderr, "hostsetup-bench: "+err.Error())
				return 2
			}
			shapeFlags.in = &v
		case arg == "--block", strings.HasPrefix(arg, "--block="):
			var raw string
			if arg == "--block" {
				if i+1 >= len(argv) {
					fmt.Fprintln(stderr, "hostsetup-bench: --block needs a value")
					return 2
				}
				i++
				raw = argv[i]
			} else {
				raw = strings.TrimPrefix(arg, "--block=")
			}
			v, err := parsePositiveInt("--block", raw)
			if err != nil {
				fmt.Fprintln(stderr, "hostsetup-bench: "+err.Error())
				return 2
			}
			shapeFlags.block = &v
		case arg == "--backend":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "hostsetup-bench: --backend needs a value")
				return 2
			}
			i++
			backend = argv[i]
		case strings.HasPrefix(arg, "--backend="):
			backend = strings.TrimPrefix(arg, "--backend=")
		case arg == "--warmup":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "hostsetup-bench: --warmup needs a value")
				return 2
			}
			i++
			v, err := strconv.Atoi(argv[i])
			if err != nil || v < 0 {
				fmt.Fprintln(stderr, "hostsetup-bench: --warmup needs a non-negative integer")
				return 2
			}
			warmup = v
		case strings.HasPrefix(arg, "--warmup="):
			v, err := strconv.Atoi(strings.TrimPrefix(arg, "--warmup="))
			if err != nil || v < 0 {
				fmt.Fprintln(stderr, "hostsetup-bench: --warmup needs a non-negative integer")
				return 2
			}
			warmup = v
		case arg == "--iters":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "hostsetup-bench: --iters needs a value")
				return 2
			}
			i++
			v, err := strconv.Atoi(argv[i])
			if err != nil || v < 1 {
				fmt.Fprintln(stderr, "hostsetup-bench: --iters needs a positive integer")
				return 2
			}
			iters = v
		case strings.HasPrefix(arg, "--iters="):
			v, err := strconv.Atoi(strings.TrimPrefix(arg, "--iters="))
			if err != nil || v < 1 {
				fmt.Fprintln(stderr, "hostsetup-bench: --iters needs a positive integer")
				return 2
			}
			iters = v
		case arg == "--threshold":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "hostsetup-bench: --threshold needs a value")
				return 2
			}
			i++
			v, err := strconv.ParseFloat(argv[i], 64)
			if err != nil || v <= 0 || v >= 1 {
				fmt.Fprintln(stderr, "hostsetup-bench: --threshold needs a float in (0,1)")
				return 2
			}
			threshold = v
		case strings.HasPrefix(arg, "--threshold="):
			v, err := strconv.ParseFloat(strings.TrimPrefix(arg, "--threshold="), 64)
			if err != nil || v <= 0 || v >= 1 {
				fmt.Fprintln(stderr, "hostsetup-bench: --threshold needs a float in (0,1)")
				return 2
			}
			threshold = v
		case arg == "--gate":
			gate = true
		case arg == "--json":
			asJSON = true
		default:
			fmt.Fprintln(stderr, hostSetupBenchUsage)
			return 2
		}
	}

	var shape HostSetupBenchShape
	if shapeJSON != "" {
		if err := json.Unmarshal([]byte(shapeJSON), &shape); err != nil {
			fmt.Fprintf(stderr, "hostsetup-bench: parse --model-shape: %v\n", err)
			return 2
		}
		if shape.Name == "" {
			shape.Name = "custom"
		}
	} else {
		ps, ok := hostSetupBenchPresets[preset]
		if !ok {
			names := make([]string, 0, len(hostSetupBenchPresets))
			for n := range hostSetupBenchPresets {
				names = append(names, n)
			}
			sort.Strings(names)
			fmt.Fprintf(stderr, "hostsetup-bench: unknown preset %q (known: %s)\n", preset, strings.Join(names, ", "))
			return 2
		}
		shape = ps
	}
	shapeFlags.applyTo(&shape)
	normed, err := shape.normalize()
	if err != nil {
		fmt.Fprintf(stderr, "hostsetup-bench: invalid shape: %v\n", err)
		return 2
	}
	shape = normed

	be, planner, skips, err := resolveHostSetupBackend(backend)
	if err != nil {
		fmt.Fprintf(stderr, "hostsetup-bench: backend resolution: %v\n", err)
		return 2
	}
	requested := backend
	if requested == "" || requested == "auto" {
		requested = "auto"
	}

	layers, x, err := buildHostSetupLayers(be, shape)
	if err != nil {
		fmt.Fprintf(stderr, "hostsetup-bench: build layers: %v\n", err)
		return 2
	}
	if err := validateHostSetupSeam(planner, layers, x); err != nil {
		fmt.Fprintf(stderr, "hostsetup-bench: seam validation: %v\n", err)
		return 2
	}

	repsSetup, err := calibrateRepsPerSample(be, planner, layers, x, false)
	if err != nil {
		fmt.Fprintf(stderr, "hostsetup-bench: setup-only calibration: %v\n", err)
		return 2
	}
	repsLaunch, err := calibrateRepsPerSample(be, planner, layers, x, true)
	if err != nil {
		fmt.Fprintf(stderr, "hostsetup-bench: setup+launch calibration: %v\n", err)
		return 2
	}

	// Warm up BOTH arms at their calibrated reps: the first PrepareMatMul/Run pay
	// allocation and icache costs that would otherwise land in the measured means.
	if _, err := benchHostSetupArm(be, planner, layers, x, warmup, repsSetup, false); err != nil {
		fmt.Fprintf(stderr, "hostsetup-bench: setup-only warmup: %v\n", err)
		return 2
	}
	if _, err := benchHostSetupArm(be, planner, layers, x, warmup, repsLaunch, true); err != nil {
		fmt.Fprintf(stderr, "hostsetup-bench: setup+launch warmup: %v\n", err)
		return 2
	}
	setupDurs, err := benchHostSetupArm(be, planner, layers, x, iters, repsSetup, false)
	if err != nil {
		fmt.Fprintf(stderr, "hostsetup-bench: setup-only arm: %v\n", err)
		return 2
	}
	launchDurs, err := benchHostSetupArm(be, planner, layers, x, iters, repsLaunch, true)
	if err != nil {
		fmt.Fprintf(stderr, "hostsetup-bench: setup+launch arm: %v\n", err)
		return 2
	}

	// Each sample spans its arm's reps sequential N-layer walks; divide back to
	// the per-walk (per-step) figure before reducing to per-layer microseconds.
	setupMean := meanDuration(setupDurs) / time.Duration(repsSetup)
	launchMean := meanDuration(launchDurs) / time.Duration(repsLaunch)
	kernelMean := launchMean - setupMean
	if kernelMean < 0 {
		kernelMean = 0
	}
	hostFraction := 0.0
	if launchMean > 0 {
		hostFraction = float64(setupMean) / float64(launchMean)
	}
	verdict := hostSetupVerdictBalanced
	if hostFraction > threshold {
		verdict = hostSetupVerdictDominant
	}

	verification := verificationDevice
	if be.Class() == compute.Reference {
		verification = verificationSWVerified
	}

	result := HostSetupBenchResult{
		Schema:           hostSetupBenchSchema,
		Root:             repoRoot(),
		HeadSHA:          gitHeadSHA(repoRoot()),
		BackendRequested: requested,
		BackendMeasured:  be.Name(),
		BackendTier:      be.Tier(),
		BackendClass:     be.Class().String(),
		Verification:     verification,
		Skipped:          skips,
		Shape:            shape,
		Warmup:           warmup,
		Iters:            iters,
		RepsPerSample: HostSetupBenchReps{
			SetupOnly:       repsSetup,
			SetupPlusLaunch: repsLaunch,
		},
		SetupOnly: HostSetupBenchArmStats{
			MeanUsPerLayer:   usPerLayer(setupMean, shape.Layers),
			MedianUsPerLayer: usPerLayer(medianDuration(setupDurs)/time.Duration(repsSetup), shape.Layers),
		},
		SetupPlusLaunch: HostSetupBenchArmStats{
			MeanUsPerLayer:   usPerLayer(launchMean, shape.Layers),
			MedianUsPerLayer: usPerLayer(medianDuration(launchDurs)/time.Duration(repsLaunch), shape.Layers),
		},
		KernelUsPerLayer: usPerLayer(kernelMean, shape.Layers),
		HostFraction:     hostFraction,
		Threshold:        threshold,
		Gate:             gate,
		Verdict:          verdict,
	}

	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "hostsetup-bench: encode: %v\n", err)
			return 2
		}
	} else {
		writeHostSetupBenchText(stdout, &result)
	}

	if gate && verdict == hostSetupVerdictDominant {
		return 1
	}
	return 0
}

func writeHostSetupBenchText(w io.Writer, r *HostSetupBenchResult) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	label := ""
	if r.Verification == verificationSWVerified {
		label = " [SW-VERIFIED]"
	}
	fmt.Fprintf(tw, "HOSTSETUP-BENCH\troot=%s head=%s\n", r.Root, shortSHA(r.HeadSHA))
	fmt.Fprintf(tw, "BACKEND\trequested=%s measured=%s tier=%s class=%s%s\n",
		r.BackendRequested, r.BackendMeasured, r.BackendTier, r.BackendClass, label)
	for _, s := range r.Skipped {
		fmt.Fprintf(tw, "SKIPPED\t%s: %s\n", s.Backend, s.Reason)
	}
	fmt.Fprintf(tw, "SHAPE\tlayers=%d out=%d in=%d block=%d batch=1 preset=%s\n",
		r.Shape.Layers, r.Shape.Out, r.Shape.In, r.Shape.Block, r.Shape.Name)
	fmt.Fprintf(tw, "ARM\tMEAN us/layer\tMEDIAN us/layer\t(warmup=%d iters=%d reps/sample setup=%d launch=%d)\n",
		r.Warmup, r.Iters, r.RepsPerSample.SetupOnly, r.RepsPerSample.SetupPlusLaunch)
	fmt.Fprintf(tw, "setup-only\t%.3f\t%.3f\t\n", r.SetupOnly.MeanUsPerLayer, r.SetupOnly.MedianUsPerLayer)
	fmt.Fprintf(tw, "setup+launch\t%.3f\t%.3f\t\n", r.SetupPlusLaunch.MeanUsPerLayer, r.SetupPlusLaunch.MedianUsPerLayer)
	fmt.Fprintf(tw, "setup_us_per_layer\t%.3f\n", r.SetupOnly.MeanUsPerLayer)
	fmt.Fprintf(tw, "kernel_us_per_layer\t%.3f\n", r.KernelUsPerLayer)
	fmt.Fprintf(tw, "host_fraction\t%.4f (threshold %.2f)\n", r.HostFraction, r.Threshold)
	gateState := "off"
	if r.Gate {
		gateState = "ON (exit 1 on LAUNCH_SETUP_DOMINANT)"
	}
	fmt.Fprintf(tw, "verdict\t%s (gate %s)\n", r.Verdict, gateState)
	if err := tw.Flush(); err != nil {
		fmt.Fprintf(w, "(table flush failed: %v)\n", err)
	}
}

// shortSHA trims a commit SHA to its citation form.
func shortSHA(full string) string {
	if len(full) > 9 {
		return full[:9]
	}
	return full
}
