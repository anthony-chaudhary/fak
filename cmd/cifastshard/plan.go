package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
)

// modulePath prefixes the workflows' relative package names (./cmd/fak) so they
// can be compared with the weights keys, which are full import paths.
const modulePath = "github.com/anthony-chaudhary/fak"

// CostModel prices one CI unit. A unit is one matrix job: it pays a fixed
// UnitOverhead (setup, cache restore, compile and link residual), then runs its
// packages with Parallel-way package parallelism. Its test portion is the larger
// of the serial share and the single heaviest package.
type CostModel struct {
	Weights      Weights
	Parallel     int
	UnitOverhead float64
	// Skew scales a bucket's share of its package's time. Buckets split by test
	// name, not by measured cost, so one bucket can run well above weight/k.
	Skew float64
}

// Budgets caps a unit. Total bounds its predicted wall, overhead included; Test
// bounds only the test portion. A zero field disables that limit.
type Budgets struct {
	Total float64
	Test  float64
}

// testLimit is the largest test portion a unit may carry under both limits.
func (b Budgets) testLimit(overhead float64) float64 {
	lim := math.Inf(1)
	if b.Test > 0 {
		lim = b.Test
	}
	if b.Total > 0 {
		lim = math.Min(lim, b.Total-overhead)
	}
	return lim
}

// PackageSeconds is the cost of one whole package in a shard: its measured or
// default test time plus the per-package compile and link charge.
func (m CostModel) PackageSeconds(pkg string) float64 {
	return float64(m.Weights.cost(pkg)) / 1000
}

// BucketCompile is the test-binary compile and link charge a split package's
// bucket pays. Every bucket recompiles the package, so the charge is per bucket.
func (m CostModel) BucketCompile(pkg string) float64 {
	if c, ok := m.Weights.CompilePackages[pkg]; ok {
		return c
	}
	return m.Weights.CompileDefault
}

// BucketSeconds is the test portion of one of k disjoint -run buckets of pkg:
// its share of the package's test time, the per-package overhead, and one
// compile and link charge.
func (m CostModel) BucketSeconds(pkg string, k int) float64 {
	s, ok := m.Weights.Packages[pkg]
	if !ok || s < 0 {
		s = m.Weights.Default
	}
	skew := m.Skew
	if skew == 0 {
		skew = 1
	}
	return skew*s/float64(k) + m.Weights.Overhead + m.BucketCompile(pkg)
}

// TestSeconds is a unit's test portion: max(sum/p, max) over its package costs.
func (m CostModel) TestSeconds(costs []float64) float64 {
	p := m.Parallel
	if p < 1 {
		p = 1
	}
	var sum, top float64
	for _, c := range costs {
		sum += c
		top = math.Max(top, c)
	}
	return math.Max(sum/float64(p), top)
}

// UnitSeconds predicts a unit's wall: UnitOverhead + TestSeconds.
func (m CostModel) UnitSeconds(costs []float64) float64 {
	return m.UnitOverhead + m.TestSeconds(costs)
}

// Unit is one predicted job: a shard of whole packages, or one bucket of a split package.
type Unit struct {
	Name         string   `json:"name"`
	Kind         string   `json:"kind"`
	Index        int      `json:"index"`
	Count        int      `json:"count"`
	Package      string   `json:"package,omitempty"`
	Packages     []string `json:"packages,omitempty"`
	OverheadSecs float64  `json:"overhead_seconds"`
	CompileSecs  float64  `json:"compile_seconds"`
	TestSecs     float64  `json:"test_seconds"`
	Seconds      float64  `json:"predicted_seconds"`
}

// Violation is a unit over a limit, or a package over the test limit that no split covers.
type Violation struct {
	Kind      string  `json:"kind"`
	Unit      string  `json:"unit,omitempty"`
	Package   string  `json:"package,omitempty"`
	Portion   string  `json:"portion"`
	Predicted float64 `json:"predicted_seconds"`
	Budget    float64 `json:"budget_seconds"`
}

// Plan is the planner's output and the JSON report.
type Plan struct {
	Shards            int            `json:"shards"`
	Split             map[string]int `json:"split"`
	UnitBudget        float64        `json:"unit_budget_seconds"`
	TestBudget        float64        `json:"test_budget_seconds"`
	Parallel          int            `json:"parallel"`
	UnitOverhead      float64        `json:"unit_overhead_seconds"`
	BucketSkew        float64        `json:"bucket_skew"`
	MaxParallel       int            `json:"max_parallel"`
	UnitCount         int            `json:"unit_count"`
	MaxUnit           float64        `json:"max_unit"`
	MaxTest           float64        `json:"max_test"`
	SummedUnitSeconds float64        `json:"summed_unit_seconds"`
	MakespanSeconds   float64        `json:"makespan_seconds"`
	Units             []Unit         `json:"units"`
	Violations        []Violation    `json:"violations"`
}

// BuildPlan partitions the non-split packages into shards and prices every
// unit against b. maxParallel <= 0 means every unit runs at once.
func BuildPlan(pkgs []string, m CostModel, shards int, split map[string]int, b Budgets, maxParallel int) (Plan, error) {
	if shards < 1 {
		return Plan{}, errors.New("shards must be >= 1")
	}
	lim := b.testLimit(m.UnitOverhead)
	seen := map[string]bool{}
	rest := []string{}
	for _, p := range pkgs {
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		if _, ok := split[p]; !ok {
			rest = append(rest, p)
		}
	}
	units := []Unit{}
	viol := []Violation{}
	for i, s := range Partition(rest, m.Weights, shards) {
		costs := make([]float64, len(s))
		for j, p := range s {
			costs[j] = m.PackageSeconds(p)
		}
		u := newUnit(fmt.Sprintf("shard %d/%d", i+1, shards), "shard", i+1, shards, "", s, m.UnitOverhead, m.TestSeconds(costs)+m.Weights.CompileShard, m.Weights.CompileShard)
		if len(s) == 0 {
			viol = append(viol, Violation{Kind: "empty_unit", Unit: u.Name, Portion: "total", Budget: b.Total})
		}
		units = append(units, u)
	}
	for _, p := range sortedKeys(split) {
		k := split[p]
		if k < 1 {
			return Plan{}, fmt.Errorf("split %s: bucket count %d must be >= 1", p, k)
		}
		for j := 1; j <= k; j++ {
			units = append(units, newUnit(fmt.Sprintf("bucket %s %d/%d", p, j, k), "bucket", j, k, p, nil, m.UnitOverhead, m.BucketSeconds(p, k), m.BucketCompile(p)))
		}
	}
	secs := make([]float64, 0, len(units))
	var summed, top, topTest float64
	for _, u := range units {
		secs = append(secs, u.Seconds)
		summed += u.Seconds
		top = math.Max(top, u.Seconds)
		topTest = math.Max(topTest, u.TestSecs)
		if u.TestSecs > lim {
			viol = append(viol, unitViolation(u, b, m.UnitOverhead))
		}
	}
	for _, p := range rest {
		if c := m.PackageSeconds(p) + m.Weights.CompileShard; c > lim {
			viol = append(viol, packageViolation(p, c, b, m.UnitOverhead))
		}
	}
	return Plan{
		Shards:            shards,
		Split:             split,
		UnitBudget:        b.Total,
		TestBudget:        b.Test,
		Parallel:          m.Parallel,
		UnitOverhead:      m.UnitOverhead,
		BucketSkew:        m.Skew,
		MaxParallel:       maxParallel,
		UnitCount:         len(units),
		MaxUnit:           round3(top),
		MaxTest:           round3(topTest),
		SummedUnitSeconds: round3(summed),
		MakespanSeconds:   round3(Makespan(secs, maxParallel)),
		Units:             units,
		Violations:        viol,
	}, nil
}

// newUnit prices one unit. test already includes the compile charge; compile is
// reported separately so the plan shows how much of the test portion it is.
func newUnit(name, kind string, index, count int, pkg string, pkgs []string, overhead, test, compile float64) Unit {
	return Unit{Name: name, Kind: kind, Index: index, Count: count, Package: pkg, Packages: pkgs,
		OverheadSecs: round3(overhead), CompileSecs: round3(compile), TestSecs: round3(test), Seconds: round3(overhead + test)}
}

// bindingPortion reports which limit is the tighter one: the test portion or the total.
func bindingPortion(b Budgets, overhead float64) string {
	if b.Test > 0 && (b.Total <= 0 || b.Test <= b.Total-overhead) {
		return "test"
	}
	return "total"
}

func unitViolation(u Unit, b Budgets, overhead float64) Violation {
	if bindingPortion(b, overhead) == "test" {
		return Violation{Kind: "unit_over_budget", Unit: u.Name, Portion: "test", Predicted: u.TestSecs, Budget: b.Test}
	}
	return Violation{Kind: "unit_over_budget", Unit: u.Name, Portion: "total", Predicted: u.Seconds, Budget: b.Total}
}

func packageViolation(pkg string, test float64, b Budgets, overhead float64) Violation {
	if bindingPortion(b, overhead) == "test" {
		return Violation{Kind: "unsplit_package_over_budget", Package: pkg, Portion: "test", Predicted: round3(test), Budget: b.Test}
	}
	return Violation{Kind: "unsplit_package_over_budget", Package: pkg, Portion: "total", Predicted: round3(test + overhead), Budget: b.Total}
}

// Makespan is the longest-processing-time schedule of unit seconds on slots
// concurrent runners: largest unit first, each to the currently least-loaded slot.
func Makespan(secs []float64, slots int) float64 {
	var top float64
	for _, s := range secs {
		top = math.Max(top, s)
	}
	if slots < 1 || slots >= len(secs) {
		return top
	}
	sorted := append([]float64(nil), secs...)
	sort.Sort(sort.Reverse(sort.Float64Slice(sorted)))
	load := make([]float64, slots)
	for _, s := range sorted {
		best := 0
		for i := 1; i < slots; i++ {
			if load[i] < load[best] {
				best = i
			}
		}
		load[best] += s
	}
	var out float64
	for _, l := range load {
		out = math.Max(out, l)
	}
	return out
}

// MinShards returns the smallest shard count whose plan has no violations.
func MinShards(pkgs []string, m CostModel, split map[string]int, b Budgets, maxN int) (int, error) {
	for n := 1; n <= maxN; n++ {
		p, err := BuildPlan(pkgs, m, n, split, b, 0)
		if err != nil {
			return 0, err
		}
		if len(p.Violations) == 0 {
			return n, nil
		}
	}
	return 0, fmt.Errorf("no shard count up to %d fits the limits", maxN)
}

// AutoSplit gives every package whose own test portion exceeds the limit the
// smallest bucket count whose bucket fits. Packages already in split keep their count.
func AutoSplit(pkgs []string, m CostModel, b Budgets, split map[string]int) (map[string]int, error) {
	lim := b.testLimit(m.UnitOverhead)
	out := map[string]int{}
	for p, k := range split {
		out[p] = k
	}
	seen := map[string]bool{}
	for _, p := range pkgs {
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		// A package that cannot share a shard within the limit, compile charge
		// included, must be split into buckets.
		if _, ok := out[p]; ok || m.PackageSeconds(p)+m.Weights.CompileShard <= lim {
			continue
		}
		s, ok := m.Weights.Packages[p]
		if !ok || s < 0 {
			s = m.Weights.Default
		}
		skew := m.Skew
		if skew == 0 {
			skew = 1
		}
		room := lim - m.Weights.Overhead - m.BucketCompile(p)
		if room <= 0 {
			return nil, fmt.Errorf("%s: no bucket fits: limit %.0fs leaves %.0fs after the per-bucket charges", p, lim, room)
		}
		out[p] = int(math.Ceil(skew * s / room))
	}
	return out, nil
}

// MatrixLines renders the plan as the workflows' matrix entries.
func MatrixLines(p Plan) []string {
	var out []string
	for _, u := range p.Units {
		switch u.Kind {
		case "shard":
			out = append(out, fmt.Sprintf("- { shard: %d, shards: %d }", u.Index, u.Count))
		case "bucket":
			out = append(out, fmt.Sprintf("- { split: %s, bucket: %d, buckets: %d }", relPackage(u.Package), u.Index, u.Count))
		}
	}
	return out
}

func relPackage(pkg string) string {
	return "./" + strings.TrimPrefix(pkg, modulePath+"/")
}

// ParseSplit reads "pkg=k,pkg=k" where pkg is a ./relative path or full import path.
func ParseSplit(spec string) (map[string]int, error) {
	out := map[string]int{}
	for _, item := range strings.Split(spec, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		name, count, ok := strings.Cut(item, "=")
		if !ok {
			return nil, fmt.Errorf("split %q: want pkg=buckets", item)
		}
		k, err := strconv.Atoi(strings.TrimSpace(count))
		if err != nil || k < 1 {
			return nil, fmt.Errorf("split %q: bucket count must be a positive integer", item)
		}
		name = strings.TrimSpace(name)
		if strings.HasPrefix(name, "./") {
			name = modulePath + "/" + strings.TrimPrefix(name, "./")
		}
		out[name] = k
	}
	return out, nil
}

func sortedKeys(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func round3(x float64) float64 { return math.Round(x*1000) / 1000 }

// runPlan prints the plan report, the smallest shard count, or matrix entries.
func runPlan(args []string, in io.Reader, out io.Writer) error {
	fs := flag.NewFlagSet("plan", flag.ContinueOnError)
	weightsPath := fs.String("weights", "", "weights JSON file")
	unitBudget := fs.Float64("unit-budget", 240, "predicted seconds one unit may take, overhead included (0 = no total limit)")
	testBudget := fs.Float64("test-budget", 0, "seconds one unit's test portion may take (0 = no test limit)")
	parallel := fs.Int("parallel", 2, "package parallelism inside one unit (-p)")
	overhead := fs.Float64("overhead", 150, "seconds per unit that are not test time (setup, restore, compile residual)")
	maxParallel := fs.Int("max-parallel", 0, "concurrent units for the makespan (0 = all at once)")
	skew := fs.Float64("bucket-skew", 1, "multiplier on a bucket's weight/k share")
	splitSpec := fs.String("split", "", "split packages as pkg=buckets,...")
	shards := fs.Int("shards", 0, "shard count to report")
	minShards := fs.Bool("min-shards", false, "print the smallest shard count with no violations")
	maxShards := fs.Int("max-shards", 512, "upper bound for -min-shards")
	autoSplit := fs.Bool("auto-split", false, "add the smallest bucket count to every package too heavy for one unit")
	matrix := fs.Bool("matrix", false, "print workflow matrix entries instead of the JSON report")
	check := fs.Bool("check", false, "exit non-zero when the plan has violations")
	if err := fs.Parse(args); err != nil {
		return err
	}
	w, err := loadWeights(*weightsPath)
	if err != nil {
		return err
	}
	split, err := ParseSplit(*splitSpec)
	if err != nil {
		return err
	}
	pkgLines, err := readLines(in)
	if err != nil {
		return err
	}
	m := CostModel{Weights: w, Parallel: *parallel, UnitOverhead: *overhead, Skew: *skew}
	b := Budgets{Total: *unitBudget, Test: *testBudget}
	if *autoSplit {
		if split, err = AutoSplit(pkgLines, m, b, split); err != nil {
			return err
		}
	}
	if *minShards {
		n, err := MinShards(pkgLines, m, split, b, *maxShards)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintln(out, n)
		return err
	}
	if *shards < 1 {
		return errors.New("plan needs -shards N or -min-shards")
	}
	p, err := BuildPlan(pkgLines, m, *shards, split, b, *maxParallel)
	if err != nil {
		return err
	}
	if *matrix {
		for _, line := range MatrixLines(p) {
			fmt.Fprintln(out, line)
		}
	} else {
		data, err := json.MarshalIndent(p, "", "  ")
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "%s\n", data)
	}
	if *check && len(p.Violations) > 0 {
		return fmt.Errorf("plan has %d violations", len(p.Violations))
	}
	return nil
}
