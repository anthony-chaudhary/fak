package main

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The committed planner parameters. The workflows record the same numbers in
// their env and comments; these tests fail when the matrix, the env and the
// weights disagree with what the planner produces from them.
const (
	ciFastParallel      = 2
	ciFastMaxParallel   = 20
	raceParallel        = 1
	raceMaxParallel     = 12
	committedBucketSkew = 1.3
)

var (
	shardEntryRE = regexp.MustCompile(`(?m)^\s*- \{ shard: (\d+), shards: (\d+) \}\s*$`)
	splitEntryRE = regexp.MustCompile(`(?m)^\s*- \{ split: (\./\S+), bucket: (\d+), buckets: (\d+) \}\s*$`)
)

// fak-test:runtime fast est=10ms
func TestCostModelUnitSeconds(t *testing.T) {
	cases := []struct {
		name     string
		parallel int
		overhead float64
		costs    []float64
		want     float64
	}{
		{"serial share dominates", 2, 150, []float64{10, 10, 10, 10}, 170},
		{"heaviest package dominates", 2, 150, []float64{100, 5}, 250},
		{"serial runs sum", 1, 100, []float64{30, 50}, 180},
		{"empty unit is overhead only", 2, 150, nil, 150},
		{"parallel below one is serial", 0, 10, []float64{4, 4}, 18},
	}
	for _, c := range cases {
		m := CostModel{Parallel: c.parallel, UnitOverhead: c.overhead}
		if got := m.UnitSeconds(c.costs); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestCostModelBucketSeconds(t *testing.T) {
	w := Weights{Default: 1, Overhead: 2, Packages: map[string]float64{"p/big": 100}}
	cases := []struct {
		skew float64
		k    int
		want float64
	}{
		{1.3, 4, 34.5},
		{0, 4, 27},
		{1, 1, 102},
	}
	for _, c := range cases {
		m := CostModel{Weights: w, Skew: c.skew}
		if got := m.BucketSeconds("p/big", c.k); got != c.want {
			t.Errorf("skew %v k %d: got %v, want %v", c.skew, c.k, got, c.want)
		}
	}
	if got := (CostModel{Weights: w}).BucketSeconds("p/unknown", 2); got != 2.5 {
		t.Errorf("unknown package must price at the default: got %v", got)
	}
}

// fak-test:runtime fast est=10ms
func TestMakespanIsLongestProcessingTime(t *testing.T) {
	cases := []struct {
		secs  []float64
		slots int
		want  float64
	}{
		{[]float64{5, 5, 5, 5}, 2, 10},
		{[]float64{5, 5, 5, 5}, 1, 20},
		{[]float64{5, 5, 5, 5}, 0, 5},
		{[]float64{5, 5, 5, 5}, 8, 5},
		{[]float64{9, 4, 4, 3, 3, 3}, 2, 14},
		{nil, 2, 0},
	}
	for _, c := range cases {
		if got := Makespan(c.secs, c.slots); got != c.want {
			t.Errorf("secs %v slots %d: got %v, want %v", c.secs, c.slots, got, c.want)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestMinShardsAndAutoSplitOnSyntheticWeights(t *testing.T) {
	w := Weights{Schema: weightsSchema, Default: 1, Packages: map[string]float64{
		"p/a": 10, "p/b": 10, "p/c": 10, "p/d": 10, "p/e": 10, "p/f": 10,
		"p/big": 100,
	}}
	m := CostModel{Weights: w, Parallel: 2, UnitOverhead: 0, Skew: 1}
	six := []string{"p/a", "p/b", "p/c", "p/d", "p/e", "p/f"}
	if n, err := MinShards(six, m, nil, Budgets{Total: 25}, 64); err != nil || n != 2 {
		t.Fatalf("MinShards: got %d, %v; want 2", n, err)
	}

	heavy := CostModel{Weights: w, Parallel: 2, UnitOverhead: 10, Skew: 1}
	split, err := AutoSplit([]string{"p/big", "p/a"}, heavy, Budgets{Total: 60}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := map[string]int{"p/big": 2}; !reflect.DeepEqual(split, want) {
		t.Fatalf("AutoSplit: got %v, want %v", split, want)
	}
	p, err := BuildPlan([]string{"p/big", "p/a"}, heavy, 1, split, Budgets{Total: 60}, 0)
	if err != nil || len(p.Violations) != 0 {
		t.Fatalf("split plan must fit: %v %+v", err, p.Violations)
	}
}

// fak-test:runtime fast est=10ms
func TestBuildPlanReportsUnsplitHeavyPackage(t *testing.T) {
	w := Weights{Schema: weightsSchema, Default: 1, Packages: map[string]float64{"p/big": 200, "p/a": 1}}
	m := CostModel{Weights: w, Parallel: 2, UnitOverhead: 10}
	p, err := BuildPlan([]string{"p/big", "p/a"}, m, 2, nil, Budgets{Total: 100}, 0)
	if err != nil {
		t.Fatal(err)
	}
	var kinds []string
	for _, v := range p.Violations {
		kinds = append(kinds, v.Kind)
	}
	sort.Strings(kinds)
	if want := []string{"unit_over_budget", "unsplit_package_over_budget"}; !reflect.DeepEqual(kinds, want) {
		t.Fatalf("violation kinds %v, want %v", kinds, want)
	}
	if err := runPlan([]string{"-weights", writeWeights(t, w), "-unit-budget", "100", "-overhead", "10", "-shards", "2", "-check"}, strings.NewReader("p/big\np/a\n"), &bytes.Buffer{}); err == nil {
		t.Fatal("-check must fail on a violating plan")
	}
}

// fak-test:runtime fast est=10ms
func TestPlanCommandMinShardsPrintsCount(t *testing.T) {
	w := writeWeights(t, Weights{Schema: weightsSchema, Default: 10, Packages: map[string]float64{}})
	var out bytes.Buffer
	err := runPlan([]string{"-weights", w, "-unit-budget", "25", "-parallel", "2", "-overhead", "0", "-min-shards"}, strings.NewReader("p/a\np/b\np/c\np/d\np/e\np/f\n"), &out)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(out.String()); got != "2" {
		t.Fatalf("min-shards printed %q, want 2", got)
	}
}

// fak-test:runtime fast est=10ms
func TestParseSplitAcceptsRelativeAndFullNames(t *testing.T) {
	got, err := ParseSplit("./cmd/fak=4, " + modulePath + "/internal/x=2")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]int{modulePath + "/cmd/fak": 4, modulePath + "/internal/x": 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for _, bad := range []string{"./cmd/fak", "./cmd/fak=0", "./cmd/fak=x"} {
		if _, err := ParseSplit(bad); err == nil {
			t.Errorf("%q must be refused", bad)
		}
	}
}

// TestCommittedCIFastPlanFitsBudget grades the committed ci-fast matrix: the
// planner, run on the committed weights with the workflow's overhead, must
// produce this shard count and these split buckets, with no violations.
// fak-test:runtime fast est=50ms
func TestCommittedCIFastPlanFitsBudget(t *testing.T) {
	text := readWorkflow(t, "ci-fast.yml")
	w := committedWeights(t)
	m := CostModel{Weights: w, Parallel: ciFastParallel, UnitOverhead: envNumber(t, text, "CI_FAST_UNIT_OVERHEAD_S"), Skew: committedBucketSkew}
	shape := parseShape(t, jobBlock(t, text, "go-test-shard"))
	forced := map[string]int{modulePath + "/internal/model": 3}
	assertCommittedPlan(t, m, weightPackages(w), Budgets{Test: envNumber(t, text, "CI_FAST_TEST_BUDGET_S")}, ciFastMaxParallel, forced, shape)
	assertEnvMatchesShape(t, text, "CI_FAST_SHARDS", "CI_FAST_BUCKETS", "CI_FAST_SPLIT_PACKAGES", shape)
}

// TestCommittedRacePlanFitsBudget grades the committed race-detector matrix the
// same way, against the race weights.
// fak-test:runtime fast est=50ms
func TestCommittedRacePlanFitsBudget(t *testing.T) {
	text := readWorkflow(t, "ci.yml")
	w, err := loadWeights(repoFile(t, ".github/race-package-weights.json"))
	if err != nil {
		t.Fatal(err)
	}
	m := CostModel{Weights: w, Parallel: raceParallel, UnitOverhead: envNumber(t, text, "RACE_UNIT_OVERHEAD_S"), Skew: committedBucketSkew}
	shape := parseShape(t, jobBlock(t, text, "race-detector"))
	assertCommittedPlan(t, m, weightPackages(w), Budgets{Total: envNumber(t, text, "RACE_UNIT_BUDGET_S")}, raceMaxParallel, nil, shape)
	assertEnvMatchesShape(t, text, "", "", "RACE_SPLIT_PACKAGES", shape)
}

// The unit-wall doctrine caps a job at 15 minutes. ci-fast keeps a recorded
// interim exception: its predicted units (up to about 880 s) still exceed the
// line until the test-variant cache replaces the coverage-superset restore.
const (
	doctrineJobMinutes       = 15
	ciFastInterimJobMinutes  = 20
	ciFastInterimStepMinutes = 18
	ciFastInterimTestMinutes = 12
	raceTestMinutes          = 12
)

// fak-test:runtime fast est=10ms
func TestCommittedWorkflowTimeoutsWithinUnitDoctrine(t *testing.T) {
	ci := readWorkflow(t, "ci-fast.yml")
	shard := jobBlock(t, ci, "go-test-shard")
	if n := envNumberIn(t, shard, "timeout-minutes"); n != ciFastInterimJobMinutes {
		t.Errorf("go-test-shard timeout-minutes %d, want the recorded interim %d", n, ciFastInterimJobMinutes)
	}
	if n := envNumberIn(t, jobBlock(t, ci, "build-vet-fast"), "timeout-minutes"); n > doctrineJobMinutes {
		t.Errorf("build-vet-fast timeout-minutes %d exceeds %d", n, doctrineJobMinutes)
	}
	for _, m := range regexp.MustCompile(`timeout-minutes: (\d+)`).FindAllStringSubmatch(ci, -1) {
		if n, _ := strconv.Atoi(m[1]); n > ciFastInterimJobMinutes {
			t.Errorf("ci-fast.yml timeout-minutes %d exceeds the recorded interim %d", n, ciFastInterimJobMinutes)
		}
	}
	if !strings.Contains(ci, fmt.Sprintf("go test -timeout=%dm -skip", ciFastInterimTestMinutes)) {
		t.Errorf("ci-fast.yml go test -timeout must be %dm", ciFastInterimTestMinutes)
	}
	if strings.Contains(ci, "-timeout=10m") || strings.Contains(ci, "-timeout=20m") {
		t.Error("ci-fast.yml must not keep a superseded -timeout")
	}
	race := jobBlock(t, readWorkflow(t, "ci.yml"), "race-detector")
	if n := envNumberIn(t, race, "timeout-minutes"); n > doctrineJobMinutes {
		t.Errorf("race-detector timeout-minutes %d exceeds %d", n, doctrineJobMinutes)
	}
	if !strings.Contains(race, fmt.Sprintf("-timeout=%dm", raceTestMinutes)) {
		t.Errorf("race-detector go test -timeout must be %dm", raceTestMinutes)
	}
}

// The race job's -timeout=12m is only honest while every predicted unit plus the
// 120 s tail fits the 15-minute job, so the committed plan must prove it.
// fak-test:runtime fast est=50ms
func TestCommittedRaceUnitsFitTwelveMinuteTestBound(t *testing.T) {
	text := readWorkflow(t, "ci.yml")
	w, err := loadWeights(repoFile(t, ".github/race-package-weights.json"))
	if err != nil {
		t.Fatal(err)
	}
	m := CostModel{Weights: w, Parallel: raceParallel, UnitOverhead: envNumber(t, text, "RACE_UNIT_OVERHEAD_S"), Skew: committedBucketSkew}
	shape := parseShape(t, jobBlock(t, text, "race-detector"))
	plan, err := BuildPlan(weightPackages(w), m, shape.shards, shape.split, Budgets{Total: envNumber(t, text, "RACE_UNIT_BUDGET_S")}, raceMaxParallel)
	if err != nil {
		t.Fatal(err)
	}
	const tailSeconds = 120
	if plan.MaxUnit+tailSeconds > float64(doctrineJobMinutes*60) {
		t.Fatalf("slowest race unit %.1fs + %ds tail must fit %d s", plan.MaxUnit, tailSeconds, doctrineJobMinutes*60)
	}
}

// Every bucket and every shard of a split package pays its package compile, so
// the compile charge must enter the predicted seconds the limits are checked against.
// fak-test:runtime fast est=10ms
func TestCompileChargeEntersBucketAndShardSeconds(t *testing.T) {
	w := Weights{Schema: weightsSchema, Default: 1, CompileDefault: 50, CompileShard: 40,
		CompilePackages: map[string]float64{"p/warm": 90}, Packages: map[string]float64{"p/warm": 100, "p/cold": 100}}
	m := CostModel{Weights: w, Parallel: 1, UnitOverhead: 10, Skew: 1}
	if got := m.BucketSeconds("p/warm", 2); got != 50+90 {
		t.Errorf("warm bucket: got %v, want 140", got)
	}
	if got := m.BucketSeconds("p/cold", 2); got != 50+50 {
		t.Errorf("cold bucket: got %v, want 100", got)
	}
	p, err := BuildPlan([]string{"p/a", "p/b"}, m, 1, nil, Budgets{Total: 1000}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := p.Units[0].CompileSecs; got != 40 {
		t.Errorf("shard compile charge: got %v, want 40", got)
	}
	// A package whose whole-unit cost plus the shard charge exceeds the limit
	// must be split, or no shard count can ever fit it.
	split, err := AutoSplit([]string{"p/cold"}, CostModel{Weights: Weights{Schema: weightsSchema, Default: 1, CompileShard: 40,
		Packages: map[string]float64{"p/cold": 200}}, Parallel: 1, UnitOverhead: 0, Skew: 1}, Budgets{Total: 200}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := split["p/cold"]; !ok {
		t.Fatalf("a package of 200 s with a 40 s shard charge must split under a 200 s limit; got %v", split)
	}
}

type matrixShape struct {
	shards int
	split  map[string]int
	lines  []string
}

// parseShape reads a job's matrix entries in the form the planner prints them.
func parseShape(t *testing.T, block string) matrixShape {
	t.Helper()
	var shape matrixShape
	shape.split = map[string]int{}
	shardTotal := -1
	for _, m := range shardEntryRE.FindAllStringSubmatch(block, -1) {
		n, _ := strconv.Atoi(m[2])
		if shardTotal >= 0 && n != shardTotal {
			t.Fatalf("shard entries disagree on the total: %d and %d", shardTotal, n)
		}
		shardTotal = n
		shape.lines = append(shape.lines, fmt.Sprintf("- { shard: %s, shards: %s }", m[1], m[2]))
	}
	shape.shards = shardTotal
	for _, m := range splitEntryRE.FindAllStringSubmatch(block, -1) {
		k, _ := strconv.Atoi(m[3])
		pkg := modulePath + "/" + strings.TrimPrefix(m[1], "./")
		shape.split[pkg] = k
		shape.lines = append(shape.lines, fmt.Sprintf("- { split: %s, bucket: %s, buckets: %s }", m[1], m[2], m[3]))
	}
	if shape.shards < 1 {
		t.Fatal("job has no `{ shard: i, shards: N }` entries")
	}
	return shape
}

func assertCommittedPlan(t *testing.T, m CostModel, pkgs []string, b Budgets, maxParallel int, forced map[string]int, got matrixShape) {
	t.Helper()
	auto, err := AutoSplit(pkgs, m, b, forced)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(auto, got.split) {
		t.Fatalf("matrix splits %v, planner wants %v (weights or overhead changed without regenerating the matrix)", got.split, auto)
	}
	n, err := MinShards(pkgs, m, got.split, b, 512)
	if err != nil {
		t.Fatal(err)
	}
	if n != got.shards {
		t.Fatalf("matrix has %d shards, the smallest plan under budget has %d", got.shards, n)
	}
	plan, err := BuildPlan(pkgs, m, got.shards, got.split, b, maxParallel)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Violations) != 0 {
		t.Fatalf("committed plan violates its limits %+v: %+v", b, plan.Violations)
	}
	want := MatrixLines(plan)
	have := append([]string(nil), got.lines...)
	sort.Strings(want)
	sort.Strings(have)
	if !reflect.DeepEqual(have, want) {
		t.Fatalf("matrix drifted from planner output:\nmatrix:  %v\nplanner: %v", have, want)
	}
	var shardPkgs []string
	for _, u := range plan.Units {
		if u.Kind == "shard" {
			shardPkgs = append(shardPkgs, u.Packages...)
		}
	}
	var rest []string
	for _, p := range pkgs {
		if _, ok := got.split[p]; !ok {
			rest = append(rest, p)
		}
	}
	requireExactlyOnce(t, [][]string{shardPkgs}, rest)
	t.Logf("shards %d, split %v, units %d, max unit %.1fs (test %.1fs), summed %.0fs, makespan at %d: %.0fs",
		plan.Shards, plan.Split, plan.UnitCount, plan.MaxUnit, plan.MaxTest, plan.SummedUnitSeconds, maxParallel, plan.MakespanSeconds)
}

// assertEnvMatchesShape checks that the workflow env the verdict and the shard
// step read agrees with the matrix. Empty names are skipped.
func assertEnvMatchesShape(t *testing.T, text, shardsEnv, bucketsEnv, splitEnv string, shape matrixShape) {
	t.Helper()
	if shardsEnv != "" {
		if got := envNumber(t, text, shardsEnv); got != float64(shape.shards) {
			t.Errorf("%s = %v, matrix has %d shards", shardsEnv, got, shape.shards)
		}
	}
	if bucketsEnv != "" {
		total := 0
		for _, k := range shape.split {
			total += k
		}
		if got := envNumber(t, text, bucketsEnv); got != float64(total) {
			t.Errorf("%s = %v, matrix has %d bucket jobs", bucketsEnv, got, total)
		}
	}
	if splitEnv != "" {
		var rels []string
		for pkg := range shape.split {
			rels = append(rels, "./"+strings.TrimPrefix(pkg, modulePath+"/"))
		}
		sort.Strings(rels)
		listed := strings.Fields(envValue(t, text, splitEnv))
		sort.Strings(listed)
		if !reflect.DeepEqual(listed, rels) {
			t.Errorf("%s = %v, matrix split packages are %v", splitEnv, listed, rels)
		}
	}
}

func readWorkflow(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile(repoFile(t, ".github/workflows/"+name))
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// jobBlock returns the text of one top-level job: from its `  <id>:` line to
// the next job key at the same indent.
func jobBlock(t *testing.T, text, id string) string {
	t.Helper()
	lines := strings.Split(text, "\n")
	var out []string
	in := false
	for _, line := range lines {
		if line == "  "+id+":" {
			in = true
			continue
		}
		if in && regexp.MustCompile(`^  \S.*:\s*$`).MatchString(line) {
			break
		}
		if in {
			out = append(out, line)
		}
	}
	if !in {
		t.Fatalf("no job %q", id)
	}
	return strings.Join(out, "\n")
}

func envValue(t *testing.T, text, name string) string {
	t.Helper()
	m := regexp.MustCompile(`(?m)^\s+` + regexp.QuoteMeta(name) + `: "([^"]*)"\s*$`).FindStringSubmatch(text)
	if m == nil {
		t.Fatalf("no env %s", name)
	}
	return m[1]
}

func envNumber(t *testing.T, text, name string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(envValue(t, text, name), 64)
	if err != nil {
		t.Fatalf("env %s: %v", name, err)
	}
	return v
}

func envNumberIn(t *testing.T, block, key string) int {
	t.Helper()
	m := regexp.MustCompile(`(?m)^\s+` + regexp.QuoteMeta(key) + `: (\d+)\s*$`).FindStringSubmatch(block)
	if m == nil {
		t.Fatalf("no %s", key)
	}
	n, _ := strconv.Atoi(m[1])
	return n
}

func weightPackages(w Weights) []string {
	var pkgs []string
	for p := range w.Packages {
		pkgs = append(pkgs, p)
	}
	sort.Strings(pkgs)
	return pkgs
}

func writeWeights(t *testing.T, w Weights) string {
	t.Helper()
	path := t.TempDir() + "/w.json"
	data := fmt.Sprintf(`{"schema":%q,"default_seconds":%v,"per_package_overhead_seconds":%v,"packages":{`, w.Schema, w.Default, w.Overhead)
	var parts []string
	for p, s := range w.Packages {
		parts = append(parts, fmt.Sprintf("%q:%v", p, s))
	}
	sort.Strings(parts)
	data += strings.Join(parts, ",") + "}}"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
