package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

const modulePrefix = "github.com/anthony-chaudhary/fak/"

func repoFile(t *testing.T, rel string) string {
	t.Helper()
	return filepath.Join("..", "..", filepath.FromSlash(rel))
}

func committedWeights(t *testing.T) Weights {
	t.Helper()
	w, err := loadWeights(repoFile(t, ".github/ci-fast-package-weights.json"))
	if err != nil {
		t.Fatal(err)
	}
	return w
}

// workflowShape reads the shard count and split packages the workflow really
// uses, so these tests grade the committed configuration rather than a copy.
func workflowShape(t *testing.T) (int, map[string]bool) {
	t.Helper()
	data, err := os.ReadFile(repoFile(t, ".github/workflows/ci-fast.yml"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	m := regexp.MustCompile(`\{ shard: 1, shards: (\d+) \}`).FindStringSubmatch(text)
	if m == nil {
		t.Fatal("ci-fast.yml has no `{ shard: 1, shards: N }` matrix entry")
	}
	shards, _ := strconv.Atoi(m[1])
	s := regexp.MustCompile(`(?m)^  CI_FAST_SPLIT_PACKAGES: "([^"]*)"$`).FindStringSubmatch(text)
	if s == nil {
		t.Fatal("ci-fast.yml has no CI_FAST_SPLIT_PACKAGES")
	}
	split := map[string]bool{}
	for _, rel := range strings.Fields(s[1]) {
		split[modulePrefix+strings.TrimPrefix(rel, "./")] = true
	}
	return shards, split
}

func requireExactlyOnce(t *testing.T, shards [][]string, want []string) {
	t.Helper()
	seen := map[string]int{}
	for _, s := range shards {
		if len(s) == 0 {
			t.Error("a shard is empty")
		}
		for _, p := range s {
			seen[p]++
		}
	}
	for _, p := range want {
		if seen[p] != 1 {
			t.Errorf("package %s assigned %d times", p, seen[p])
		}
		delete(seen, p)
	}
	for p := range seen {
		t.Errorf("shards invented package %s", p)
	}
}

func committedPackages(t *testing.T, w Weights, split map[string]bool) []string {
	t.Helper()
	var pkgs []string
	for p := range w.Packages {
		if !split[p] {
			pkgs = append(pkgs, p)
		}
	}
	sort.Strings(pkgs)
	return pkgs
}

// fak-test:runtime fast est=50ms
func TestPartitionCoversEveryPackageOnceIncludingUnknown(t *testing.T) {
	w := committedWeights(t)
	n, split := workflowShape(t)
	pkgs := append(committedPackages(t, w, split), modulePrefix+"internal/not-yet-weighed", modulePrefix+"cmd/also-new")
	shards := Partition(pkgs, w, n)
	if len(shards) != n {
		t.Fatalf("got %d shards, want %d", len(shards), n)
	}
	requireExactlyOnce(t, shards, pkgs)
}

// fak-test:runtime fast est=50ms
func TestPartitionIgnoresInputOrderAndDuplicates(t *testing.T) {
	w := committedWeights(t)
	n, split := workflowShape(t)
	pkgs := committedPackages(t, w, split)
	reversed := make([]string, 0, len(pkgs)*2)
	for i := len(pkgs) - 1; i >= 0; i-- {
		reversed = append(reversed, pkgs[i], pkgs[i])
	}
	if a, b := Partition(pkgs, w, n), Partition(reversed, w, n); !reflect.DeepEqual(a, b) {
		t.Fatal("partition depends on input order or duplicates")
	}
}

// The point of weighting: the slowest shard sets the gate's wall, so on the
// committed timings the heaviest shard must stay close to the lightest.
// fak-test:runtime fast est=50ms
func TestPartitionBalancesCommittedWeights(t *testing.T) {
	w := committedWeights(t)
	n, split := workflowShape(t)
	pkgs := committedPackages(t, w, split)
	loads := ShardLoads(Partition(pkgs, w, n), w)
	lo, hi := loads[0], loads[0]
	var total float64
	for _, l := range loads {
		lo, hi = min(lo, l), max(hi, l)
		total += l
	}
	var heaviest float64
	for _, p := range pkgs {
		heaviest = max(heaviest, float64(w.cost(p))/1000)
	}
	// Greedy largest-first can overshoot the mean by at most one package.
	if hi > total/float64(n)+heaviest {
		t.Fatalf("heaviest shard %.0fs exceeds mean %.0fs plus heaviest package %.0fs", hi, total/float64(n), heaviest)
	}
	if hi/lo > 1.15 {
		t.Fatalf("shard loads %v: max/min %.2f exceeds 1.15", loads, hi/lo)
	}
	t.Logf("modeled shard seconds: %v", loads)
}

// fak-test:runtime fast est=10ms
func TestBucketsSplitListedTestsDisjointly(t *testing.T) {
	var names []string
	for i := 0; i < 103; i++ {
		names = append(names, fmt.Sprintf("Test%03d", i))
	}
	names = append(names, "ExampleRun", "FuzzParse", "BenchmarkHot", "ok  \tgithub.com/x\t0.1s", "Test003")
	for _, k := range []int{1, 2, 4, 7} {
		buckets := Buckets(names, k)
		seen := map[string]int{}
		for _, b := range buckets {
			if !sort.StringsAreSorted(b) {
				t.Errorf("k=%d: bucket not sorted: %v", k, b)
			}
			for _, n := range b {
				seen[n]++
			}
		}
		for _, n := range names {
			want := 1
			if !testName.MatchString(n) {
				want = 0
			}
			if seen[n] != want {
				t.Errorf("k=%d: %q in %d buckets, want %d", k, n, seen[n], want)
			}
		}
		if len(buckets[0])-len(buckets[k-1]) > 1 {
			t.Errorf("k=%d: bucket sizes differ by more than one", k)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestParseWeightsReadsRunLogLines(t *testing.T) {
	log := strings.Join([]string{
		"go test · shard 1/4\tUNKNOWN STEP\t2026-01-01T00:00:00.0Z ok  \texample.invalid/a\t12.345s",
		"ok  \texample.invalid/b\t(cached)",
		"FAIL\texample.invalid/c\t3.0s",
		"ok  \texample.invalid/d\t0.04s [no tests to run]",
		"?   \texample.invalid/e\t[no test files]",
	}, "\n")
	got, err := ParseWeights(strings.NewReader(log))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]float64{"example.invalid/a": 12.3, "example.invalid/c": 3, "example.invalid/d": 0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

// fak-test:runtime fast est=10ms
func TestParseWeightsKeepsMaximumOfRepeatedLines(t *testing.T) {
	log := "ok  \texample.invalid/a\t5.0s\nok  \texample.invalid/a\t9.0s\nFAIL\texample.invalid/a\t7.0s\n"
	got, err := ParseWeights(strings.NewReader(log))
	if err != nil {
		t.Fatal(err)
	}
	if got["example.invalid/a"] != 9 {
		t.Fatalf("repeated lines must keep the maximum, got %v", got["example.invalid/a"])
	}
}

// fak-test:runtime fast est=10ms
func TestWeightsNeverRecordsBucketPartials(t *testing.T) {
	log := "ok  \texample.invalid/cmd/fak\t300.0s\nok  \texample.invalid/whole\t12.0s\n"
	var out bytes.Buffer
	if err := runWeights([]string{"-partial", "example.invalid/cmd/fak"}, strings.NewReader(log), &out); err != nil {
		t.Fatal(err)
	}
	var w Weights
	if err := json.Unmarshal(out.Bytes(), &w); err != nil {
		t.Fatal(err)
	}
	if _, ok := w.Packages["example.invalid/cmd/fak"]; ok {
		t.Fatal("a bucket's partial time must not become the package weight")
	}
	if w.Packages["example.invalid/whole"] != 12 {
		t.Fatalf("whole package lost: %v", w.Packages)
	}
}

// fak-test:runtime fast est=50ms
func TestPartitionCommandHonorsExcludeFile(t *testing.T) {
	dir := t.TempDir()
	weights := filepath.Join(dir, "w.json")
	if err := os.WriteFile(weights, []byte(`{"schema":"`+weightsSchema+`","default_seconds":1,"per_package_overhead_seconds":0,"packages":{"p/a":5,"p/b":4}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	exclude := filepath.Join(dir, "x.txt")
	if err := os.WriteFile(exclude, []byte("p/big\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var union []string
	for i := 0; i < 2; i++ {
		var out bytes.Buffer
		err := runPartition([]string{"-weights", weights, "-shards", "2", "-index", strconv.Itoa(i), "-exclude-file", exclude}, strings.NewReader("p/a\np/b\np/big\np/c\n"), &out)
		if err != nil {
			t.Fatal(err)
		}
		union = append(union, strings.Fields(out.String())...)
	}
	sort.Strings(union)
	if want := []string{"p/a", "p/b", "p/c"}; !reflect.DeepEqual(union, want) {
		t.Fatalf("union %v, want %v", union, want)
	}
	if err := runPartition([]string{"-weights", weights, "-shards", "2", "-index", "2"}, strings.NewReader("p/a\n"), &bytes.Buffer{}); err == nil {
		t.Fatal("out-of-range index must be refused")
	}
}
