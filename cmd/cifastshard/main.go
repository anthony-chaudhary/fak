// Command cifastshard divides ci-fast's `go test` work into parallel jobs.
//
//	cifastshard partition -weights F -shards N -index I [-exclude-file X] < go-list-output
//	cifastshard buckets -buckets K -index I < go-test-list-output
//	cifastshard weights [-default S] [-overhead S] < go-test-log
//
// partition prints shard I's packages: every package from stdin lands in exactly
// one of N shards by a deterministic greedy fill on committed per-package
// seconds, so shards finish together instead of in package-name order.
// buckets prints bucket I's top-level test names for one package too slow to
// share a shard; the buckets split the package's listed tests disjointly.
// weights turns a `go test` log (gh run view --log works) into the weights file.
//
// The workflow's evidence check never trusts this program: it re-proves from the
// per-job receipts that every package and every split test ran exactly once.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: cifastshard partition|buckets|weights [flags]")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "partition":
		err = runPartition(os.Args[2:], os.Stdin, os.Stdout)
	case "buckets":
		err = runBuckets(os.Args[2:], os.Stdin, os.Stdout)
	case "weights":
		err = runWeights(os.Args[2:], os.Stdin, os.Stdout)
	default:
		err = fmt.Errorf("unknown subcommand %q", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "cifastshard:", err)
		os.Exit(1)
	}
}

// Weights is the committed per-package cost model. Overhead is charged to every
// package because a shard of many small packages is bound by their compile,
// link and vet time, which the `ok <pkg> <secs>s` lines do not include.
type Weights struct {
	Schema   string             `json:"schema"`
	Source   string             `json:"source"`
	Default  float64            `json:"default_seconds"`
	Overhead float64            `json:"per_package_overhead_seconds"`
	Packages map[string]float64 `json:"packages"`
}

const weightsSchema = "fak.ci-fast-package-weights/v1"

func loadWeights(path string) (Weights, error) {
	var w Weights
	data, err := os.ReadFile(path)
	if err != nil {
		return w, err
	}
	if err := json.Unmarshal(data, &w); err != nil {
		return w, fmt.Errorf("%s: %w", path, err)
	}
	if w.Schema != weightsSchema {
		return w, fmt.Errorf("%s: schema %q, want %q", path, w.Schema, weightsSchema)
	}
	if w.Default < 0 || w.Overhead < 0 {
		return w, fmt.Errorf("%s: negative default or overhead", path)
	}
	return w, nil
}

// millis keeps the greedy fill in integers so the assignment cannot depend on
// floating-point summation order.
func millis(seconds float64) int64 { return int64(math.Round(seconds * 1000)) }

func (w Weights) cost(pkg string) int64 {
	s, ok := w.Packages[pkg]
	if !ok || s < 0 {
		s = w.Default
	}
	return millis(s + w.Overhead)
}

// Partition assigns every distinct package to exactly one of n shards: heaviest
// first (ties by name), each into the currently lightest shard (ties by lowest
// index). Each shard's packages come back sorted by name.
func Partition(packages []string, w Weights, n int) [][]string {
	uniq := map[string]bool{}
	var pkgs []string
	for _, p := range packages {
		if p != "" && !uniq[p] {
			uniq[p] = true
			pkgs = append(pkgs, p)
		}
	}
	sort.Slice(pkgs, func(i, j int) bool {
		ci, cj := w.cost(pkgs[i]), w.cost(pkgs[j])
		if ci != cj {
			return ci > cj
		}
		return pkgs[i] < pkgs[j]
	})
	shards := make([][]string, n)
	load := make([]int64, n)
	for _, p := range pkgs {
		best := 0
		for i := 1; i < n; i++ {
			if load[i] < load[best] {
				best = i
			}
		}
		shards[best] = append(shards[best], p)
		load[best] += w.cost(p)
	}
	for _, s := range shards {
		sort.Strings(s)
	}
	return shards
}

// ShardLoads reports each shard's modeled seconds.
func ShardLoads(shards [][]string, w Weights) []float64 {
	out := make([]float64, len(shards))
	for i, s := range shards {
		var ms int64
		for _, p := range s {
			ms += w.cost(p)
		}
		out[i] = float64(ms) / 1000
	}
	return out
}

var testName = regexp.MustCompile(`^(Test|Benchmark|Example|Fuzz)[A-Za-z0-9_]*$`)

// Buckets splits a package's top-level test names into k disjoint buckets:
// name i of the sorted distinct list goes to bucket i mod k.
func Buckets(names []string, k int) [][]string {
	uniq := map[string]bool{}
	var sorted []string
	for _, n := range names {
		if testName.MatchString(n) && !uniq[n] {
			uniq[n] = true
			sorted = append(sorted, n)
		}
	}
	sort.Strings(sorted)
	out := make([][]string, k)
	for i, n := range sorted {
		out[i%k] = append(out[i%k], n)
	}
	return out
}

func readLines(r io.Reader) ([]string, error) {
	var lines []string
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		if line := strings.TrimSpace(sc.Text()); line != "" {
			lines = append(lines, line)
		}
	}
	return lines, sc.Err()
}

func checkIndex(index, total int, what string) error {
	if total < 1 {
		return fmt.Errorf("-%s must be >= 1", what)
	}
	if index < 0 || index >= total {
		return fmt.Errorf("-index %d out of range for %d %s", index, total, what)
	}
	return nil
}

func runPartition(args []string, in io.Reader, out io.Writer) error {
	fs := flag.NewFlagSet("partition", flag.ContinueOnError)
	weightsPath := fs.String("weights", "", "weights JSON file")
	shards := fs.Int("shards", 0, "number of package shards")
	index := fs.Int("index", -1, "0-based shard to print")
	excludeFile := fs.String("exclude-file", "", "file of packages run by dedicated bucket jobs")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := checkIndex(*index, *shards, "shards"); err != nil {
		return err
	}
	w, err := loadWeights(*weightsPath)
	if err != nil {
		return err
	}
	exclude := map[string]bool{}
	if *excludeFile != "" {
		f, err := os.Open(*excludeFile)
		if err != nil {
			return err
		}
		lines, err := readLines(f)
		f.Close()
		if err != nil {
			return err
		}
		for _, l := range lines {
			exclude[l] = true
		}
	}
	all, err := readLines(in)
	if err != nil {
		return err
	}
	var pkgs []string
	for _, p := range all {
		if !exclude[p] {
			pkgs = append(pkgs, p)
		}
	}
	for _, p := range Partition(pkgs, w, *shards)[*index] {
		fmt.Fprintln(out, p)
	}
	return nil
}

func runBuckets(args []string, in io.Reader, out io.Writer) error {
	fs := flag.NewFlagSet("buckets", flag.ContinueOnError)
	buckets := fs.Int("buckets", 0, "number of buckets")
	index := fs.Int("index", -1, "0-based bucket to print")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := checkIndex(*index, *buckets, "buckets"); err != nil {
		return err
	}
	names, err := readLines(in)
	if err != nil {
		return err
	}
	for _, n := range Buckets(names, *buckets)[*index] {
		fmt.Fprintln(out, n)
	}
	return nil
}

var resultLine = regexp.MustCompile(`(?:^|\s)ok\s+(\S+)\s+([0-9]+(?:\.[0-9]+)?)s(?:\s|$)`)

// ParseWeights reads `ok <pkg> <secs>s` lines; cached results carry no timing.
func ParseWeights(r io.Reader) (map[string]float64, error) {
	out := map[string]float64{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		m := resultLine.FindStringSubmatch(sc.Text())
		if m == nil {
			continue
		}
		var s float64
		if _, err := fmt.Sscan(m[2], &s); err != nil {
			return nil, err
		}
		out[m[1]] = math.Round(s*10) / 10
	}
	return out, sc.Err()
}

func runWeights(args []string, in io.Reader, out io.Writer) error {
	fs := flag.NewFlagSet("weights", flag.ContinueOnError)
	def := fs.Float64("default", 1, "seconds charged to a package with no recorded time")
	overhead := fs.Float64("overhead", 2, "compile/link/vet seconds charged to every package")
	source := fs.String("source", "", "where the timings came from")
	if err := fs.Parse(args); err != nil {
		return err
	}
	pkgs, err := ParseWeights(in)
	if err != nil {
		return err
	}
	if len(pkgs) == 0 {
		return fmt.Errorf("no `ok <pkg> <secs>s` lines on stdin")
	}
	data, err := json.MarshalIndent(Weights{Schema: weightsSchema, Source: *source, Default: *def, Overhead: *overhead, Packages: pkgs}, "", "  ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(out, "%s\n", data)
	return err
}
