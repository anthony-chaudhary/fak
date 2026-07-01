package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
)

const testShardPlanSchema = "fak.test_shard_plan.v1"

type testShardPlan struct {
	Schema            string      `json:"schema"`
	Source            string      `json:"source,omitempty"`
	Shards            []testShard `json:"shards"`
	ShardCount        int         `json:"shard_count"`
	TotalPackages     int         `json:"total_packages"`
	TotalElapsedMS    int64       `json:"total_elapsed_ms"`
	MaxShardElapsedMS int64       `json:"max_shard_elapsed_ms"`
	MinShardElapsedMS int64       `json:"min_shard_elapsed_ms"`
	ImbalanceMS       int64       `json:"imbalance_ms"`
	CommandPrefix     []string    `json:"command_prefix"`
}

type testShard struct {
	Index        int      `json:"index"`
	Packages     []string `json:"packages"`
	PackageCount int      `json:"package_count"`
	EstimatedMS  int64    `json:"estimated_ms"`
	Command      []string `json:"command"`
}

type testShardOptions struct {
	ShardCount    int
	CommandPrefix []string
}

func runTestShards(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak test shards", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", ".fak/test-duration-ledger.json", "duration ledger JSON from fak test durations")
	shards := fs.Int("shards", 2, "number of balanced shards to produce")
	var goArgs pathList
	fs.Var(&goArgs, "go-arg", "go test arg to put before package names in each command (repeatable)")
	fs.Usage = func() {
		fmt.Fprint(stderr, `fak test shards -- balance packages from a duration ledger

  fak test shards --input .fak/test-duration-ledger.json --shards 4 --go-arg -short

The output schema is fak.test_shard_plan.v1. Packages are assigned by measured
elapsed time using deterministic longest-processing-time balancing.
`)
	}
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak test shards: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *shards <= 0 {
		fmt.Fprintln(stderr, "fak test shards: --shards must be positive")
		return 2
	}

	f, err := os.Open(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak test shards: open input: %v\n", err)
		return 1
	}
	defer f.Close()
	ledger, err := readTestDurationLedger(f)
	if err != nil {
		fmt.Fprintf(stderr, "fak test shards: %v\n", err)
		return 1
	}
	prefix := append([]string{"go", "test"}, []string(goArgs)...)
	plan, err := buildTestShardPlan(ledger, testShardOptions{
		ShardCount:    *shards,
		CommandPrefix: prefix,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak test shards: %v\n", err)
		return 1
	}
	if err := writeIndentedJSONNoEscape(stdout, plan); err != nil {
		fmt.Fprintf(stderr, "fak test shards: encode json: %v\n", err)
		return 1
	}
	return 0
}

func readTestDurationLedger(r io.Reader) (testDurationLedger, error) {
	var ledger testDurationLedger
	dec := json.NewDecoder(r)
	if err := dec.Decode(&ledger); err != nil {
		return testDurationLedger{}, fmt.Errorf("parse duration ledger: %w", err)
	}
	if ledger.Schema != testDurationLedgerSchema {
		return testDurationLedger{}, fmt.Errorf("ledger schema = %q, want %s", ledger.Schema, testDurationLedgerSchema)
	}
	return ledger, nil
}

func buildTestShardPlan(ledger testDurationLedger, opts testShardOptions) (testShardPlan, error) {
	if opts.ShardCount <= 0 {
		return testShardPlan{}, fmt.Errorf("shard count must be positive")
	}
	packages := make([]testDurationPackage, 0, len(ledger.Packages))
	for _, row := range ledger.Packages {
		if row.Package == "" {
			continue
		}
		packages = append(packages, row)
	}
	if len(packages) == 0 {
		return testShardPlan{}, fmt.Errorf("duration ledger has no package rows")
	}
	sort.Slice(packages, func(i, j int) bool {
		if packages[i].ElapsedMS != packages[j].ElapsedMS {
			return packages[i].ElapsedMS > packages[j].ElapsedMS
		}
		return packages[i].Package < packages[j].Package
	})

	prefix := append([]string(nil), opts.CommandPrefix...)
	if len(prefix) == 0 {
		prefix = []string{"go", "test"}
	}
	shards := make([]testShard, opts.ShardCount)
	for i := range shards {
		shards[i].Index = i + 1
	}
	for _, row := range packages {
		idx := lightestShard(shards)
		shards[idx].Packages = append(shards[idx].Packages, row.Package)
		shards[idx].PackageCount++
		shards[idx].EstimatedMS += row.ElapsedMS
	}

	var total, max, min int64
	for i := range shards {
		sort.Strings(shards[i].Packages)
		shards[i].Command = append(append([]string(nil), prefix...), shards[i].Packages...)
		total += shards[i].EstimatedMS
		if i == 0 || shards[i].EstimatedMS > max {
			max = shards[i].EstimatedMS
		}
		if i == 0 || shards[i].EstimatedMS < min {
			min = shards[i].EstimatedMS
		}
	}
	return testShardPlan{
		Schema:            testShardPlanSchema,
		Source:            ledger.Source,
		Shards:            shards,
		ShardCount:        len(shards),
		TotalPackages:     len(packages),
		TotalElapsedMS:    total,
		MaxShardElapsedMS: max,
		MinShardElapsedMS: min,
		ImbalanceMS:       max - min,
		CommandPrefix:     prefix,
	}, nil
}

func lightestShard(shards []testShard) int {
	best := 0
	for i := 1; i < len(shards); i++ {
		if shards[i].EstimatedMS < shards[best].EstimatedMS {
			best = i
			continue
		}
		if shards[i].EstimatedMS == shards[best].EstimatedMS {
			if shards[i].PackageCount < shards[best].PackageCount || (shards[i].PackageCount == shards[best].PackageCount && shards[i].Index < shards[best].Index) {
				best = i
			}
		}
	}
	return best
}
