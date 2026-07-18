package main

// toolproc_repeats.go — the thin shell for `fak toolproc repeats` (#5121): feed
// native Codex rollout JSONL files into internal/toolproc's repeat classifier
// (IngestRolloutFiles → AttachReadDigests → ClassifyRepeats) and print the
// RepeatReport. All logic is in the leaf; this file owns only flags and exit
// codes. --json emits the machine report; by default immutable-read targets are
// digested from the local filesystem so the invalidation-after-mutation key
// model runs on real data (--no-digest keeps the conservative path-only fold).

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

func runToolprocRepeats(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("toolproc repeats", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the RepeatReport as JSON (machine form)")
	noDigest := fs.Bool("no-digest", false, "skip resolving immutable-read content digests from the local filesystem")
	root := fs.String("root", "", "root to resolve relative read paths against (default: current directory)")
	top := fs.Int("top", 15, "group rows rendered in the human table (0 = all)")
	pollMin := fs.Int("poll-min-repeats", 0, "mutable-query repeats before POLL_STORM promotion (0 = default)")
	pollSpacingMS := fs.Int64("poll-max-median-spacing-ms", 0, "max median spacing for POLL_STORM promotion (0 = default)")
	freshMS := fs.Int64("fresh-ms", 0, "default freshness window for mutable queries (0 = default)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak toolproc repeats: at least one rollout JSONL file is required")
		return 2
	}
	cfg := toolproc.RepeatConfig{
		PollingMinRepeats:         *pollMin,
		PollingMaxMedianSpacingMS: *pollSpacingMS,
		DefaultFreshnessMS:        *freshMS,
	}
	recs, err := toolproc.IngestRolloutFiles(fs.Args())
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc repeats: %v\n", err)
		return 1
	}
	var digest toolproc.DigestFn
	if !*noDigest {
		digest = toolproc.FileDigest(*root)
	}
	recs = toolproc.AttachReadDigests(recs, cfg, digest)
	rep := toolproc.ClassifyRepeats(recs, cfg)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak toolproc repeats")
	}
	toolproc.RenderRepeatReport(stdout, rep, *top)
	return 0
}
