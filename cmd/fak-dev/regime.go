package main

// regime — decode-step overhead-regime reader for perf receipts and profiles.
// It classifies each input document as LAUNCH_BOUND / COMPUTE_BOUND /
// MEM_BOUND / UNKNOWN from inter-kernel gap sums, device-busy fraction, and
// memory-traffic counters when present (internal/regime), replacing
// single-reading prose verdicts with a reproducible multi-signal reading over
// every receipt. Exit codes: 0 classified with no --fail-on match, 1 fail-on
// match, 2 usage or unreadable/undecodable input.

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/regime"
)

const (
	regimeUsage = "usage: fak-dev regime [PATH...] [--latest N] [--dir DIR] [--fail-on REGIME[,REGIME]] [--json]"
	// regimeSchema versions the --json document.
	regimeSchema = "fak.dev.regime.v1"
	// regimeDefaultDir is where committed benchmark receipts live.
	regimeDefaultDir = "docs/benchmarks/receipts"
)

type regimeArgs struct {
	paths   []string
	latest  int
	dir     string
	failOn  []regime.Regime
	asJSON  bool
	help    bool
	badFlag string
}

// parseRegimeArgs scans argv for the closed flag set in any position; the
// stdlib flag package would stop at the first positional path and swallow
// later flags. Everything that is not a known flag is a receipt/profile path.
func parseRegimeArgs(argv []string) regimeArgs {
	a := regimeArgs{dir: regimeDefaultDir}
	i := 0
	for i < len(argv) {
		arg := argv[i]
		switch {
		case arg == "-h" || arg == "--help":
			a.help = true
			i++
		case arg == "--json":
			a.asJSON = true
			i++
		case arg == "--latest":
			if i+1 >= len(argv) {
				return regimeArgs{badFlag: "--latest needs a value"}
			}
			n, err := strconv.Atoi(argv[i+1])
			if err != nil || n <= 0 {
				return regimeArgs{badFlag: "--latest needs a positive integer"}
			}
			a.latest = n
			i += 2
		case strings.HasPrefix(arg, "--latest="):
			n, err := strconv.Atoi(strings.TrimPrefix(arg, "--latest="))
			if err != nil || n <= 0 {
				return regimeArgs{badFlag: "--latest needs a positive integer"}
			}
			a.latest = n
			i++
		case arg == "--dir":
			if i+1 >= len(argv) {
				return regimeArgs{badFlag: "--dir needs a value"}
			}
			a.dir = argv[i+1]
			i += 2
		case strings.HasPrefix(arg, "--dir="):
			a.dir = strings.TrimPrefix(arg, "--dir=")
			i++
		case arg == "--fail-on":
			if i+1 >= len(argv) {
				return regimeArgs{badFlag: "--fail-on needs a value"}
			}
			for _, part := range strings.Split(argv[i+1], ",") {
				r, ok := regime.ParseRegime(part)
				if !ok {
					return regimeArgs{badFlag: fmt.Sprintf("unknown --fail-on regime %q", part)}
				}
				a.failOn = append(a.failOn, r)
			}
			i += 2
		case strings.HasPrefix(arg, "--fail-on="):
			for _, part := range strings.Split(strings.TrimPrefix(arg, "--fail-on="), ",") {
				r, ok := regime.ParseRegime(part)
				if !ok {
					return regimeArgs{badFlag: fmt.Sprintf("unknown --fail-on regime %q", part)}
				}
				a.failOn = append(a.failOn, r)
			}
			i++
		case strings.HasPrefix(arg, "-"):
			return regimeArgs{badFlag: fmt.Sprintf("unknown flag %q", arg)}
		default:
			a.paths = append(a.paths, arg)
			i++
		}
	}
	return a
}

type regimeFileDoc struct {
	Path     string        `json:"path"`
	Regime   regime.Regime `json:"regime"`
	Evidence []string      `json:"evidence"`
	Missing  []string      `json:"missing,omitempty"`
	Error    string        `json:"error,omitempty"`
}

type regimeDoc struct {
	Schema string          `json:"schema"`
	Files  []regimeFileDoc `json:"files"`
	FailOn []string        `json:"fail_on,omitempty"`
}

func runRegime(stdout, stderr io.Writer, argv []string) int {
	a := parseRegimeArgs(argv)
	if a.help {
		fmt.Fprintln(stdout, regimeUsage)
		return 0
	}
	if a.badFlag != "" {
		fmt.Fprintln(stderr, a.badFlag)
		fmt.Fprintln(stderr, regimeUsage)
		return 2
	}
	if len(a.paths) == 0 && a.latest == 0 {
		fmt.Fprintln(stderr, "regime: pass receipt/profile paths or --latest N")
		fmt.Fprintln(stderr, regimeUsage)
		return 2
	}
	if len(a.paths) > 0 && a.latest > 0 {
		fmt.Fprintln(stderr, "regime: choose either explicit paths or --latest N, not both")
		return 2
	}
	if a.latest > 0 {
		latest, err := latestReceipts(a.dir, a.latest)
		if err != nil {
			fmt.Fprintf(stderr, "regime: %v\n", err)
			return 2
		}
		if len(latest) == 0 {
			fmt.Fprintf(stderr, "regime: no .json receipts under %s\n", a.dir)
			return 2
		}
		a.paths = latest
	}

	docs := make([]regimeFileDoc, 0, len(a.paths))
	failed := false
	counts := map[regime.Regime]int{}
	for _, path := range a.paths {
		doc := regimeFileDoc{Path: path}
		data, err := os.ReadFile(path)
		if err != nil {
			doc.Error = err.Error()
			docs = append(docs, doc)
			fmt.Fprintf(stderr, "regime: %s: %v\n", path, err)
			continue
		}
		p, err := regime.DecodeProfile(data)
		if err != nil {
			doc.Error = err.Error()
			docs = append(docs, doc)
			fmt.Fprintf(stderr, "regime: %s: %v\n", path, err)
			continue
		}
		v := regime.Classify(p)
		doc.Regime = v.Regime
		doc.Evidence = v.Evidence
		doc.Missing = v.Missing
		counts[v.Regime]++
		for _, want := range a.failOn {
			if v.Regime == want {
				failed = true
			}
		}
		docs = append(docs, doc)
	}

	if a.asJSON {
		out := regimeDoc{Schema: regimeSchema, Files: docs}
		for _, r := range a.failOn {
			out.FailOn = append(out.FailOn, string(r))
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fmt.Fprintf(stderr, "regime: encode: %v\n", err)
			return 2
		}
	} else {
		for _, d := range docs {
			if d.Error != "" {
				fmt.Fprintf(stdout, "ERROR\t%s\n", d.Path)
				continue
			}
			fmt.Fprintf(stdout, "%s\t%s\n", d.Regime, d.Path)
			for _, e := range d.Evidence {
				fmt.Fprintf(stdout, "\tevidence: %s\n", e)
			}
			if len(d.Missing) > 0 {
				fmt.Fprintf(stdout, "\tmissing: %s\n", strings.Join(d.Missing, ", "))
			}
		}
		order := []regime.Regime{regime.RegimeLaunchBound, regime.RegimeComputeBound, regime.RegimeMemBound, regime.RegimeUnknown}
		var parts []string
		for _, r := range order {
			if n := counts[r]; n > 0 {
				parts = append(parts, fmt.Sprintf("%d %s", n, strings.ToLower(string(r))))
			}
		}
		if len(parts) > 0 {
			fmt.Fprintf(stdout, "summary: %d profile(s): %s\n", len(docs), strings.Join(parts, ", "))
		}
	}
	if failed {
		return 1
	}
	return 0
}

// latestReceipts returns up to n newest .json files under dir (recursive),
// ordered by modification time descending then path for determinism.
func latestReceipts(dir string, n int) ([]string, error) {
	type entry struct {
		path    string
		modTime int64
	}
	var entries []entry
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(strings.ToLower(path), ".json") {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		entries = append(entries, entry{path: path, modTime: info.ModTime().UnixNano()})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].modTime != entries[j].modTime {
			return entries[i].modTime > entries[j].modTime
		}
		return entries[i].path < entries[j].path
	})
	if len(entries) > n {
		entries = entries[:n]
	}
	paths := make([]string, len(entries))
	for i, e := range entries {
		paths[i] = e.path
	}
	return paths, nil
}
