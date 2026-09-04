// Command perfscout searches, scores, and inventories fresh, performance-specific
// open-source repositories running and proving Qwen 3.8 Flash and GLM 5.3 Flash models.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/perfscout"
)

func main() {
	os.Exit(run(os.Stdout, os.Stderr, os.Args[1:]))
}

func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("perfscout", flag.ContinueOnError)
	fs.SetOutput(stderr)

	maxStars := fs.Int("max-stars", 500, "upper bound for stars (unpopular indie filter; 0 for unlimited)")
	minScore := fs.Int("min-score", 10, "minimum performance score to retain")
	maxAgeDays := fs.Int("max-age-days", 30, "maximum age since last update in days")
	family := fs.String("family", "all", "target family filter (all, qwen, glm)")
	limit := fs.Int("limit", 25, "maximum results fetched per search query")
	cohorts := fs.Int("cohorts", 4, "number of subagent cohorts to partition results into")
	fixture := fs.String("fixture", "", "path to offline JSON fixture to replay")
	outPath := fs.String("out", "", "write Markdown report to this path")
	jsonOut := fs.Bool("json", false, "emit report as JSON to stdout")
	crossMatrix := fs.Bool("cross", false, "display cross-architecture innovation translation matrix")
	sourcePlatform := fs.String("source", "", "filter cross innovations by source platform (e.g. mlx, cuda, rocm)")
	targetPlatform := fs.String("target", "", "filter cross innovations by target platform (e.g. cuda, metal, rocm)")
	saveFixture := fs.String("save-fixture", "", "save raw fetched repo candidates to this fixture path")

	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "perfscout: unexpected positional arguments")
		return 2
	}

	if *crossMatrix {
		innovations := perfscout.FilterInnovations(*sourcePlatform, *targetPlatform, "")
		if *jsonOut {
			b, err := perfscout.RenderCrossInnovationsJSON(innovations)
			if err != nil {
				fmt.Fprintf(stderr, "perfscout: %v\n", err)
				return 1
			}
			_, _ = stdout.Write(b)
			_, _ = stdout.Write([]byte("\n"))
			return 0
		}
		md := perfscout.RenderCrossInnovationsMarkdown(innovations)
		if *outPath != "" {
			if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
				fmt.Fprintf(stderr, "perfscout: %v\n", err)
				return 1
			}
			if err := os.WriteFile(*outPath, []byte(md), 0o644); err != nil {
				fmt.Fprintf(stderr, "perfscout: %v\n", err)
				return 1
			}
			fmt.Fprintf(stdout, "perfscout: wrote cross-architecture matrix to %s (%d innovations)\n", *outPath, len(innovations))
			return 0
		}
		_, _ = stdout.Write([]byte(md))
		return 0
	}

	var famFilter perfscout.ModelFamily
	switch strings.ToLower(*family) {
	case "qwen", "qwen_flash":
		famFilter = perfscout.FamilyQwenFlash
	case "glm", "glm_flash":
		famFilter = perfscout.FamilyGLMFlash
	default:
		famFilter = perfscout.FamilyUnknown
	}

	opts := perfscout.SearchOptions{
		MaxStars:      *maxStars,
		MinScore:      *minScore,
		MaxAgeDays:    *maxAgeDays,
		FamilyFilter:  famFilter,
		LimitPerQuery: *limit,
		CohortCount:   *cohorts,
		FixturePath:   *fixture,
		Now:           time.Now().UTC(),
	}

	report, err := perfscout.Run(opts)
	if err != nil {
		fmt.Fprintf(stderr, "perfscout: %v\n", err)
		return 1
	}

	if *saveFixture != "" {
		if err := os.MkdirAll(filepath.Dir(*saveFixture), 0o755); err != nil {
			fmt.Fprintf(stderr, "perfscout: create fixture dir: %v\n", err)
			return 1
		}
		rawBytes, err := json.MarshalIndent(report.Repositories, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "perfscout: marshal fixture: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*saveFixture, rawBytes, 0o644); err != nil {
			fmt.Fprintf(stderr, "perfscout: write fixture: %v\n", err)
			return 1
		}
		fmt.Fprintf(stderr, "perfscout: saved fixture to %s\n", *saveFixture)
	}

	if *jsonOut {
		b, err := perfscout.RenderJSON(report)
		if err != nil {
			fmt.Fprintf(stderr, "perfscout: %v\n", err)
			return 1
		}
		_, _ = stdout.Write(b)
		_, _ = stdout.Write([]byte("\n"))
		return 0
	}

	md := perfscout.RenderMarkdown(report)
	if *outPath != "" {
		if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
			fmt.Fprintf(stderr, "perfscout: create out dir: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*outPath, []byte(md), 0o644); err != nil {
			fmt.Fprintf(stderr, "perfscout: write output: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "perfscout: wrote inventory report to %s (%d repos, %d Qwen, %d GLM, %d Dual across %d cohorts)\n",
			*outPath, report.RetainedCount, report.QwenCount, report.GLMCount, report.DualCount, len(report.Cohorts))
		return 0
	}

	_, _ = stdout.Write([]byte(md))
	return 0
}
