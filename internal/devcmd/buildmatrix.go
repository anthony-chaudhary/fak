package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/buildwitness"
)

// MatrixCompileResult records the execution outcome of one variant + target combination.
type MatrixCompileResult struct {
	Target     string `json:"target"`
	GOOS       string `json:"goos"`
	GOARCH     string `json:"goarch"`
	Variant    string `json:"variant"`
	Tags       string `json:"tags"`
	CGOEnabled string `json:"cgo_enabled"`
	Advisory   bool   `json:"advisory"`
	Command    string `json:"command"`
	Env        string `json:"env"`
	Verdict    string `json:"verdict"` // PASS, PLAN, ADVISORY_FAIL, FAIL
	Output     string `json:"output,omitempty"`
	ElapsedMS  int64  `json:"elapsed_ms"`
}

// BuildMatrixReport is the machine-readable output format of fak-dev build-matrix.
type BuildMatrixReport struct {
	Schema         string                `json:"schema"`
	Manifest       string                `json:"manifest"`
	Package        string                `json:"package"`
	Total          int                   `json:"total"`
	Passed         int                   `json:"passed"`
	Failed         int                   `json:"failed"`
	AdvisoryFailed int                   `json:"advisory_failed"`
	OK             bool                  `json:"ok"`
	Results        []MatrixCompileResult `json:"results"`
}

// RunBuildMatrix executes the compile-check matrix for declared build variants across release targets.
func RunBuildMatrix(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak-dev build-matrix", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "build-matrix")

	manifestPath := fs.String("manifest", "", "path to build-variants.json manifest (defaults to repo discovery)")
	pureOnly := fs.Bool("pure-only", true, "only validate pure-Go variants (CGO_ENABLED=0)")
	targetFilter := fs.String("target", "", "filter by target GOOS/GOARCH (e.g. 'linux/amd64', 'darwin/arm64')")
	variantFilter := fs.String("variant", "", "filter by variant name (e.g. 'default', 'wip_sessionfleet')")
	pkg := fs.String("pkg", buildwitness.TargetPackage, "target package to compile-check (defaults to ./cmd/fak)")
	dryRun := fs.Bool("dry-run", false, "plan compile commands without executing go build")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON report")
	emitGHA := fs.Bool("emit-gha-matrix", false, "emit GitHub Actions JSON matrix { include: [...] } and exit")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	root := repoRoot()

	var manifest *buildwitness.VariantManifest
	var resolvedPath string
	var err error

	if strings.TrimSpace(*manifestPath) != "" {
		data, readErr := os.ReadFile(*manifestPath)
		if readErr != nil {
			fmt.Fprintf(stderr, "fak-dev build-matrix: read manifest: %v\n", readErr)
			return 2
		}
		var m buildwitness.VariantManifest
		if jsonErr := json.Unmarshal(data, &m); jsonErr != nil {
			fmt.Fprintf(stderr, "fak-dev build-matrix: parse manifest: %v\n", jsonErr)
			return 2
		}
		manifest = &m
		resolvedPath = *manifestPath
	} else {
		manifest, resolvedPath, err = buildwitness.LoadVariantManifest(root)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev build-matrix: load manifest: %v\n", err)
			return 2
		}
	}

	if *emitGHA {
		matrix := buildwitness.BuildGHAMatrix(manifest, *pureOnly)
		data, err := json.Marshal(matrix)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev build-matrix: marshal GHA matrix: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(data))
		return 0
	}

	variants := manifest.Variants
	if *pureOnly {
		variants = manifest.PureGoVariants()
	}

	var targets []buildwitness.ReleaseTarget
	if strings.TrimSpace(*targetFilter) != "" {
		parts := strings.Split(strings.TrimSpace(*targetFilter), "/")
		if len(parts) != 2 {
			fmt.Fprintf(stderr, "fak-dev build-matrix: invalid target filter %q (expected GOOS/GOARCH)\n", *targetFilter)
			return 2
		}
		targets = append(targets, buildwitness.ReleaseTarget{GOOS: parts[0], GOARCH: parts[1]})
	} else {
		targets = manifest.ReleaseTargets
	}

	report := BuildMatrixReport{
		Schema:   "fak.build_matrix_report.v1",
		Manifest: resolvedPath,
		Package:  *pkg,
	}

	allOK := true
	for _, v := range variants {
		if strings.TrimSpace(*variantFilter) != "" && !strings.EqualFold(v.Name, strings.TrimSpace(*variantFilter)) {
			continue
		}
		for _, t := range targets {
			if v.TargetOSArch != "" && v.TargetOSArch != "all" && v.TargetOSArch != t.String() {
				continue
			}

			plan := buildwitness.PlanCompile(*pkg, v, t, buildwitness.NullDevice())
			res := MatrixCompileResult{
				Target:     t.String(),
				GOOS:       t.GOOS,
				GOARCH:     t.GOARCH,
				Variant:    v.Name,
				Tags:       v.Tags,
				CGOEnabled: v.CGOEnabled,
				Advisory:   v.IsAdvisory(),
				Command:    strings.Join(plan.Command, " "),
				Env:        strings.Join(plan.Env, " "),
			}

			start := time.Now()
			if *dryRun {
				res.Verdict = "PLAN"
				res.ElapsedMS = 0
			} else {
				out, err := buildwitness.RunCompileCheck("", root, plan)
				res.ElapsedMS = time.Since(start).Milliseconds()
				if err != nil {
					res.Output = strings.TrimSpace(out)
					if v.IsAdvisory() {
						res.Verdict = "ADVISORY_FAIL"
						report.AdvisoryFailed++
					} else {
						res.Verdict = "FAIL"
						report.Failed++
						allOK = false
					}
				} else {
					res.Verdict = "PASS"
					report.Passed++
				}
			}
			report.Total++
			report.Results = append(report.Results, res)

			if !*asJSON {
				switch res.Verdict {
				case "PASS":
					fmt.Fprintf(stdout, "[PASS] %s (%s, tags=%q) [%dms]\n", res.Variant, res.Target, res.Tags, res.ElapsedMS)
				case "PLAN":
					fmt.Fprintf(stdout, "[PLAN] %s (%s, tags=%q) -> %s %s\n", res.Variant, res.Target, res.Tags, res.Env, res.Command)
				case "ADVISORY_FAIL":
					fmt.Fprintf(stdout, "[ADVISORY_FAIL] %s (%s, tags=%q) (advisory)\n%s\n", res.Variant, res.Target, res.Tags, res.Output)
				case "FAIL":
					fmt.Fprintf(stdout, "[FAIL] %s (%s, tags=%q)\n%s\n", res.Variant, res.Target, res.Tags, res.Output)
				}
			}
		}
	}

	report.OK = allOK
	if *asJSON {
		data, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev build-matrix: marshal JSON: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(data))
	} else if !*dryRun {
		fmt.Fprintf(stdout, "\nMatrix Summary: %d total, %d passed, %d failed (%d advisory)\n",
			report.Total, report.Passed, report.Failed, report.AdvisoryFailed)
	}

	if !allOK {
		return 1
	}
	return 0
}
