package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func runNewModel(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("new-model", flag.ContinueOnError)
	fs.SetOutput(stderr)
	family := fs.String("family", "", "family name, lowercase (e.g. myfamily)")
	topology := fs.String("topology", "identity", "topology: prenorm, postnorm, parallel, or identity")
	dryRun := fs.Bool("dry-run", false, "print scaffold without writing files")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	fromManifest := fs.String("from-manifest", "", "compile a pinned release manifest into an onboarding packet")
	gguf := fs.String("gguf", "", "with --from-manifest: also run the header preflight (fak model preflight) on this GGUF")
	hfConfig := fs.String("hf-config", "", "with --gguf: the HF config.json the GGUF keys must agree with")
	hfRepo := fs.String("hf", "", "emit a family spine from <owner>/<repo>@<40-hex commit> headers (needs --family)") //pathlint:not-path a Hub repo pin, never opened on disk
	hfEndpoint := fs.String("hf-endpoint", hfhub.DefaultBaseURL, "with --hf: Hub origin or mirror")
	outDir := fs.String("out", ".", "with --hf: directory the spine tree is written under")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak new-model: positional arguments are not supported")
		return 2
	}
	*gguf = pathutil.ExpandTilde(*gguf)
	*hfConfig = pathutil.ExpandTilde(*hfConfig)
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })

	if *hfRepo != "" {
		if *fromManifest != "" || *gguf != "" || *hfConfig != "" || set["topology"] || set["dry-run"] {
			fmt.Fprintln(stderr, "fak new-model: --hf takes only --family, --out, --hf-endpoint and --json")
			return 2
		}
		return newModelHFSpine(stdout, stderr, *hfRepo, *family, *hfEndpoint, pathutil.ExpandTilde(*outDir), *asJSON)
	}
	if set["out"] || set["hf-endpoint"] {
		fmt.Fprintln(stderr, "fak new-model: --out and --hf-endpoint require --hf")
		return 2
	}

	if *fromManifest != "" {
		if *family != "" || set["topology"] || set["dry-run"] || !*asJSON || (*hfConfig != "" && *gguf == "") {
			fmt.Fprintln(stderr, "fak new-model: --from-manifest requires --json and cannot be combined with scaffold flags")
			return 2
		}
		data, err := os.ReadFile(*fromManifest)
		if err != nil {
			fmt.Fprintf(stderr, "fak new-model: read manifest: %v\n", err)
			return 1
		}
		packet, err := newmodel.CompileManifestJSON(data)
		if err != nil {
			var refusal *newmodel.Refusal
			if errors.As(err, &refusal) {
				enc := json.NewEncoder(stderr)
				enc.SetIndent("", "  ")
				_ = enc.Encode(refusal)
				return 3
			}
			fmt.Fprintf(stderr, "fak new-model: compile manifest: %v\n", err)
			return 1
		}
		if *gguf != "" {
			if code := newModelHeaderPreflight(stderr, *gguf, *hfConfig); code != 0 {
				return code
			}
		}
		if _, err := stdout.Write(packet); err != nil {
			fmt.Fprintf(stderr, "fak new-model: write packet: %v\n", err)
			return 1
		}
		return 0
	}

	if *gguf != "" || *hfConfig != "" {
		fmt.Fprintln(stderr, "fak new-model: --gguf and --hf-config require --from-manifest")
		return 2
	}
	if *family == "" {
		fmt.Fprintln(stderr, "fak new-model: --family or --from-manifest is required")
		fmt.Fprintln(stderr, "usage: fak new-model (--family <name> [--topology <topology>] [--dry-run] | --family <name> --hf <owner>/<repo>@<commit> [--out <dir>] | --from-manifest <file>) [--json]")
		return 2
	}

	res, err := newmodel.Run(newmodel.Scaffold{Family: *family, Topology: *topology, DryRun: *dryRun})
	if err != nil {
		fmt.Fprintf(stderr, "fak new-model: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak new-model: encode scaffold: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "=== Scaffolding model family '%s' (topology: %s) ===\n\n", res.Family, res.Topology)
	fmt.Fprintln(stdout, "Files to edit:")
	for _, edit := range res.Edits {
		fmt.Fprintf(stdout, "  - %s\n", edit)
	}
	fmt.Fprintln(stdout, "\nNext steps:")
	for _, step := range res.NextSteps {
		fmt.Fprintln(stdout, step)
	}
	return 0
}

// newModelHeaderPreflight runs the same one-pass header check as `fak model
// preflight`; any refusal is written as the full report and exits 3.
func newModelHeaderPreflight(stderr io.Writer, gguf, hfConfig string) int {
	f, err := ggufload.OpenHeaders(gguf)
	if err != nil {
		fmt.Fprintf(stderr, "fak new-model: gguf header: %v\n", err)
		return 1
	}
	in := ggufload.ManifestPreflightInput{Path: gguf, File: f}
	if hfConfig != "" {
		if in.HFConfig, err = os.ReadFile(hfConfig); err != nil {
			fmt.Fprintf(stderr, "fak new-model: read hf config: %v\n", err)
			return 1
		}
	}
	report := ggufload.BuildManifestPreflight(in)
	if !report.Refused() {
		return 0
	}
	enc := json.NewEncoder(stderr)
	enc.SetIndent("", "  ")
	_ = enc.Encode(report)
	return 3
}

// newModelHFSpine range-reads a pinned Hub checkpoint's config and safetensors
// headers and writes the family spine under outDir. Refusals exit 3 as JSON.
func newModelHFSpine(stdout, stderr io.Writer, hf, family, endpoint, outDir string, asJSON bool) int {
	refused := func(err error) bool {
		var refusal *newmodel.Refusal
		if !errors.As(err, &refusal) {
			return false
		}
		enc := json.NewEncoder(stderr)
		enc.SetIndent("", "  ")
		_ = enc.Encode(refusal)
		return true
	}
	if family == "" {
		fmt.Fprintln(stderr, "fak new-model: --hf requires --family")
		return 2
	}
	ref, err := newmodel.ParseHFPin(hf)
	if err != nil {
		fmt.Fprintf(stderr, "fak new-model: %v\n", err)
		return 2
	}
	client := hfhub.NewClient()
	client.BaseURL = endpoint
	census, err := newmodel.CensusHF(context.Background(), newmodel.HubRangeSource{Client: client, Ref: ref}, ref.Repo, ref.Revision)
	if err != nil {
		if refused(err) {
			return 3
		}
		fmt.Fprintf(stderr, "fak new-model: hf census: %v\n", err)
		return 1
	}
	spine, err := newmodel.EmitHFSpine(census, family)
	if err != nil {
		if refused(err) {
			return 3
		}
		fmt.Fprintf(stderr, "fak new-model: %v\n", err)
		return 2
	}
	if err := newmodel.WriteHFSpine(outDir, spine); err != nil {
		fmt.Fprintf(stderr, "fak new-model: write spine: %v\n", err)
		return 1
	}
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(struct {
			Family     string                    `json:"family"`
			Out        string                    `json:"out"`
			Files      []string                  `json:"files"`
			Insertions []newmodel.SpineInsertion `json:"insertions"`
		}{family, outDir, spine.Paths(), spine.Insertions}); err != nil {
			fmt.Fprintf(stderr, "fak new-model: encode spine: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "=== %s spine from %s@%s written under %s ===\n", family, ref.Repo, ref.Revision, outDir)
	for _, p := range spine.Paths() {
		fmt.Fprintf(stdout, "  %s\n", p)
	}
	fmt.Fprintf(stdout, "Apply the hand edits in internal/model/testdata/%s/INSERTION.md.\n", family)
	return 0
}
