package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelaccept"
)

func runModelReadinessInventory(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model readiness-inventory", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "-", "acceptance JSON artifact path ('-' reads stdin)")
	artifact := fs.String("artifact", "", "provenance path recorded in inventory (defaults to --input)")
	revision := fs.String("artifact-revision", "", "required immutable module@rev provenance")
	expectedCorpus := fs.String("expected-corpus", "", "required corpus ID when set")
	ladderDir := fs.String("ladder-evidence-dir", "", "explicit immutable Qwen3.8 ladder evidence directory")
	ladderManifest := fs.String("ladder-manifest", "", "ladder checksum manifest (defaults to <ladder-evidence-dir>/checksums.json)")
	asOfText := fs.String("as-of", "", "inventory time in RFC3339 (defaults to current UTC time)")
	maxAge := fs.Duration("max-evidence-age", 30*24*time.Hour, "maximum acceptance evidence age")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak model readiness-inventory: unexpected positional arguments")
		return 2
	}
	if *revision == "" {
		fmt.Fprintln(stderr, "fak model readiness-inventory: --artifact-revision is required")
		return 2
	}
	if *maxAge <= 0 {
		fmt.Fprintln(stderr, "fak model readiness-inventory: --max-evidence-age must be positive")
		return 2
	}
	asOf := time.Now().UTC()
	if *asOfText != "" {
		parsed, err := time.Parse(time.RFC3339, *asOfText)
		if err != nil {
			fmt.Fprintf(stderr, "fak model readiness-inventory: --as-of must be RFC3339: %v\n", err)
			return 2
		}
		asOf = parsed
	}
	if *ladderManifest != "" && *ladderDir == "" {
		fmt.Fprintln(stderr, "fak model readiness-inventory: --ladder-manifest requires --ladder-evidence-dir")
		return 2
	}
	if *ladderDir != "" {
		if *input != "-" {
			fmt.Fprintln(stderr, "fak model readiness-inventory: --input and --ladder-evidence-dir are mutually exclusive")
			return 2
		}
		manifest := *ladderManifest
		if manifest == "" {
			manifest = filepath.Join(*ladderDir, "checksums.json")
		}
		out, _ := modelaccept.BuildQwen38LadderReadinessInventory(modelaccept.InventoryOptions{
			Artifact: *artifact, ArtifactRevision: *revision, ExpectedCorpusID: *expectedCorpus,
		}, modelaccept.LadderEvidenceOptions{Directory: *ladderDir, Manifest: manifest})
		return writeModelReadinessInventory(stdout, stderr, out)
	}

	var r io.Reader = os.Stdin
	var f *os.File
	if *input != "-" {
		var err error
		f, err = os.Open(*input)
		if err != nil {
			fmt.Fprintf(stderr, "fak model readiness-inventory: %v\n", err)
			return 2
		}
		defer f.Close()
		r = f
	}
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var in modelaccept.Input
	if err := dec.Decode(&in); err != nil {
		fmt.Fprintf(stderr, "fak model readiness-inventory: decode input: %v\n", err)
		return 2
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		fmt.Fprintf(stderr, "fak model readiness-inventory: decode input: %v\n", err)
		return 2
	}
	artifactPath := *artifact
	if artifactPath == "" {
		artifactPath = *input
	}
	out := modelaccept.BuildInventory(in, modelaccept.InventoryOptions{
		Artifact: artifactPath, ArtifactRevision: *revision, ExpectedCorpusID: *expectedCorpus,
		AsOf: asOf, MaxEvidenceAge: *maxAge,
	})
	return writeModelReadinessInventory(stdout, stderr, out)
}

func writeModelReadinessInventory(stdout, stderr io.Writer, out modelaccept.Inventory) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		fmt.Fprintf(stderr, "fak model readiness-inventory: encode inventory: %v\n", err)
		return 2
	}
	if out.Verdict == modelaccept.Pass {
		return 0
	}
	return 4
}
