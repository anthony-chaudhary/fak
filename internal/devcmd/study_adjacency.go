package devcmd

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/studyadjacency"
)

const defaultStudyAdjacencyManifest = "docs/research/inventory/vllm-related-system-adjacency-v1.json"

func RunStudyAdjacency(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		studyAdjacencyUsage(stderr)
		return 2
	}
	switch args[0] {
	case "validate":
		return runStudyAdjacencyValidate(stdout, stderr, args[1:])
	case "render":
		return runStudyAdjacencyRender(stdout, stderr, args[1:])
	default:
		studyAdjacencyUsage(stderr)
		return 2
	}
}

func runStudyAdjacencyValidate(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("study-adjacency validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath := fs.String("manifest", defaultStudyAdjacencyManifest, "versioned adjacency manifest path")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak study-adjacency validate [--manifest PATH]")
		return 2
	}
	manifest, err := studyadjacency.Load(*manifestPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-adjacency: %v\n", err)
		return 1
	}
	candidates := 0
	for _, member := range manifest.Members {
		candidates += len(member.Candidates)
	}
	fmt.Fprintf(stdout, "valid study adjacency manifest %s: members=%d candidates=%d cutoff=%s\n",
		manifest.ID, len(manifest.Members), candidates, manifest.Anchor.Pin.Cutoff)
	return 0
}

func runStudyAdjacencyRender(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("study-adjacency render", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath := fs.String("manifest", defaultStudyAdjacencyManifest, "versioned adjacency manifest path")
	outPath := fs.String("out", "", "write Markdown summary to PATH instead of stdout")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak study-adjacency render [--manifest PATH] [--out PATH]")
		return 2
	}
	manifest, err := studyadjacency.Load(*manifestPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-adjacency: %v\n", err)
		return 1
	}
	rendered, err := studyadjacency.RenderMarkdown(manifest)
	if err != nil {
		fmt.Fprintf(stderr, "study-adjacency: %v\n", err)
		return 1
	}
	if *outPath == "" {
		if _, err := stdout.Write(rendered); err != nil {
			fmt.Fprintf(stderr, "study-adjacency: write summary: %v\n", err)
			return 1
		}
		return 0
	}
	if err := writeStudyAdjacencySummary(*outPath, rendered); err != nil {
		fmt.Fprintf(stderr, "study-adjacency: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "rendered study adjacency summary to %s\n", *outPath)
	return 0
}

func writeStudyAdjacencySummary(path string, data []byte) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create summary directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".study-adjacency-*.tmp")
	if err != nil {
		return fmt.Errorf("create summary temporary file: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write summary temporary file: %w", err)
	}
	if err := tmp.Chmod(0o644); err != nil {
		tmp.Close()
		return fmt.Errorf("chmod summary temporary file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close summary temporary file: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("replace summary: %w", err)
	}
	return nil
}

func studyAdjacencyUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak study-adjacency <validate|render> [flags]")
	fmt.Fprintln(w, "  validate [--manifest PATH]")
	fmt.Fprintln(w, "  render [--manifest PATH] [--out PATH]")
}
