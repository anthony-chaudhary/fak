package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scratchjanitor"
)

type scratchJanitorResumePacket struct {
	Candidates []scratchJanitorReference `json:"candidates"`
	Scan       *struct {
		Candidates []scratchJanitorReference `json:"candidates"`
	} `json:"scan"`
}

type scratchJanitorReference struct {
	ScratchpadPath string `json:"scratchpad_path"`
}

func runScratchJanitor(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("scratch-janitor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "root containing project/session scratchpad directories (required)")
	maxAge := fs.Duration("max-age", scratchjanitor.DefaultMaxAge, "minimum session age to select")
	resumeJSON := fs.String("resume-json", "", "resume JSON packet whose scratchpad references must be kept")
	apply := fs.Bool("apply", false, "remove selected sessions (default is dry-run)")
	jsonOutput := fs.Bool("json", false, "print the result as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak scratch-janitor: unexpected positional arguments")
		return 2
	}
	if strings.TrimSpace(*root) == "" {
		fmt.Fprintln(stderr, "fak scratch-janitor: --root is required")
		return 2
	}

	referenced := map[string]bool{}
	if *resumeJSON != "" {
		paths, err := readScratchJanitorReferences(*resumeJSON)
		if err != nil {
			fmt.Fprintln(stderr, "fak scratch-janitor:", err)
			return 1
		}
		for _, path := range paths {
			referenced[path] = true
		}
	}

	result, err := scratchjanitor.Scan(scratchjanitor.Config{
		Root:       *root,
		MaxAge:     *maxAge,
		Referenced: referenced,
		Apply:      *apply,
	})
	if err != nil {
		fmt.Fprintln(stderr, "fak scratch-janitor:", err)
		return 1
	}

	// The command's stable output is JSON. --json is accepted explicitly for
	// scripts that require machine-readable mode, while preserving that format
	// as the safe default.
	_ = *jsonOutput
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		fmt.Fprintln(stderr, "fak scratch-janitor: encode result:", err)
		return 1
	}
	return 0
}

func readScratchJanitorReferences(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open resume JSON: %w", err)
	}
	defer file.Close()

	var packet scratchJanitorResumePacket
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&packet); err != nil {
		return nil, fmt.Errorf("decode resume JSON: %w", err)
	}
	if err := ensureScratchJanitorJSONEOF(decoder); err != nil {
		return nil, err
	}

	candidates := packet.Candidates
	if packet.Scan != nil {
		candidates = append(candidates, packet.Scan.Candidates...)
	}
	paths := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		if strings.TrimSpace(candidate.ScratchpadPath) != "" {
			paths = append(paths, candidate.ScratchpadPath)
		}
	}
	return paths, nil
}

func ensureScratchJanitorJSONEOF(decoder *json.Decoder) error {
	var trailing any
	err := decoder.Decode(&trailing)
	switch {
	case errors.Is(err, io.EOF):
		return nil
	case err != nil:
		return fmt.Errorf("decode trailing resume JSON: %w", err)
	default:
		return fmt.Errorf("resume JSON contains multiple values")
	}
}
