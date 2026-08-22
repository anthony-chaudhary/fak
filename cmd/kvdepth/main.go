// Command kvdepth measures reusable prefix depth without collapsing the
// result into a single cache-hit scalar.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }

func run(stdout, stderr io.Writer, args []string) int {
	flags := flag.NewFlagSet("kvdepth", flag.ContinueOnError)
	flags.SetOutput(stderr)
	manifestPath := flags.String("manifest", "", "campaign manifest JSON")
	observationsPath := flags.String("observations", "", "request-level observation JSONL")
	outputPath := flags.String("output", "", "also write the JSON report to this path")
	selfcheck := flags.Bool("selfcheck", false, "run the deterministic 8k/12k cliff and missing-evidence fixtures")
	emitFixturesDir := flags.String("emit-fixtures", "", "write deterministic raw fixtures and captured witness to DIR")
	if err := flags.Parse(args); err != nil {
		printUsage(stderr)
		return 2
	}
	if flags.NArg() != 0 || *manifestPath == "" || (*selfcheck && *observationsPath != "") || (*emitFixturesDir != "" && (*selfcheck || *observationsPath != "" || *outputPath != "")) {
		printUsage(stderr)
		return 2
	}
	manifest, err := readManifest(*manifestPath)
	if err != nil {
		fmt.Fprintf(stderr, "kvdepth: %v\n", err)
		return 1
	}
	if *emitFixturesDir != "" {
		if err := emitFixtures(*emitFixturesDir, manifest); err != nil {
			fmt.Fprintf(stderr, "kvdepth: emit fixtures: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "PASS fixtures=%s\n", *emitFixturesDir)
		return 0
	}
	if *selfcheck {
		witness, err := BuildSelfcheck(manifest)
		if err != nil {
			fmt.Fprintf(stderr, "kvdepth: selfcheck: %v\n", err)
			return 1
		}
		if err := writeJSON(stdout, *outputPath, witness); err != nil {
			fmt.Fprintf(stderr, "kvdepth: write selfcheck: %v\n", err)
			return 1
		}
		return 0
	}
	if *observationsPath == "" {
		printUsage(stderr)
		return 2
	}
	observations, err := readObservations(*observationsPath)
	if err != nil {
		fmt.Fprintf(stderr, "kvdepth: %v\n", err)
		return 1
	}
	report, err := Analyze(manifest, observations)
	if err != nil {
		fmt.Fprintf(stderr, "kvdepth: analyze: %v\n", err)
		return 1
	}
	if err := writeJSON(stdout, *outputPath, report); err != nil {
		fmt.Fprintf(stderr, "kvdepth: write report: %v\n", err)
		return 1
	}
	return 0
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: kvdepth -manifest CAMPAIGN.json (-observations RAW.jsonl | -selfcheck | -emit-fixtures DIR) [-output REPORT.json]")
	fmt.Fprintln(w, "recovery: provide one readable manifest and choose exactly one analysis mode; -output is valid with observations or selfcheck")
}

func readManifest(path string) (Manifest, error) {
	body, err := os.ReadFile(path)
	if err != nil {
		return Manifest{}, fmt.Errorf("read manifest: %w; recovery: provide an existing readable campaign manifest path", err)
	}
	var manifest Manifest
	if err := decodeStrict(body, &manifest); err != nil {
		return Manifest{}, fmt.Errorf("decode manifest: %w; recovery: provide exactly one valid JSON object matching the campaign schema", err)
	}
	if err := manifest.Validate(); err != nil {
		return Manifest{}, fmt.Errorf("validate manifest: %w", err)
	}
	return manifest, nil
}

func readObservations(path string) ([]Observation, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open observations: %w; recovery: provide an existing readable request-level observation JSONL path", err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	var observations []Observation
	for line := 1; scanner.Scan(); line++ {
		body := bytes.TrimSpace(scanner.Bytes())
		if len(body) == 0 {
			continue
		}
		var observation Observation
		if err := decodeStrict(body, &observation); err != nil {
			return nil, fmt.Errorf("decode observations line %d: %w; recovery: replace that line with exactly one valid observation JSON object", line, err)
		}
		observations = append(observations, observation)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan observations: %w; recovery: make every JSONL record at most 4 MiB and ensure the file remains readable", err)
	}
	return observations, nil
}

func decodeStrict(body []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("decode JSON: %w; recovery: provide exactly one valid JSON object with only declared fields", err)
	}
	if decoder.More() {
		return errors.New("multiple JSON values; recovery: keep exactly one JSON object in each manifest or JSONL record")
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("multiple JSON values; recovery: keep exactly one JSON object in each manifest or JSONL record")
		}
		return fmt.Errorf("decode trailing JSON: %w; recovery: remove trailing non-whitespace bytes after the single JSON object", err)
	}
	return nil
}

func writeJSON(stdout io.Writer, outputPath string, value any) error {
	body, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode JSON report: %w; recovery: remove unsupported values from the report before writing it", err)
	}
	body = append(body, '\n')
	if outputPath != "" {
		if err := os.WriteFile(outputPath, body, 0o644); err != nil {
			return fmt.Errorf("write output file: %w; recovery: choose a writable file path whose parent directory exists", err)
		}
	}
	if _, err := stdout.Write(body); err != nil {
		return fmt.Errorf("write stdout: %w; recovery: restore the output stream or use -output with a writable file", err)
	}
	return nil
}

func emitFixtures(dir string, manifest Manifest) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create fixture directory: %w; recovery: choose a writable directory path not occupied by a file", err)
	}
	known, err := SyntheticObservations(manifest, true)
	if err != nil {
		return err
	}
	missing, err := SyntheticObservations(manifest, false)
	if err != nil {
		return err
	}
	if err := writeObservationJSONL(filepath.Join(dir, "known-cliff.jsonl"), known); err != nil {
		return fmt.Errorf("write known-cliff fixture: %w", err)
	}
	if err := writeObservationJSONL(filepath.Join(dir, "missing-cache-evidence.jsonl"), missing); err != nil {
		return fmt.Errorf("write missing-evidence fixture: %w", err)
	}
	witness, err := BuildSelfcheck(manifest)
	if err != nil {
		return err
	}
	body, err := json.MarshalIndent(witness, "", "  ")
	if err != nil {
		return fmt.Errorf("encode selfcheck witness: %w; recovery: remove unsupported values from the selfcheck report before emitting fixtures", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "known-cliff-witness.json"), append(body, '\n'), 0o644); err != nil {
		return fmt.Errorf("write selfcheck witness: %w; recovery: make known-cliff-witness.json a writable file path", err)
	}
	return nil
}

func writeObservationJSONL(path string, observations []Observation) error {
	var output bytes.Buffer
	encoder := json.NewEncoder(&output)
	encoder.SetEscapeHTML(false)
	for _, observation := range observations {
		if err := encoder.Encode(observation); err != nil {
			return fmt.Errorf("encode observation JSONL: %w; recovery: remove unsupported values from the observation before writing the fixture", err)
		}
	}
	if err := os.WriteFile(path, output.Bytes(), 0o644); err != nil {
		return fmt.Errorf("write observation JSONL: %w; recovery: choose a writable fixture file path whose parent directory exists", err)
	}
	return nil
}
