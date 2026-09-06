package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/resultstier"
)

const (
	resultsTierSchema = "fak-results-tier/1"
	defaultStoreURI   = "blob://fak-results-payload"
)

func cmdResults(argv []string) { os.Exit(runResults(os.Stdout, os.Stderr, argv)) }

func runResults(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		resultsUsage(stderr)
		return 2
	}
	sub := argv[0]
	if sub == "-h" || sub == "--help" || sub == "help" {
		resultsUsage(stdout)
		return 0
	}
	if sub != "tier" {
		fmt.Fprintf(stderr, "fak results: unknown subcommand %q\n", sub)
		resultsUsage(stderr)
		return 2
	}

	fs := flag.NewFlagSet("results tier", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "results", "target results directory to inspect, mint, or verify")
	mint := fs.Bool("mint", false, "mint payload-index.json into target directory")
	verify := fs.Bool("verify", false, "verify payload-index.json against files on disk")
	store := fs.String("store", defaultStoreURI, "external payload store URI")
	unknown := fs.Bool("unknown", false, "list unknown files in the human report")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON structure")

	if err := fs.Parse(argv[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			resultsUsage(stdout)
			return 0
		}
		return 2
	}

	if *mint && *verify {
		fmt.Fprintln(stderr, "fak results tier: --mint and --verify are mutually exclusive")
		return 2
	}

	info, err := os.Stat(*dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak results tier: %v\n", err)
		return 1
	}
	if !info.IsDir() {
		fmt.Fprintf(stderr, "fak results tier: target is not a directory: %s\n", *dir)
		return 1
	}

	if *mint {
		return runResultsMint(stdout, stderr, *dir, *store, *asJSON)
	}
	if *verify {
		return runResultsVerify(stdout, stderr, *dir, *asJSON)
	}
	return runResultsCensus(stdout, stderr, *dir, *store, *unknown, *asJSON)
}

func runResultsMint(stdout, stderr io.Writer, dir, storeURI string, asJSON bool) int {
	idx, _, err := resultstier.MintPayloadIndex(dir, storeURI)
	if err != nil {
		fmt.Fprintf(stderr, "fak results tier: mint failed: %v\n", err)
		return 1
	}
	data, err := json.MarshalIndent(idx, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak results tier: failed to encode index: %v\n", err)
		return 1
	}
	data = append(data, '\n')
	indexPath := filepath.Join(dir, "payload-index.json")
	if err := os.WriteFile(indexPath, data, 0644); err != nil {
		fmt.Fprintf(stderr, "fak results tier: failed to write %s: %v\n", indexPath, err)
		return 1
	}

	if asJSON {
		res := map[string]any{
			"schema":     resultsTierSchema,
			"action":     "mint",
			"dir":        dir,
			"store_uri":  storeURI,
			"entries":    len(idx.Entries),
			"bytes":      idx.TotalBytes(),
			"index_path": filepath.ToSlash(indexPath),
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak results tier: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "minted %s: %d entries (%d bytes)\n", filepath.ToSlash(indexPath), len(idx.Entries), idx.TotalBytes())
	}
	return 0
}

func runResultsVerify(stdout, stderr io.Writer, dir string, asJSON bool) int {
	indexPath := filepath.Join(dir, "payload-index.json")
	data, err := os.ReadFile(indexPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak results tier: failed to read %s: %v\n", indexPath, err)
		return 1
	}
	var idx resultstier.PayloadIndex
	if err := json.Unmarshal(data, &idx); err != nil {
		fmt.Fprintf(stderr, "fak results tier: failed to parse %s: %v\n", indexPath, err)
		return 1
	}
	discrepancies, err := resultstier.VerifyPayloadIndex(dir, idx)
	if err != nil {
		fmt.Fprintf(stderr, "fak results tier: verify failed: %v\n", err)
		return 1
	}

	if len(discrepancies) > 0 {
		if asJSON {
			res := map[string]any{
				"schema":        resultsTierSchema,
				"action":        "verify",
				"dir":           dir,
				"status":        "discrepancy",
				"discrepancies": discrepancies,
			}
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			_ = enc.Encode(res)
		} else {
			for _, d := range discrepancies {
				fmt.Fprintf(stdout, "discrepancy: %s\n", d)
			}
		}
		return 1
	}

	if asJSON {
		res := map[string]any{
			"schema":  resultsTierSchema,
			"action":  "verify",
			"dir":     dir,
			"status":  "ok",
			"entries": len(idx.Entries),
			"bytes":   idx.TotalBytes(),
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
	} else {
		fmt.Fprintf(stdout, "OK: payload index verified (%d entries, %d bytes)\n", len(idx.Entries), idx.TotalBytes())
	}
	return 0
}

func runResultsCensus(stdout, stderr io.Writer, dir, storeURI string, showUnknown, asJSON bool) int {
	_, census, err := resultstier.MintPayloadIndex(dir, storeURI)
	if err != nil {
		fmt.Fprintf(stderr, "fak results tier: %v\n", err)
		return 1
	}
	unknownFiles, err := collectUnknownFiles(dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak results tier: %v\n", err)
		return 1
	}

	if asJSON {
		report := map[string]any{
			"schema":        resultsTierSchema,
			"dir":           dir,
			"store_uri":     storeURI,
			"total_files":   census.TotalFiles(),
			"total_bytes":   census.TotalBytes(),
			"claim_files":   census.ClaimFiles,
			"claim_bytes":   census.ClaimBytes,
			"payload_files": census.PayloadFiles,
			"payload_bytes": census.PayloadBytes,
			"unknown_files": census.UnknownFiles,
			"unknown_bytes": census.UnknownBytes,
			"payload_share": census.PayloadShare(),
			"shrink":        census.Shrink(),
			"unknown_exts":  census.UnknownExts,
			"unknown_paths": unknownFiles,
			"census": map[string]any{
				"claim_files":   census.ClaimFiles,
				"claim_bytes":   census.ClaimBytes,
				"payload_files": census.PayloadFiles,
				"payload_bytes": census.PayloadBytes,
				"unknown_files": census.UnknownFiles,
				"unknown_bytes": census.UnknownBytes,
				"total_files":   census.TotalFiles(),
				"total_bytes":   census.TotalBytes(),
				"payload_share": census.PayloadShare(),
				"shrink":        census.Shrink(),
				"unknown_exts":  census.UnknownExts,
			},
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak results tier: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "Results Tier Census: %s\n", dir)
	fmt.Fprintf(stdout, "Store URI: %s\n\n", storeURI)
	fmt.Fprintf(stdout, "Tiers:\n")
	fmt.Fprintf(stdout, "  Claim:   %d files, %d bytes\n", census.ClaimFiles, census.ClaimBytes)
	fmt.Fprintf(stdout, "  Payload: %d files, %d bytes\n", census.PayloadFiles, census.PayloadBytes)
	fmt.Fprintf(stdout, "  Unknown: %d files, %d bytes\n\n", census.UnknownFiles, census.UnknownBytes)
	fmt.Fprintf(stdout, "Total: %d files, %d bytes\n", census.TotalFiles(), census.TotalBytes())
	fmt.Fprintf(stdout, "Payload share: %.2f%%\n", census.PayloadShare()*100)
	fmt.Fprintf(stdout, "Externalize shrink: %.2fx\n", census.Shrink())

	if len(census.UnknownExts) > 0 {
		fmt.Fprintf(stdout, "\nUnknown extensions:\n")
		exts := make([]string, 0, len(census.UnknownExts))
		for ext := range census.UnknownExts {
			exts = append(exts, ext)
		}
		sort.Strings(exts)
		for _, ext := range exts {
			fmt.Fprintf(stdout, "  %s: %d\n", ext, census.UnknownExts[ext])
		}
	}

	if showUnknown {
		fmt.Fprintf(stdout, "\nUnknown files (%d):\n", len(unknownFiles))
		if len(unknownFiles) == 0 {
			fmt.Fprintf(stdout, "  (none)\n")
		} else {
			for _, u := range unknownFiles {
				fmt.Fprintf(stdout, "  %s\n", u)
			}
		}
	}

	return 0
}

func collectUnknownFiles(dir string) ([]string, error) {
	var unknown []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Type()&os.ModeSymlink != 0 || !d.Type().IsRegular() {
			return nil
		}
		name := d.Name()
		if name == ".DS_Store" || name == ".gitignore" || name == ".gitkeep" || name == ".git" {
			return nil
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		relSlash := filepath.ToSlash(rel)
		relSlash = strings.TrimPrefix(relSlash, "./")
		tier, _ := resultstier.TierOf(relSlash)
		if tier == resultstier.TierUnknown {
			unknown = append(unknown, relSlash)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(unknown)
	return unknown, nil
}

func resultsUsage(w io.Writer) {
	fmt.Fprint(w, `fak results — manage storage tiers and external payload indexing for results

Usage:
  fak results tier [--dir <dir>] [--mint] [--verify] [--store <uri>] [--unknown] [--json]

Modes:
  default       Walks --dir, computes census, and prints a human report (or JSON).
  --mint        Mints payload-index.json in --dir with hashed payload files.
  --verify      Verifies existing payload-index.json against files on disk.

Flags:
  --dir <dir>   Directory to inspect or mint (default: "results").
  --store <uri> External payload store URI (default: "blob://fak-results-payload").
  --unknown     List unknown files in the human report.
  --json        Emit machine-readable JSON structure.
  -h, --help    Show this help message.
`)
}
