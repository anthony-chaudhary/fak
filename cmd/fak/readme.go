package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/readmenext"
)

var readmeExit = os.Exit

func cmdReadme(args []string) {
	readmeExit(runReadme(os.Stdout, os.Stderr, args))
}

func runReadme(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		printReadmeHelp(stdout)
		return 0
	}

	switch args[0] {
	case "lint-staged":
		return runReadmeLintStaged(stdout, stderr, args[1:])
	case "preview-next":
		return runReadmePreviewNext(stdout, stderr, args[1:])
	case "publish":
		return runReadmePublish(stdout, stderr, args[1:])
	case "init-fragment":
		return runReadmeInitFragment(stdout, stderr, args[1:])
	case "help", "-h", "--help":
		printReadmeHelp(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown readme command: %q. Run 'fak readme help' for usage.\n", args[0])
		return 2
	}
}

func printReadmeHelp(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak readme <subcommand> [options]")
	fmt.Fprintln(w, "   or: fak readmenext <subcommand> [options]")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Subcommands:")
	fmt.Fprintln(w, "  lint-staged     Validate all candidate fragments in directory")
	fmt.Fprintln(w, "  preview-next    Preview README synthesis with staged fragments applied")
	fmt.Fprintln(w, "  publish         Publish staged fragments to README.md and archives")
	fmt.Fprintln(w, "  init-fragment   Initialize a clean candidate fragment template")
	fmt.Fprintln(w, "  help            Show this help message")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Options:")
	fmt.Fprintln(w, "  Run 'fak readme <subcommand> -h' for subcommand-specific flags.")
}

func resolveRepoAndDir(repoFlag, dirFlag string) (string, string) {
	repoRoot := repoFlag
	if repoRoot != "" {
		repoRoot = filepath.Clean(repoRoot)
	} else {
		if dirFlag != "" {
			if absDir, err := filepath.Abs(dirFlag); err == nil {
				cur := absDir
				for {
					if _, err := os.Stat(filepath.Join(cur, readmenext.DefaultReadmePath)); err == nil {
						repoRoot = cur
						break
					}
					parent := filepath.Dir(cur)
					if parent == cur {
						break
					}
					cur = parent
				}
			}
		}
		if repoRoot == "" {
			root := findRepoRoot(".")
			if _, err := os.Stat(filepath.Join(root, readmenext.DefaultReadmePath)); err == nil {
				repoRoot = root
			} else if cwd, err := os.Getwd(); err == nil {
				repoRoot = cwd
			} else {
				repoRoot = "."
			}
		}
	}

	stagingDir := dirFlag
	if !filepath.IsAbs(stagingDir) {
		stagingDir = filepath.Join(repoRoot, filepath.Clean(stagingDir))
	} else {
		stagingDir = filepath.Clean(stagingDir)
	}
	return repoRoot, stagingDir
}

type loadedFragmentFile struct {
	Path     string
	Filename string
	Frag     *readmenext.CandidateFragment
	ParseErr error
}

func scanFragmentFiles(stagingDir string) ([]loadedFragmentFile, error) {
	entries, err := os.ReadDir(stagingDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var files []loadedFragmentFile
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") {
			continue
		}
		if !strings.HasSuffix(strings.ToLower(name), ".json") {
			continue
		}
		if name == "candidate-template.json" || strings.HasSuffix(name, ".template.json") {
			continue
		}
		p := filepath.Join(stagingDir, name)
		frag, parseErr := readmenext.LoadCandidateFragmentFile(p)
		files = append(files, loadedFragmentFile{
			Path:     p,
			Filename: name,
			Frag:     frag,
			ParseErr: parseErr,
		})
	}
	sort.Slice(files, func(i, j int) bool {
		return files[i].Filename < files[j].Filename
	})
	return files, nil
}

func runReadmeLintStaged(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak readme lint-staged", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var dirFlag string
	var asJSON bool
	var repoFlag string
	fs.StringVar(&dirFlag, "dir", "docs/readme-next", "Directory containing staged fragment files")
	fs.BoolVar(&asJSON, "json", false, "Output validation results in JSON format")
	fs.StringVar(&repoFlag, "repo", "", "Repository root directory (defaults to auto-detected root)")

	if err := fs.Parse(args); err != nil {
		return 2
	}

	repoRoot, stagingDir := resolveRepoAndDir(repoFlag, dirFlag)
	files, err := scanFragmentFiles(stagingDir)
	if err != nil {
		if asJSON {
			_ = json.NewEncoder(stdout).Encode(map[string]interface{}{
				"valid": false,
				"error": err.Error(),
			})
		} else {
			fmt.Fprintf(stderr, "error scanning fragments in %s: %v\n", stagingDir, err)
		}
		return 1
	}

	type fragmentLintItem struct {
		File          string `json:"file"`
		Valid         bool   `json:"valid"`
		Issue         int    `json:"issue,omitempty"`
		Topic         string `json:"topic,omitempty"`
		TargetSection string `json:"target_section,omitempty"`
		Error         string `json:"error,omitempty"`
	}

	type lintReport struct {
		Valid     bool                `json:"valid"`
		Total     int                 `json:"total"`
		Passed    int                 `json:"passed"`
		Failed    int                 `json:"failed"`
		Fragments []*fragmentLintItem `json:"fragments"`
	}

	report := lintReport{
		Valid:     true,
		Total:     len(files),
		Fragments: make([]*fragmentLintItem, 0, len(files)),
	}

	for _, f := range files {
		item := &fragmentLintItem{
			File: f.Filename,
		}
		if f.ParseErr != nil {
			item.Valid = false
			item.Error = fmt.Sprintf("parse error: %v", f.ParseErr)
		} else {
			item.Issue = f.Frag.Issue
			item.Topic = f.Frag.Topic
			item.TargetSection = f.Frag.TargetSection
			if valErr := readmenext.ValidateFragment(f.Frag, repoRoot); valErr != nil {
				item.Valid = false
				item.Error = valErr.Error()
			} else {
				item.Valid = true
			}
		}

		if item.Valid {
			report.Passed++
		} else {
			report.Failed++
			report.Valid = false
		}
		report.Fragments = append(report.Fragments, item)
	}

	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "error encoding JSON report: %v\n", err)
			return 1
		}
	} else {
		if report.Total == 0 {
			fmt.Fprintf(stdout, "README-NEXT Lint: 0 fragments found in %s (clean)\n", stagingDir)
		} else {
			fmt.Fprintf(stdout, "README-NEXT Lint: %d/%d fragments valid (%d passed, %d failed)\n",
				report.Passed, report.Total, report.Passed, report.Failed)
			for _, item := range report.Fragments {
				if item.Valid {
					fmt.Fprintf(stdout, "  [PASS] %s: issue #%d (topic: %s, section: %s)\n",
						item.File, item.Issue, item.Topic, item.TargetSection)
				} else {
					fmt.Fprintf(stdout, "  [FAIL] %s: %s\n", item.File, item.Error)
				}
			}
		}
	}

	if !report.Valid {
		return 1
	}
	return 0
}

func runReadmePreviewNext(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak readme preview-next", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var dirFlag string
	var outPath string
	var repoFlag string
	fs.StringVar(&dirFlag, "dir", "docs/readme-next", "Directory containing staged fragment files")
	fs.StringVar(&outPath, "out", "", "Output path for synthesized README (or stdout summary if empty)")
	fs.StringVar(&repoFlag, "repo", "", "Repository root directory (defaults to auto-detected root)")

	if err := fs.Parse(args); err != nil {
		return 2
	}

	repoRoot, stagingDir := resolveRepoAndDir(repoFlag, dirFlag)

	readmePath := filepath.Join(repoRoot, readmenext.DefaultReadmePath)
	readmeBytes, err := os.ReadFile(readmePath)
	if err != nil {
		fmt.Fprintf(stderr, "failed to read %s: %v\n", readmePath, err)
		return 1
	}

	files, err := scanFragmentFiles(stagingDir)
	if err != nil {
		fmt.Fprintf(stderr, "error scanning fragments in %s: %v\n", stagingDir, err)
		return 1
	}

	var validFragments []*readmenext.CandidateFragment
	for _, f := range files {
		if f.ParseErr != nil {
			continue
		}
		if err := readmenext.ValidateFragment(f.Frag, repoRoot); err != nil {
			continue
		}
		validFragments = append(validFragments, f.Frag)
	}

	simulated, changes, err := readmenext.PreviewNext(string(readmeBytes), validFragments)
	if err != nil {
		fmt.Fprintf(stderr, "preview synthesis failed: %v\n", err)
		return 1
	}

	if outPath == "-" {
		_, _ = io.WriteString(stdout, simulated)
		return 0
	}

	if outPath != "" {
		targetFile := outPath
		if !filepath.IsAbs(targetFile) {
			targetFile = filepath.Join(repoRoot, filepath.Clean(targetFile))
		}
		if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
			fmt.Fprintf(stderr, "failed to create directory for %s: %v\n", targetFile, err)
			return 1
		}
		if err := os.WriteFile(targetFile, []byte(simulated), 0644); err != nil {
			fmt.Fprintf(stderr, "failed to write preview to %s: %v\n", targetFile, err)
			return 1
		}
		fmt.Fprintf(stdout, "Preview written to %s (%d fragments applied, %d changes)\n",
			outPath, len(validFragments), len(changes))
		if len(changes) > 0 {
			fmt.Fprintln(stdout, "Changes:")
			for _, c := range changes {
				fmt.Fprintf(stdout, "  - %s\n", c)
			}
		}
		return 0
	}

	// Default: print diff/preview summary to stdout
	fmt.Fprintf(stdout, "README-NEXT Preview: %d fragments applied\n", len(validFragments))
	if len(changes) == 0 {
		fmt.Fprintln(stdout, "No staged changes to apply.")
	} else {
		fmt.Fprintf(stdout, "Changes (%d):\n", len(changes))
		for _, c := range changes {
			fmt.Fprintf(stdout, "  - %s\n", c)
		}
	}
	return 0
}

func runReadmePublish(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak readme publish", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var dirFlag string
	var dryRunFlag bool
	var applyFlag bool
	var asJSON bool
	var repoFlag string
	fs.StringVar(&dirFlag, "dir", "docs/readme-next", "Directory containing staged fragment files")
	fs.BoolVar(&dryRunFlag, "dry-run", false, "Simulate publish without modifying files")
	fs.BoolVar(&applyFlag, "apply", false, "Commit changes to README.md and archive docs")
	fs.BoolVar(&asJSON, "json", false, "Output publish results in JSON format")
	fs.StringVar(&repoFlag, "repo", "", "Repository root directory (defaults to auto-detected root)")

	if err := fs.Parse(args); err != nil {
		return 2
	}

	dryRun := true
	if applyFlag {
		dryRun = false
	} else if dryRunFlag {
		dryRun = true
	}

	repoRoot, stagingDir := resolveRepoAndDir(repoFlag, dirFlag)

	files, err := scanFragmentFiles(stagingDir)
	if err != nil {
		if asJSON {
			_ = json.NewEncoder(stdout).Encode(map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			})
		} else {
			fmt.Fprintf(stderr, "error scanning fragments in %s: %v\n", stagingDir, err)
		}
		return 1
	}

	var fragments []*readmenext.CandidateFragment
	for _, f := range files {
		if f.ParseErr != nil {
			if asJSON {
				_ = json.NewEncoder(stdout).Encode(map[string]interface{}{
					"success": false,
					"error":   fmt.Sprintf("error parsing %s: %v", f.Filename, f.ParseErr),
				})
			} else {
				fmt.Fprintf(stderr, "error parsing fragment %s: %v\n", f.Filename, f.ParseErr)
			}
			return 1
		}
		fragments = append(fragments, f.Frag)
	}

	res, err := readmenext.Publish(repoRoot, fragments, dryRun)
	if err != nil {
		if asJSON {
			_ = json.NewEncoder(stdout).Encode(map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			})
		} else {
			fmt.Fprintf(stderr, "publish failed: %v\n", err)
		}
		return 1
	}

	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "error encoding JSON: %v\n", err)
			return 1
		}
		return 0
	}

	if dryRun {
		fmt.Fprintf(stdout, "[DRY-RUN] README-NEXT Publish: %d fragments evaluated (no files modified, use --apply to commit)\n", len(fragments))
	} else {
		fmt.Fprintf(stdout, "[SUCCESS] README-NEXT Publish: %d fragments applied to %s\n", len(fragments), res.ReadmePath)
	}
	if len(res.Changes) > 0 {
		fmt.Fprintln(stdout, "Changes:")
		for _, c := range res.Changes {
			fmt.Fprintf(stdout, "  - %s\n", c)
		}
	}
	if len(res.RetiredItems) > 0 {
		fmt.Fprintln(stdout, "Retired items:")
		for _, r := range res.RetiredItems {
			fmt.Fprintf(stdout, "  - %s\n", r)
		}
	}
	if res.HardwareJSONUpdated {
		fmt.Fprintf(stdout, "Hardware manifest updated: %s\n", res.HardwareJSONPath)
	}
	if res.LegacyPath != "" && !dryRun {
		fmt.Fprintf(stdout, "Legacy archive updated: %s\n", res.LegacyPath)
	}
	return 0
}

func runReadmeInitFragment(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak readme init-fragment", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var issue int
	var topic string
	var section string
	var outPath string
	var dirFlag string
	fs.IntVar(&issue, "issue", 0, "GitHub issue number (required, positive integer)")
	fs.StringVar(&topic, "topic", "", "Topic slug (required, e.g. nvidia-h100-q8)")
	fs.StringVar(&section, "section", readmenext.TargetWhyFak, "Target section (hardware_table, hero_headline, memory_overflow, why_fak, default_priorities, custom)")
	fs.StringVar(&outPath, "out", "", "Output file path (default: <dir>/issue-<N>-<topic>.json)")
	fs.StringVar(&dirFlag, "dir", "docs/readme-next", "Directory to write fragment to when --out is omitted")

	if err := fs.Parse(args); err != nil {
		return 2
	}

	if issue <= 0 {
		fmt.Fprintln(stderr, "error: --issue must be a positive integer")
		return 1
	}
	topic = strings.TrimSpace(topic)
	if topic == "" {
		fmt.Fprintln(stderr, "error: --topic cannot be empty")
		return 1
	}

	switch section {
	case readmenext.TargetHardwareTable, readmenext.TargetHeroHeadline,
		readmenext.TargetMemoryOverflow, readmenext.TargetWhyFak,
		readmenext.TargetDefaultPriorities, readmenext.TargetCustom:
		// Valid section
	default:
		fmt.Fprintf(stderr, "error: unsupported section %q (valid: hardware_table, hero_headline, memory_overflow, why_fak, default_priorities, custom)\n", section)
		return 1
	}

	today := time.Now().UTC().Format("2006-01-02")
	frag := &readmenext.CandidateFragment{
		Schema:        readmenext.SchemaCandidate,
		Issue:         issue,
		Topic:         topic,
		TargetSection: section,
		Date:          today,
		LawsChecklist: readmenext.LawsChecklist{
			SOTAComparison: true,
			FeynmanGloss:   true,
			WideAudience:   true,
		},
	}

	switch section {
	case readmenext.TargetHardwareTable:
		frag.CandidateContent = fmt.Sprintf("| Platform | %s: witnessed result | Verified | [Details](docs/...) |", topic)
		frag.RetireTarget = readmenext.RetireTarget{
			Action:     readmenext.RetireActionReplaceRow,
			TargetText: "| Platform | Old result row to replace |",
		}
	case readmenext.TargetHeroHeadline:
		frag.CandidateContent = fmt.Sprintf("**fak is an agent runtime: %s.**", topic)
		frag.RetireTarget = readmenext.RetireTarget{
			Action:     readmenext.RetireActionReplaceRow,
			TargetText: "**fak is an agent runtime: one binary puts a fast, cache-accelerated boundary between your coding agent and every tool call.**",
		}
	case readmenext.TargetWhyFak:
		frag.CandidateContent = fmt.Sprintf("- **%s:** Concise summary of capability and verified impact.", topic)
		frag.RetireTarget = readmenext.RetireTarget{
			Action: readmenext.RetireActionNone,
		}
	case readmenext.TargetMemoryOverflow:
		frag.CandidateContent = fmt.Sprintf("- **%s:** Memory tiering and cache paging details.", topic)
		frag.RetireTarget = readmenext.RetireTarget{
			Action: readmenext.RetireActionNone,
		}
	case readmenext.TargetDefaultPriorities:
		frag.CandidateContent = fmt.Sprintf("5. **%s**", topic)
		frag.RetireTarget = readmenext.RetireTarget{
			Action: readmenext.RetireActionNone,
		}
	case readmenext.TargetCustom:
		frag.CandidateContent = fmt.Sprintf("### %s\n\nCustom section content.", topic)
		frag.RetireTarget = readmenext.RetireTarget{
			Action: readmenext.RetireActionNone,
		}
	}

	data, err := json.MarshalIndent(frag, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "failed to marshal fragment: %v\n", err)
		return 1
	}
	data = append(data, '\n')

	if outPath == "-" {
		_, _ = stdout.Write(data)
		return 0
	}

	if outPath == "" {
		slug := strings.ReplaceAll(strings.ToLower(topic), " ", "-")
		filename := fmt.Sprintf("issue-%d-%s.json", issue, slug)
		outPath = filepath.Join(dirFlag, filename)
	}

	if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
		fmt.Fprintf(stderr, "failed to create directory for %s: %v\n", outPath, err)
		return 1
	}

	if err := os.WriteFile(outPath, data, 0644); err != nil {
		fmt.Fprintf(stderr, "failed to write fragment to %s: %v\n", outPath, err)
		return 1
	}

	fmt.Fprintf(stdout, "Created fragment template: %s\n", outPath)
	return 0
}
