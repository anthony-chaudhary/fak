package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/factorymigrate"
)

// RunFactoryMigrate is the main entrypoint for `fak-dev factory-migrate`.
func RunFactoryMigrate(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return runFactoryMigrateStatus(stdout, stderr, nil)
	}

	sub := argv[0]
	if sub == "-h" || sub == "--help" || sub == "help" {
		writeFactoryMigrateHelp(stdout)
		return 0
	}

	// If first argument is a flag, default to status subcommand
	if strings.HasPrefix(sub, "-") {
		return runFactoryMigrateStatus(stdout, stderr, argv)
	}

	subArgs := argv[1:]
	switch sub {
	case "status":
		return runFactoryMigrateStatus(stdout, stderr, subArgs)
	case "list":
		return runFactoryMigrateList(stdout, stderr, subArgs)
	case "next":
		return runFactoryMigrateNext(stdout, stderr, subArgs)
	case "audit-boundary", "boundary-audit":
		return runFactoryMigrateAuditBoundary(stdout, stderr, subArgs)
	case "scaffold":
		return runFactoryMigrateScaffold(stdout, stderr, subArgs)
	default:
		fmt.Fprintf(stderr, "fak-dev factory-migrate: unknown subcommand %q\n", sub)
		fmt.Fprintln(stderr, "Run 'fak-dev factory-migrate help' for usage.")
		return 2
	}
}

func writeFactoryMigrateHelp(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak-dev factory-migrate [command] [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Commands:")
	fmt.Fprintln(w, "  status         Display overall migration progress and cohort breakdown (default)")
	fmt.Fprintln(w, "  list           List all tools with optional --cohort and --status filters")
	fmt.Fprintln(w, "  next           Show the next unmigrated/partial candidates in priority order")
	fmt.Fprintln(w, "  audit-boundary Verify 5-Gate import boundaries in fak-private")
	fmt.Fprintln(w, "  scaffold <num> Generate scaffolding skeleton for a catalog item")
	fmt.Fprintln(w, "  help           Show this help message")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Global Flags:")
	fmt.Fprintln(w, "  --fak-root <path>     Public fak repository root (auto-detected by default)")
	fmt.Fprintln(w, "  --private-root <path> Private fak-private root (auto-detected or FAK_PRIVATE_ROOT)")
	fmt.Fprintln(w, "  --inventory <path>    Path to dev-process-top-100-tools-inventory.md")
	fmt.Fprintln(w, "  --json                Emit output in JSON format")
}

func resolveFactoryRoots(fakRoot, privateRoot, inventoryPath string) (string, string, string) {
	if fakRoot == "" {
		fakRoot = findFakRepoRoot()
	}
	if privateRoot == "" {
		if env := os.Getenv("FAK_PRIVATE_ROOT"); env != "" {
			privateRoot = env
		} else if fakRoot != "" {
			candidate := filepath.Join(fakRoot, "..", "fak-private")
			if fi, err := os.Stat(candidate); err == nil && fi.IsDir() {
				privateRoot = candidate
			}
		}
	}
	if inventoryPath == "" && fakRoot != "" {
		candidate := filepath.Join(fakRoot, "docs", "dev-process-top-100-tools-inventory.md")
		if _, err := os.Stat(candidate); err == nil {
			inventoryPath = candidate
		}
	}
	return fakRoot, privateRoot, inventoryPath
}

func findFakRepoRoot() string {
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	dir := cwd
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return cwd
}

func runFactoryMigrateStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("factory-migrate status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fakRoot := fs.String("fak-root", "", "path to public fak repository root")
	privateRoot := fs.String("private-root", "", "path to private fak-private repository root")
	inventory := fs.String("inventory", "", "path to dev-process-top-100-tools-inventory.md")
	asJSON := fs.Bool("json", false, "emit JSON format")

	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	rFak, rPriv, rInv := resolveFactoryRoots(*fakRoot, *privateRoot, *inventory)
	items, err := factorymigrate.ParseInventory(rInv)
	if err != nil {
		fmt.Fprintf(stderr, "factory-migrate: failed to parse inventory: %v\n", err)
		return 1
	}

	report := factorymigrate.AuditStatus(rFak, rPriv, items)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "factory-migrate: JSON encode error: %v\n", err)
			return 1
		}
		return 0
	}

	printStatusReport(stdout, report)
	return 0
}

func runFactoryMigrateList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("factory-migrate list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fakRoot := fs.String("fak-root", "", "path to public fak repository root")
	privateRoot := fs.String("private-root", "", "path to private fak-private repository root")
	inventory := fs.String("inventory", "", "path to dev-process-top-100-tools-inventory.md")
	cohortFilter := fs.String("cohort", "", "filter by cohort name or substring")
	statusFilter := fs.String("status", "", "filter by status (MIGRATED, PARTIAL, UNMIGRATED)")
	asJSON := fs.Bool("json", false, "emit JSON format")

	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	rFak, rPriv, rInv := resolveFactoryRoots(*fakRoot, *privateRoot, *inventory)
	items, err := factorymigrate.ParseInventory(rInv)
	if err != nil {
		fmt.Fprintf(stderr, "factory-migrate: failed to parse inventory: %v\n", err)
		return 1
	}

	report := factorymigrate.AuditStatus(rFak, rPriv, items)

	var filtered []factorymigrate.Item
	cLower := strings.ToLower(strings.TrimSpace(*cohortFilter))
	sUpper := strings.ToUpper(strings.TrimSpace(*statusFilter))

	for _, it := range report.Items {
		if cLower != "" && !strings.Contains(strings.ToLower(it.Cohort), cLower) &&
			!strings.Contains(strings.ToLower(it.TargetPkg), cLower) {
			continue
		}
		if sUpper != "" && string(it.Status) != sUpper {
			continue
		}
		filtered = append(filtered, it)
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(filtered); err != nil {
			fmt.Fprintf(stderr, "factory-migrate: JSON encode error: %v\n", err)
			return 1
		}
		return 0
	}

	printItemList(stdout, filtered)
	return 0
}

func runFactoryMigrateNext(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("factory-migrate next", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fakRoot := fs.String("fak-root", "", "path to public fak repository root")
	privateRoot := fs.String("private-root", "", "path to private fak-private repository root")
	inventory := fs.String("inventory", "", "path to dev-process-top-100-tools-inventory.md")
	cohortFilter := fs.String("cohort", "", "filter by cohort name or substring")
	count := fs.Int("count", 10, "number of candidates to return")
	asJSON := fs.Bool("json", false, "emit JSON format")

	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	rFak, rPriv, rInv := resolveFactoryRoots(*fakRoot, *privateRoot, *inventory)
	items, err := factorymigrate.ParseInventory(rInv)
	if err != nil {
		fmt.Fprintf(stderr, "factory-migrate: failed to parse inventory: %v\n", err)
		return 1
	}

	report := factorymigrate.AuditStatus(rFak, rPriv, items)
	candidates := factorymigrate.NextCandidates(report, *count, *cohortFilter)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(candidates); err != nil {
			fmt.Fprintf(stderr, "factory-migrate: JSON encode error: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "Next %d Migration Candidates (Priority Order):\n", len(candidates))
	printItemList(stdout, candidates)
	return 0
}

func runFactoryMigrateAuditBoundary(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("factory-migrate audit-boundary", flag.ContinueOnError)
	fs.SetOutput(stderr)
	privateRoot := fs.String("private-root", "", "path to private fak-private repository root")
	asJSON := fs.Bool("json", false, "emit JSON format")

	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	_, rPriv, _ := resolveFactoryRoots("", *privateRoot, "")
	if rPriv == "" {
		fmt.Fprintln(stderr, "factory-migrate audit-boundary: fak-private repository root not found (specify --private-root or set FAK_PRIVATE_ROOT)")
		return 1
	}

	violations, err := factorymigrate.AuditBoundary(rPriv)
	if err != nil {
		fmt.Fprintf(stderr, "factory-migrate audit-boundary: error auditing boundary: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(violations); err != nil {
			fmt.Fprintf(stderr, "factory-migrate: JSON encode error: %v\n", err)
			return 1
		}
		if len(violations) > 0 {
			return 1
		}
		return 0
	}

	if len(violations) == 0 {
		fmt.Fprintln(stdout, "Boundary audit passed: all Go files in fak-private/platform satisfy 5-Gate import invariants (0 violations).")
		return 0
	}

	fmt.Fprintf(stdout, "Found %d boundary violation(s) in %s:\n", len(violations), rPriv)
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "FILE:LINE\tRULE\tIMPORT\tREASON")
	for _, v := range violations {
		fmt.Fprintf(tw, "%s:%d\t%s\t%s\t%s\n", v.File, v.Line, v.Rule, v.ImportPath, v.Reason)
	}
	tw.Flush()
	return 1
}

func reorderFlagsAndArgs(argv []string, valueFlags map[string]bool) []string {
	var flags []string
	var positional []string

	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if strings.HasPrefix(arg, "-") {
			flags = append(flags, arg)
			flagName := strings.TrimLeft(arg, "-")
			if strings.Contains(flagName, "=") {
				continue
			}
			if valueFlags[flagName] && i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "-") {
				i++
				flags = append(flags, argv[i])
			}
		} else {
			positional = append(positional, arg)
		}
	}
	return append(flags, positional...)
}

func runFactoryMigrateScaffold(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("factory-migrate scaffold", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fakRoot := fs.String("fak-root", "", "path to public fak repository root")
	privateRoot := fs.String("private-root", "", "path to private fak-private repository root")
	inventory := fs.String("inventory", "", "path to dev-process-top-100-tools-inventory.md")
	dryRun := fs.Bool("dry-run", false, "simulate scaffolding without writing to disk")
	asJSON := fs.Bool("json", false, "emit JSON format")

	reordered := reorderFlagsAndArgs(argv, map[string]bool{
		"fak-root":     true,
		"private-root": true,
		"inventory":    true,
	})

	if err := fs.Parse(reordered); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	args := fs.Args()
	if len(args) == 0 {
		fmt.Fprintln(stderr, "factory-migrate scaffold: requires item number (e.g. 'fak-dev factory-migrate scaffold 1')")
		return 2
	}

	num, err := strconv.Atoi(args[0])
	if err != nil || num < 1 || num > 100 {
		fmt.Fprintf(stderr, "factory-migrate scaffold: invalid item number %q (must be between 1 and 100)\n", args[0])
		return 2
	}

	rFak, rPriv, rInv := resolveFactoryRoots(*fakRoot, *privateRoot, *inventory)
	if rPriv == "" {
		fmt.Fprintln(stderr, "factory-migrate scaffold: fak-private repository root not found (specify --private-root or set FAK_PRIVATE_ROOT)")
		return 1
	}

	items, err := factorymigrate.ParseInventory(rInv)
	if err != nil {
		fmt.Fprintf(stderr, "factory-migrate: failed to parse inventory: %v\n", err)
		return 1
	}

	var targetItem *factorymigrate.Item
	for i := range items {
		if items[i].Number == num {
			targetItem = &items[i]
			break
		}
	}
	if targetItem == nil {
		fmt.Fprintf(stderr, "factory-migrate scaffold: item %d not found in inventory\n", num)
		return 2
	}

	files, err := factorymigrate.Scaffold(rFak, rPriv, *targetItem, *dryRun)
	if err != nil {
		fmt.Fprintf(stderr, "factory-migrate scaffold: %v\n", err)
		return 1
	}

	if *asJSON {
		out := map[string]interface{}{
			"number":           num,
			"source_path":      targetItem.SourcePath,
			"target_path":      targetItem.TargetPath,
			"scaffolded_files": files,
			"dry_run":          *dryRun,
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fmt.Fprintf(stderr, "factory-migrate: JSON encode error: %v\n", err)
			return 1
		}
		return 0
	}

	if *dryRun {
		fmt.Fprintf(stdout, "[dry-run] Would scaffold item %d (%s -> %s):\n", num, targetItem.SourcePath, targetItem.TargetPath)
		for _, f := range files {
			fmt.Fprintf(stdout, "  would create: %s\n", f)
		}
	} else {
		fmt.Fprintf(stdout, "Scaffolded item %d (%s -> %s):\n", num, targetItem.SourcePath, targetItem.TargetPath)
		for _, f := range files {
			fmt.Fprintf(stdout, "  created: %s\n", f)
		}
	}

	return 0
}

func printStatusReport(w io.Writer, report factorymigrate.Report) {
	fmt.Fprintln(w, "=========================================================================================")
	fmt.Fprintln(w, "Autonomous Dev Factory Migration Status (Top 100 Tools)")
	fmt.Fprintln(w, "=========================================================================================")
	fmt.Fprintf(w, "Total: %d | Migrated: %d | Partial: %d | Unmigrated: %d | Progress: %.1f%%\n\n",
		report.Total, report.Migrated, report.Partial, report.Unmigrated, report.Percent)

	fmt.Fprintln(w, "Cohort Breakdown:")
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "COHORT\tTARGET PKG\tTOTAL\tMIGRATED\tPARTIAL\tUNMIGRATED\tPROGRESS")
	for _, c := range report.Cohorts {
		fmt.Fprintf(tw, "%s\t%s\t%d\t%d\t%d\t%d\t%.1f%%\n",
			c.Name, c.TargetPkg, c.Total, c.Migrated, c.Partial, c.Unmigrated, c.Percent)
	}
	tw.Flush()
	fmt.Fprintln(w, "=========================================================================================")
}

func printItemList(w io.Writer, items []factorymigrate.Item) {
	if len(items) == 0 {
		fmt.Fprintln(w, "No items found matching criteria.")
		return
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "NUM\tSTATUS\tTARGET PKG\tSOURCE PATH\tTARGET PATH\tNOTES")
	for _, it := range items {
		note := it.Notes
		if len(note) > 45 {
			note = note[:42] + "..."
		}
		fmt.Fprintf(tw, "%d\t%s\t%s\t%s\t%s\t%s\n",
			it.Number, it.Status, it.TargetPkg, it.SourcePath, it.TargetPath, note)
	}
	tw.Flush()
}
