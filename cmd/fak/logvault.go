package main

// fak logvault — central, chain-aware backup of every durable fak log store
// (guard decision journals, DOS kernel state, dispatch run logs, loop/usage
// ledgers, the Claude Code harness store) into one vault directory.
//
// Parked scaffold: wired into main.go's verb switch by the scaffold wave.
//
//	fak logvault plan      diff live sources against the vault (dry-run, default)
//	fak logvault capture   copy new/appended/rewritten files + append chained manifest rows
//	fak logvault verify    re-derive the manifest hash chain + re-hash mirrors
//	fak logvault sources   print the source registry resolved for this box
//
// The vault defaults to a sibling directory of the repo root (<parent>/fak-log-vault,
// override with -vault or FAK_LOG_VAULT) so it never lives inside any git tree it
// captures from. Sources are read-only; a file that cannot be read is recorded as
// a skip-error manifest row and retried next capture.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/logvault"
)

func cmdLogvault(argv []string) { os.Exit(runLogvault(os.Stdout, os.Stderr, argv)) }

func runLogvault(w, ew io.Writer, argv []string) int {
	verb := "plan"
	if len(argv) > 0 && argv[0] != "" && argv[0][0] != '-' {
		verb, argv = argv[0], argv[1:]
	}
	fs := flag.NewFlagSet("logvault", flag.ContinueOnError)
	fs.SetOutput(ew)
	repo := fs.String("repo", "", "repo root holding the state dirs (default: current directory)")
	vaultDir := fs.String("vault", "", "vault directory (default: $FAK_LOG_VAULT, else <repo-parent>/fak-log-vault)")
	sample := fs.Int("sample", 250, "verify: mirrors to re-hash (0 = all)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		// `fak logvault -vault X capture` must not silently run the default verb.
		fmt.Fprintf(ew, "logvault: unexpected argument %q — the verb comes first: fak logvault %s [flags]\n", fs.Arg(0), fs.Arg(0))
		return 2
	}
	if *repo == "" {
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(ew, "logvault: %v\n", err)
			return 1
		}
		*repo = cwd
	}
	absRepo, err := filepath.Abs(*repo)
	if err != nil {
		fmt.Fprintf(ew, "logvault: %v\n", err)
		return 1
	}
	if *vaultDir == "" {
		*vaultDir = os.Getenv("FAK_LOG_VAULT")
	}
	if *vaultDir == "" {
		*vaultDir = filepath.Join(filepath.Dir(absRepo), "fak-log-vault")
	}
	home, _ := os.UserHomeDir()
	v := &logvault.Vault{Dir: *vaultDir, Sources: logvault.DefaultSources(absRepo, home)}

	switch verb {
	case "sources":
		for _, s := range v.Sources {
			fmt.Fprintf(w, "%-22s %s\n", s.ID, s.Root)
			fmt.Fprintf(w, "%-22s   %s\n", "", s.Note)
		}
		return 0
	case "plan", "capture":
		var stats []logvault.SourceStats
		if verb == "plan" {
			stats, err = v.Plan()
		} else {
			stats, err = v.Capture()
		}
		if err != nil {
			fmt.Fprintf(ew, "logvault %s: %v\n", verb, err)
			return 1
		}
		fmt.Fprintf(w, "logvault %s  vault=%s\n", verb, v.Dir)
		var files int
		var bytes int64
		var errs int
		for _, st := range stats {
			if st.Missing {
				fmt.Fprintf(w, "  %-22s (absent on this box)\n", st.Source)
				continue
			}
			fmt.Fprintf(w, "  %-22s files=%-6d unchanged=%-6d full=%-5d append=%-4d rewrite=%-4d errors=%-3d copy=%s\n",
				st.Source, st.Files, st.Unchanged, st.Full, st.Append, st.Rewrite, st.Errors, fmtBytesLV(st.CopyBytes))
			files += st.Files
			bytes += st.CopyBytes
			errs += st.Errors
		}
		fmt.Fprintf(w, "TOTAL files=%d copy=%s errors=%d (WITNESSED: sizes stat'd, hashes computed, by this run)\n",
			files, fmtBytesLV(bytes), errs)
		if errs > 0 {
			return 1
		}
		return 0
	case "verify":
		rows, checked, problems, err := v.Verify(*sample)
		if err != nil {
			fmt.Fprintf(ew, "logvault verify: manifest chain BROKEN: %v\n", err)
			return 1
		}
		fmt.Fprintf(w, "logvault verify  vault=%s\n", v.Dir)
		fmt.Fprintf(w, "  manifest chain OK: %d rows\n", rows)
		fmt.Fprintf(w, "  mirrors re-hashed: %d, mismatches: %d\n", checked, len(problems))
		for _, p := range problems {
			fmt.Fprintf(w, "  PROBLEM %s/%s: %s\n", p.Source, p.RelPath, p.Reason)
		}
		if len(problems) > 0 {
			return 1
		}
		return 0
	default:
		fmt.Fprintf(ew, "logvault: unknown verb %q (plan|capture|verify|sources)\n", verb)
		return 2
	}
}

func fmtBytesLV(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1fGB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1fMB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1fKB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%dB", n)
	}
}
