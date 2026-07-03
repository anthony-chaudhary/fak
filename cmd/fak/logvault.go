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
	"github.com/anthony-chaudhary/fak/internal/slackenv"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
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
	notifySlack := fs.Bool("notify-slack", false, "capture/verify: enqueue a durable Slack digest (counts + the vault-head chain anchor) through the slack outbox")
	slackChannel := fs.String("slack-channel", "", "channel for -notify-slack (default: $FAK_DISPATCH_CHANNEL, then $FAK_SCOREBOARD_CHANNEL)")
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
		if verb == "capture" && *notifySlack {
			logvaultNotifySlack(w, ew, *slackChannel, logvaultCaptureDigest(v.Dir, files, bytes, errs))
		}
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
		if *notifySlack {
			logvaultNotifySlack(w, ew, *slackChannel, logvaultVerifyDigest(v.Dir, rows, checked, problems))
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

// logvaultAnchorSuffix renders the chain-head anchor for a digest line — the
// off-box tamper-evidence witness: even a fully hijacked vault (local anchor
// file rewritten to match a corrupted manifest) cannot un-post a PAST digest
// naming the prior honest (seq, hash).
func logvaultAnchorSuffix(vaultDir string) string {
	seq, hash, ok, err := logvault.ReadAnchor(vaultDir)
	if err != nil {
		return fmt.Sprintf(" · anchor unreadable: %v", err)
	}
	if !ok {
		return " · anchor: none yet"
	}
	if len(hash) > 12 {
		hash = hash[:12]
	}
	return fmt.Sprintf(" · anchor seq=%d hash=%s…", seq, hash)
}

// logvaultCaptureDigest renders the one-line Slack digest for a capture run.
func logvaultCaptureDigest(vaultDir string, files int, bytes int64, errs int) string {
	glyph := "✅"
	if errs > 0 {
		glyph = "⚠️"
	}
	return fmt.Sprintf("%s *logvault capture* — files=%d copy=%s errors=%d%s",
		glyph, files, fmtBytesLV(bytes), errs, logvaultAnchorSuffix(vaultDir))
}

// logvaultVerifyDigest renders the one-line Slack digest for a verify run.
func logvaultVerifyDigest(vaultDir string, rows, checked int, problems []logvault.VerifyProblem) string {
	glyph := "✅"
	if len(problems) > 0 {
		glyph = "🔴"
	}
	line := fmt.Sprintf("%s *logvault verify* — chain rows=%d mirrors_checked=%d mismatches=%d%s",
		glyph, rows, checked, len(problems), logvaultAnchorSuffix(vaultDir))
	for i, p := range problems {
		if i >= 5 {
			line += fmt.Sprintf("\n… +%d more", len(problems)-5)
			break
		}
		line += fmt.Sprintf("\n  PROBLEM %s/%s: %s", p.Source, p.RelPath, p.Reason)
	}
	return line
}

// logvaultNotifySlack enqueues text durably through the shared slack outbox
// (never blocks on the network — Enqueue only appends to the local spool)
// and then attempts one same-tick drain pass so a healthy channel sees the
// digest immediately; a failed or skipped drain leaves the row queued for the
// next `fak slack outbox drain` (scheduled or manual), so this call can never
// lose the digest, only delay it.
func logvaultNotifySlack(w, ew io.Writer, channel, text string) {
	ch := channel
	if ch == "" {
		if r := slackenv.Lookup("FAK_DISPATCH_CHANNEL"); r.Set() {
			ch = r.Value
		} else if r := slackenv.Lookup("FAK_SCOREBOARD_CHANNEL"); r.Set() {
			ch = r.Value
		}
	}
	if ch == "" {
		fmt.Fprintln(w, "  slack: skipped — no channel resolved (set -slack-channel, FAK_DISPATCH_CHANNEL, or FAK_SCOREBOARD_CHANNEL)")
		return
	}
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(ew, "  slack: outbox open failed: %v\n", err)
		return
	}
	nonce, err := ob.Enqueue(slackoutbox.Row{Channel: ch, Text: text, Source: "logvault"})
	if err != nil {
		fmt.Fprintf(ew, "  slack: enqueue failed: %v\n", err)
		return
	}
	fmt.Fprintf(w, "  slack: enqueued %s to %s\n", nonce, ch)
	wire, err := outboxWire("", "")
	if err != nil {
		fmt.Fprintf(w, "  slack: drain deferred — %v (row stays durably queued)\n", err)
		return
	}
	rep, err := ob.Drain(ctx(), wire, slackoutbox.DrainOpts{Root: "."})
	switch {
	case err == slackoutbox.ErrDrainBusy:
		fmt.Fprintln(w, "  slack: another drainer holds the lock — row stays queued")
	case err != nil:
		fmt.Fprintf(w, "  slack: drain attempt failed: %v (row stays durably queued for retry)\n", err)
	default:
		fmt.Fprintf(w, "  slack: drained — posted %d updated %d remaining %d\n", rep.Posted, rep.Updated, rep.Remaining)
	}
}
