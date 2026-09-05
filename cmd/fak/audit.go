package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// cmdAudit handles `fak audit <subcommand>` over the durable DECISION JOURNAL —
// the tamper-evident, hash-chained ledger `fak guard` (and FAK_AUDIT_JOURNAL)
// write a verdict to per kernel decision. It is the consumer end of the audit
// trail: a self-report is not a witness, so this is how an operator (or an
// auditor who never trusted the running process) re-verifies the record offline.
//
//	verify PATH — re-read the file and validate the hash chain end to end; exit 1
//	              naming the FIRST broken link if a single byte changed since it
//	              was written. Also covers a usagelog journal (internal/usagelog,
//	              the `fak usage` CLI-invocation trail) — verify auto-detects it
//	              by its schema field and dispatches to usagelog.Verify.
//	export PATH — re-emit the journal as JSONL on stdout (a sound copy of a sound
//	              journal), for archival or piping to another tool.
//	diagnose PATH — reconstruct the per-session chains from the hash links and tell a
//	              benign concurrent-writer INTERLEAVE apart from real TAMPERING, so a
//	              shared default journal is not mis-reported as broken (see audit_diagnose.go).
//	replay PATH — re-drive the recorded call sequence against the floor's own verdicts and
//	              assert determinism: every identical recorded call (same tool + args digest)
//	              must carry one verdict; a divergence is a structured non-determinism finding
//	              (the reproducible-trajectory witness of #2905; see audit_replay.go).
//	rsl PATH    — replay a git Reference State Log (append-only, hash-chained record of observed
//	              trunk ref transitions) and exit 1 on a tampered chain OR a non-fast-forward gap,
//	              naming the offending ref — the forge-independent no-force-push proof (#3190;
//	              borrowed from gittuf's RSL; see audit_rsl.go / internal/rsl).
func cmdAudit(args []string) {
	if len(args) == 0 {
		auditUsage()
		os.Exit(2)
	}
	switch args[0] {
	case "verify":
		cmdAuditVerify(args[1:])
	case "export":
		cmdAuditExport(args[1:])
	case "dataset":
		cmdAuditDataset(args[1:])
	case "diagnose":
		cmdAuditDiagnose(args[1:])
	case "replay":
		cmdAuditReplay(args[1:])
	case "rsl":
		cmdAuditRSL(args[1:])
	case "usage":
		cmdAuditUsage(args[1:])
	case "-h", "--help", "help":
		auditUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak audit: unknown subcommand %q\n", args[0])
		auditUsage()
		os.Exit(2)
	}
}

func auditUsage() {
	fmt.Fprintln(os.Stderr, "usage: fak audit verify <journal.jsonl>   (validate the tamper-evident hash chain; exit 1 if edited)")
	fmt.Fprintln(os.Stderr, "       fak audit export <journal.jsonl>   (re-emit the journal as JSONL on stdout)")
	fmt.Fprintln(os.Stderr, "       fak audit dataset <journal.jsonl>  (emit fak-decision-outcome/1 JSONL, one row per adjudicated call)")
	fmt.Fprintln(os.Stderr, "       fak audit diagnose [<journal.jsonl>] (tell concurrent-writer interleave apart from real tampering)")
	fmt.Fprintln(os.Stderr, "       fak audit replay [--json] <journal.jsonl> (re-drive recorded decisions; assert every identical call replayed to one verdict)")
	fmt.Fprintln(os.Stderr, "       fak audit rsl <rsl.jsonl>          (replay a git Reference State Log; exit 1 on a tampered chain or non-fast-forward gap)")
	fmt.Fprintln(os.Stderr, "       fak audit usage [--since DUR] [--json] [--root DIR ...] (cross-session usage rollup over every durable journal/ledger)")
}

// cmdAuditVerify re-reads a decision journal and validates its hash chain. A clean
// chain prints the row count and exits 0; ANY edit since it was written (a flipped
// byte, a dropped row, a resequence) breaks the link and exits 1, naming the first
// broken row — the property that lets the journal stand in for trust in the process.
// auditJournalPathArg parses the single <journal.jsonl> positional shared by the audit
// subcommands, exiting 2 on misuse. name + usage tailor the flag set and the usage line.
func auditJournalPathArg(name, usage string, args []string) string {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	fs.Usage = func() { fmt.Fprintln(os.Stderr, usage) }
	_ = fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	return fs.Arg(0)
}

func cmdAuditVerify(args []string) {
	os.Exit(runCmdAuditVerify(os.Stdout, os.Stderr, args))
}

func runCmdAuditVerify(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("audit verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprintln(stderr, "usage: fak audit verify <journal.jsonl>") }
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return 2
	}
	return runAuditVerify(stdout, stderr, fs.Arg(0))
}

func runAuditVerify(stdout, stderr io.Writer, path string) int {
	switch auditVerifySchema(path) {
	case usagelog.SchemaV1:
		n, err := usagelog.Verify(path)
		return renderAuditVerifyVerdict(stdout, stderr, path, "usage ", n, err)
	case modelroute.AuditReceiptLedgerRowSchema:
		v, err := modelroute.VerifyAuditReceiptLedger(path)
		if err != nil {
			n := auditReceiptSoundRows(err)
			fmt.Fprintf(stderr, "fak audit verify: %s — TAMPERED/BROKEN after %d sound row(s): %v\n", path, n, err)
			return 1
		}
		fmt.Fprintf(stdout, "fak audit verify: %s — OK: %d hash-chained cross-audit receipt row(s), unique_audits=%d verdicts=%v head_hash=%s\n",
			path, v.Rows, v.UniqueAudits, v.VerdictCounts, v.HeadHash)
		return 0
	default:
		n, err := journal.Verify(path)
		return renderAuditVerifyVerdict(stdout, stderr, path, "", n, err)
	}
}

func auditReceiptSoundRows(err error) int {
	var integrity *modelroute.AuditReceiptLedgerIntegrityError
	if errors.As(err, &integrity) {
		return integrity.Integrity.Recovered
	}
	return 0
}

// renderAuditVerifyVerdict writes the one-line verdict for a chain verification and
// returns the process exit code: 1 (with the tampered/broken text on stderr) when the
// chain failed to verify, 0 (with the sound-row count on stdout) when it held.
func renderAuditVerifyVerdict(stdout, stderr io.Writer, path, rowKind string, n int, err error) int {
	if err != nil {
		fmt.Fprintf(stderr, "fak audit verify: %s — TAMPERED/BROKEN after %d sound row(s): %v\n", path, n, err)
		return 1
	}
	fmt.Fprintf(stdout, "fak audit verify: %s — OK: %d hash-chained %srow(s), chain intact (no edit since written)\n", path, n, rowKind)
	return 0
}

// auditVerifySchema peeks the first well-formed row's schema so audit verify
// dispatches each chain dialect to its own verifier. Schema-less and unreadable
// inputs return "" and retain the decision-journal path's existing error text.
func auditVerifySchema(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	probe, found, err := jsonlledger.First[struct {
		Schema string `json:"schema"`
	}](f)
	if err != nil || !found {
		return ""
	}
	return probe.Schema
}

// cmdAuditExport re-emits a journal as JSONL on stdout. It opens the file-backed
// journal (append mode, recovering the chain head) and streams its durable history
// re-read from disk, so an export of a sound journal is itself a sound journal.
func cmdAuditExport(args []string) {
	path := auditJournalPathArg("audit export", "usage: fak audit export <journal.jsonl>", args)
	j, err := journal.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak audit export: %v\n", err)
		os.Exit(1)
	}
	defer j.Close()
	if _, err := j.ExportTo(os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "fak audit export: %v\n", err)
		os.Exit(1)
	}
}
