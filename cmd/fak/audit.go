package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/journal"
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
	case "diagnose":
		cmdAuditDiagnose(args[1:])
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
	fmt.Fprintln(os.Stderr, "       fak audit diagnose [<journal.jsonl>] (tell concurrent-writer interleave apart from real tampering)")
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
	path := auditJournalPathArg("audit verify", "usage: fak audit verify <journal.jsonl>", args)
	if isUsageLog(path) {
		n, err := usagelog.Verify(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak audit verify: %s — TAMPERED/BROKEN after %d sound row(s): %v\n", path, n, err)
			os.Exit(1)
		}
		fmt.Printf("fak audit verify: %s — OK: %d hash-chained usage row(s), chain intact (no edit since written)\n", path, n)
		return
	}
	n, err := journal.Verify(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak audit verify: %s — TAMPERED/BROKEN after %d sound row(s): %v\n", path, n, err)
		os.Exit(1)
	}
	fmt.Printf("fak audit verify: %s — OK: %d hash-chained row(s), chain intact (no edit since written)\n", path, n)
}

// isUsageLog peeks the first well-formed line of path to tell a usage journal
// (internal/usagelog, schema "fak-usage-log/1") apart from a decision journal
// (internal/journal, no schema field) so 'fak audit verify' dispatches to the
// matching Verify without a separate --kind flag. A file that can't be opened
// or whose first line doesn't parse falls through to the decision-journal
// path, which reports the real error.
func isUsageLog(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var probe struct {
			Schema string `json:"schema"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			return false
		}
		return probe.Schema == usagelog.SchemaV1
	}
	return false
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
