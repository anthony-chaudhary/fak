package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// cmdAudit handles `fak audit <subcommand>` over the durable DECISION JOURNAL —
// the tamper-evident, hash-chained ledger `fak guard` (and FAK_AUDIT_JOURNAL)
// write a verdict to per kernel decision. It is the consumer end of the audit
// trail: a self-report is not a witness, so this is how an operator (or an
// auditor who never trusted the running process) re-verifies the record offline.
//
//	verify PATH — re-read the file and validate the hash chain end to end; exit 1
//	              naming the FIRST broken link if a single byte changed since it
//	              was written.
//	export PATH — re-emit the journal as JSONL on stdout (a sound copy of a sound
//	              journal), for archival or piping to another tool.
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
}

// cmdAuditVerify re-reads a decision journal and validates its hash chain. A clean
// chain prints the row count and exits 0; ANY edit since it was written (a flipped
// byte, a dropped row, a resequence) breaks the link and exits 1, naming the first
// broken row — the property that lets the journal stand in for trust in the process.
func cmdAuditVerify(args []string) {
	fs := flag.NewFlagSet("audit verify", flag.ExitOnError)
	fs.Usage = func() { fmt.Fprintln(os.Stderr, "usage: fak audit verify <journal.jsonl>") }
	_ = fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	path := fs.Arg(0)
	n, err := journal.Verify(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak audit verify: %s — TAMPERED/BROKEN after %d sound row(s): %v\n", path, n, err)
		os.Exit(1)
	}
	fmt.Printf("fak audit verify: %s — OK: %d hash-chained row(s), chain intact (no edit since written)\n", path, n)
}

// cmdAuditExport re-emits a journal as JSONL on stdout. It opens the file-backed
// journal (append mode, recovering the chain head) and streams its durable history
// re-read from disk, so an export of a sound journal is itself a sound journal.
func cmdAuditExport(args []string) {
	fs := flag.NewFlagSet("audit export", flag.ExitOnError)
	fs.Usage = func() { fmt.Fprintln(os.Stderr, "usage: fak audit export <journal.jsonl>") }
	_ = fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	path := fs.Arg(0)
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
