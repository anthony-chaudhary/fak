package main

// session_teleport.go — `fak session export` / `fak session import`, and the
// ledger-trace arm of `fak session fork` (issue #2419, part of #2392). Generation:
// gen/next.
//
// These are the operator front end to the teleport verbs the gateway serves on
// /v1/fak/session/{id}/{verb} (internal/gateway/session_teleport.go): a served
// session moves between hosts as a verifiable hash closure rather than a file copy.
//
//	fak session export <trace> [--out FILE] [--turns N] [--tokens N] ...
//	fak session import [--in FILE]
//	fak session fork   <trace> [--to ID] [--json]
//
// They are OFFLINE verbs, like `fak session branch` / `checkpoint`: they read and
// write the DURABLE ledger directory on this host (--ledger-dir, default
// $FAK_SESSION_LEDGER_DIR or the user config dir) and never dial a live gateway.
// That is deliberate — a host whose gateway has already stopped is exactly the host
// an operator needs to export from.
//
// NAME COLLISION, read this before editing. `fak session fork` was already taken by
// #2761, which forks a session IMAGE DIRECTORY (internal/sessionimage.ForkDir,
// session_fork.go). That verb REQUIRES --out and --checkpoint, so the two arms are
// told apart by shape and never by guesswork: with --out/--checkpoint it is the
// image fork, and with a bare trace and neither flag it is this one. The bare form
// was a usage error (exit 2) before this file existed, so no invocation that worked
// yesterday changes meaning today.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

// maxTeleportBundleBytes bounds a bundle the CLI will read off disk or stdin. The
// ledger's own entry and node ceilings bound what an honest export can produce; this
// is the reading side's independent bound so a corrupt or hostile file cannot be
// streamed unbounded into the operator's process.
const maxTeleportBundleBytes = 256 << 20

// teleportIsLedgerFork reports whether a `fak session fork` invocation is the
// ledger-trace arm (#2419) rather than the image-directory arm (#2761). The image
// arm requires --out and --checkpoint; naming neither, with a leading positional, is
// the trace arm. Both spellings of a Go flag (--flag and -flag, bare or =value) are
// recognised so the routing cannot be side-stepped by how the operator typed it.
func teleportIsLedgerFork(argv []string) bool {
	if len(argv) == 0 || strings.HasPrefix(argv[0], "-") {
		return false
	}
	for _, a := range argv[1:] {
		name := strings.TrimLeft(a, "-")
		if i := strings.IndexByte(name, '='); i >= 0 {
			name = name[:i]
		}
		if name == "out" || name == "checkpoint" {
			return false
		}
	}
	return true
}

// teleportLedgerFlag registers --ledger-dir on a flagset. Empty resolves to the
// package default at open time, so the flag's printed default stays stable across
// hosts instead of baking one machine's config path into the help text.
func teleportLedgerFlag(fs *flag.FlagSet) *string {
	return fs.String("ledger-dir", "", "durable session ledger directory (default $FAK_SESSION_LEDGER_DIR, else the user config dir)")
}

func openTeleportLedger(stderr io.Writer, label, dir string) (*sessionledger.Ledger, int) {
	if strings.TrimSpace(dir) == "" {
		dir = sessionledger.DefaultDir()
	} else {
		dir = pathutil.ExpandTilde(dir)
	}
	l, err := sessionledger.Open(dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak session %s: open ledger %q: %v\n", label, dir, err)
		return nil, 1
	}
	return l, 0
}

// runTeleportExport writes one trace's portable closure — the ledger head, the
// entries that reach it, and the re-arm state — to a file or stdout.
func runTeleportExport(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || strings.HasPrefix(argv[0], "-") {
		fmt.Fprintln(stderr, "fak session export: want <trace> first, then flags")
		return 2
	}
	trace := argv[0]

	fs := flag.NewFlagSet("session export", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "session")
	ledgerDir := teleportLedgerFlag(fs)
	out := fs.String("out", "", "write the bundle here (default: stdout)")
	turns := fs.Int("turns", 0, "re-arm: remaining turns (-1 = unbounded)")
	tokens := fs.Int("tokens", 0, "re-arm: remaining output tokens (-1 = unbounded)")
	contextTokens := fs.Int("context-tokens", 0, "re-arm: remaining prompt/context tokens (0 = off)")
	taint := fs.String("taint", "", "re-arm: the session's taint HIGH-WATER mark, carried so a hop cannot launder it clean")
	generation := fs.Int("generation", 0, "re-arm: how many times this lineage has been re-armed")
	if rc, ok := parseFlagsOrHelp(fs, argv[1:]); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session export: unexpected argument %q (the trace comes first, then flags)\n", fs.Arg(0))
		return 2
	}

	l, code := openTeleportLedger(stderr, "export", *ledgerDir)
	if code != 0 {
		return code
	}
	bundle, err := gateway.ExportTeleport(l, trace, gateway.TeleportArm{
		Budget: gateway.SessionBudget{
			TurnsLeft:         *turns,
			TokensLeft:        *tokens,
			ContextTokensLeft: *contextTokens,
		},
		TaintHighWater: strings.TrimSpace(*taint),
		Generation:     *generation,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak session export: %v\n", err)
		return 1
	}

	// Compact, never MarshalIndent. Indenting re-flows the entries' embedded content
	// bytes, and those exact bytes are what each entry's hash covers — a pretty-printed
	// bundle would verify on nobody's host, including this one.
	raw, err := json.Marshal(bundle)
	if err != nil {
		fmt.Fprintf(stderr, "fak session export: encode bundle: %v\n", err)
		return 1
	}
	raw = append(raw, '\n')
	if dest := strings.TrimSpace(*out); dest != "" {
		if err := os.WriteFile(pathutil.ExpandTilde(dest), raw, 0600); err != nil {
			fmt.Fprintf(stderr, "fak session export: write %q: %v\n", dest, err)
			return 1
		}
		fmt.Fprintf(stdout, "exported %s -> %s\n", bundle.TraceID, dest)
		fmt.Fprintf(stdout, "  head:    %s\n", bundle.Head)
		fmt.Fprintf(stdout, "  entries: %d (closure %s)\n", len(bundle.Entries), bundle.Closure)
		return 0
	}
	if _, err := stdout.Write(raw); err != nil {
		fmt.Fprintf(stderr, "fak session export: write bundle: %v\n", err)
		return 1
	}
	return 0
}

// runTeleportImport re-arms a session on THIS host from a portable closure. The
// bundle is verified and then RE-DERIVED entry by entry, so a document that was
// altered in flight cannot reproduce the head it declares and is refused.
func runTeleportImport(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session import", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "session")
	ledgerDir := teleportLedgerFlag(fs)
	in := fs.String("in", "", "read the bundle from here (default: stdin)")
	asJSON := fs.Bool("json", false, "emit the re-derived bundle as JSON instead of the human summary")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session import: unexpected argument %q (the bundle arrives via --in or stdin)\n", fs.Arg(0))
		return 2
	}

	src := stdin
	if path := strings.TrimSpace(*in); path != "" {
		f, err := os.Open(pathutil.ExpandTilde(path))
		if err != nil {
			fmt.Fprintf(stderr, "fak session import: open %q: %v\n", path, err)
			return 1
		}
		defer f.Close()
		src = f
	}
	raw, err := io.ReadAll(io.LimitReader(src, maxTeleportBundleBytes))
	if err != nil {
		fmt.Fprintf(stderr, "fak session import: read bundle: %v\n", err)
		return 1
	}
	var bundle gateway.TeleportBundle
	if err := json.Unmarshal(raw, &bundle); err != nil {
		fmt.Fprintf(stderr, "fak session import: decode bundle: %v\n", err)
		return 1
	}

	l, code := openTeleportLedger(stderr, "import", *ledgerDir)
	if code != 0 {
		return code
	}
	got, err := gateway.ImportTeleport(l, bundle)
	if err != nil {
		// A refused import wrote nothing: the verify runs before the ledger is touched.
		fmt.Fprintf(stderr, "fak session import: refused: %v\n", err)
		return 1
	}
	if *asJSON {
		return emitSessionJSON(stdout, stderr, got)
	}
	fmt.Fprintf(stdout, "imported %s\n", got.TraceID)
	fmt.Fprintf(stdout, "  head:    %s\n", got.Head)
	fmt.Fprintf(stdout, "  entries: %d re-derived (closure %s)\n", len(got.Entries), got.Closure)
	fmt.Fprintf(stdout, "  re-arm:  turns=%d tokens=%d context=%d taint=%s gen=%d\n",
		got.Arm.Budget.TurnsLeft, got.Arm.Budget.TokensLeft, got.Arm.Budget.ContextTokensLeft,
		orNoneTaint(got.Arm.TaintHighWater), got.Arm.Generation)
	return 0
}

// runTeleportFork mints a new trace whose head points at the source's current head
// and prints the new trace id and the shared-prefix hash — issue #2419's CLI
// witness. Nothing is copied; the two traces are two sessions sharing one immutable
// prefix.
func runTeleportFork(stdout, stderr io.Writer, argv []string) int {
	trace := argv[0]

	fs := flag.NewFlagSet("session fork", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "session")
	ledgerDir := teleportLedgerFlag(fs)
	to := fs.String("to", "", "the fork's trace id (default: minted from the source and the shared prefix)")
	asJSON := fs.Bool("json", false, "emit the fork as JSON instead of the human summary")
	if rc, ok := parseFlagsOrHelp(fs, argv[1:]); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session fork: unexpected argument %q (the trace comes first, then flags)\n", fs.Arg(0))
		return 2
	}

	l, code := openTeleportLedger(stderr, "fork", *ledgerDir)
	if code != 0 {
		return code
	}
	f, err := gateway.ForkTeleport(l, trace, strings.TrimSpace(*to))
	if err != nil {
		fmt.Fprintf(stderr, "fak session fork: %v\n", err)
		return 1
	}
	if *asJSON {
		return emitSessionJSON(stdout, stderr, f)
	}
	fmt.Fprintf(stdout, "forked %s -> %s\n", f.TraceID, f.ForkTraceID)
	fmt.Fprintf(stdout, "  shared prefix: %s\n", f.SharedPrefix)
	return 0
}

// orNoneTaint renders an unstamped taint high-water mark as an explicit "none" so a
// blank column is never misread as "clean".
func orNoneTaint(s string) string {
	if strings.TrimSpace(s) == "" {
		return "none"
	}
	return s
}
