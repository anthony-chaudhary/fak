package main

// session_checkpoint_witness.go — `fak session checkpoint-witness`, the operator verb
// that mints and re-checks #2425's two-axis checkpoint: the pair {ledger_head_hash,
// tree_witness} bound in ONE append-only ledger record.
//
//	fak session checkpoint-witness <trace> [--repo DIR] [--ledger-dir DIR] [--untracked] [--json]
//	fak session checkpoint-witness <trace> --verify [...]
//
// It is an OFFLINE verb (like log / branch / checkpoint): it never dials a live gateway.
// The two halves it binds:
//
//   - the TRANSCRIPT axis — the head hash of the append-only session ledger
//     (internal/sessionledger), read at append time so the record's parent IS the head;
//   - the TREE axis — a git witness over the workspace: HEAD SHA plus a digest of the
//     dirty working set, the same evidence class internal/sessionobs/ingest.go's
//     commitMarker already trusts (a SHA git printed, never a timestamp).
//
// WHY THIS IS NOT THE `checkpoint` VERB. `fak session checkpoint` (#2760) writes a
// durable session IMAGE — a copy. This one is deliberately NOT a copy: a checkpoint here
// is three hashes and a path count, which is what makes it cheap enough to mint
// automatically. Two different artifacts, two verbs; the shared word is the concept, not
// the payload.
//
// WHY THE GIT READ LIVES HERE. internal/sessionledger stays pure and stdlib-only — it
// never shells out — so the subprocess boundary is confined to this CLI shell. That also
// makes the binding hermetically testable without a repo, and makes THIS file the only
// place a git flag has to be right.

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

// maxWitnessFileBytes caps how much of one dirty file is hashed into the tree witness. A
// checkpoint must stay cheap; a multi-gigabyte artifact in the dirty set would otherwise
// turn an "automatic, cheap" mint into a disk crawl. Over the cap we digest the file's
// size instead of its bytes and say so in the status, so the witness still MOVES when the
// file does — it just stops claiming to have read all of it.
const maxWitnessFileBytes = 8 << 20

// checkpointWitnessReport is the --json shape: the receipt plus the verify verdict when
// one was asked for. Axis is empty on a pass, and names the failing half on a failure, so
// a caller can branch without parsing prose.
type checkpointWitnessReport struct {
	Trace      string                       `json:"trace"`
	Checkpoint sessionledger.Checkpoint     `json:"checkpoint"`
	Verified   *bool                        `json:"verified,omitempty"`
	Axis       sessionledger.CheckpointAxis `json:"axis,omitempty"`
	Detail     string                       `json:"detail,omitempty"`
}

// runSessionCheckpointWitness is the testable core: exit code 0 ok, 1 a runtime error or
// a failed verification, 2 a usage error.
func runSessionCheckpointWitness(stdout, stderr io.Writer, argv []string) int {
	// The trace is the one leading positional and comes BEFORE any flag: Go's flag
	// package stops parsing at the first non-flag token, so it has to be split off ahead
	// of Parse — the same positionals-first discipline `fak session checkpoint` uses. A
	// leading flag (e.g. --help) still parses, and then fails the empty-trace check.
	trace, flagArgs := "", argv
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		trace, flagArgs = strings.TrimSpace(argv[0]), argv[1:]
	}

	fs := flag.NewFlagSet("session checkpoint-witness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "session")
	repo := fs.String("repo", ".", "workspace whose git tree is witnessed")
	ledgerDir := fs.String("ledger-dir", "", "session-ledger directory (default: the operator's ledger)")
	untracked := fs.String("untracked", "no", "include untracked paths in the dirty set: no|normal|all")
	verify := fs.Bool("verify", false, "re-check the trace's latest checkpoint against the ledger and the tree now")
	asJSON := fs.Bool("json", false, "emit the checkpoint as JSON")
	if rc, ok := parseFlagsOrHelp(fs, flagArgs); !ok {
		return rc
	}
	if trace == "" || fs.NArg() > 0 {
		fmt.Fprintln(stderr, "usage: fak session checkpoint-witness <trace> [--repo DIR] [--ledger-dir DIR] [--untracked no|normal|all] [--verify] [--json]")
		return 2
	}

	root := pathutil.ExpandTilde(strings.TrimSpace(*repo))
	tree, err := liveTreeWitness(root, *untracked)
	if err != nil {
		fmt.Fprintf(stderr, "fak session checkpoint-witness: %v\n", err)
		return 1
	}
	ledger, err := openCheckpointLedger(*ledgerDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak session checkpoint-witness: %v\n", err)
		return 1
	}

	if *verify {
		return verifyCheckpointWitness(stdout, stderr, ledger, trace, tree, *asJSON)
	}
	cp, err := ledger.Checkpoint(trace, tree)
	if err != nil {
		fmt.Fprintf(stderr, "fak session checkpoint-witness: %v\n", err)
		return 1
	}
	if *asJSON {
		return emitSessionJSON(stdout, stderr, checkpointWitnessReport{Trace: trace, Checkpoint: cp})
	}
	renderCheckpointWitness(stdout, cp)
	return 0
}

// verifyCheckpointWitness re-checks the trace's latest checkpoint. A failure exits 1 and
// NAMES the axis: "tree" means the workspace drifted from the conversation that described
// it; "transcript" means the ledger no longer matches the record — different problems,
// different responses.
func verifyCheckpointWitness(stdout, stderr io.Writer, ledger *sessionledger.Ledger, trace string, tree sessionledger.TreeWitness, asJSON bool) int {
	cp, err := ledger.LatestCheckpoint(trace)
	if err != nil {
		fmt.Fprintf(stderr, "fak session checkpoint-witness: %v\n", err)
		return 1
	}
	rep := checkpointWitnessReport{Trace: trace, Checkpoint: cp}
	err = ledger.VerifyCheckpoint(cp, tree)
	ok := err == nil
	rep.Verified = &ok
	if !ok {
		var mm *sessionledger.CheckpointMismatch
		if errors.As(err, &mm) {
			rep.Axis, rep.Detail = mm.Axis, mm.Detail
		} else {
			rep.Detail = err.Error()
		}
	}
	if asJSON {
		rc := emitSessionJSON(stdout, stderr, rep)
		if rc == 0 && !ok {
			return 1
		}
		return rc
	}
	if !ok {
		fmt.Fprintf(stderr, "checkpoint %s FAILED on the %s axis: %s\n", cp.Hash, rep.Axis, rep.Detail)
		return 1
	}
	fmt.Fprintf(stdout, "checkpoint %s verified (both axes hold)\n", cp.Hash)
	renderCheckpointWitness(stdout, cp)
	return 0
}

// renderCheckpointWitness prints BOTH hashes — the acceptance the ticket names — plus the
// record id that binds them.
func renderCheckpointWitness(stdout io.Writer, cp sessionledger.Checkpoint) {
	fmt.Fprintf(stdout, "checkpoint  %s (trace %s)\n", cp.Hash, cp.Trace)
	fmt.Fprintf(stdout, "  transcript: ledger head %s\n", checkpointHashOrRoot(cp.LedgerHead))
	fmt.Fprintf(stdout, "  tree:       %s + dirty %s (%d paths)\n",
		cp.Tree.HeadSHA, cp.Tree.DirtySHA256, cp.Tree.DirtyCount)
}

// checkpointHashOrRoot renders the empty parent of a trace's first record readably rather
// than as a blank line.
func checkpointHashOrRoot(h sessionledger.Hash) string {
	if h == "" {
		return "(root — first record on this trace)"
	}
	return string(h)
}

// openCheckpointLedger opens the named ledger directory, or the operator's default when
// none is named. The explicit directory is what lets a test (and a fixture repo) mint
// checkpoints without touching the live ledger.
func openCheckpointLedger(dir string) (*sessionledger.Ledger, error) {
	if d := strings.TrimSpace(dir); d != "" {
		return sessionledger.Open(pathutil.ExpandTilde(d))
	}
	return sessionledger.OpenDefault()
}

// liveTreeWitness reads the workspace's git state and folds it into a TreeWitness: the
// committed anchor from `git rev-parse HEAD`, and the dirty set from `git status
// --porcelain -z` (NUL-framed, so a path with a space or a quote cannot be mis-parsed the
// way the quoting non-`-z` form invites).
//
// untracked selects git's --untracked-files mode. The default is "no": in a shared trunk
// the untracked set is dominated by other sessions' in-flight files and build artifacts,
// so including it by default would make a checkpoint fail its own tree axis for reasons
// that have nothing to do with the session that minted it. An operator who wants the
// wider claim asks for it.
func liveTreeWitness(root, untracked string) (sessionledger.TreeWitness, error) {
	switch untracked {
	case "no", "normal", "all":
	default:
		return sessionledger.TreeWitness{}, fmt.Errorf("--untracked must be no|normal|all, got %q", untracked)
	}
	head, err := gitOut(root, "rev-parse", "HEAD")
	if err != nil {
		return sessionledger.TreeWitness{}, fmt.Errorf("read HEAD in %s: %w", root, err)
	}
	status, err := gitOut(root, "status", "--porcelain", "-z", "--untracked-files="+untracked)
	if err != nil {
		return sessionledger.TreeWitness{}, fmt.Errorf("read status in %s: %w", root, err)
	}
	dirty, err := parseDirtySet(root, status)
	if err != nil {
		return sessionledger.TreeWitness{}, err
	}
	return sessionledger.NewTreeWitness(strings.TrimSpace(head), dirty)
}

// parseDirtySet decodes NUL-framed porcelain v1 records into digested dirty entries. Each
// record is "XY <path>"; a rename/copy (R/C in either column) is followed by an EXTRA
// NUL-terminated field carrying the ORIGIN path, which is folded into the entry so a
// rename cannot read as a no-op.
func parseDirtySet(root, status string) ([]sessionledger.DirtyEntry, error) {
	fields := strings.Split(status, "\x00")
	var out []sessionledger.DirtyEntry
	for i := 0; i < len(fields); i++ {
		rec := fields[i]
		if len(rec) < 4 { // "XY " + at least one path byte
			continue
		}
		code, path := rec[:2], rec[3:]
		entry := sessionledger.DirtyEntry{Path: path, Status: code}
		if strings.ContainsAny(code, "RC") {
			if i+1 < len(fields) {
				i++
				entry.Origin = fields[i]
			}
		}
		sum, status, err := digestWorktreeFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			return nil, err
		}
		entry.SHA256 = sum
		if status != "" {
			entry.Status = code + " " + status
		}
		out = append(out, entry)
	}
	return out, nil
}

// digestWorktreeFile hashes one dirty path's bytes as they are on disk. It returns an
// empty digest plus a state note for the cases where there are no bytes to read — a
// deletion, a directory (git collapses a wholly-untracked directory into one record), or
// a path that is not a regular file. Those notes are inside the witness digest, so a
// deletion is still a CHANGE the tree axis sees.
func digestWorktreeFile(path string) (sum string, note string, err error) {
	info, err := os.Lstat(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		return "", "absent", nil
	case err != nil:
		return "", "", fmt.Errorf("stat %s: %w", path, err)
	case info.IsDir():
		return "", "dir", nil
	case !info.Mode().IsRegular():
		return "", "irregular", nil
	case info.Size() > maxWitnessFileBytes:
		// Too big to be worth reading on every mint: witness its size instead. The
		// note records that we did so, so the claim is never overstated.
		return fmt.Sprintf("size:%d", info.Size()), "oversize", nil
	}
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", "absent", nil // raced with a delete: still an honest note
		}
		return "", "", fmt.Errorf("read %s: %w", path, err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, io.LimitReader(f, maxWitnessFileBytes)); err != nil {
		return "", "", fmt.Errorf("hash %s: %w", path, err)
	}
	return hex.EncodeToString(h.Sum(nil)), "", nil
}
