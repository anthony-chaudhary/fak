package main

// session_branch.go — `fak session branch`, the operator verb that FORKS a session from
// its checkpoint into a new durable id (issue #1200, part of Pillar 2 / #1193). It is the
// net-new lifecycle move beyond #748's restore-in-place: a restore brings a session back
// as itself; a branch mints a SECOND session from the same checkpoint so an operator can
// "try a risky path without losing my place" while the parent keeps running.
//
//	fak session branch <parent-image-dir> --out <branch-dir> [--id <branch-id>]
//	    [--reason R] [--to-model M] [--to-host H] [--registry PATH] [--json]
//
// It is an OFFLINE verb (like `fak session reset-diff`): it never dials a live gateway. It
// composes internal/sessionimage.BranchDir — which shares the parent's content-addressed
// recall pages copy-on-write (no deep copy of unchanged pages) and records the parent_id +
// "branched from <parent> at <sha>" lineage in the branch's image.json migration log — and,
// when --registry is given, writes a NEW durable descriptor (new id, parent_id link) into
// the C1 session registry. The parent bundle is only read, never written, so the parent
// session's state / lease / descriptor are unaffected.

import (
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionimage"
)

// runSessionBranch is the testable core of `fak session branch`: it returns the process
// exit code (0 ok, 1 a runtime error, 2 a usage error) and takes its streams explicitly.
func runSessionBranch(stdout, stderr io.Writer, argv []string) int {
	// The parent checkpoint is the one leading positional; it comes before any flags so
	// `fak session branch <dir> --out <dir>` parses cleanly (Go's flag package stops at the
	// first non-flag token, so the id must be split off before Parse — the same positionals-
	// first discipline the gateway verbs use).
	if len(argv) == 0 || strings.HasPrefix(argv[0], "-") {
		fmt.Fprintln(stderr, "fak session branch: want <parent-image-dir> first, then flags")
		return 2
	}
	parentArg := argv[0]
	flagArgs := argv[1:]

	fs := flag.NewFlagSet("session branch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "session")
	out := fs.String("out", "", "destination directory for the new branch image (required)")
	id := fs.String("id", "", "the branch's durable session id (default: <parent-id>-branch)")
	reason := fs.String("reason", "", "operator note folded into the branch's migration-log entry")
	toModel := fs.String("to-model", "", "re-home the branch to a different model (default: inherit the parent's)")
	toHost := fs.String("to-host", "", "re-home the branch to a different host (default: inherit the parent's)")
	registry := fs.String("registry", "", "also register the branch as a new C1 descriptor in this registry file (with a parent_id link)")
	asJSON := fs.Bool("json", false, "emit the branch Meta as JSON instead of the human summary")
	if rc, ok := parseFlagsOrHelp(fs, flagArgs); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session branch: unexpected argument %q (the parent dir comes first, then flags)\n", fs.Arg(0))
		return 2
	}
	parentDir := pathutil.ExpandTilde(parentArg)
	branchDir, ok := requiredSessionChildDir(stderr, "branch", *out)
	if !ok {
		return 2
	}

	// The parent's checkpoint must be a whole bundle dir; loading it (BranchDir does this)
	// fails closed on a truncated/tampered image. Derive the branch id when not pinned.
	parent, ok := loadSessionParent(stderr, "branch", parentDir)
	if !ok {
		return 1
	}
	target := newSessionChildTarget(*id, parent.Meta.SessionID, "branch", *toModel, *toHost, *reason)

	meta, code := runSessionBranchOperation(stderr, "branch", *registry, func() (sessionimage.Meta, sessionimage.Meta, error) {
		meta, err := sessionimage.BranchDir(parentDir, branchDir, sessionimage.BranchOptions{
			BranchID: target.id, ToModel: target.model, ToHost: target.host, Reason: target.reason,
		})
		return meta, meta, err
	})
	if code != 0 {
		return code
	}

	if *asJSON {
		return emitSessionJSON(stdout, stderr, meta)
	}
	fmt.Fprintf(stdout, "branched %s -> %s (parent_id=%s)\n", meta.ParentID, meta.SessionID, meta.ParentID)
	if len(meta.Migrations) > 0 {
		fmt.Fprintf(stdout, "  lineage: %s\n", meta.Migrations[len(meta.Migrations)-1].Reason)
	}
	fmt.Fprintf(stdout, "  image:   %s (%d shared parts, copy-on-write)\n", branchDir, len(meta.Parts))
	if strings.TrimSpace(*registry) != "" {
		fmt.Fprintf(stdout, "  registry: new descriptor %s registered with parent_id=%s\n", meta.SessionID, meta.ParentID)
	}
	return 0
}

func loadSessionParent(stderr io.Writer, command, parentDir string) (*sessionimage.Image, bool) {
	parent, err := sessionimage.LoadDir(parentDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak session %s: load parent %q: %v\n", command, parentDir, err)
		return nil, false
	}
	return parent, true
}

func requiredSessionChildDir(stderr io.Writer, command, value string) (string, bool) {
	if strings.TrimSpace(value) == "" {
		fmt.Fprintf(stderr, "fak session %s: --out <%s-dir> is required\n", command, command)
		return "", false
	}
	return pathutil.ExpandTilde(value), true
}

// registerBranchDescriptor writes a NEW durable descriptor for the branch into the C1
// registry file, carrying the parent_id link — the registry half of "a branch creates a new
// durable descriptor (new id) with a parent_id link". The branch resumes at the parent's
// drive state (inherited into the branch image's session.json), registered under the branch
// id so the two sessions are distinct rows. Returns 0 ok, 1 on a registry error.
func registerBranchDescriptor(stderr io.Writer, path string, meta sessionimage.Meta) int {
	reg := session.NewRegistry(session.NewFileStore(path))
	st := session.State{TraceID: meta.SessionID, Run: session.Running}
	if _, err := reg.RegisterWithMeta(meta.SessionID, meta.Host, st, session.DefaultDescriptorTTL, time.Now(),
		session.DescriptorMeta{ParentID: meta.ParentID}); err != nil {
		fmt.Fprintf(stderr, "fak session branch: register branch descriptor: %v\n", err)
		return 1
	}
	return 0
}

func sessionChildID(explicit, parent, kind string) string {
	if id := strings.TrimSpace(explicit); id != "" {
		return id
	}
	return parent + "-" + kind
}

type sessionChildTarget struct{ id, model, host, reason string }

func newSessionChildTarget(explicit, parent, kind, model, host, reason string) sessionChildTarget {
	return sessionChildTarget{id: sessionChildID(explicit, parent, kind), model: model, host: host, reason: reason}
}

func runSessionBranchOperation[T any](stderr io.Writer, verb, registry string, operation func() (T, sessionimage.Meta, error)) (T, int) {
	result, meta, err := operation()
	if err != nil {
		fmt.Fprintf(stderr, "fak session %s: %v\n", verb, err)
		return result, 1
	}
	if reg := strings.TrimSpace(registry); reg != "" {
		if code := registerBranchDescriptor(stderr, pathutil.ExpandTilde(reg), meta); code != 0 {
			return result, code
		}
	}
	return result, 0
}
