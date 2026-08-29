package main

// leaseref_release.go — `fak leaseref release`, the CLI release twin of `acquire`
// (the named follow-on of docs/region-admission.md): hand a region back the moment
// the work is done instead of waiting out the TTL, so a finished exclusive-lane
// lease stops stalling the fleet. The policy lives in leaseref.ReleaseFenced
// (holder-checked, generation-aware, CAS delete); --force is the operator escape
// that maps to the unconditional Store.Release for a wedged record whose holder
// identity is unrecoverable.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func runLeaserefRelease(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref release", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	id := fs.String("id", "", "lease id to release")
	holder := fs.String("holder", "", "the holder identity that owns the lease")
	gen := fs.Int64("generation", 0, "the fencing token from acquire (0 = don't check the token)")
	force := fs.Bool("force", false, "operator override: delete the record without the holder check")
	announce := fs.String("announce", "", "public-safe lifecycle announcement: on, off, or offline")
	announceIssue := fs.Int("announce-issue", 0, "coordination issue number for --announce=on")
	announceRepo := fs.String("announce-repo", "", "owner/repo for --announce=on")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *id == "" {
		fmt.Fprintln(stderr, "fak leaseref release: --id is required")
		return 2
	}
	if !*force && *holder == "" {
		fmt.Fprintln(stderr, "fak leaseref release: --holder is required (or --force to delete a wedged record without the holder check)")
		return 2
	}
	store := leaseref.NewInDir(*dir)
	// Capture the public-safe projection inputs before deletion. Get failures are ignored
	// here because release remains authoritative and announcement is only advisory.
	announceRec := leaseref.Record{ID: *id, Holder: *holder, Generation: *gen}
	if current, ok, getErr := store.Get(context.Background(), *id); getErr == nil && ok {
		announceRec = current
	}
	if *force {
		// The unconditional single-ref delete — idempotent on an absent record. This is
		// the operator's manual reap for a record whose holder identity is lost; the
		// fenced path below is what agents and loops should run.
		if err := store.Release(context.Background(), *id); err != nil {
			fmt.Fprintf(stderr, "fak leaseref release: %v\n", err)
			return 1
		}
		ambientLeaserefAnnounce(stderr, *dir, leaseref.AnnounceRelease, announceRec, resolveAmbientLeaserefConfig(*announce, *announceIssue, *announceRepo))
		return emitLeaserefJSON(stdout, stderr, leaseref.FenceVerdict{
			OK:     true,
			Detail: "lease " + *id + " force-released (holder check skipped)",
		}, "release")
	}
	v, err := store.ReleaseFenced(context.Background(), *id, *holder, *gen, time.Now())
	if err == nil && v.OK {
		ambientLeaserefAnnounce(stderr, *dir, leaseref.AnnounceRelease, announceRec, resolveAmbientLeaserefConfig(*announce, *announceIssue, *announceRepo))
	}
	return emitLeaserefResult(stdout, stderr, v, err, "fak leaseref release", "release", func(v leaseref.FenceVerdict) bool { return v.OK })
}
