package main

// fak loop region -- the surface-neutral region-admission question: "may ACTOR
// act on this (lane, tree) right now?", answered by internal/regionadmit
// against the live lease fabric (refs/fak/locks/* via internal/leaseref) and
// the dos.toml lane taxonomy.
//
// This is the decision every coordinated surface shares — the dispatch tick
// asks it before spawning a worker, `fak loop drive` asks it before holding a
// GOAL loop's region — exposed as one verb so the surfaces WITHOUT a built-in
// admission step (a manual operator session, a super-loop enter path, a
// script) can ask the same question before touching a region:
//
//   fak loop region --lane gateway --actor session:me
//   fak loop region --tree 'internal/gateway/**' --json
//
// Exit 0 = admitted, 3 = refused (COLLISION_RISK with the conflicting lease
// named), 1 = the lease store or taxonomy could not be read (an error, never a
// silent admit), 2 = usage. It still holds NOTHING — holding a lease stays with
// `fak leaseref acquire`, and the same honest boundary applies: cross-machine
// this is visibility after a fetch, not atomic acquisition.
//
// #5505: a refusal no longer EVAPORATES. It mints a waiter ticket
// (internal/leasequeue) whose enqueue clock survives every retry, and reports
// this caller's place in line, the blocker holding it there and a bounded poll
// schedule — so repeated attempts are ordered by arrival instead of re-raced,
// and a four-hour waiter no longer has the same odds as one that arrived 200ms
// ago. The queue is best-effort and OFF the decision: the verdict and the exit
// code are computed first and are never changed by it. `--no-queue` opts out
// (a pure query that leaves no trace), and `--class interactive` declares an
// operator so it does not queue behind a wall of background loops.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
)

func runLoopRegion(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop region", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lane := fs.String("lane", "", "dos.toml lane to admit against (its canonical tree is the region when --tree is absent)")
	var tree repeatedString
	fs.Var(&tree, "tree", "region glob to admit (repeatable)")
	actor := fs.String("actor", "", "who is asking (defaults to the lease-holder identity: FAK_LEASE_OWNER, session id, or host:pid)")
	selfID := fs.String("self", "", "the caller's own lease id, never counted as a conflict (re-admission/renew)")
	dir := fs.String("dir", "", "repo whose refs/fak/locks/* and dos.toml are read (default: cwd)")
	readOnly := fs.Bool("read-only", false, "the act writes nothing (a provably empty footprint): admitted against every live lease, distinct from an absent region (unknown blast radius, which still collides)")
	jsonOut := fs.Bool("json", false, "emit the decision as JSON")
	class := fs.String("class", "", "priority class of this waiter: interactive (an operator) or loop (a background driver); unset ranks as a loop")
	noQueue := fs.Bool("no-queue", false, "do not mint a waiter ticket on a refusal (a pure query that leaves no trace and takes no place in line)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak loop region: unexpected positional arguments")
		return 2
	}
	// A read-only act declares an EMPTY write footprint, so it needs no region: --lane/--tree
	// are optional for it (an absent region without --read-only stays unknown blast radius and
	// is rejected as before).
	if strings.TrimSpace(*lane) == "" && len(tree) == 0 && !*readOnly {
		fmt.Fprintln(stderr, "fak loop region: --lane or --tree is required (an empty region is unknown blast radius)")
		return 2
	}
	who := strings.TrimSpace(*actor)
	if who == "" {
		who = dispatchLeaseHolder()
	}
	*dir = pathutil.ExpandTilde(*dir)
	taxRoot := *dir
	if taxRoot == "" {
		taxRoot = "."
	}
	tax, err := regionadmit.LoadTaxonomy(taxRoot)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop region: read lane taxonomy: %v\n", err)
		return 1
	}
	live, _, err := leaseref.NewInDir(*dir).Live(context.Background(), time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak loop region: read live leases: %v\n", err)
		return 1
	}
	req := regionadmit.Request{Actor: who, Lane: strings.TrimSpace(*lane), Tree: tree, SelfID: strings.TrimSpace(*selfID), ReadOnly: *readOnly}
	dec := regionadmit.Decide(req, regionLeases(live), tax)

	// The waiter plane (#5505). It runs AFTER the verdict and can only add reporting: a refusal
	// takes a place in line, an admit gives one up. Both are best-effort and silent on failure,
	// so an unwritable queue degrades the report and never the decision or the exit code.
	var queued *loopRegionQueueReport
	if !*noQueue {
		if dec.Admit {
			loopRegionDequeue(taxRoot, req, tax)
		} else {
			queued = loopRegionEnqueue(taxRoot, req, tax, live, strings.TrimSpace(*class), time.Now())
		}
	}

	if *jsonOut {
		payload := map[string]any{
			"schema":     "fak.loop-region.v1",
			"admit":      dec.Admit,
			"actor":      who,
			"lane":       req.Lane,
			"tree":       append([]string(nil), regionadmit.ResolveTree(req, tax)...),
			"read_only":  req.ReadOnly,
			"live_count": len(live),
		}
		if !dec.Admit {
			payload["reason"] = dec.Reason
			payload["rung"] = dec.Rung
			payload["detail"] = dec.Detail
			if dec.Conflict != nil {
				payload["conflict"] = map[string]any{
					"id":     dec.Conflict.ID,
					"holder": dec.Conflict.Holder,
					"tree":   append([]string(nil), dec.Conflict.Tree...),
				}
			}
			if q := queued.payload(); q != nil {
				payload["queue"] = q
			}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(payload); err != nil {
			fmt.Fprintf(stderr, "fak loop region: %v\n", err)
			return 1
		}
	} else if dec.Admit {
		fmt.Fprintf(stdout, "ADMIT %s may act on %s (%d live lease(s), none conflict)\n",
			who, regionLabel(req, tax), len(live))
	} else {
		fmt.Fprintf(stdout, "REFUSE %s: %s [%s] %s\n", who, dec.Reason, dec.Rung, dec.Detail)
		if line := queued.line(); line != "" {
			fmt.Fprintln(stdout, line)
		}
	}
	if dec.Admit {
		return 0
	}
	return leaserefRefused
}

func regionLabel(req regionadmit.Request, tax regionadmit.Taxonomy) string {
	tree := regionadmit.ResolveTree(req, tax)
	if req.ReadOnly && req.Lane == "" && len(tree) == 0 {
		return "a read-only region (writes nothing)"
	}
	if req.Lane != "" {
		return fmt.Sprintf("lane %q %v", req.Lane, tree)
	}
	return fmt.Sprintf("%v", tree)
}
