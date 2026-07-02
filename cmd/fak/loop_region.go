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
// silent admit), 2 = usage. It DECIDES only — holding a lease stays with
// `fak leaseref acquire`, and the same honest boundary applies: cross-machine
// this is visibility after a fetch, not atomic acquisition.

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
	jsonOut := fs.Bool("json", false, "emit the decision as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak loop region: unexpected positional arguments")
		return 2
	}
	if strings.TrimSpace(*lane) == "" && len(tree) == 0 {
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
	req := regionadmit.Request{Actor: who, Lane: strings.TrimSpace(*lane), Tree: tree, SelfID: strings.TrimSpace(*selfID)}
	dec := regionadmit.Decide(req, regionLeases(live), tax)

	if *jsonOut {
		payload := map[string]any{
			"schema":     "fak.loop-region.v1",
			"admit":      dec.Admit,
			"actor":      who,
			"lane":       req.Lane,
			"tree":       append([]string(nil), regionadmit.ResolveTree(req, tax)...),
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
	}
	if dec.Admit {
		return 0
	}
	return leaserefRefused
}

func regionLabel(req regionadmit.Request, tax regionadmit.Taxonomy) string {
	if req.Lane != "" {
		return fmt.Sprintf("lane %q %v", req.Lane, regionadmit.ResolveTree(req, tax))
	}
	return fmt.Sprintf("%v", regionadmit.ResolveTree(req, tax))
}
