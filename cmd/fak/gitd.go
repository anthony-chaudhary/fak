package main

// gitd.go is `fak gitd` (#5622, rung 3 of epic #5619): the resident, per-repo git
// query broker that separate short-lived client processes share over one
// Unix-domain socket.
//
// WHY A COMMAND AND NOT A LIBRARY CALL. Rung 2 (#5621) batches git reads WITHIN one
// process. The churn this fleet generates comes from MANY short-lived ones -- eight
// resident sessions firing per-turn hooks plus N python dispatchers on timers -- so
// the warm backend has to outlive any single client. That means a process, and a
// process needs a verb.
//
// DISCOVERY IS DERIVED, NOT CONFIGURED. Both halves compute the socket and token
// paths from the canonical repo root ALONE (gitbroker.RendezvousFor), so a client
// that knows which repo it is asking about already knows where to knock. There is
// no env var to hand-wire and no address to pass around.
//
// THE STATUS SURFACE SPELLS "DOWN" DIFFERENTLY FROM "IDLE". Per the internal/
// stallscan lesson, `--status` against a dead broker reports status=down with a
// non-zero exit, which is NOT the same output as a live broker that has served
// nothing (status=up, served=0). Collapsing those two is the bug that lesson names.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/anthony-chaudhary/fak/internal/gitbroker"
)

func cmdGitd(argv []string) { os.Exit(runGitd(os.Stdout, os.Stderr, argv)) }

// runGitd serves the broker in the foreground until interrupted, or -- under
// --status -- reports on a broker that is already running and exits.
func runGitd(stdout, stderr io.Writer, argv []string) int {
	const prefix = "fak gitd"
	fs := flag.NewFlagSet(prefix, flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "repository to broker for (default: the repo containing the working directory)")
	status := fs.Bool("status", false, "report on the broker already serving this repo and exit; non-zero when none is running")
	read := fs.String("read", "", "act as a CLIENT: resolve one revision through the broker and print it with its provenance")
	asJSON := fs.Bool("json", false, "emit machine-readable output")
	cacheBytes := fs.Int64("cache-bytes", gitbroker.DefaultCacheBytes, "byte ceiling for the content-addressed (Class A) object cache")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s: unexpected argument %q\n", prefix, fs.Arg(0))
		return 2
	}

	root := *repo
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "%s: resolve working directory: %v\n", prefix, err)
			return 2
		}
		root = findRepoRoot(wd)
	}

	if *status && *read != "" {
		fmt.Fprintf(stderr, "%s: --status and --read are separate modes; pass one\n", prefix)
		return 2
	}
	if *status {
		return gitdStatus(stdout, root, *asJSON)
	}
	if *read != "" {
		return gitdRead(stdout, stderr, prefix, root, *read, *asJSON)
	}
	return gitdServe(stdout, stderr, prefix, root, *cacheBytes, *asJSON)
}

// gitdRead is the CLIENT half, and it is what makes the cross-process claim
// checkable by hand: run it twice against a live broker and the second run prints
// provenance=cache, proving a SEPARATE process reused the warm entry. It is also
// the operator's provenance window -- per the internal/stallscan lesson, "the
// broker is down" (fallback-spawn) reads differently from a served answer, so a
// silently-dead broker cannot masquerade as a healthy one.
//
// It never fails because the broker is unhealthy: gitbroker.Client falls back to
// spawning git, so this exits non-zero only when the REVISION genuinely does not
// resolve.
func gitdRead(stdout, stderr io.Writer, prefix, root, rev string, asJSON bool) int {
	c := &gitbroker.Client{RepoRoot: root}
	res, err := c.Object(context.Background(), rev)
	if err != nil {
		fmt.Fprintf(stderr, "%s: read %s: %v\n", prefix, rev, err)
		return 1
	}
	if asJSON {
		emitGitdJSON(stdout, map[string]any{
			"provenance": string(res.Provenance),
			"oid":        res.OID,
			"type":       res.Type,
			"size":       res.Size,
			"cacheable":  gitbroker.IsOID(rev),
		})
		return 0
	}
	fmt.Fprintf(stdout, "gitd: provenance=%s oid=%s type=%s size=%d cacheable=%t\n",
		res.Provenance, res.OID, res.Type, res.Size, gitbroker.IsOID(rev))
	return 0
}

// gitdStatus asks the live broker for its counters. The ok=false path is the
// stallscan rule: a broker that is DOWN gets its own spelling and its own exit
// code, so no operator (and no script) can read it as a quiet, healthy zero.
func gitdStatus(stdout io.Writer, root string, asJSON bool) int {
	rv := gitbroker.RendezvousFor(root)
	c := &gitbroker.Client{RepoRoot: root}
	st, ok := c.Stats(context.Background())
	if !ok {
		if asJSON {
			emitGitdJSON(stdout, map[string]any{"status": "down", "repo": root, "socket": rv.Socket})
		} else {
			fmt.Fprintf(stdout, "gitd: status=down repo=%s socket=%s (no broker answered)\n", root, rv.Socket)
		}
		return 1
	}
	if asJSON {
		emitGitdJSON(stdout, map[string]any{"status": "up", "repo": root, "socket": rv.Socket, "stats": st})
	} else {
		fmt.Fprintf(stdout, "gitd: status=up repo=%s socket=%s served=%d cache_hits=%d cache_misses=%d uncacheable=%d entries=%d cache_bytes=%d\n",
			root, rv.Socket, st.Served, st.Hits, st.Misses, st.Uncached, st.Entries, st.CacheSize)
	}
	return 0
}

// gitdServe binds the rendezvous and blocks. The backend is left nil so the
// package picks its SpawnRunner default -- the seam #5621's warm pool drops into
// without this command changing at all.
func gitdServe(stdout, stderr io.Writer, prefix, root string, cacheBytes int64, asJSON bool) int {
	srv, err := gitbroker.Serve(gitbroker.Config{RepoRoot: root, CacheBytes: cacheBytes})
	if err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", prefix, err)
		return 1
	}
	defer srv.Close()

	rv := srv.Rendezvous()
	if asJSON {
		emitGitdJSON(stdout, map[string]any{"status": "serving", "repo": root, "socket": rv.Socket})
	} else {
		fmt.Fprintf(stdout, "gitd: serving repo=%s socket=%s (ctrl-c to stop)\n", root, rv.Socket)
	}
	// Flush the readiness line before blocking: a supervisor that waits for it
	// must not deadlock behind a buffered writer.
	if f, ok := stdout.(*os.File); ok {
		_ = f.Sync()
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig
	// Close removes the socket and token so the NEXT broker binds cleanly rather
	// than inheriting a corpse.
	_ = srv.Close()
	return 0
}

func emitGitdJSON(w io.Writer, payload map[string]any) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(payload)
}
