package main

// trunk_red_claim.go is the shared RED-TRUNK REPAIR CLAIM surface (2026-10-10 ctxmmu
// incident: two sessions landed the same rename fix under different names within the hour).
//
//   - The commit gate, on proving a red PRE-EXISTING at HEAD, prints any live repair claim on
//     the failing package(s) and any upstream commit after HEAD already touching them
//     (formatRedRepairAdvisory), so the next session converges instead of re-fixing.
//   - `fak trunk-red claim <pkg>` takes the claim when a repair starts (an intent lease under
//     refs/fak/locks/, keyed by package and base OID, default one-hour TTL), and
//     `fak trunk-red release <pkg>` drops it on land. Push the lock refs as usual so peers on
//     other clones see it.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// redRepairNow is the clock for claim liveness; tests pin it.
var redRepairNow = time.Now

// redRepairHolder names the claimant: the session id when one is exported, else the host.
func redRepairHolder() string {
	if s := trunkRedSession(); s != "" {
		return s
	}
	if h, err := os.Hostname(); err == nil && h != "" {
		return h
	}
	return "unknown"
}

// redRepairUpstream is the upstream ref a repair should be compared against: the current
// branch's configured upstream, else origin/main.
func redRepairUpstream(root string) string {
	if u, err := gitOut(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"); err == nil {
		if u = strings.TrimSpace(u); u != "" && u != "@{upstream}" {
			return u
		}
	}
	return "origin/main"
}

// redRepairKeys maps failing import paths to repo-relative package directories, the claim
// key; synthetic or foreign paths fall back to the import path itself.
func redRepairKeys(root string, pkgs []string) []string {
	if dirs := trunkRedPackageDirs(root, pkgs); len(dirs) > 0 {
		return dirs
	}
	return pkgs
}

// formatRedRepairAdvisory renders the claim + already-landed-fix lines for a pre-existing red
// at base. Fail-open: any lookup error simply omits its lines.
func formatRedRepairAdvisory(root, base string, pkgs []string) string {
	keys := redRepairKeys(root, pkgs)
	if len(keys) == 0 {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	now := redRepairNow()
	var b strings.Builder
	claims, _ := leaseref.NewInDir(root).RedRepairClaims(ctx, keys, now)
	for _, c := range claims {
		pkg, cbase, _ := leaseref.ParseRedRepairTarget(c.Target)
		age := now.Unix() - c.AcquiredAt
		if c.RenewedAt > c.AcquiredAt {
			age = now.Unix() - c.RenewedAt
		}
		fmt.Fprintf(&b, "  repair already CLAIMED: %s by %s (base %s, %ds ago", pkg, c.Holder, shortRedSHA(cbase), age)
		if c.SessionID != "" && c.SessionID != c.Holder {
			b.WriteString(", session " + c.SessionID)
		}
		b.WriteString(") — coordinate with that session instead of landing a second fix.\n")
	}
	upstream := redRepairUpstream(root)
	if peers, err := safesync.PeerCommitsTouching(ctx, nil, root, base, upstream, keys); err == nil {
		for _, p := range peers {
			fmt.Fprintf(&b, "  %s already has %s touching the failing package after HEAD: %s — fetch and check it before repairing.\n", upstream, shortRedSHA(p.SHA), p.Subject)
		}
	}
	if len(claims) == 0 {
		fmt.Fprintf(&b, "  repairing it? claim it first so peers see you: fak trunk-red claim %s --base %s\n", strings.Join(keys, " "), shortRedSHA(base))
	}
	return b.String()
}

func shortRedSHA(sha string) string {
	sha = strings.TrimSpace(sha)
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

// runTrunkRedClaim implements `fak trunk-red claim|release <pkg>...`.
func runTrunkRedClaim(stdout, stderr io.Writer, verb string, argv []string) int {
	fs := flag.NewFlagSet("trunk-red "+verb, flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: fak trunk-red %s [--base SHA] [--ttl 1h] <pkg-dir>...\n", verb)
		fmt.Fprintln(stderr, "  claim: record that this session is repairing the red package(s) at base (default HEAD).")
		fmt.Fprintln(stderr, "  release: drop that claim once the repair lands.")
		fs.PrintDefaults()
	}
	repo := fs.String("repo", "", "repo path (default: the enclosing repo)")
	base := fs.String("base", "HEAD", "the red base commit the repair starts from")
	ttl := fs.Duration("ttl", time.Hour, "claim: lifetime before the claim lapses")
	holder := fs.String("holder", "", "claim: claimant identity (default: session id or hostname)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	pkgs := fs.Args()
	if len(pkgs) == 0 {
		fs.Usage()
		return 2
	}
	root := strings.TrimSpace(*repo)
	if root == "" {
		root = repoRoot()
	}
	baseSHA, err := gitRevParse(root, *base)
	if err != nil {
		fmt.Fprintf(stderr, "fak trunk-red %s: resolve --base %s: %v\n", verb, *base, err)
		return 1
	}
	h := strings.TrimSpace(*holder)
	if h == "" {
		h = redRepairHolder()
	}
	store := leaseref.NewInDir(root)
	ctx := context.Background()
	now := redRepairNow()
	code := 0
	for _, pkg := range pkgs {
		pkg = strings.Trim(strings.TrimSpace(pkg), "/")
		if verb == "release" {
			if err := store.ReleaseRedRepair(ctx, pkg, baseSHA); err != nil {
				fmt.Fprintf(stderr, "fak trunk-red release %s: %v\n", pkg, err)
				code = 1
				continue
			}
			fmt.Fprintf(stdout, "released repair claim: %s @ %s\n", pkg, shortRedSHA(baseSHA))
			continue
		}
		_, v, err := store.ClaimRedRepair(ctx, pkg, baseSHA, h, trunkRedSession(), int64(ttl.Seconds()), now)
		if err != nil {
			fmt.Fprintf(stderr, "fak trunk-red claim %s: %v\n", pkg, err)
			code = 1
			continue
		}
		if !v.OK {
			fmt.Fprintf(stderr, "fak trunk-red claim %s: %s: %s\n", pkg, v.Reason, v.Detail)
			code = 3
			continue
		}
		fmt.Fprintf(stdout, "claimed repair: %s @ %s by %s (ttl %s); release on land: fak trunk-red release %s --base %s\n",
			pkg, shortRedSHA(baseSHA), h, *ttl, pkg, shortRedSHA(baseSHA))
	}
	return code
}

// syncDuplicateFixAdvisory prints the pre-push DUPLICATE_FIX advisory for HEAD vs
// <remote>/<branch>. Advisory and fail-open: it never changes the push outcome.
var syncDuplicateFixes = safesync.DuplicateFixAdvisories

func syncDuplicateFixAdvisory(stderr io.Writer, repo, remote, branch string) {
	b := strings.TrimSpace(branch)
	if b == "" {
		cur, err := gitOut(repo, "rev-parse", "--abbrev-ref", "HEAD")
		if err != nil || strings.TrimSpace(cur) == "" || strings.TrimSpace(cur) == "HEAD" {
			return
		}
		b = strings.TrimSpace(cur)
	}
	upstream := remote + "/" + b
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	fixes, err := syncDuplicateFixes(ctx, nil, repo, upstream)
	if err != nil {
		return
	}
	safesync.RenderDuplicateFixes(stderr, upstream, fixes)
}
