package main

// leaseref_announce.go is the caller-side edge of PLANE 2 (the GH-comment backup channel,
// #2300 under epic #2254). The pure render/parse/fold lives in internal/leaseref
// (announce.go); THIS file does the `gh` I/O the unwitnessedclaim discipline keeps at the
// edge — posting a rendered comment and reading comments back to fold.
//
//   fak leaseref announce --issue N --id ID --holder H --action acquire|renew|release \
//       [--generation N] [--tree GLOB ...] [--ttl SEC] [--repo OWNER/REPO] [--dry-run]
//       Post a structured one-line announcement of a lease transition to coordination
//       issue N. NEVER BLOCKS the underlying lease op: a gh failure is a WARNING on
//       stderr and a 0 exit, never a refusal (#2300 acceptance). --dry-run prints the
//       comment body without posting (and needs no --issue).
//
//   fak leaseref announce-view --issue N [--repo OWNER/REPO] [--dir DIR]
//       Read issue N's comments and FOLD the announcements into the advisory held-set
//       view (JSON [{lease_id,holder,generation,tree,ttl_seconds,action}, ...]). This is
//       EVIDENCE, never an admission input — advisory visibility when planes 0/1 are down.

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// leaserefAnnounceTimeout bounds each gh subprocess so a hung network never wedges a
// best-effort announce (the whole point of plane 2 is a node whose faster planes are
// already unreachable).
const leaserefAnnounceTimeout = 60 * time.Second

// ambientLeaserefAnnouncePost is the network-edge test seam. Ambient lifecycle
// announcements deliberately reuse the explicit announce verb's gh body-file edge.
var ambientLeaserefAnnouncePost = postLeaserefAnnounce

const (
	ambientAnnounceSuccess = "success"
	ambientAnnounceRefusal = "refusal"
	ambientAnnounceError   = "error"
)

var ambientLeaserefAnnounceStderrMu sync.Mutex

// reportAmbientLeaserefAnnounce preserves the existing human diagnostic while exposing a
// stable, one-hot count on the command's existing stderr surface. Fixed field order lets
// callers fold output deterministically without learning any lease or key material. A
// single guarded write keeps the record intact when commands share a stderr writer.
func reportAmbientLeaserefAnnounce(stderr io.Writer, message, outcome string) {
	success, refusal, announceError := 0, 0, 0
	switch outcome {
	case ambientAnnounceSuccess:
		success = 1
	case ambientAnnounceRefusal:
		refusal = 1
	case ambientAnnounceError:
		announceError = 1
	}
	entry := fmt.Sprintf("%s\nfak leaseref: ambient-announcement-outcomes success=%d refusal=%d error=%d\n",
		message, success, refusal, announceError)
	ambientLeaserefAnnounceStderrMu.Lock()
	defer ambientLeaserefAnnounceStderrMu.Unlock()
	_, _ = io.WriteString(stderr, entry)
}

// ambientLeaserefAnnounce publishes a public-safe projection after a successful local
// lifecycle transition. It is opt-in and best-effort: every invocation reports exactly
// one public-safe outcome on stderr, and no configuration or transport failure is returned
// to the lease operation. The key itself is read only from a file; it is never accepted in
// argv or environment.
type ambientLeaserefConfig struct {
	Mode  string
	Issue int
	Repo  string
}

var ambientLeaserefDefaultConfig ambientLeaserefConfig

func resolveAmbientLeaserefConfig(mode string, issue int, repo string) ambientLeaserefConfig {
	if strings.TrimSpace(mode) == "" && issue == 0 && strings.TrimSpace(repo) == "" {
		return ambientLeaserefDefaultConfig
	}
	return ambientLeaserefConfig{Mode: mode, Issue: issue, Repo: repo}
}

func ambientLeaserefAnnounce(stderr io.Writer, dir, action string, rec leaseref.Record, configs ...ambientLeaserefConfig) {
	config := ambientLeaserefDefaultConfig
	if len(configs) > 0 {
		config = configs[0]
	}
	mode := strings.ToLower(strings.TrimSpace(config.Mode))
	if mode == "" || mode == "0" || mode == "off" || mode == "false" || mode == "no" || mode == "disabled" {
		reportAmbientLeaserefAnnounce(stderr, "fak leaseref: public-safe issue announcement disabled (pass --announce=on to opt in)", ambientAnnounceRefusal)
		return
	}
	if mode == "offline" {
		reportAmbientLeaserefAnnounce(stderr, "fak leaseref: public-safe issue announcement offline; local lease operation succeeded", ambientAnnounceRefusal)
		return
	}
	if mode != "1" && mode != "on" && mode != "true" && mode != "yes" && mode != "enabled" {
		reportAmbientLeaserefAnnounce(stderr, "fak leaseref: WARNING: public-safe issue announcement disabled by an unrecognized --announce value", ambientAnnounceRefusal)
		return
	}

	issue := config.Issue
	if issue <= 0 {
		reportAmbientLeaserefAnnounce(stderr, "fak leaseref: WARNING: public-safe issue announcement not posted: --announce-issue is missing or invalid", ambientAnnounceRefusal)
		return
	}
	repo := strings.TrimSpace(config.Repo)
	if repo == "" {
		reportAmbientLeaserefAnnounce(stderr, "fak leaseref: WARNING: public-safe issue announcement not posted: --announce-repo is missing", ambientAnnounceRefusal)
		return
	}
	keyFile := strings.TrimSpace(os.Getenv("FAK_LEASEREF_ANNOUNCE_KEY_FILE"))
	if keyFile == "" {
		reportAmbientLeaserefAnnounce(stderr, "fak leaseref: WARNING: public-safe issue announcement not posted: FAK_LEASEREF_ANNOUNCE_KEY_FILE is missing", ambientAnnounceRefusal)
		return
	}
	key, err := os.ReadFile(pathutil.ExpandTilde(keyFile))
	if err != nil || len(bytes.TrimSpace(key)) == 0 {
		// Do not include the path or underlying error: secret-store paths and helper
		// diagnostics do not belong in logs.
		reportAmbientLeaserefAnnounce(stderr, "fak leaseref: WARNING: public-safe issue announcement not posted: key file is unavailable or empty", ambientAnnounceRefusal)
		return
	}
	projected, err := leaseref.PublicSafeAnnounce(leaseref.AnnounceRecord{
		LeaseID: rec.ID, Holder: rec.Holder, Generation: rec.Generation,
		Tree: rec.TreeGlobs, TTLSeconds: rec.TTLSeconds, Action: action,
	}, bytes.TrimSpace(key))
	if err != nil {
		reportAmbientLeaserefAnnounce(stderr, "fak leaseref: WARNING: public-safe issue announcement projection failed; local lease operation succeeded", ambientAnnounceError)
		return
	}
	body := hooks.ScrubHardwareNames(leaseref.RenderAnnounce(projected))
	if err := ambientLeaserefAnnouncePost(dir, repo, issue, body); err != nil {
		// Never print gh output here: even a hostile helper must not turn its diagnostics
		// into a secret/log exfiltration edge.
		reportAmbientLeaserefAnnounce(stderr, "fak leaseref: WARNING: public-safe issue announcement post failed; local lease operation succeeded", ambientAnnounceError)
		return
	}
	reportAmbientLeaserefAnnounce(stderr, "fak leaseref: public-safe issue announcement posted", ambientAnnounceSuccess)
}

// runLeaserefAnnounce renders a lease transition (leaseref.RenderAnnounce — pure) and
// posts it to a coordination issue via `gh issue comment` at the edge. The load-bearing
// acceptance property: a post failure is a WARNING, never a refusal — it returns 0 so a
// chained caller (`fak leaseref acquire && fak leaseref announce`) is never broken by a
// plane-2 outage. Only a caller MISTAKE (bad flags / out-of-vocabulary action) is a usage
// error (exit 2).
func runLeaserefAnnounce(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref announce", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	issue := fs.Int("issue", 0, "coordination issue number to comment on")
	id := fs.String("id", "", "lease id being announced")
	holder := fs.String("holder", "", "holder identity (machine/session)")
	gen := fs.Int64("generation", 0, "fencing token (generation) of the lease")
	ttl := fs.Int64("ttl", 0, "lease lifetime in seconds (0 = no expiry)")
	action := fs.String("action", "", "lifecycle action: acquire|renew|release")
	repo := fs.String("repo", "", "gh repo (OWNER/REPO); default the current repo")
	dryRun := fs.Bool("dry-run", false, "render the comment body to stdout without posting")
	publicSafeKeyFile := fs.String("public-safe-key-file", "", "file containing the shared key used to fingerprint public lease fields")
	var trees repeatedString
	fs.Var(&trees, "tree", "repo-relative tree glob this lease covers (repeatable)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *id == "" || *holder == "" {
		fmt.Fprintln(stderr, "fak leaseref announce: --id and --holder are required")
		return 2
	}
	if !leaseref.ValidAnnounceAction(*action) {
		fmt.Fprintf(stderr, "fak leaseref announce: --action must be one of acquire|renew|release (got %q)\n", *action)
		return 2
	}
	rec := leaseref.AnnounceRecord{LeaseID: *id, Holder: *holder, Generation: *gen, Tree: trees, TTLSeconds: *ttl, Action: *action}
	if *publicSafeKeyFile != "" {
		key, err := os.ReadFile(pathutil.ExpandTilde(*publicSafeKeyFile))
		if err != nil {
			fmt.Fprintf(stderr, "fak leaseref announce: read --public-safe-key-file: %v\n", err)
			return 2
		}
		rec, err = leaseref.PublicSafeAnnounce(rec, bytes.TrimSpace(key))
		if err != nil {
			fmt.Fprintf(stderr, "fak leaseref announce: %v\n", err)
			return 2
		}
	}
	body := hooks.ScrubHardwareNames(leaseref.RenderAnnounce(rec))
	if *dryRun {
		fmt.Fprintln(stdout, body)
		return 0
	}
	if *issue <= 0 {
		fmt.Fprintln(stderr, "fak leaseref announce: --issue is required to post (or use --dry-run)")
		return 2
	}
	if err := postLeaserefAnnounce(*dir, *repo, *issue, body); err != nil {
		// Announce NEVER refuses the underlying op: warn and exit 0 so a plane-2 outage
		// never cascades into the lease operation it merely records.
		fmt.Fprintf(stderr, "fak leaseref announce: WARNING: could not post to #%d: %v (the lease operation is unaffected)\n", *issue, err)
		return 0
	}
	fmt.Fprintf(stdout, "announced %s of lease %q to #%d\n", *action, *id, *issue)
	return 0
}

// runLeaserefAnnounceView reads issue N's comments (gh at the edge) and folds the
// announcements into the advisory held-set view (leaseref.FoldAnnouncements — pure). A
// fetch failure IS an error here (exit 1): unlike the write path, a read that could not
// reach gh has no advisory answer to offer, so it must not print an empty (misleadingly
// "nothing held") view as if it succeeded.
func runLeaserefAnnounceView(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref announce-view", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	issue := fs.Int("issue", 0, "coordination issue number to read comments from")
	repo := fs.String("repo", "", "gh repo (OWNER/REPO); default the current repo")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *issue <= 0 {
		fmt.Fprintln(stderr, "fak leaseref announce-view: --issue is required")
		return 2
	}
	bodies, err := fetchLeaserefAnnounceComments(*dir, *repo, *issue)
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref announce-view: %v\n", err)
		return 1
	}
	view := leaseref.FoldAnnouncements(bodies)
	if view == nil {
		view = []leaseref.AnnounceRecord{}
	}
	return emitLeaserefJSON(stdout, stderr, view, "announce-view")
}

// postLeaserefAnnounce writes the rendered body to a temp file and posts it with
// `gh issue comment N --body-file` — the same body-file discipline issuecatalog uses so a
// multi-line body with special characters never hits an argv/quoting limit.
func postLeaserefAnnounce(dir, repo string, issue int, body string) error {
	f, err := os.CreateTemp("", "fak-leaseref-announce-*.md")
	if err != nil {
		return fmt.Errorf("temp body file: %w", err)
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err := f.WriteString(body); err != nil {
		f.Close()
		return fmt.Errorf("write body file: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close body file: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), leaserefAnnounceTimeout)
	defer cancel()
	args := []string{"issue", "comment", strconv.Itoa(issue), "--body-file", tmp}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	cmd := exec.CommandContext(ctx, "gh", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("gh issue comment: %w (%s)", err, strings.TrimSpace(string(out)))
	}
	return nil
}

// fetchLeaserefAnnounceComments reads issue N's comments via `gh issue view --json
// comments` and returns each comment body oldest-first (the order FoldAnnouncements
// expects).
func fetchLeaserefAnnounceComments(dir, repo string, issue int) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), leaserefAnnounceTimeout)
	defer cancel()
	args := []string{"issue", "view", strconv.Itoa(issue), "--json", "comments"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	cmd := exec.CommandContext(ctx, "gh", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("gh issue view --json comments: %w (%s)", err, strings.TrimSpace(string(out)))
	}
	var payload struct {
		Comments []struct {
			Body string `json:"body"`
		} `json:"comments"`
	}
	if uerr := json.Unmarshal(out, &payload); uerr != nil {
		return nil, fmt.Errorf("parse gh comments json: %w", uerr)
	}
	bodies := make([]string, 0, len(payload.Comments))
	for _, c := range payload.Comments {
		bodies = append(bodies, c.Body)
	}
	return bodies, nil
}
