package main

// wip_ticket.go — `fak wip reconcile --file-ticket` (#5337). When reconcile
// classifies a crashed session's orphaned checkpoint as QUARANTINE (the delta is
// unlanded AND does not apply cleanly, so it is neither auto-reclaimed nor
// discarded), that orphaned work has no durable follow-up: the verdict is printed
// and forgotten. This adds an opt-in path that binds each QUARANTINE verdict to ONE
// idempotent GitHub tracking ticket, so orphaned WIP always has a linked issue.
//
// It is SAFE by construction:
//   - It acts ONLY on QUARANTINE verdicts; DISCARD_WITNESSED/RECLAIM are untouched.
//   - Every ticket is keyed by session + start-SHA and carries an embedded marker
//     (<!-- fak-wip-ticket-key: wip-orphan-<session>-<sha12> -->). Before filing it
//     dedups against existing issues by that marker, so a re-run never double-files.
//   - It is DRY-RUN by default when uncertain: with --dry-run, or when `gh` is not
//     available, it PRINTS the exact ticket it would file and returns without error.
//     It never wedges the reconcile pass it rides on.
//
// The git I/O (listing checkpoints, resolving the delta's file set) lives here; the
// GitHub calls are behind an injected seam (wipTicketGH) so the fold and the
// dedup/file logic are testable with no network.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/wipref"
	"github.com/anthony-chaudhary/fak/internal/wiprecon"
)

// wipOrphanTicket is the fully-rendered tracking ticket for one QUARANTINE orphan.
// Key is the idempotency key; Title/Body are exactly what would be filed. The Body
// carries the Key as an embedded HTML-comment marker so a later dedup can recover it.
type wipOrphanTicket struct {
	Session string
	SHA12   string
	Files   []string
	Reason  string
	Key     string
	Title   string
	Body    string
}

// wipTicketGH is the injected GitHub seam so the dedup/file logic needs no network in
// tests. A nil `available` (or one returning false) is treated as offline — the driver
// falls back to a dry-run print, never an error.
type wipTicketGH struct {
	available func() bool
	find      func(ctx context.Context, key string) ([]int, error)
	create    func(ctx context.Context, title, body string) (int, error)
}

// wipOrphanTicketKey is the per-orphan idempotency key: session + short start-SHA.
// One crashed checkpoint (session at a fixed start commit) maps to exactly one key,
// so re-running reconcile --file-ticket converges on a single ticket.
func wipOrphanTicketKey(session, sha12 string) string {
	return fmt.Sprintf("wip-orphan-%s-%s", session, sha12)
}

// wipOrphanTicketMarker is the exact string embedded in (and searched for in) a
// ticket body — the durable, machine-recoverable binding of a ticket to its key.
func wipOrphanTicketMarker(key string) string {
	return "<!-- fak-wip-ticket-key: " + key + " -->"
}

// buildWipOrphanTicket renders the ticket for one QUARANTINE orphan. Pure: no git, no
// network. A missing start-SHA (unparseable stamp) degrades to a stable "unknown" so
// the key is still deterministic.
func buildWipOrphanTicket(session, startSHA string, files []string, reason string) wipOrphanTicket {
	sha12 := shortWipSHA(strings.TrimSpace(startSHA))
	if sha12 == "" {
		sha12 = "unknown"
	}
	t := wipOrphanTicket{
		Session: session,
		SHA12:   sha12,
		Files:   files,
		Reason:  strings.TrimSpace(reason),
		Key:     wipOrphanTicketKey(session, sha12),
	}
	t.Title = fmt.Sprintf("WIP orphan (QUARANTINE): crashed session %s checkpoint @ %s", session, sha12)
	t.Body = wipOrphanTicketBody(t)
	return t
}

// wipOrphanTicketBody renders the issue body: the key marker, the disposition
// (QUARANTINE), the attributed file set, and a concrete next-step line.
func wipOrphanTicketBody(t wipOrphanTicket) string {
	reason := t.Reason
	if reason == "" {
		reason = "delta unlanded and does not apply cleanly to the current tree"
	}
	var b strings.Builder
	b.WriteString(wipOrphanTicketMarker(t.Key))
	b.WriteString("\n\n")
	b.WriteString("A crashed session left an orphaned working-tree checkpoint that `fak wip reconcile` classified **QUARANTINE**: the delta is unlanded AND does not apply cleanly to the current tree, so it is neither auto-reclaimed nor discarded. This ticket tracks that orphaned WIP so it is not silently lost.\n\n")
	fmt.Fprintf(&b, "- Session: `%s`\n", t.Session)
	fmt.Fprintf(&b, "- Start SHA: `%s`\n", t.SHA12)
	fmt.Fprintf(&b, "- Disposition: QUARANTINE — %s\n\n", reason)
	b.WriteString("Attributed file set:\n")
	if len(t.Files) == 0 {
		b.WriteString("- (file set unavailable — inspect with `fak wip status`)\n")
	} else {
		for _, f := range t.Files {
			fmt.Fprintf(&b, "- `%s`\n", f)
		}
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "Next step: restore this delta in isolation with `fak wip restore %s`, review it against HEAD, then either re-land it with `fak wip land %s` once it applies cleanly, or discard the checkpoint if its content is confirmed superseded. Do NOT auto-apply — the delta conflicts with the current tree.\n", t.Session, t.Session)
	return b.String()
}

// wipPrintWouldFile prints the exact ticket that would be filed — the dry-run /
// offline rendering, and the fallback whenever a live gh call cannot be made.
func wipPrintWouldFile(w io.Writer, t wipOrphanTicket) {
	fmt.Fprintf(w, "would file ticket [%s]:\n", t.Key)
	fmt.Fprintf(w, "  title: %s\n", t.Title)
	for _, line := range strings.Split(strings.TrimRight(t.Body, "\n"), "\n") {
		fmt.Fprintf(w, "  | %s\n", line)
	}
}

// wipCollectOrphanTickets re-lists the checkpoints and renders one ticket per
// QUARANTINE decision, resolving each orphan's start-SHA (from its stamp) and file set
// (the checkpoint delta). Non-QUARANTINE decisions are ignored. A checkpoint whose
// file set can't be resolved still yields a ticket (file set unavailable), so an
// orphan is never dropped just because its delta is unreadable.
func wipCollectOrphanTickets(ctx context.Context, repo string, decisions []wiprecon.Decision) ([]wipOrphanTicket, error) {
	recs, err := wipListRecords(ctx, repo)
	if err != nil {
		return nil, err
	}
	bySession := make(map[string]wipref.RefRecord, len(recs))
	for _, r := range recs {
		bySession[wipSessionOf(r)] = r
	}
	var tickets []wipOrphanTicket
	for _, d := range decisions {
		if d.Action != wiprecon.ActQuarantine {
			continue
		}
		var startSHA string
		var files []string
		if rec, ok := bySession[d.Session]; ok {
			startSHA = rec.Stamp.StartSHA
			if fs, ferr := wipDeltaFiles(ctx, repo, rec.Object); ferr == nil {
				files = fs
			}
		}
		tickets = append(tickets, buildWipOrphanTicket(d.Session, startSHA, files, d.Reason))
	}
	return tickets, nil
}

// wipEmitOrphanTickets is the dedup/file/print fold over the gh seam. Offline (dry-run,
// or gh unavailable) it prints each ticket and files nothing. Online it dedups by the
// ticket key: an existing marker prints "already tracked: #N" and files no duplicate;
// otherwise it files and prints "filed: #N". A gh error at any step degrades to a
// printed ticket (never wedges reconcile). Pure over the seam — no git.
func wipEmitOrphanTickets(ctx context.Context, stdout, stderr io.Writer, tickets []wipOrphanTicket, dryRun bool, gh wipTicketGH) {
	offline := dryRun || gh.available == nil || !gh.available() || gh.find == nil || gh.create == nil
	for _, t := range tickets {
		if offline {
			wipPrintWouldFile(stdout, t)
			continue
		}
		nums, err := gh.find(ctx, t.Key)
		if err != nil {
			fmt.Fprintf(stderr, "fak wip reconcile: gh lookup failed for %s (%v) — printing ticket instead of filing\n", t.Key, err)
			wipPrintWouldFile(stdout, t)
			continue
		}
		if len(nums) > 0 {
			fmt.Fprintf(stdout, "already tracked: #%d (%s)\n", nums[0], t.Key)
			continue
		}
		n, err := gh.create(ctx, t.Title, t.Body)
		if err != nil {
			fmt.Fprintf(stderr, "fak wip reconcile: gh issue create failed for %s (%v) — printing ticket instead of filing\n", t.Key, err)
			wipPrintWouldFile(stdout, t)
			continue
		}
		fmt.Fprintf(stdout, "filed: #%d (%s)\n", n, t.Key)
	}
}

// wipReconcileFileTickets composes the collect + emit passes: build a ticket per
// QUARANTINE decision, then dedup/file/print. A record-listing failure is reported to
// stderr and returns without wedging reconcile.
func wipReconcileFileTickets(ctx context.Context, stdout, stderr io.Writer, repo string, decisions []wiprecon.Decision, dryRun bool, gh wipTicketGH) {
	tickets, err := wipCollectOrphanTickets(ctx, repo, decisions)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip reconcile: file-ticket collect failed: %v\n", err)
		return
	}
	wipEmitOrphanTickets(ctx, stdout, stderr, tickets, dryRun, gh)
}

// ---- real GitHub seam (gh CLI) ----

// newWipTicketGH wires the production seam over the `gh` CLI. `available` is a plain
// LookPath so an absent gh routes the driver to its dry-run print path.
func newWipTicketGH() wipTicketGH {
	return wipTicketGH{
		available: func() bool { _, err := exec.LookPath("gh"); return err == nil },
		find:      wipGHFindByKey,
		create:    wipGHCreateIssue,
	}
}

// wipGHFindByKey returns the issue numbers whose body carries the exact key marker.
// It searches (state:all) for the key, then confirms the marker in each candidate's
// body — the search index tokenizes the key, so the body-substring check is the
// authoritative dedup, not the search hit alone.
func wipGHFindByKey(ctx context.Context, key string) ([]int, error) {
	cmd := exec.CommandContext(ctx, "gh", "issue", "list",
		"--state", "all", "--search", key, "--limit", "100", "--json", "number,body")
	configureDispatchHelperCommand(cmd)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("gh issue list: %v: %s", err, strings.TrimSpace(errb.String()))
	}
	var items []struct {
		Number int    `json:"number"`
		Body   string `json:"body"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(out.String())), &items); err != nil {
		return nil, fmt.Errorf("parse gh issue list json: %w", err)
	}
	marker := wipOrphanTicketMarker(key)
	var nums []int
	for _, it := range items {
		if strings.Contains(it.Body, marker) {
			nums = append(nums, it.Number)
		}
	}
	return nums, nil
}

// wipGHCreateIssue files the issue and returns its number, parsed from the issue URL
// gh prints on success.
func wipGHCreateIssue(ctx context.Context, title, body string) (int, error) {
	cmd := exec.CommandContext(ctx, "gh", "issue", "create", "--title", title, "--body", body)
	configureDispatchHelperCommand(cmd)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return 0, fmt.Errorf("gh issue create: %v: %s", err, strings.TrimSpace(errb.String()))
	}
	return wipParseIssueNumberFromURL(strings.TrimSpace(out.String()))
}

// wipParseIssueNumberFromURL extracts the trailing issue number from a gh-printed
// issue URL (…/issues/1234). It scans the last whitespace-delimited token so surrounding
// prose does not defeat it.
func wipParseIssueNumberFromURL(s string) (int, error) {
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return 0, fmt.Errorf("no issue url in gh output: %q", s)
	}
	last := fields[len(fields)-1]
	last = strings.TrimRight(last, "/")
	if i := strings.LastIndexByte(last, '/'); i >= 0 {
		last = last[i+1:]
	}
	n, err := strconv.Atoi(last)
	if err != nil {
		return 0, fmt.Errorf("parse issue number from %q: %w", s, err)
	}
	return n, nil
}
