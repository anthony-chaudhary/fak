package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuefanout"
)

// issueDraftSchema identifies the machine-readable draft result.
const issueDraftSchema = "fak.issue-draft.v1"

// issueDraftMarkerKeyRE is the frozen marker-key grammar from
// internal/issuepolicy (contract.go:80): `<!-- fak-<domain>-key: <value> -->`.
// A draft is "keyed" the moment this marker is in its body, because the
// create-time contract and the fanout rerun-dedupe scan both read it.
var issueDraftMarkerKeyRE = regexp.MustCompile(`<!--\s*fak-[A-Za-z0-9_-]+-key:\s*([^>\s]+)\s*-->`)

// issueDraftLaneSanitizeRE strips anything outside [A-Za-z0-9_-] from a lane.
var issueDraftLaneSanitizeRE = regexp.MustCompile(`[^A-Za-z0-9_-]`)

// issueDraftDashRunRE collapses a run of sanitized dashes so a lane like
// dis patch/../x yields a single well-formed dis-patch-x, not a dash run.
var issueDraftDashRunRE = regexp.MustCompile(`-{2,}`)

// issueDraftGHRunner executes one gh invocation and reports stdout, stderr,
// and an ok flag (true when the process exited 0). It mirrors
// issuefanout.Runner so the leaf check and this verb share a shape; the
// package-level var is injectable so tests never touch a real gh.
var issueDraftGHRunner issueCreateRunner = runTaskHandoffGH

// issueDraftDedupe is the backlog-check sub-record of the JSON result.
type issueDraftDedupe struct {
	Checked bool   `json:"checked"`
	Scanned int    `json:"scanned"`
	Cap     int    `json:"cap"`
	Refused bool   `json:"refused"`
	Error   string `json:"error,omitempty"`
}

// issueDraftResult is the --json shape for one authoring run.
type issueDraftResult struct {
	Schema  string           `json:"schema"`
	Lane    string           `json:"lane"`
	Key     string           `json:"key"`
	Marker  string           `json:"marker"`
	Slug    string           `json:"slug"`
	Path    string           `json:"path,omitempty"`
	Wrote   bool             `json:"wrote"`
	DryRun  bool             `json:"dry_run"`
	Dedupe  issueDraftDedupe `json:"dedupe"`
	Title   string           `json:"title,omitempty"`
	Body    string           `json:"body,omitempty"`
	Message string           `json:"message,omitempty"`
}

// runIssueDraft scaffolds a contract-valid issue draft and mints a
// freshly-minted, optionally backlog-checked `fak-<lane>-key` marker. It
// writes nothing to GitHub: it produces a local markdown file (or stdout on
// --dry-run) that `fak-dev issue create --body-file` will later accept.
//
// Exit codes: 0 ok, 2 bad flags/io, 3 duplicate key in the backlog,
// 1 encode failure.
func runIssueDraft(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue draft", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lane := fs.String("lane", "", "issue domain lane; forms the marker key fak-<lane>-key (required)")
	title := fs.String("title", "", "optional draft title")
	slug := fs.String("slug", "", "deterministic key value (default: derived from --title, else a timestamp token)")
	out := fs.String("out", ".issue-scratch", "directory to write the draft into (filename = <slug>.md)")
	dedupeChecked := fs.Bool("dedupe-checked", false, "scan the backlog and refuse if the minted key already appears in an issue body")
	dedupeCap := fs.Int("dedupe-cap", 0, "bound the backlog scan size (0 = issuefanout.DefaultDedupeCap)")
	repo := fs.String("repo", "", "owner/name for gh (default: gh infers from cwd)")
	dryRun := fs.Bool("dry-run", false, "render the draft + key without writing a file or calling gh")
	asJSON := fs.Bool("json", false, "emit the machine-readable result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue draft: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	rawLane := strings.TrimSpace(*lane)
	if rawLane == "" {
		fmt.Fprintln(stderr, "fak-dev issue draft: --lane is required")
		return 2
	}
	safeLane := issueDraftLaneSanitizeRE.ReplaceAllString(rawLane, "-")
	safeLane = issueDraftDashRunRE.ReplaceAllString(safeLane, "-")
	safeLane = strings.Trim(safeLane, "-")
	if safeLane == "" {
		fmt.Fprintln(stderr, "fak-dev issue draft: --lane must contain at least one [A-Za-z0-9_-] character")
		return 2
	}
	if *dedupeCap <= 0 {
		*dedupeCap = issuefanout.DefaultDedupeCap
	}

	draftTitle := strings.TrimSpace(*title)
	keyValue := strings.TrimSpace(*slug)
	if keyValue == "" {
		keyValue = issueDraftSlug(draftTitle)
	}
	keyValue = sanitizeIssueDraftKey(keyValue)

	key := "fak-" + safeLane + "-key"
	marker := fmt.Sprintf("<!-- %s: %s -->", key, keyValue)

	result := &issueDraftResult{
		Schema: issueDraftSchema,
		Lane:   safeLane,
		Key:    key,
		Marker: marker,
		Slug:   keyValue,
		DryRun: *dryRun,
		Title:  draftTitle,
	}
	result.Dedupe = issueDraftDedupe{Checked: *dedupeChecked, Cap: *dedupeCap}

	// Backlog check (opt-in): refuse a key already present in any issue body.
	// A gh failure fails OPEN - an offline authoring pass must still work.
	if *dedupeChecked && !*dryRun {
		existing, err := issueDraftFetchExisting(*repo, *dedupeCap)
		result.Dedupe.Scanned = len(existing)
		if err != nil {
			result.Dedupe.Error = err.Error()
			fmt.Fprintf(stderr, "fak-dev issue draft: warning: backlog dedupe scan failed (%v); proceeding without it\n", err)
		} else if n, seen := issueDraftSeenIn(existing, keyValue); seen {
			result.Dedupe.Refused = true
			result.Message = fmt.Sprintf("ISSUE_DUPLICATE_KEY: key %s already appears in issue #%d", keyValue, n)
			if *asJSON {
				encodeJSONOrFail(stdout, stderr, result, "fak-dev issue draft")
			} else {
				fmt.Fprintf(stderr, "fak-dev issue draft: %s\n", result.Message)
			}
			return 3
		}
	}

	body := issueDraftBody(draftTitle, safeLane, marker)
	result.Body = body

	if *dryRun {
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, result, "fak-dev issue draft")
		}
		fmt.Fprintf(stdout, "%s\n\n%s\n", marker, body)
		fmt.Fprintf(stdout, "# key: %s\n", keyValue)
		return 0
	}

	path, err := issueDraftWrite(*out, keyValue, body)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue draft: %v\n", err)
		return 2
	}
	result.Path = path
	result.Wrote = true
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, result, "fak-dev issue draft")
	}
	fmt.Fprintln(stdout, path)
	fmt.Fprintf(stdout, "# key: %s\n", keyValue)
	return 0
}

// issueDraftWrite mkdir -p's out and writes <out>/<slug>.md.
func issueDraftWrite(out, slug, body string) (string, error) {
	if strings.TrimSpace(out) == "" {
		out = ".issue-scratch"
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return "", fmt.Errorf("create --out dir: %w", err)
	}
	path := filepath.Join(out, slug+".md")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		return "", fmt.Errorf("write draft: %w", err)
	}
	return path, nil
}

// issueDraftFetchExisting composes the bounded backlog scan argv via
// issuefanout.ListExistingArgs and runs it through the injectable gh runner.
func issueDraftFetchExisting(repo string, cap int) ([]issuefanout.Issue, error) {
	args := issuefanout.ListExistingArgs(repo, cap)
	run := issueDraftGHRunner
	if run == nil {
		run = runTaskHandoffGH
	}
	stdout, errOut, ok := run(args)
	if !ok {
		return nil, fmt.Errorf("gh issue list failed: %s", strings.TrimSpace(errOut))
	}
	var issues []issuefanout.Issue
	if err := json.Unmarshal([]byte(stdout), &issues); err != nil {
		return nil, fmt.Errorf("parse gh issue list: %w", err)
	}
	return issues, nil
}

// issueDraftSeenIn returns the first existing issue whose body already carries
// keyValue. issuefanout.seenIn is unexported, so this substring scan (the same
// marker-key match the fanout rerun contract uses) is mirrored locally.
func issueDraftSeenIn(existing []issuefanout.Issue, keyValue string) (int, bool) {
	for _, issue := range existing {
		if strings.Contains(issue.Body, keyValue) {
			return issue.Number, true
		}
	}
	return 0, false
}

// issueDraftBody renders a short, contract-valid draft: the marker line first,
// then exactly one `## Core through-line` and one `## Gold-plating boundary`
// (issueCreateShiftLeftScope requires one of each), plus the standard witness
// headings. The author fills in the scaffold prose.
func issueDraftBody(title, lane, marker string) string {
	var b strings.Builder
	b.WriteString(marker)
	b.WriteString("\n\n")
	if title != "" {
		fmt.Fprintf(&b, "# %s\n\n", title)
	}
	b.WriteString("## Core through-line\n\n")
	b.WriteString("TODO: name the shortest change -> real seam -> observable outcome -> witness path.\n\n")
	b.WriteString("## Gold-plating boundary\n\n")
	b.WriteString("TODO: name tempting work this outcome and witness still work without (or state why none exists).\n\n")
	b.WriteString("## Definition of done\n\n")
	b.WriteString("- [ ] TODO: the specific artifact/behavior that must exist\n")
	b.WriteString("- [ ] TODO: the acceptance check that turns green\n\n")
	b.WriteString("## Witness\n\n")
	b.WriteString("TODO: the exact command + observable output that proves it.\n\n")
	b.WriteString("## Acceptance gate\n\n")
	b.WriteString("TODO: the single check that must pass after the last edit.\n\n")
	b.WriteString("## Lane\n\n")
	b.WriteString(lane)
	b.WriteString("\n")
	return b.String()
}

// issueDraftSlug derives a deterministic key value: a slug of the title when
// present, else a timestamp token so concurrent drafts never collide.
func issueDraftSlug(title string) string {
	title = strings.TrimSpace(title)
	if title == "" {
		return "draft-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	lower := strings.ToLower(title)
	var b strings.Builder
	prevDash := false
	for _, r := range lower {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			prevDash = false
		default:
			if !prevDash {
				b.WriteRune('-')
				prevDash = true
			}
		}
	}
	slug := strings.Trim(b.String(), "-")
	if slug == "" {
		return "draft-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	if len(slug) > 100 {
		slug = strings.Trim(slug[:100], "-")
	}
	return slug
}

// sanitizeIssueDraftKey keeps the key value inside the marker grammar
// ([^>\s]+) and the contract key RE
// ([A-Za-z0-9][A-Za-z0-9._:/-]{0,119}).
func sanitizeIssueDraftKey(v string) string {
	v = strings.TrimSpace(v)
	v = strings.NewReplacer(" ", "-", "\t", "-", "\n", "-", "\r", "-").Replace(v)
	v = issueDraftLaneSanitizeRE.ReplaceAllString(v, "-")
	v = issueDraftDashRunRE.ReplaceAllString(v, "-")
	v = strings.Trim(v, "-")
	if v == "" {
		return "draft-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	if len(v) > 119 {
		v = strings.Trim(v[:119], "-")
	}
	return v
}
