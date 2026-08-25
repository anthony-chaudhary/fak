package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/negframe"
)

const (
	guardRefusalCarryForwardSchema = "fak.guard.refusal-carry-forward.v1"
	guardRefusalCarryForwardTopN   = 3
)

type guardRefusalCarry struct {
	Reason string `json:"reason"`
	Count  int    `json:"count"`
	Fix    string `json:"fix,omitempty"`
}

type guardReasonDoc struct {
	Summary string
	Fix     string
}

type guardRefusalCarryForwardFile struct {
	Schema        string              `json:"schema"`
	TraceID       string              `json:"trace_id"`
	AuditPath     string              `json:"audit_path"`
	WrittenAtUnix int64               `json:"written_at_unix"`
	Refusals      []guardRefusalCarry `json:"refusals"`
}

func guardReadPriorAuditEntries(auditPath string) (string, []os.DirEntry, bool) {
	dir := guardPriorAuditDir(auditPath)
	if dir == "" {
		return "", nil, false
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", nil, false
	}
	return dir, entries, true
}

type guardPriorAuditEntry struct {
	path  string
	entry os.DirEntry
}

func guardFilteredPriorAuditEntries(auditPath, current, suffix string) ([]guardPriorAuditEntry, bool) {
	dir, entries, ok := guardReadPriorAuditEntries(auditPath)
	if !ok {
		return nil, false
	}
	current = filepath.Clean(current)
	filtered := make([]guardPriorAuditEntry, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), suffix) {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		if current != "." && filepath.Clean(path) == current {
			continue
		}
		filtered = append(filtered, guardPriorAuditEntry{path: path, entry: entry})
	}
	return filtered, true
}

func guardReadRefusalCarryForward(auditPath, traceID, root string) []guardRefusalCarry {
	path := guardRefusalCarryForwardPath(auditPath)
	if path == "" {
		return nil
	}
	out, _, ok := guardLoadRefusalCarryForwardFile(path, traceID, root)
	if !ok {
		return nil
	}
	return out
}

func guardReadPriorRefusalCarryForward(auditPath, traceID, root string) []guardRefusalCarry {
	path := guardRefusalCarryForwardPath(auditPath)
	if path != "" {
		if out, _, ok := guardLoadRefusalCarryForwardFile(path, traceID, root); ok {
			return out
		}
	}
	if out, ok := guardReadLatestRefusalCarryForwardSidecar(auditPath, traceID, root); ok {
		return out
	}
	if out, ok := guardReadLatestRefusalCarryForwardJournal(auditPath, traceID, root); ok {
		return out
	}
	return nil
}

func guardLoadRefusalCarryForwardFile(path, traceID, root string) ([]guardRefusalCarry, int64, bool) {
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) == 0 {
		return nil, 0, false
	}
	var file guardRefusalCarryForwardFile
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, 0, false
	}
	if file.Schema != guardRefusalCarryForwardSchema || strings.TrimSpace(file.TraceID) != strings.TrimSpace(traceID) {
		return nil, 0, false
	}
	docs := guardReadReasonDocs(root)
	out := append([]guardRefusalCarry(nil), file.Refusals...)
	for i := range out {
		if strings.TrimSpace(out[i].Fix) == "" {
			out[i].Fix = guardReasonFix(out[i].Reason, docs)
		}
	}
	return out, file.WrittenAtUnix, true
}

func guardReadLatestRefusalCarryForwardSidecar(auditPath, traceID, root string) ([]guardRefusalCarry, bool) {
	entries, ok := guardFilteredPriorAuditEntries(auditPath, guardRefusalCarryForwardPath(auditPath), ".jsonl.refusals.json")
	if !ok {
		return nil, false
	}
	type candidate struct {
		path string
		when int64
		out  []guardRefusalCarry
	}
	var candidates []candidate
	for _, ent := range entries {
		out, when, ok := guardLoadRefusalCarryForwardFile(ent.path, traceID, root)
		if !ok {
			continue
		}
		if when == 0 {
			if info, err := ent.entry.Info(); err == nil {
				when = info.ModTime().Unix()
			}
		}
		candidates = append(candidates, candidate{path: ent.path, when: when, out: out})
	}
	if len(candidates) == 0 {
		return nil, false
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].when != candidates[j].when {
			return candidates[i].when > candidates[j].when
		}
		return candidates[i].path > candidates[j].path
	})
	return candidates[0].out, true
}

func guardReadLatestRefusalCarryForwardJournal(auditPath, traceID, root string) ([]guardRefusalCarry, bool) {
	entries, ok := guardFilteredPriorAuditEntries(auditPath, auditPath, ".jsonl")
	if !ok {
		return nil, false
	}
	type candidate struct {
		path string
		mod  time.Time
	}
	var candidates []candidate
	for _, ent := range entries {
		info, err := ent.entry.Info()
		if err != nil {
			continue
		}
		candidates = append(candidates, candidate{path: ent.path, mod: info.ModTime()})
	}
	if len(candidates) == 0 {
		return nil, false
	}
	sort.Slice(candidates, func(i, j int) bool {
		if !candidates[i].mod.Equal(candidates[j].mod) {
			return candidates[i].mod.After(candidates[j].mod)
		}
		return candidates[i].path > candidates[j].path
	})
	docs := guardReadReasonDocs(root)
	for _, cand := range candidates {
		// Segment-aware (#6488): the carry-forward rolls up a trace's refusals, and a
		// trace that predates a rotation cut must still be findable.
		rows, err := journal.ReadAllSegments(cand.path)
		if err != nil {
			continue
		}
		rows = journal.WithoutCutAnchors(rows)
		if len(rows) == 0 {
			return nil, true
		}
		if !guardRowsContainTrace(rows, traceID) {
			continue
		}
		return guardRefusalCarryForwardFromRows(rows, traceID, docs, guardRefusalCarryForwardTopN), true
	}
	return nil, false
}

func guardPriorAuditDir(auditPath string) string {
	auditPath = strings.TrimSpace(auditPath)
	if auditPath != "" {
		return filepath.Dir(auditPath)
	}
	return guardAuditDir(findRepoRoot("."))
}

func guardRowsContainTrace(rows []journal.Row, traceID string) bool {
	traceID = strings.TrimSpace(traceID)
	for _, row := range rows {
		if traceID == "" || strings.TrimSpace(row.TraceID) == traceID {
			return true
		}
	}
	return false
}

func guardWriteRefusalCarryForwardAndReturn(j *journal.Journal, seq0 uint64, traceID, root string) ([]guardRefusalCarry, error) {
	if j == nil {
		return nil, nil
	}
	auditPath := j.Path()
	if auditPath == "" {
		return nil, nil
	}
	if err := j.Flush(); err != nil {
		return nil, err
	}
	// Segment-aware (#6488): a session long enough to rotate would otherwise lose the
	// part of itself that sits in the sealed segment, even though seq0 still selects it.
	rows, err := journal.ReadAllSegments(auditPath)
	if err != nil {
		return nil, err
	}
	rows = journal.WithoutCutAnchors(rows)
	sessionRows := make([]journal.Row, 0, len(rows))
	for _, row := range rows {
		if row.Seq > seq0 {
			sessionRows = append(sessionRows, row)
		}
	}
	refusals := guardRefusalCarryForwardFromRows(sessionRows, traceID, guardReadReasonDocs(root), guardRefusalCarryForwardTopN)
	return refusals, guardWriteRefusalCarryForwardFile(auditPath, traceID, refusals, time.Now())
}

func guardWriteRefusalCarryForwardFile(auditPath, traceID string, refusals []guardRefusalCarry, now time.Time) error {
	path := guardRefusalCarryForwardPath(auditPath)
	if path == "" {
		return nil
	}
	file := guardRefusalCarryForwardFile{
		Schema:        guardRefusalCarryForwardSchema,
		TraceID:       strings.TrimSpace(traceID),
		AuditPath:     auditPath,
		WrittenAtUnix: now.Unix(),
		Refusals:      refusals,
	}
	raw, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func guardRefusalCarryForwardPath(auditPath string) string {
	auditPath = strings.TrimSpace(auditPath)
	if auditPath == "" {
		return ""
	}
	return auditPath + ".refusals.json"
}

func guardRefusalCarryForwardFromRows(rows []journal.Row, traceID string, docs map[string]guardReasonDoc, n int) []guardRefusalCarry {
	traceID = strings.TrimSpace(traceID)
	if n <= 0 {
		return nil
	}
	type bucket struct {
		reason string
		count  int
		order  int
	}
	buckets := map[string]*bucket{}
	order := 0
	for i := len(rows) - 1; i >= 0; i-- {
		row := rows[i]
		if traceID != "" && strings.TrimSpace(row.TraceID) != traceID {
			continue
		}
		reason := strings.TrimSpace(row.Reason)
		if reason == "" || !guardRowIsRefusal(row) {
			continue
		}
		b := buckets[reason]
		if b == nil {
			b = &bucket{reason: reason, order: order}
			buckets[reason] = b
			order++
		}
		b.count++
	}
	if len(buckets) == 0 {
		return nil
	}
	ordered := make([]*bucket, 0, len(buckets))
	for _, b := range buckets {
		ordered = append(ordered, b)
	}
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].count != ordered[j].count {
			return ordered[i].count > ordered[j].count
		}
		return ordered[i].order < ordered[j].order
	})
	if len(ordered) > n {
		ordered = ordered[:n]
	}
	out := make([]guardRefusalCarry, 0, len(ordered))
	for _, b := range ordered {
		out = append(out, guardRefusalCarry{Reason: b.reason, Count: b.count, Fix: guardReasonFix(b.reason, docs)})
	}
	return out
}

func guardRowIsRefusal(row journal.Row) bool {
	verdict := strings.ToUpper(strings.TrimSpace(row.Verdict))
	kind := strings.ToUpper(strings.TrimSpace(row.Kind))
	return verdict == "DENY" || verdict == "QUARANTINE" || kind == "DENY" || kind == "RESULT_DENY" || kind == "QUARANTINE"
}

func formatGuardRefusalCarryForward(items []guardRefusalCarry) string {
	if len(items) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("  prior run  : known refusal(s) from the last session\n")
	for _, item := range items {
		if strings.TrimSpace(item.Reason) == "" {
			continue
		}
		fmt.Fprintf(&b, "    - %s x%d", item.Reason, item.Count)
		if fix := strings.TrimSpace(item.Fix); fix != "" {
			fmt.Fprintf(&b, " — fix: %s", fix)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

func guardRecoveryPrompt(items []guardRefusalCarry) string {
	if len(items) == 0 {
		return ""
	}
	fragments := []negframe.Fragment{
		negframe.Fak("[fak] resume recovery: the previous guarded run recorded capability-floor refusal(s). Treat this resumed turn as recovery/debugging and revise the call from evidence. Clear the blocker or choose an allowed alternative, then continue with a revised call. Prior refusal(s):"),
	}
	wrote := false
	for _, item := range items {
		if strings.TrimSpace(item.Reason) == "" {
			continue
		}
		separator := " "
		if wrote {
			separator = "; "
		}
		fragments = append(fragments, negframe.Fak(fmt.Sprintf("%s%s x%d", separator, item.Reason, item.Count)))
		if fix := strings.TrimSpace(item.Fix); fix != "" {
			// Reason fixes can be operator-authored; retain them byte-identically.
			fragments = append(fragments, negframe.Fak(" (fix: "), negframe.Opaque(fix), negframe.Fak(")"))
		}
		wrote = true
	}
	if !wrote {
		return ""
	}
	fragments = append(fragments, negframe.Fak(". Keep fak guard wrapped after the blocker is cleared."))
	// #3568 lever (#5365): the resume-recovery prompt is the SECOND injected-directive emit
	// site, so it routes through the same gated seam as the SessionStart affordance rather than
	// calling ReframeFakOnly unconditionally. With FAK_ABLATE=negframe_reframe this ships
	// #3546's control arm the RAW negative-framed prose instead of quietly reframing on both.
	//
	// The telemetry row is dropped HERE by design: this composes in the `fak guard` parent
	// before the child is spawned, and the child's SessionStart BEGINS the per-session journal
	// by truncating it (guardNegframeBegin). A row appended at this point would be discarded a
	// moment later, so this site honours the arm without seeding a row the boundary eats.
	prompt, _ := guardNegframeReframe(fragments...)
	return prompt
}

var guardReasonHeaderRE = regexp.MustCompile(`^\[reasons\.([A-Z0-9_]+)\]`)

func guardReadReasonDocs(root string) map[string]guardReasonDoc {
	root = strings.TrimSpace(root)
	if root == "" {
		root = guardFindReasonRoot()
	}
	if root == "" {
		return nil
	}
	raw, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		return nil
	}
	docs := map[string]guardReasonDoc{}
	current := ""
	for _, rawLine := range strings.Split(string(raw), "\n") {
		line := strings.TrimSpace(rawLine)
		if m := guardReasonHeaderRE.FindStringSubmatch(line); m != nil {
			current = m[1]
			if _, ok := docs[current]; !ok {
				docs[current] = guardReasonDoc{}
			}
			continue
		}
		if current == "" || strings.HasPrefix(line, "[") {
			if strings.HasPrefix(line, "[") {
				current = ""
			}
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		text := guardTomlStringValue(val)
		doc := docs[current]
		switch strings.TrimSpace(key) {
		case "summary":
			doc.Summary = text
		case "fix":
			doc.Fix = text
		}
		docs[current] = doc
	}
	return docs
}

func guardReasonFix(reason string, docs map[string]guardReasonDoc) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		return ""
	}
	if doc, ok := docs[reason]; ok && strings.TrimSpace(doc.Fix) != "" {
		return strings.TrimSpace(doc.Fix)
	}
	switch reason {
	case "DEFAULT_DENY":
		return "choose an allowed tool or add a deliberate policy allow rule before retrying"
	case "POLICY_BLOCK":
		return "choose an allowed alternative, or change the policy intentionally if this tool should be permitted"
	case "SELF_MODIFY":
		return "the write reaches a guarded trust-critical tree (internal/adjudicator|kernel|abi|shipgate, dos.toml, .dos/); ordinary docs/notes/workspace writes are NOT blocked — target an unguarded path, split a compound command to isolate the guarded part, or drop it; a genuinely required self-edit is operator-gated, so route it to an operator or worktree-isolated path (#1334) and keep unrelated work moving"
	case "MALFORMED":
		return "repair the tool arguments to the declared schema, then retry"
	case "MISROUTE":
		return "pick the tool that matches the intended effect and retry with the expected argument shape"
	case "RATE_LIMITED":
		return "wait for the named limit to clear, then retry"
	case "LEASE_HELD":
		return "wait for the conflicting lease or choose a disjoint file tree"
	case "SECRET_EXFIL", "RESULT_SECRET_DISCOVERED":
		return "remove or redact the secret-shaped content before retrying"
	case "UNWITNESSED":
		return "supply the independent witness the gate asked for, then retry"
	case "OVERSIZE":
		return "shrink, page, or summarize the payload before admitting it to context"
	case "UNKNOWN_TOOL":
		return "use a tool exposed by the current harness or update the policy/harness configuration"
	}
	return "inspect the refusal reason, choose an allowed alternative, and retry only after the named blocker is cleared"
}

func guardTomlStringValue(raw string) string {
	raw = strings.TrimSpace(raw)
	if i := strings.IndexByte(raw, '#'); i >= 0 {
		raw = strings.TrimSpace(raw[:i])
	}
	if s, err := strconv.Unquote(raw); err == nil {
		return s
	}
	return strings.Trim(raw, `"`)
}

func guardFindReasonRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	dir, err := filepath.Abs(wd)
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			return dir
		}
		next := filepath.Dir(dir)
		if next == dir {
			return ""
		}
		dir = next
	}
}
