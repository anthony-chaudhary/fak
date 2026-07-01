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

func guardReadRefusalCarryForward(auditPath, traceID, root string) []guardRefusalCarry {
	path := guardRefusalCarryForwardPath(auditPath)
	if path == "" {
		return nil
	}
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) == 0 {
		return nil
	}
	var file guardRefusalCarryForwardFile
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil
	}
	if file.Schema != guardRefusalCarryForwardSchema || strings.TrimSpace(file.TraceID) != strings.TrimSpace(traceID) {
		return nil
	}
	docs := guardReadReasonDocs(root)
	out := append([]guardRefusalCarry(nil), file.Refusals...)
	for i := range out {
		if strings.TrimSpace(out[i].Fix) == "" {
			out[i].Fix = guardReasonFix(out[i].Reason, docs)
		}
	}
	return out
}

func guardWriteRefusalCarryForward(j *journal.Journal, seq0 uint64, traceID, root string) error {
	if j == nil {
		return nil
	}
	auditPath := j.Path()
	if auditPath == "" {
		return nil
	}
	if err := j.Flush(); err != nil {
		return err
	}
	rows, err := journal.ReadRows(auditPath)
	if err != nil {
		return err
	}
	sessionRows := make([]journal.Row, 0, len(rows))
	for _, row := range rows {
		if row.Seq > seq0 {
			sessionRows = append(sessionRows, row)
		}
	}
	refusals := guardRefusalCarryForwardFromRows(sessionRows, traceID, guardReadReasonDocs(root), guardRefusalCarryForwardTopN)
	return guardWriteRefusalCarryForwardFile(auditPath, traceID, refusals, time.Now())
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
		return "redirect the write away from guarded kernel/agent files, split the command, or drop the guarded edit"
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
