// Command relativity-frame-loss folds captured Claude Code session transcripts
// into the master-frame vs subagent-frame comparison required by fak issue #6574.
//
// It is stdlib-only and reads nothing but transcripts. Run it as:
//
//	go run ./_scratch/relativity-6574 -homes 'C:\Users\<you>' -project C--work-fak -out fold.json
//
// Corpus definition (stated so a second reader can reproduce it):
//
//	For every home directory matching `.claude*` under -homes whose name does not
//	contain "DELETED", take <home>/projects/<project>. A SESSION qualifies iff the
//	directory <root>/<id> exists, contains a `subagents` subdirectory, and the master
//	transcript <root>/<id>.jsonl exists. Subagent transcripts are every `agent-*.jsonl`
//	found recursively under <root>/<id>, de-duplicated by base name (some sessions
//	carry a nested duplicate copy of their own directory).
package framevisibility

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------- transcript

type record struct {
	Type        string `json:"type"`
	IsSidechain bool   `json:"isSidechain"`
	Message     *struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"message"`
}

type block struct {
	Type    string          `json:"type"`
	Name    string          `json:"name"`
	Input   json.RawMessage `json:"input"`
	IsError bool            `json:"is_error"`
}

// blocks decodes message.content, which is either a JSON string (plain text) or
// an array of content blocks. A string yields a single synthetic text block.
func blocks(raw json.RawMessage) ([]block, bool) {
	if len(raw) == 0 {
		return nil, false
	}
	if raw[0] == '"' {
		return []block{{Type: "text"}}, true // string form: operator-authored text
	}
	var bs []block
	if err := json.Unmarshal(raw, &bs); err != nil {
		return nil, false
	}
	return bs, false
}

// ------------------------------------------------------------------- rubric

// spawnTools are the delegation verbs: a tool_use of one of these creates a
// subagent frame. Both the current (`Agent`) and legacy (`Task`) names count.
var spawnTools = map[string]bool{"Agent": true, "Task": true, "Workflow": true}

// mutatingTools write repository state directly.
var mutatingTools = map[string]bool{
	"Write": true, "Edit": true, "MultiEdit": true, "NotebookEdit": true,
}

// shellTools carry their mutation intent in a command string rather than in the
// tool name, so they are classified by matching mutatePattern against it.
var shellTools = map[string]bool{"Bash": true, "PowerShell": true}

// mutatePattern is the shell-command mutation predicate (rule R3). It is broad
// on purpose: the rubric is deliberately generous toward "decision-relevant", so
// the reported relevance fraction is an UPPER bound.
var mutatePattern = regexp.MustCompile(`(?i)(git\s+(commit|push|merge|rebase|reset|checkout|tag|revert|rm|mv|add|stash|cherry-pick)|dos\s+lease-lane|fak\s+(commit|sync)|Set-Content|Out-File|New-Item|Remove-Item|Copy-Item|Move-Item|Rename-Item|\brm\b|\bmv\b|\bcp\b|\bmkdir\b|\btouch\b|\btee\b|>>|[^>|]>[^>]|\bgo\s+(build|install|generate)\b|\bmake\b|\bnpm\s+(i|install|run)\b)`)

func commandOf(in json.RawMessage) string {
	if len(in) == 0 {
		return ""
	}
	var m struct {
		Command string `json:"command"`
	}
	_ = json.Unmarshal(in, &m)
	return m.Command
}

// masterRelevant applies the decision-relevance rubric to one master-frame
// event. `terminal` reports whether the assistant record holding this block
// carried no tool_use block at all (rule R5).
func masterRelevant(b block, terminal bool) bool {
	switch b.Type {
	case "tool_result":
		return b.IsError // R1: a failure the operator may have to unblock
	case "tool_use":
		if spawnTools[b.Name] {
			return true // R4: delegation is a control event
		}
		if mutatingTools[b.Name] {
			return true // R2: a direct repository state change
		}
		if shellTools[b.Name] {
			return mutatePattern.MatchString(commandOf(b.Input)) // R3
		}
		return false
	case "text":
		return terminal // R5: the master's answer turn, not progress narration
	}
	return false // thinking, images, everything else
}

// -------------------------------------------------------------------- counts

// Counts tracks event, relevance, and visibility metrics across master and subagent frames.
type Counts struct {
	Sessions int `json:"sessions"`
	Homes    int `json:"homes"`

	// Master frame.
	MasterEvents   int `json:"master_events"`
	MasterRelevant int `json:"master_relevant"`
	MasterToolUse  int `json:"master_tool_use"`
	MasterSpawns   int `json:"master_spawns"`
	MasterByRule   map[string]int
	MasterKind     map[string]int

	// Subagent frame.
	SubFiles       int `json:"subagent_files"`
	SubEvents      int `json:"subagent_events"`
	SubVisible     int `json:"subagent_visible"`
	SubCountedOnly int `json:"subagent_counted_only"`
	SubToolUse     int `json:"subagent_tool_use"`
	SubKind        map[string]int
}

func newCounts() *Counts {
	return &Counts{
		MasterByRule: map[string]int{},
		MasterKind:   map[string]int{},
		SubKind:      map[string]int{},
	}
}

func (c *Counts) add(o *Counts) {
	c.Sessions += o.Sessions
	c.MasterEvents += o.MasterEvents
	c.MasterRelevant += o.MasterRelevant
	c.MasterToolUse += o.MasterToolUse
	c.MasterSpawns += o.MasterSpawns
	c.SubFiles += o.SubFiles
	c.SubEvents += o.SubEvents
	c.SubVisible += o.SubVisible
	c.SubCountedOnly += o.SubCountedOnly
	c.SubToolUse += o.SubToolUse
	for k, v := range o.MasterByRule {
		c.MasterByRule[k] += v
	}
	for k, v := range o.MasterKind {
		c.MasterKind[k] += v
	}
	for k, v := range o.SubKind {
		c.SubKind[k] += v
	}
}

// ---------------------------------------------------------------- the folds

func scanner(path string) (*os.File, *bufio.Scanner, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 0, 1<<20), 64<<20) // transcript lines run to megabytes
	return f, s, nil
}

// foldMaster classifies every event in the master's own stream.
//
// Denominator: every content block of a non-sidechain assistant record, plus
// every tool_result block of a non-sidechain user record. Operator-authored user
// text (content encoded as a bare JSON string) is EXCLUDED — the operator wrote
// it, so it costs no reading attention.
func foldMaster(path string, c *Counts) error {
	f, s, err := scanner(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for s.Scan() {
		var r record
		if json.Unmarshal(s.Bytes(), &r) != nil || r.Message == nil || r.IsSidechain {
			continue
		}
		bs, isString := blocks(r.Message.Content)
		if isString {
			continue // operator-authored input
		}
		switch r.Type {
		case "assistant":
			terminal := true
			for _, b := range bs {
				if b.Type == "tool_use" {
					terminal = false
				}
			}
			for _, b := range bs {
				if b.Type == "tool_result" {
					continue // never appears on an assistant record
				}
				c.MasterEvents++
				c.MasterKind[b.Type]++
				if b.Type == "tool_use" {
					c.MasterToolUse++
					if spawnTools[b.Name] {
						c.MasterSpawns++
					}
				}
				if masterRelevant(b, terminal) {
					c.MasterRelevant++
					c.MasterByRule[ruleName(b, terminal)]++
				}
			}
		case "user":
			for _, b := range bs {
				if b.Type != "tool_result" {
					continue
				}
				c.MasterEvents++
				c.MasterKind["tool_result"]++
				if masterRelevant(b, false) {
					c.MasterRelevant++
					c.MasterByRule["R1_tool_error"]++
				}
			}
		}
	}
	return s.Err()
}

func ruleName(b block, terminal bool) string {
	switch {
	case b.Type == "tool_result":
		return "R1_tool_error"
	case b.Type == "tool_use" && spawnTools[b.Name]:
		return "R4_delegation"
	case b.Type == "tool_use" && mutatingTools[b.Name]:
		return "R2_file_write"
	case b.Type == "tool_use" && shellTools[b.Name]:
		return "R3_mutating_shell"
	case b.Type == "text" && terminal:
		return "R5_answer_turn"
	}
	return "unclassified"
}

// foldSubagent classifies every event in one subagent's own frame.
//
// Denominator: the spawn prompt (1), every content block of its assistant
// records, and every tool_result block of its user records.
//
// VISIBLE (reconstructible by a consumer reading only the master's stream):
//   - V1 the spawn prompt — the master authored it and it is echoed back.
//   - V2 every text block of the subagent's LAST assistant record — returned
//     verbatim to the master as the Agent tool_result content.
//
// COUNTED-ONLY: tool_use blocks. `toolUseResult.toolStats` gives the master four
// coarse category counters (read/search/bash/editFile), so the master can learn
// HOW MANY calls of a broad kind happened but never which, with what arguments,
// or with what outcome. Tracked separately as a generous upper bound.
func foldSubagent(path string, c *Counts) error {
	f, s, err := scanner(path)
	if err != nil {
		return err
	}
	defer f.Close()
	c.SubFiles++
	sawPrompt := false
	pendingText := 0 // text blocks of the most recent assistant record
	for s.Scan() {
		var r record
		if json.Unmarshal(s.Bytes(), &r) != nil || r.Message == nil {
			continue
		}
		bs, isString := blocks(r.Message.Content)
		if isString {
			if !sawPrompt && r.Type == "user" {
				sawPrompt = true
				c.SubEvents++
				c.SubKind["spawn_prompt"]++
				c.SubVisible++ // V1
			}
			continue
		}
		switch r.Type {
		case "assistant":
			// The previous assistant record is now known to be non-final.
			pendingText = 0
			for _, b := range bs {
				c.SubEvents++
				c.SubKind[b.Type]++
				switch b.Type {
				case "tool_use":
					c.SubToolUse++
					c.SubCountedOnly++
				case "text":
					pendingText++
				}
			}
		case "user":
			for _, b := range bs {
				if b.Type != "tool_result" {
					continue
				}
				c.SubEvents++
				c.SubKind["tool_result"]++
			}
		}
	}
	c.SubVisible += pendingText // V2: the returned blob
	return s.Err()
}

// ------------------------------------------------------------------- corpus

// SessionRow captures per-session visibility and relevance metrics.
type SessionRow struct {
	Home           string `json:"home"`
	Session        string `json:"session"`
	Subagents      int    `json:"subagents"`
	Spawns         int    `json:"spawns"`
	MasterEvents   int    `json:"master_events"`
	MasterRelevant int    `json:"master_relevant"`
	MasterToolUse  int    `json:"master_tool_use"`
	SubEvents      int    `json:"subagent_events"`
	SubVisible     int    `json:"subagent_visible"`
	SubCountedOnly int    `json:"subagent_counted_only"`
	SubToolUse     int    `json:"subagent_tool_use"`
}

func subagentFiles(dir string) []string {
	seen := map[string]string{}
	_ = filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil //nolint:nilerr // unreadable subtrees are skipped, not fatal
		}
		base := d.Name()
		if !strings.HasPrefix(base, "agent-") || !strings.HasSuffix(base, ".jsonl") {
			return nil
		}
		if _, dup := seen[base]; !dup {
			seen[base] = p
		}
		return nil
	})
	out := make([]string, 0, len(seen))
	for _, p := range seen {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// Report aggregates totals, rule breakdowns, and individual session rows.
type Report struct {
	Totals      *Counts        `json:"totals"`
	ByRule      map[string]int `json:"by_rule"`
	MasterKinds map[string]int `json:"master_kinds"`
	SubKinds    map[string]int `json:"sub_kinds"`
	Sessions    []SessionRow   `json:"sessions"`
}

// Fold measures what a reader confined to each master transcript can reconstruct
// about events recorded in descendant subagent transcripts.
func Fold(homesRoot, project string) (*Counts, []SessionRow, error) {
	entries, err := os.ReadDir(homesRoot)
	if err != nil {
		return nil, nil, err
	}
	total := newCounts()
	var rows []SessionRow
	homes := 0
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), ".claude") || strings.Contains(e.Name(), "DELETED") {
			continue
		}
		root := filepath.Join(homesRoot, e.Name(), "projects", project)
		sessions, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		used := false
		for _, session := range sessions {
			if !session.IsDir() {
				continue
			}
			dir := filepath.Join(root, session.Name())
			if _, err := os.Stat(filepath.Join(dir, "subagents")); err != nil {
				continue
			}
			master := filepath.Join(root, session.Name()+".jsonl")
			if _, err := os.Stat(master); err != nil {
				continue
			}
			c := newCounts()
			c.Sessions = 1
			if err := foldMaster(master, c); err != nil {
				continue
			}
			for _, sf := range subagentFiles(dir) {
				_ = foldSubagent(sf, c)
			}
			used = true
			total.add(c)
			rows = append(rows, SessionRow{Home: e.Name(), Session: session.Name(), Subagents: c.SubFiles, Spawns: c.MasterSpawns, MasterEvents: c.MasterEvents, MasterRelevant: c.MasterRelevant, MasterToolUse: c.MasterToolUse, SubEvents: c.SubEvents, SubVisible: c.SubVisible, SubCountedOnly: c.SubCountedOnly, SubToolUse: c.SubToolUse})
		}
		if used {
			homes++
		}
	}
	total.Homes = homes
	return total, rows, nil
}

// Run parses command-line flags, folds qualifying session transcripts, and writes the report.
func Run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("frame-visibility", flag.ContinueOnError)
	fs.SetOutput(stderr)
	homesRoot := fs.String("homes", `C:\Users\USER`, "directory holding the .claude* home directories")
	project := fs.String("project", "C--work-fak", "project slug under <home>/projects")
	out := fs.String("out", "fold.json", "machine-readable fold output")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	total, rows, err := Fold(*homesRoot, *project)
	if err != nil {
		fmt.Fprintln(stderr, "fold:", err)
		return 1
	}

	pct := func(n, m int) float64 {
		if m == 0 {
			return 0
		}
		return float64(n) / float64(m) * 100
	}
	fmt.Fprintf(stdout, "homes=%d sessions=%d subagent_files=%d spawns=%d\n", total.Homes, total.Sessions, total.SubFiles, total.MasterSpawns)
	fmt.Fprintf(stdout, "SUBAGENT VISIBILITY  : %d of %d  (%.2f%%)\n", total.SubVisible, total.SubEvents, pct(total.SubVisible, total.SubEvents))
	fmt.Fprintf(stdout, "  + counted-only     : %d of %d  (%.2f%%) generous upper bound\n", total.SubVisible+total.SubCountedOnly, total.SubEvents, pct(total.SubVisible+total.SubCountedOnly, total.SubEvents))
	fmt.Fprintf(stdout, "MASTER RELEVANCE     : %d of %d  (%.2f%%)\n", total.MasterRelevant, total.MasterEvents, pct(total.MasterRelevant, total.MasterEvents))
	fmt.Fprintf(stdout, "tool calls classified: master=%d subagent=%d total=%d\n", total.MasterToolUse, total.SubToolUse, total.MasterToolUse+total.SubToolUse)
	fmt.Fprintf(stdout, "master by rule: %v\n", total.MasterByRule)
	fmt.Fprintf(stdout, "master kinds  : %v\n", total.MasterKind)
	fmt.Fprintf(stdout, "sub kinds     : %v\n", total.SubKind)

	blob := Report{Totals: total, ByRule: total.MasterByRule, MasterKinds: total.MasterKind, SubKinds: total.SubKind, Sessions: rows}
	buf, _ := json.MarshalIndent(blob, "", "  ")
	if err := os.WriteFile(*out, buf, 0o644); err != nil {
		fmt.Fprintln(stderr, "write:", err)
		return 1
	}
	fmt.Fprintln(stdout, "wrote", *out)
	return 0
}
