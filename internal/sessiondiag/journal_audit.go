package sessiondiag

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

const JournalAuditSchema = "fak.sessiondiag.journal-audit.v1"

const (
	JournalVerdictGreen = "green"
	JournalVerdictRed   = "red"

	JournalStatusAdvanced                    = "advanced"
	JournalStatusMissingTranscript           = "missing_transcript"
	JournalStatusPresentNoPostLaunchProgress = "present_no_post_launch_progress"
)

// JournalLaunchIdentity is one provider-session identity observed at SessionStart.
// Callers source these rows from the live resume_identity.jsonl authority.
type JournalLaunchIdentity struct {
	Identity string
	Trace    string
	LaunchAt time.Time
	Provider string
	Account  string
	Via      string
	Source   string
}

// JournalAuditOptions makes the filesystem shell reproducible in tests. UserHome is
// the root under which all .claude* and .codex* homes are discovered. The explicit
// provider homes are additive and normally carry CLAUDE_CONFIG_DIR/CODEX_HOME.
type JournalAuditOptions struct {
	Now             time.Time
	Window          time.Duration
	IdentityPath    string
	Identities      []JournalLaunchIdentity
	UserHome        string
	CodexHome       string
	ClaudeConfigDir string
	AuthorityErrors []JournalAuthorityError
}

type JournalAuditWindow struct {
	Duration string `json:"duration"`
	Start    string `json:"start"`
	End      string `json:"end"`
}

type JournalRoot struct {
	Provider   string `json:"provider"`
	Path       string `json:"path"`
	Provenance string `json:"provenance"`
}

type JournalAuthorityError struct {
	Provider string `json:"provider,omitempty"`
	Path     string `json:"path"`
	Code     string `json:"code"`
	Detail   string `json:"detail,omitempty"`
}

type JournalIdentityProvenance struct {
	Path     string `json:"path"`
	Via      string `json:"via,omitempty"`
	Account  string `json:"account,omitempty"`
	Provider string `json:"provider,omitempty"`
	Source   string `json:"source,omitempty"`
}

type JournalCursor struct {
	ID   string `json:"id"`
	At   string `json:"at"`
	Kind string `json:"kind"`
}

type JournalAuditRow struct {
	Identity           string                    `json:"identity"`
	Trace              string                    `json:"trace,omitempty"`
	LaunchAt           string                    `json:"launch_at"`
	Provider           string                    `json:"provider"`
	IdentityProvenance JournalIdentityProvenance `json:"identity_provenance"`
	TranscriptPath     string                    `json:"transcript_path,omitempty"`
	BaselineCursor     *JournalCursor            `json:"baseline_cursor,omitempty"`
	PostLaunchCursor   *JournalCursor            `json:"post_launch_cursor,omitempty"`
	Status             string                    `json:"status"`
}

type JournalAuditCounts struct {
	Identities                  int            `json:"identities"`
	ExactTranscripts            int            `json:"exact_transcripts"`
	Advanced                    int            `json:"advanced"`
	MissingTranscript           int            `json:"missing_transcript"`
	PresentNoPostLaunchProgress int            `json:"present_no_post_launch_progress"`
	ExcludedUnsupportedProvider int            `json:"excluded_unsupported_provider"`
	AuthorityErrors             int            `json:"authority_errors"`
	ByProvider                  map[string]int `json:"by_provider"`
}

type JournalAuditReport struct {
	Schema          string                  `json:"schema"`
	ObservedAt      string                  `json:"observed_at"`
	Window          JournalAuditWindow      `json:"window"`
	Verdict         string                  `json:"verdict"`
	Summary         string                  `json:"summary"`
	IdentityJournal string                  `json:"identity_journal"`
	Roots           []JournalRoot           `json:"roots"`
	Counts          JournalAuditCounts      `json:"counts"`
	Rows            []JournalAuditRow       `json:"rows"`
	AuthorityErrors []JournalAuthorityError `json:"authority_errors,omitempty"`
}

type journalEvidence struct {
	provider string
	path     string
	cursors  []journalEvidenceCursor
}

type journalEvidenceCursor struct {
	id   string
	kind string
	at   time.Time
	path string
}

// AuditRecentLaunches exact-joins recent resume-identity rows to every local Claude
// project transcript and Codex rollout. It is read-only: provider stores are opened
// only for scanning, and no cursor or receipt is written back.
func AuditRecentLaunches(opt JournalAuditOptions) JournalAuditReport {
	now := opt.Now.UTC()
	if now.IsZero() {
		now = time.Now().UTC()
	}
	window := opt.Window
	if window <= 0 {
		window = 24 * time.Hour
	}
	cutoff := now.Add(-window)
	rep := JournalAuditReport{
		Schema:          JournalAuditSchema,
		ObservedAt:      now.Format(time.RFC3339Nano),
		Window:          JournalAuditWindow{Duration: window.String(), Start: cutoff.Format(time.RFC3339Nano), End: now.Format(time.RFC3339Nano)},
		Verdict:         JournalVerdictGreen,
		IdentityJournal: opt.IdentityPath,
		Counts:          JournalAuditCounts{ByProvider: map[string]int{}},
		Rows:            []JournalAuditRow{},
		AuthorityErrors: append([]JournalAuthorityError(nil), opt.AuthorityErrors...),
	}

	latest := make(map[string]JournalLaunchIdentity)
	order := make(map[string]int)
	for i, launch := range opt.Identities {
		launch.Identity = strings.TrimSpace(launch.Identity)
		launch.Trace = strings.TrimSpace(launch.Trace)
		launch.Provider = normalizeJournalProvider(launch.Provider)
		launch.LaunchAt = launch.LaunchAt.UTC()
		if launch.Identity == "" || launch.LaunchAt.IsZero() {
			rep.AuthorityErrors = append(rep.AuthorityErrors, JournalAuthorityError{Path: opt.IdentityPath, Code: "INVALID_IDENTITY_ROW", Detail: "identity or launch timestamp is missing"})
			continue
		}
		if launch.LaunchAt.After(now) {
			rep.AuthorityErrors = append(rep.AuthorityErrors, JournalAuthorityError{Path: opt.IdentityPath, Code: "FUTURE_IDENTITY_ROW", Detail: "launch timestamp is after the audit window"})
			continue
		}
		if launch.LaunchAt.Before(cutoff) {
			continue
		}
		if launch.Provider != "" && launch.Provider != "claude" && launch.Provider != "codex" {
			rep.Counts.ExcludedUnsupportedProvider++
			continue
		}
		key := journalIdentityKey(launch.Identity)
		prev, ok := latest[key]
		if !ok || launch.LaunchAt.After(prev.LaunchAt) || (launch.LaunchAt.Equal(prev.LaunchAt) && i > order[key]) {
			latest[key] = launch
			order[key] = i
		}
	}

	roots, rootErrors := discoverJournalRoots(opt.UserHome, opt.CodexHome, opt.ClaudeConfigDir)
	rep.Roots = roots
	rep.AuthorityErrors = append(rep.AuthorityErrors, rootErrors...)
	targets := make(map[string]bool, len(latest))
	for key := range latest {
		targets[key] = true
	}
	evidence, scanErrors := collectJournalEvidence(roots, targets)
	rep.AuthorityErrors = append(rep.AuthorityErrors, scanErrors...)

	keys := make([]string, 0, len(latest))
	for key := range latest {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := latest[keys[i]], latest[keys[j]]
		if !a.LaunchAt.Equal(b.LaunchAt) {
			return a.LaunchAt.Before(b.LaunchAt)
		}
		return keys[i] < keys[j]
	})
	for _, key := range keys {
		launch := latest[key]
		row := journalAuditRow(opt.IdentityPath, launch, evidence[key], &rep.AuthorityErrors)
		rep.Rows = append(rep.Rows, row)
		rep.Counts.Identities++
		rep.Counts.ByProvider[row.Provider]++
		switch row.Status {
		case JournalStatusAdvanced:
			rep.Counts.ExactTranscripts++
			rep.Counts.Advanced++
		case JournalStatusPresentNoPostLaunchProgress:
			rep.Counts.ExactTranscripts++
			rep.Counts.PresentNoPostLaunchProgress++
		default:
			rep.Counts.MissingTranscript++
		}
	}

	rep.AuthorityErrors = dedupeJournalErrors(rep.AuthorityErrors)
	rep.Counts.AuthorityErrors = len(rep.AuthorityErrors)
	switch {
	case rep.Counts.AuthorityErrors > 0:
		rep.Verdict = JournalVerdictRed
		rep.Summary = fmt.Sprintf("authority unreadable (%d error(s)); no healthy verdict is possible", rep.Counts.AuthorityErrors)
	case rep.Counts.MissingTranscript > 0 || rep.Counts.PresentNoPostLaunchProgress > 0:
		rep.Verdict = JournalVerdictRed
		rep.Summary = fmt.Sprintf("%d/%d recent launch identities advanced; missing=%d present_without_post_launch_progress=%d", rep.Counts.Advanced, rep.Counts.Identities, rep.Counts.MissingTranscript, rep.Counts.PresentNoPostLaunchProgress)
	default:
		rep.Summary = fmt.Sprintf("all %d recent launch identities advanced in provider journals", rep.Counts.Identities)
	}
	return rep
}

func journalAuditRow(identityPath string, launch JournalLaunchIdentity, evidence []journalEvidence, authorityErrors *[]JournalAuthorityError) JournalAuditRow {
	row := JournalAuditRow{
		Identity: launch.Identity,
		Trace:    launch.Trace,
		LaunchAt: launch.LaunchAt.Format(time.RFC3339Nano),
		Provider: launch.Provider,
		IdentityProvenance: JournalIdentityProvenance{
			Path: identityPath, Via: launch.Via, Account: launch.Account,
			Provider: launch.Provider, Source: launch.Source,
		},
		Status: JournalStatusMissingTranscript,
	}
	byProvider := map[string][]journalEvidence{}
	for _, item := range evidence {
		byProvider[item.provider] = append(byProvider[item.provider], item)
	}
	if row.Provider == "" {
		switch len(byProvider) {
		case 0:
			row.Provider = "unknown"
		case 1:
			for provider := range byProvider {
				row.Provider = provider
			}
		default:
			row.Provider = "unknown"
			*authorityErrors = append(*authorityErrors, JournalAuthorityError{Path: identityPath, Code: "PROVIDER_IDENTITY_CONFLICT", Detail: "the same full identity exists in Claude and Codex journals"})
			return row
		}
	} else {
		for provider := range byProvider {
			if provider != row.Provider {
				*authorityErrors = append(*authorityErrors, JournalAuthorityError{Provider: row.Provider, Path: identityPath, Code: "PROVIDER_IDENTITY_CONFLICT", Detail: "recorded provider disagrees with an exact journal match"})
			}
		}
	}
	matches := byProvider[row.Provider]
	if len(matches) == 0 {
		return row
	}
	row.Status = JournalStatusPresentNoPostLaunchProgress
	row.TranscriptPath = matches[0].path
	var baseline, post *journalEvidenceCursor
	for _, match := range matches {
		if row.TranscriptPath == "" || match.path < row.TranscriptPath {
			row.TranscriptPath = match.path
		}
		for i := range match.cursors {
			cursor := match.cursors[i]
			if cursor.at.After(launch.LaunchAt) {
				if post == nil || cursor.at.After(post.at) || (cursor.at.Equal(post.at) && cursor.id > post.id) {
					copy := cursor
					post = &copy
				}
				continue
			}
			if baseline == nil || cursor.at.After(baseline.at) || (cursor.at.Equal(baseline.at) && cursor.id > baseline.id) {
				copy := cursor
				baseline = &copy
			}
		}
	}
	if baseline != nil {
		row.BaselineCursor = publicJournalCursor(*baseline)
		row.TranscriptPath = baseline.path
	}
	if post != nil {
		row.PostLaunchCursor = publicJournalCursor(*post)
		row.TranscriptPath = post.path
		row.Status = JournalStatusAdvanced
	}
	return row
}

func publicJournalCursor(cursor journalEvidenceCursor) *JournalCursor {
	return &JournalCursor{ID: cursor.id, At: cursor.at.UTC().Format(time.RFC3339Nano), Kind: cursor.kind}
}

func discoverJournalRoots(userHome, codexHome, claudeConfigDir string) ([]JournalRoot, []JournalAuthorityError) {
	var roots []JournalRoot
	var errs []JournalAuthorityError
	seen := map[string]bool{}
	add := func(provider, path, provenance string, explicit bool) {
		path = strings.TrimSpace(path)
		if path == "" {
			return
		}
		path = filepath.Clean(path)
		key := provider + "\x00" + journalPathKey(path)
		if seen[key] {
			return
		}
		info, err := os.Stat(path)
		if err != nil || !info.IsDir() {
			if explicit {
				errs = append(errs, JournalAuthorityError{Provider: provider, Path: path, Code: "ROOT_UNAVAILABLE", Detail: "configured provider journal root is not a readable directory"})
			}
			return
		}
		seen[key] = true
		roots = append(roots, JournalRoot{Provider: provider, Path: path, Provenance: provenance})
	}
	if strings.TrimSpace(codexHome) != "" {
		add("codex", codexHome, "CODEX_HOME", true)
	}
	if strings.TrimSpace(claudeConfigDir) != "" {
		add("claude", filepath.Join(claudeConfigDir, "projects"), "CLAUDE_CONFIG_DIR", true)
	}
	if strings.TrimSpace(userHome) == "" {
		errs = append(errs, JournalAuthorityError{Path: userHome, Code: "USER_HOME_UNAVAILABLE", Detail: "cannot discover local Claude and Codex homes"})
	} else {
		entries, err := os.ReadDir(userHome)
		if err != nil {
			errs = append(errs, JournalAuthorityError{Path: userHome, Code: "USER_HOME_READ_FAILED", Detail: "cannot discover local Claude and Codex homes"})
		} else {
			for _, entry := range entries {
				if !entry.IsDir() {
					continue
				}
				name := strings.ToLower(entry.Name())
				switch {
				case strings.HasPrefix(name, ".codex"):
					add("codex", filepath.Join(userHome, entry.Name()), "user_home_glob", false)
				case strings.HasPrefix(name, ".claude"):
					add("claude", filepath.Join(userHome, entry.Name(), "projects"), "user_home_glob", false)
				}
			}
		}
	}
	sort.Slice(roots, func(i, j int) bool {
		if roots[i].Provider != roots[j].Provider {
			return roots[i].Provider < roots[j].Provider
		}
		return roots[i].Path < roots[j].Path
	})
	return roots, errs
}

func collectJournalEvidence(roots []JournalRoot, targets map[string]bool) (map[string][]journalEvidence, []JournalAuthorityError) {
	out := make(map[string][]journalEvidence)
	var errs []JournalAuthorityError
	for _, root := range roots {
		var found map[string][]journalEvidence
		var scanErrs []JournalAuthorityError
		if root.Provider == "claude" {
			found, scanErrs = scanClaudeProjectRoot(root.Path, targets)
		} else {
			found, scanErrs = scanCodexHome(root.Path, targets)
		}
		for key, rows := range found {
			out[key] = append(out[key], rows...)
		}
		errs = append(errs, scanErrs...)
	}
	return out, errs
}

func scanClaudeProjectRoot(root string, targets map[string]bool) (map[string][]journalEvidence, []JournalAuthorityError) {
	out := make(map[string][]journalEvidence)
	projects, err := os.ReadDir(root)
	if err != nil {
		return out, []JournalAuthorityError{{Provider: "claude", Path: root, Code: "READ_FAILED", Detail: "cannot enumerate Claude project transcripts"}}
	}
	var errs []JournalAuthorityError
	for _, project := range projects {
		if !project.IsDir() {
			continue
		}
		projectPath := filepath.Join(root, project.Name())
		files, err := os.ReadDir(projectPath)
		if err != nil {
			errs = append(errs, JournalAuthorityError{Provider: "claude", Path: projectPath, Code: "READ_FAILED", Detail: "cannot enumerate Claude project transcripts"})
			continue
		}
		for _, file := range files {
			if file.IsDir() || !strings.HasSuffix(strings.ToLower(file.Name()), ".jsonl") {
				continue
			}
			identity := strings.TrimSuffix(file.Name(), filepath.Ext(file.Name()))
			key := journalIdentityKey(identity)
			if !targets[key] {
				continue
			}
			path := filepath.Join(projectPath, file.Name())
			cursors, readErr := readClaudeJournalCursors(path)
			if readErr != nil {
				errs = append(errs, JournalAuthorityError{Provider: "claude", Path: path, Code: "READ_FAILED", Detail: "cannot read exact Claude transcript"})
				continue
			}
			out[key] = append(out[key], journalEvidence{provider: "claude", path: path, cursors: cursors})
		}
	}
	return out, errs
}

func readClaudeJournalCursors(path string) ([]journalEvidenceCursor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var cursors []journalEvidenceCursor
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1<<20), 64<<20)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var row struct {
			UUID       string          `json:"uuid"`
			Timestamp  string          `json:"timestamp"`
			Type       string          `json:"type"`
			Role       string          `json:"role"`
			IsAPIError bool            `json:"isApiErrorMessage"`
			Content    json.RawMessage `json:"content"`
			Message    *struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		}
		if err := json.Unmarshal(line, &row); err != nil {
			return nil, fmt.Errorf("%s: invalid JSONL row %d: %w", path, lineNo, err)
		}
		role, content := row.Role, row.Content
		if row.Message != nil {
			role, content = row.Message.Role, row.Message.Content
		}
		if !strings.EqualFold(strings.TrimSpace(role), "assistant") || row.Type == "error" || row.IsAPIError || strings.TrimSpace(journalTextContent(content)) == "" {
			continue
		}
		at, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(row.Timestamp))
		if err != nil {
			return nil, fmt.Errorf("%s: invalid timestamp on row %d: %w", path, lineNo, err)
		}
		id := strings.TrimSpace(row.UUID)
		if id == "" {
			id = fmt.Sprintf("line:%d", lineNo)
		}
		cursors = append(cursors, journalEvidenceCursor{id: id, kind: "assistant", at: at.UTC(), path: path})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return cursors, nil
}

func scanCodexHome(home string, targets map[string]bool) (map[string][]journalEvidence, []JournalAuthorityError) {
	out := make(map[string][]journalEvidence)
	root := filepath.Join(home, "sessions")
	if _, err := os.Stat(root); errorsIsNotExist(err) {
		return out, nil
	} else if err != nil {
		return out, []JournalAuthorityError{{Provider: "codex", Path: root, Code: "READ_FAILED", Detail: "cannot access Codex rollout store"}}
	}
	var errs []JournalAuthorityError
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			errs = append(errs, JournalAuthorityError{Provider: "codex", Path: path, Code: "READ_FAILED", Detail: "cannot enumerate Codex rollout store"})
			return nil
		}
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".jsonl") {
			return nil
		}
		identity, cursors, matched, readErr := readCodexJournal(path, targets)
		if readErr != nil {
			errs = append(errs, JournalAuthorityError{Provider: "codex", Path: path, Code: "READ_FAILED", Detail: "cannot read Codex rollout"})
			return nil
		}
		if matched {
			key := journalIdentityKey(identity)
			out[key] = append(out[key], journalEvidence{provider: "codex", path: path, cursors: cursors})
		}
		return nil
	})
	if err != nil {
		errs = append(errs, JournalAuthorityError{Provider: "codex", Path: root, Code: "READ_FAILED", Detail: "cannot enumerate Codex rollout store"})
	}
	return out, errs
}

func readCodexJournal(path string, targets map[string]bool) (string, []journalEvidenceCursor, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", nil, false, err
	}
	defer f.Close()
	var identity, currentTurn string
	var matched bool
	var cursors []journalEvidenceCursor
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 256<<10), 64<<20)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var row struct {
			Timestamp string `json:"timestamp"`
			Type      string `json:"type"`
			Payload   struct {
				Type      string `json:"type"`
				ID        string `json:"id"`
				SessionID string `json:"session_id"`
				TurnID    string `json:"turn_id"`
			} `json:"payload"`
		}
		if err := json.Unmarshal(line, &row); err != nil {
			return "", nil, false, fmt.Errorf("%s: invalid JSONL row %d: %w", path, lineNo, err)
		}
		if row.Type == "session_meta" && identity == "" {
			identity = strings.TrimSpace(row.Payload.ID)
			if identity == "" {
				identity = strings.TrimSpace(row.Payload.SessionID)
			}
			matched = targets[journalIdentityKey(identity)]
			if !matched {
				return identity, nil, false, nil
			}
			continue
		}
		if !matched || row.Type != "event_msg" {
			continue
		}
		turnID := strings.TrimSpace(row.Payload.TurnID)
		if row.Payload.Type == "task_started" && turnID != "" {
			currentTurn = turnID
		}
		if turnID == "" && (row.Payload.Type == "task_complete" || row.Payload.Type == "turn_aborted") {
			turnID = currentTurn
		}
		if turnID == "" || (row.Payload.Type != "task_started" && row.Payload.Type != "task_complete" && row.Payload.Type != "turn_aborted") {
			continue
		}
		at, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(row.Timestamp))
		if err != nil {
			return identity, nil, matched, fmt.Errorf("%s: invalid timestamp on row %d: %w", path, lineNo, err)
		}
		cursors = append(cursors, journalEvidenceCursor{id: turnID, kind: row.Payload.Type, at: at.UTC(), path: path})
	}
	if err := scanner.Err(); err != nil {
		return identity, nil, matched, err
	}
	return identity, cursors, matched, nil
}

func journalTextContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) != nil {
		return ""
	}
	var parts []string
	for _, block := range blocks {
		if block.Type == "text" {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func normalizeJournalProvider(provider string) string {
	provider = strings.ToLower(strings.TrimSpace(provider))
	switch provider {
	case "anthropic", "claude-code":
		return "claude"
	case "openai", "chatgpt":
		return "codex"
	default:
		return provider
	}
}

func journalIdentityKey(identity string) string { return strings.ToLower(strings.TrimSpace(identity)) }

func journalPathKey(path string) string {
	path = filepath.Clean(path)
	if runtime.GOOS == "windows" {
		return strings.ToLower(path)
	}
	return path
}

func errorsIsNotExist(err error) bool { return err != nil && os.IsNotExist(err) }

func dedupeJournalErrors(in []JournalAuthorityError) []JournalAuthorityError {
	seen := map[string]bool{}
	out := make([]JournalAuthorityError, 0, len(in))
	for _, item := range in {
		key := item.Provider + "\x00" + journalPathKey(item.Path) + "\x00" + item.Code + "\x00" + item.Detail
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, item)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Provider != out[j].Provider {
			return out[i].Provider < out[j].Provider
		}
		if out[i].Path != out[j].Path {
			return out[i].Path < out[j].Path
		}
		return out[i].Code < out[j].Code
	})
	return out
}
