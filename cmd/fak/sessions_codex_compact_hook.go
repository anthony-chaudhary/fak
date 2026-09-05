package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopdrive"
)

type codexCompactHookInput struct {
	SessionID      string `json:"session_id"`
	TurnID         string `json:"turn_id"`
	TranscriptPath string `json:"transcript_path"`
	Cwd            string `json:"cwd"`
}

type codexInvariantsPrecompact struct {
	SessionID      string           `json:"session_id"`
	TurnID         string           `json:"turn_id"`
	Timestamp      string           `json:"timestamp"`
	ActiveLeases   []codexLaneLease `json:"active_leases"`
	FrozenSubtrees []string         `json:"frozen_subtrees"`
	GoalState      codexGoalState   `json:"goal_state"`
	DirtyFiles     []string         `json:"dirty_files"`
}

type codexLaneLease struct {
	Lane      string   `json:"lane"`
	TreeGlobs []string `json:"tree_globs"`
}

type codexGoalState struct {
	Objective string   `json:"objective"`
	NonGoals  []string `json:"non_goals,omitempty"`
	Witness   string   `json:"witness"`
}

type codexPendingRestoration struct {
	SessionID       string           `json:"session_id"`
	TurnID          string           `json:"turn_id"`
	Dropped         []string         `json:"dropped_invariants"`
	Objective       string           `json:"objective"`
	Witness         string           `json:"witness"`
	LaneLeases      []codexLaneLease `json:"lane_leases"`
	FrozenSubtrees  []string         `json:"frozen_subtrees"`
	DirtyFiles      []string         `json:"dirty_files"`
	RestorationNote string           `json:"restoration_note"`
}

var codexFrozenSubtrees = []string{"internal/abi", "dos.toml", ".git/", "internal/kernel/"}

func codexSessionDir(codexHome, sessionID string) (string, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return "", fmt.Errorf("empty session id")
	}
	home, err := resolvedCodexLoopHome(codexHome)
	if err == nil && home != "" {
		return filepath.Join(home, "sessions", sessionID), nil
	}
	return filepath.Join(os.TempDir(), "fak-codex-sessions", sessionID), nil
}

func codexPrecompactInvariantsPath(codexHome, sessionID string) (string, error) {
	dir, err := codexSessionDir(codexHome, sessionID)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "invariants_precompact.json"), nil
}

func codexPendingRestorationPath(codexHome, sessionID string) (string, error) {
	dir, err := codexSessionDir(codexHome, sessionID)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "pending_restoration.json"), nil
}

func sessionsCodexCompactHook(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("sessions codex-compact-hook", flag.ContinueOnError)
	fs.SetOutput(stderr)
	isPre := fs.Bool("pre", false, "execute PreCompact lifecycle hook")
	isPost := fs.Bool("post", false, "execute PostCompact lifecycle hook")
	codexHome := fs.String("codex-home", "", "Codex home directory (default: $CODEX_HOME or ~/.codex)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak sessions codex-compact-hook --pre | --post [--codex-home DIR]")
	}
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if (*isPre && *isPost) || (!*isPre && !*isPost) {
		fs.Usage()
		return 2
	}

	var in codexCompactHookInput
	if err := json.NewDecoder(io.LimitReader(stdin, 1<<20)).Decode(&in); err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-compact-hook: unreadable payload: %v\n", err)
		return 0
	}
	if in.SessionID == "" {
		in.SessionID = strings.TrimSpace(os.Getenv("CODEX_THREAD_ID"))
	}
	if in.Cwd == "" {
		in.Cwd, _ = os.Getwd()
	}

	if *isPre {
		return handleCodexPreCompactHook(stdout, stderr, *codexHome, in)
	}
	return handleCodexPostCompactHook(stdout, stderr, *codexHome, in)
}

func handleCodexPreCompactHook(stdout, stderr io.Writer, codexHome string, in codexCompactHookInput) int {
	leases := collectActiveLaneLeases(in.Cwd)
	goalState := collectGoalState(in.Cwd)
	dirtyFiles := collectDirtyFiles(in.Cwd)

	manifest := codexInvariantsPrecompact{
		SessionID:      in.SessionID,
		TurnID:         in.TurnID,
		Timestamp:      time.Now().UTC().Format(time.RFC3339Nano),
		ActiveLeases:   leases,
		FrozenSubtrees: append([]string(nil), codexFrozenSubtrees...),
		GoalState:      goalState,
		DirtyFiles:     dirtyFiles,
	}

	manifestPath, err := codexPrecompactInvariantsPath(codexHome, in.SessionID)
	if err == nil {
		if err := os.MkdirAll(filepath.Dir(manifestPath), 0o700); err == nil {
			data, err := json.MarshalIndent(manifest, "", "  ")
			if err == nil {
				_ = os.WriteFile(manifestPath, append(data, '\n'), 0o600)
			}
		}
	}

	promptFile := resolveExperimentalCompactPromptFile(codexHome)
	if promptFile != "" {
		var promptContent strings.Builder
		promptContent.WriteString("CRITICAL OPERATIONAL INVARIANTS TO RETAIN ACROSS COMPACTION:\n")
		if len(manifest.ActiveLeases) > 0 {
			promptContent.WriteString("- Active Lane Leases:\n")
			for _, l := range manifest.ActiveLeases {
				promptContent.WriteString(fmt.Sprintf("  * %s (Tree: %s)\n", l.Lane, strings.Join(l.TreeGlobs, ", ")))
			}
		}
		promptContent.WriteString("- Frozen Subtrees (DO NOT MODIFY):\n")
		for _, s := range manifest.FrozenSubtrees {
			promptContent.WriteString(fmt.Sprintf("  * %s\n", s))
		}
		if manifest.GoalState.Objective != "" {
			promptContent.WriteString(fmt.Sprintf("- Objective: %s\n", manifest.GoalState.Objective))
		}
		if manifest.GoalState.Witness != "" {
			promptContent.WriteString(fmt.Sprintf("- Witness Exit Gate: %s\n", manifest.GoalState.Witness))
		}
		if len(manifest.DirtyFiles) > 0 {
			promptContent.WriteString(fmt.Sprintf("- In-flight Files: %s\n", strings.Join(manifest.DirtyFiles, ", ")))
		}
		if err := os.MkdirAll(filepath.Dir(promptFile), 0o700); err == nil {
			_ = os.WriteFile(promptFile, []byte(promptContent.String()), 0o600)
		}
	}

	return 0
}

func handleCodexPostCompactHook(stdout, stderr io.Writer, codexHome string, in codexCompactHookInput) int {
	manifestPath, err := codexPrecompactInvariantsPath(codexHome, in.SessionID)
	var manifest codexInvariantsPrecompact
	manifestFound := false
	if err == nil {
		if data, err := os.ReadFile(manifestPath); err == nil {
			if json.Unmarshal(data, &manifest) == nil {
				manifestFound = true
			}
		}
	}
	if !manifestFound {
		fallback := filepath.Join(os.TempDir(), "fak-codex-sessions", in.SessionID, "invariants_precompact.json")
		if data, err := os.ReadFile(fallback); err == nil {
			if json.Unmarshal(data, &manifest) == nil {
				manifestFound = true
			}
		}
	}
	if !manifestFound {
		return 0
	}

	transcriptPath := in.TranscriptPath
	if transcriptPath == "" {
		if resolved, err := resolveCodexLoopSessionPath(codexHome, in.SessionID, ""); err == nil {
			transcriptPath = resolved
		}
	}

	replacementText := ""
	if transcriptPath != "" {
		replacementText, _ = extractReplacementTextFromTranscript(transcriptPath)
	}

	var dropped []string
	for _, l := range manifest.ActiveLeases {
		if l.Lane != "" && !strings.Contains(replacementText, l.Lane) {
			dropped = append(dropped, "lane_lease:"+l.Lane)
		}
	}
	for _, s := range manifest.FrozenSubtrees {
		if !strings.Contains(replacementText, s) {
			dropped = append(dropped, "frozen_subtree:"+s)
		}
	}
	if manifest.GoalState.Objective != "" && !strings.Contains(replacementText, manifest.GoalState.Objective) {
		dropped = append(dropped, "objective")
	}
	if manifest.GoalState.Witness != "" && !strings.Contains(replacementText, manifest.GoalState.Witness) {
		dropped = append(dropped, "witness")
	}
	for _, f := range manifest.DirtyFiles {
		if !strings.Contains(replacementText, f) {
			dropped = append(dropped, "dirty_file:"+f)
		}
	}

	if len(dropped) > 0 {
		if journal := strings.TrimSpace(os.Getenv("FAK_AUDIT_JOURNAL")); journal != "" {
			event := map[string]any{
				"timestamp":  time.Now().UTC().Format(time.RFC3339Nano),
				"event":      "COMPACTION_INVARIANT_DROPPED",
				"session_id": in.SessionID,
				"dropped":    dropped,
			}
			data, _ := json.Marshal(event)
			data = append(data, '\n')
			if fh, err := os.OpenFile(journal, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600); err == nil {
				_, _ = fh.Write(data)
				_ = fh.Close()
			}
		}

		pending := codexPendingRestoration{
			SessionID:      in.SessionID,
			TurnID:         in.TurnID,
			Dropped:        dropped,
			Objective:      manifest.GoalState.Objective,
			Witness:        manifest.GoalState.Witness,
			LaneLeases:     manifest.ActiveLeases,
			FrozenSubtrees: manifest.FrozenSubtrees,
			DirtyFiles:     manifest.DirtyFiles,
		}
		pending.RestorationNote = formatRestorativeContext(pending)

		pendingPath, err := codexPendingRestorationPath(codexHome, in.SessionID)
		if err == nil {
			if err := os.MkdirAll(filepath.Dir(pendingPath), 0o700); err == nil {
				if data, err := json.MarshalIndent(pending, "", "  "); err == nil {
					_ = os.WriteFile(pendingPath, append(data, '\n'), 0o600)
				}
			}
		}
	}

	return 0
}

func collectActiveLaneLeases(cwd string) []codexLaneLease {
	var leases []codexLaneLease
	leasesPath := filepath.Join(cwd, ".dos", "leases.jsonl")
	if data, err := os.ReadFile(leasesPath); err == nil {
		scanner := bufio.NewScanner(bytes.NewReader(data))
		for scanner.Scan() {
			line := bytes.TrimSpace(scanner.Bytes())
			if len(line) == 0 {
				continue
			}
			var row struct {
				Lane      string   `json:"lane"`
				TreeGlobs []string `json:"tree_globs"`
				Tree      []string `json:"tree"`
			}
			if json.Unmarshal(line, &row) == nil && row.Lane != "" {
				globs := row.TreeGlobs
				if len(globs) == 0 {
					globs = row.Tree
				}
				leases = append(leases, codexLaneLease{Lane: row.Lane, TreeGlobs: globs})
			}
		}
	}

	goalPath := filepath.Join(cwd, "GOAL.md")
	if data, err := os.ReadFile(goalPath); err == nil {
		if spec, err := loopdrive.Parse(data); err == nil && spec.Lane != "" {
			found := false
			for _, l := range leases {
				if l.Lane == spec.Lane {
					found = true
					break
				}
			}
			if !found {
				leases = append(leases, codexLaneLease{Lane: spec.Lane, TreeGlobs: spec.Region})
			}
		}
	}

	if envLane := strings.TrimSpace(os.Getenv("FAK_LANE")); envLane != "" {
		found := false
		for _, l := range leases {
			if l.Lane == envLane {
				found = true
				break
			}
		}
		if !found {
			leases = append(leases, codexLaneLease{Lane: envLane})
		}
	}
	return leases
}

func collectGoalState(cwd string) codexGoalState {
	var gs codexGoalState
	goalPath := filepath.Join(cwd, "GOAL.md")
	if data, err := os.ReadFile(goalPath); err == nil {
		if spec, err := loopdrive.Parse(data); err == nil {
			gs.Objective = strings.TrimSpace(spec.Objective)
			gs.Witness = strings.TrimSpace(spec.Witness)
		}
		if idx := strings.Index(gs.Objective, "\n#"); idx != -1 {
			gs.Objective = strings.TrimSpace(gs.Objective[:idx])
		}
		lines := strings.Split(string(data), "\n")
		inNonGoals := false
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "#") {
				lower := strings.ToLower(trimmed)
				if strings.Contains(lower, "non-goal") {
					inNonGoals = true
					continue
				} else if inNonGoals {
					break
				}
			}
			if inNonGoals && strings.HasPrefix(trimmed, "-") {
				item := strings.TrimSpace(strings.TrimPrefix(trimmed, "-"))
				if item != "" {
					gs.NonGoals = append(gs.NonGoals, item)
				}
			}
		}
	}
	if gs.Objective == "" {
		gs.Objective = strings.TrimSpace(os.Getenv("FAK_OBJECTIVE"))
	}
	if gs.Witness == "" {
		gs.Witness = strings.TrimSpace(os.Getenv("FAK_WITNESS"))
	}
	return gs
}

func collectDirtyFiles(cwd string) []string {
	cmd := exec.Command("git", "status", "--porcelain")
	configureDispatchHelperCommand(cmd)
	if cwd != "" {
		cmd.Dir = cwd
	}
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	var files []string
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) < 4 {
			continue
		}
		path := strings.TrimSpace(line[3:])
		if strings.Contains(path, " -> ") {
			parts := strings.Split(path, " -> ")
			path = parts[len(parts)-1]
		}
		path = strings.Trim(path, "\"")
		if path != "" {
			files = append(files, path)
		}
	}
	return files
}

func resolveExperimentalCompactPromptFile(codexHome string) string {
	if p := strings.TrimSpace(os.Getenv("EXPERIMENTAL_COMPACT_PROMPT_FILE")); p != "" {
		return p
	}
	if p := strings.TrimSpace(os.Getenv("CODEX_EXPERIMENTAL_COMPACT_PROMPT_FILE")); p != "" {
		return p
	}
	cfgPath := effectiveCodexConfigFile(codexHome)
	if cfgPath != "" {
		if data, err := os.ReadFile(cfgPath); err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "experimental_compact_prompt_file") {
					parts := strings.SplitN(line, "=", 2)
					if len(parts) == 2 {
						val := strings.TrimSpace(parts[1])
						val = strings.Trim(val, `"`)
						val = strings.Trim(val, `'`)
						return val
					}
				}
			}
		}
	}
	return ""
}

func extractReplacementTextFromTranscript(transcriptPath string) (string, error) {
	fh, err := os.Open(transcriptPath)
	if err != nil {
		return "", err
	}
	defer fh.Close()

	scanner := bufio.NewScanner(fh)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 32*1024*1024)

	var lastCompactedPayload []byte
	for scanner.Scan() {
		line := scanner.Bytes()
		if bytes.Contains(line, []byte(`"compacted"`)) || bytes.Contains(line, []byte(`"context_compacted"`)) {
			cpy := make([]byte, len(line))
			copy(cpy, line)
			lastCompactedPayload = cpy
		}
	}
	if err := scanner.Err(); err != nil && len(lastCompactedPayload) == 0 {
		return "", err
	}
	if len(lastCompactedPayload) == 0 {
		return "", nil
	}

	var row struct {
		Type    string          `json:"type"`
		Payload json.RawMessage `json:"payload"`
	}
	if err := json.Unmarshal(lastCompactedPayload, &row); err != nil {
		return string(lastCompactedPayload), nil
	}

	var payload struct {
		Message            string            `json:"message"`
		Summary            string            `json:"summary"`
		ReplacementHistory []json.RawMessage `json:"replacement_history"`
	}
	if err := json.Unmarshal(row.Payload, &payload); err != nil {
		return string(row.Payload), nil
	}

	var b strings.Builder
	b.WriteString(payload.Message)
	b.WriteString(" ")
	b.WriteString(payload.Summary)
	b.WriteString(" ")
	for _, item := range payload.ReplacementHistory {
		var obj map[string]any
		if json.Unmarshal(item, &obj) == nil {
			for _, v := range obj {
				b.WriteString(fmt.Sprint(v))
				b.WriteString(" ")
			}
		} else {
			var s string
			if json.Unmarshal(item, &s) == nil {
				b.WriteString(s)
				b.WriteString(" ")
			} else {
				b.Write(item)
				b.WriteString(" ")
			}
		}
	}
	return b.String(), nil
}

func formatRestorativeContext(r codexPendingRestoration) string {
	var b strings.Builder
	b.WriteString("[FAK RESTORATIVE INVARIANT RESTORATION]\n")
	b.WriteString("Context compaction omitted critical system invariants. Re-injecting:\n")

	obj := strings.TrimSpace(r.Objective)
	if obj == "" {
		obj = "none"
	}
	b.WriteString("- Objective: " + obj + "\n")

	witness := strings.TrimSpace(r.Witness)
	if witness == "" {
		witness = "none"
	}
	b.WriteString("- Witness Exit Gate: " + witness + "\n")

	laneStr := "none"
	if len(r.LaneLeases) > 0 {
		var parts []string
		for _, l := range r.LaneLeases {
			globs := strings.Join(l.TreeGlobs, ", ")
			if globs != "" {
				parts = append(parts, fmt.Sprintf("%s (Tree: %s)", l.Lane, globs))
			} else {
				parts = append(parts, l.Lane)
			}
		}
		laneStr = strings.Join(parts, "; ")
	}
	b.WriteString("- Lane Lease: " + laneStr + "\n")

	b.WriteString("- Frozen Subtrees: internal/abi must not be modified\n")

	filesStr := "none"
	if len(r.DirtyFiles) > 0 {
		filesStr = strings.Join(r.DirtyFiles, ", ")
	}
	b.WriteString("- In-flight Files: " + filesStr)
	return b.String()
}

func consumeCodexPendingRestoration(codexHome, sessionID string) (string, bool) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return "", false
	}
	path, err := codexPendingRestorationPath(codexHome, sessionID)
	if err != nil {
		return "", false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fallback := filepath.Join(os.TempDir(), "fak-codex-sessions", sessionID, "pending_restoration.json")
		data, err = os.ReadFile(fallback)
		if err != nil {
			return "", false
		}
		path = fallback
	}
	_ = os.Remove(path)

	var pending codexPendingRestoration
	if err := json.Unmarshal(data, &pending); err != nil {
		return "", false
	}
	if pending.RestorationNote != "" {
		return pending.RestorationNote, true
	}
	return formatRestorativeContext(pending), true
}
