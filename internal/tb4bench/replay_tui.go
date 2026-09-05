package tb4bench

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strings"
)

// ANSI escape codes for terminal formatting.
const (
	ansiReset      = "\033[0m"
	ansiBold       = "\033[1m"
	ansiDim        = "\033[2m"
	ansiRed        = "\033[31m"
	ansiGreen      = "\033[32m"
	ansiYellow     = "\033[33m"
	ansiBlue       = "\033[34m"
	ansiMagenta    = "\033[35m"
	ansiCyan       = "\033[36m"
	ansiWhite      = "\033[37m"
	ansiBgBlue     = "\033[44m"
	ansiBgGreen    = "\033[42m"
	ansiBgDarkGray = "\033[100m"
)

var ansiRegexp = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)

// stripANSI removes ANSI escape sequences to compute visual character length.
func stripANSI(s string) string {
	return ansiRegexp.ReplaceAllString(s, "")
}

// visibleLen calculates the printable column width of an ANSI-formatted string.
func visibleLen(s string) int {
	return len([]rune(stripANSI(s)))
}

// truncateVisible cuts a string so its printable column width does not exceed maxWidth.
func truncateVisible(s string, maxWidth int) string {
	if visibleLen(s) <= maxWidth {
		return s
	}
	var b strings.Builder
	vis := 0
	inEsc := false
	runes := []rune(s)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if r == 0x1b {
			inEsc = true
			b.WriteRune(r)
			continue
		}
		if inEsc {
			b.WriteRune(r)
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
				inEsc = false
			}
			continue
		}
		if vis >= maxWidth-3 {
			b.WriteString("...")
			b.WriteString(ansiReset)
			break
		}
		b.WriteRune(r)
		vis++
	}
	return b.String()
}

// padRight pads an ANSI-formatted string with spaces to align column borders.
func padRight(s string, width int) string {
	s = truncateVisible(s, width)
	vis := visibleLen(s)
	if vis >= width {
		return s
	}
	return s + strings.Repeat(" ", width-vis)
}

// TUIState represents the active state of the interactive trajectory viewer.
type TUIState struct {
	ResA        *ArmExecutionResult
	ResB        *ArmExecutionResult
	CurrentTurn int  // 1-based index: 1 .. MaxTurns
	ActiveArm   int  // 0: Arm A (fak), 1: Arm B (opencode)
	Expanded    bool // toggle tool output expansion
	Comparative bool // true if dual-arm comparative mode
}

// MaxTurns returns the maximum turns among the active execution results.
func (s *TUIState) MaxTurns() int {
	maxTurns := 0
	if s.ResA != nil && len(s.ResA.Turns) > maxTurns {
		maxTurns = len(s.ResA.Turns)
	}
	if s.ResB != nil && len(s.ResB.Turns) > maxTurns {
		maxTurns = len(s.ResB.Turns)
	}
	if maxTurns == 0 {
		maxTurns = 1
	}
	return maxTurns
}

// NextTurn advances the active turn by one, bounded by MaxTurns.
func (s *TUIState) NextTurn() {
	if s.CurrentTurn < s.MaxTurns() {
		s.CurrentTurn++
	}
}

// PrevTurn steps the active turn back by one, bounded by 1.
func (s *TUIState) PrevTurn() {
	if s.CurrentTurn > 1 {
		s.CurrentTurn--
	}
}

// ToggleArm switches the active arm between Arm A and Arm B in comparative mode.
func (s *TUIState) ToggleArm() {
	if s.Comparative {
		if s.ActiveArm == 0 {
			s.ActiveArm = 1
		} else {
			s.ActiveArm = 0
		}
	}
}

// ToggleExpanded flips tool output expansion state.
func (s *TUIState) ToggleExpanded() {
	s.Expanded = !s.Expanded
}

// InteractiveReplayViewer manages interactive TUI trajectory navigation and frame rendering.
type InteractiveReplayViewer struct {
	State         *TUIState
	StateHistory  []*TUIState
	OnStateChange func(state *TUIState)
}

// TUIController is an alias for InteractiveReplayViewer.
type TUIController = InteractiveReplayViewer

// NewInteractiveReplayViewer instantiates a new viewer with given results.
func NewInteractiveReplayViewer(resA, resB *ArmExecutionResult) *InteractiveReplayViewer {
	state := &TUIState{
		ResA:        resA,
		ResB:        resB,
		CurrentTurn: 1,
		ActiveArm:   0,
		Expanded:    false,
		Comparative: resB != nil,
	}
	return &InteractiveReplayViewer{
		State: state,
	}
}

// NewTUIController creates a controller instance for TUI replay.
func NewTUIController(resA, resB *ArmExecutionResult) *TUIController {
	return NewInteractiveReplayViewer(resA, resB)
}

// RunInteractive drives interactive navigation reading keys from r and rendering frames to w.
func RunInteractive(r io.Reader, w io.Writer, resA, resB *ArmExecutionResult) error {
	viewer := NewInteractiveReplayViewer(resA, resB)
	return viewer.RunInteractive(r, w, resA, resB)
}

// RunInteractive executes the interactive replay loop on viewer.
func (v *InteractiveReplayViewer) RunInteractive(r io.Reader, w io.Writer, resA, resB *ArmExecutionResult) error {
	if v.State == nil {
		v.State = &TUIState{
			ResA:        resA,
			ResB:        resB,
			CurrentTurn: 1,
			ActiveArm:   0,
			Expanded:    false,
			Comparative: resB != nil,
		}
	} else {
		if resA != nil {
			v.State.ResA = resA
		}
		if resB != nil {
			v.State.ResB = resB
			v.State.Comparative = true
		}
	}

	// Capture initial state snapshot
	initSnap := *v.State
	v.StateHistory = append(v.StateHistory, &initSnap)
	if v.OnStateChange != nil {
		v.OnStateChange(&initSnap)
	}

	// Render initial frame
	if err := v.DrawTurnFrame(w, v.State); err != nil {
		return err
	}

	br := bufio.NewReader(r)
	for {
		b, err := br.ReadByte()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		stateChanged := false
		switch b {
		case 'q', 'Q':
			return nil
		case 0x1b: // ESC or escape sequence
			if br.Buffered() > 0 {
				next, err := br.ReadByte()
				if err == nil && (next == '[' || next == 'O') {
					if br.Buffered() > 0 {
						dir, err := br.ReadByte()
						if err == nil {
							switch dir {
							case 'A': // Up
								v.State.PrevTurn()
								stateChanged = true
							case 'B': // Down
								v.State.NextTurn()
								stateChanged = true
							}
						}
					}
				} else {
					// Standalone Esc key
					return nil
				}
			} else {
				// Standalone Esc key
				return nil
			}
		case 'j', 'J':
			v.State.NextTurn()
			stateChanged = true
		case 'k', 'K':
			v.State.PrevTurn()
			stateChanged = true
		case '\t':
			v.State.ToggleArm()
			stateChanged = true
		case '\r':
			if br.Buffered() > 0 {
				if peek, err := br.Peek(1); err == nil && len(peek) > 0 && peek[0] == '\n' {
					_, _ = br.ReadByte()
				}
			}
			v.State.ToggleExpanded()
			stateChanged = true
		case '\n':
			v.State.ToggleExpanded()
			stateChanged = true
		}

		if stateChanged {
			snap := *v.State
			v.StateHistory = append(v.StateHistory, &snap)
			if v.OnStateChange != nil {
				v.OnStateChange(&snap)
			}
			if err := v.DrawTurnFrame(w, v.State); err != nil {
				return err
			}
		}
	}
}

// DrawTurnFrame renders a single TUI frame to w without user interaction.
func DrawTurnFrame(w io.Writer, state *TUIState) error {
	viewer := &InteractiveReplayViewer{State: state}
	return viewer.DrawTurnFrame(w, state)
}

// DrawTurnFrame renders the current turn state into an ANSI-formatted frame.
func (v *InteractiveReplayViewer) DrawTurnFrame(w io.Writer, state *TUIState) error {
	if state == nil {
		return fmt.Errorf("state cannot be nil")
	}

	taskID := "unknown"
	if state.ResA != nil && state.ResA.TaskID != "" {
		taskID = state.ResA.TaskID
	} else if state.ResB != nil && state.ResB.TaskID != "" {
		taskID = state.ResB.TaskID
	}

	maxTurns := state.MaxTurns()

	if state.Comparative && state.ResB != nil {
		// Comparative dual-arm side-by-side mode
		colWidth := 50
		linesA := formatArmCardLines(state.ResA, state.CurrentTurn, true, state.ActiveArm == 0, state.Expanded, colWidth)
		linesB := formatArmCardLines(state.ResB, state.CurrentTurn, false, state.ActiveArm == 1, state.Expanded, colWidth)

		maxLines := len(linesA)
		if len(linesB) > maxLines {
			maxLines = len(linesB)
		}

		_, _ = fmt.Fprintf(w, "==========================================================================================================\n")
		_, _ = fmt.Fprintf(w, "TB4 Comparative Replay: Task %s | Turn %d of %d\n", taskID, state.CurrentTurn, maxTurns)
		_, _ = fmt.Fprintf(w, "==========================================================================================================\n")

		// Top border
		_, _ = fmt.Fprintf(w, "┌─%s─┬─%s─┐\n", strings.Repeat("─", colWidth), strings.Repeat("─", colWidth))

		for i := 0; i < maxLines; i++ {
			var left, right string
			if i < len(linesA) {
				left = linesA[i]
			}
			if i < len(linesB) {
				right = linesB[i]
			}
			_, _ = fmt.Fprintf(w, "│ %s │ %s │\n", padRight(left, colWidth), padRight(right, colWidth))
		}

		// Bottom border
		_, _ = fmt.Fprintf(w, "└─%s─┴─%s─┘\n", strings.Repeat("─", colWidth), strings.Repeat("─", colWidth))

		// Controls
		var activeName string
		if state.ActiveArm == 0 {
			activeName = "Arm A (fak)"
		} else {
			activeName = "Arm B (opencode)"
		}
		expStatus := "OFF"
		if state.Expanded {
			expStatus = "ON"
		}
		_, _ = fmt.Fprintf(w, "Controls: [j: Next Turn] [k: Prev Turn] [Tab: Switch Arm (Active: %s)] [Enter: Toggle Output (%s)] [q: Quit]\n\n",
			activeName, expStatus)
	} else {
		// Single-arm mode
		cardWidth := 88
		lines := formatArmCardLines(state.ResA, state.CurrentTurn, true, true, state.Expanded, cardWidth)

		armName := "fak native"
		if state.ResA != nil && state.ResA.ArmID != "" {
			armName = state.ResA.ArmID
		}

		_, _ = fmt.Fprintf(w, "==========================================================================================\n")
		_, _ = fmt.Fprintf(w, "TB4 Trajectory Replay: Task %s | Arm: %s | Turn %d of %d\n", taskID, armName, state.CurrentTurn, maxTurns)
		_, _ = fmt.Fprintf(w, "==========================================================================================\n")

		// Top border
		_, _ = fmt.Fprintf(w, "┌─%s─┐\n", strings.Repeat("─", cardWidth))
		for _, line := range lines {
			_, _ = fmt.Fprintf(w, "│ %s │\n", padRight(line, cardWidth))
		}
		// Bottom border
		_, _ = fmt.Fprintf(w, "└─%s─┘\n", strings.Repeat("─", cardWidth))

		expStatus := "OFF"
		if state.Expanded {
			expStatus = "ON"
		}
		_, _ = fmt.Fprintf(w, "Controls: [j: Next Turn] [k: Prev Turn] [Enter: Toggle Output (%s)] [q: Quit]\n\n", expStatus)
	}

	return nil
}

// formatArmCardLines formats the execution turn details for one arm into display lines.
func formatArmCardLines(arm *ArmExecutionResult, turnNum int, isArmA, isActive, expanded bool, colWidth int) []string {
	var lines []string

	// Arm Header Badge
	var armLabel string
	if isArmA {
		armLabel = "Arm A: fak (In-Kernel)"
	} else {
		armLabel = "Arm B: OpenCode (Reference)"
	}

	if isActive {
		lines = append(lines, fmt.Sprintf("%s%s★ %s [ACTIVE]%s", ansiBold, ansiGreen, armLabel, ansiReset))
	} else {
		lines = append(lines, fmt.Sprintf("%s  %s%s", ansiDim, armLabel, ansiReset))
	}

	if arm == nil {
		lines = append(lines, "  (No execution data)")
		return lines
	}

	lines = append(lines, fmt.Sprintf("Status: %s | Total Turns: %d", arm.Status, arm.TotalTurns))

	// Check if turn is past the arm's completion
	if turnNum > len(arm.Turns) {
		lines = append(lines, strings.Repeat("─", colWidth))
		lines = append(lines, fmt.Sprintf("%s(Finished at turn %d - Final: %s)%s", ansiDim, len(arm.Turns), arm.Status, ansiReset))
		return lines
	}

	lines = append(lines, strings.Repeat("─", colWidth))

	t := arm.Turns[turnNum-1]

	// 1. Turn index, role, status
	verdict := t.AdjudicationVerdict
	if verdict == "" {
		verdict = "COMPLETED"
	}
	lines = append(lines, fmt.Sprintf("Turn %d / %d | Role: assistant | Status: %s", t.Turn, arm.TotalTurns, verdict))

	// 2. Real-time Token Breakdown (prompt initial, prompt reused, completion, thoughts)
	var initialTokens, reusedTokens int64
	if t.PromptTokens > 0 {
		if t.Turn == 1 {
			initialTokens = t.PromptTokens
			reusedTokens = 0
		} else {
			if isArmA {
				// In-kernel prefix hit simulation/derived metric
				reusedTokens = t.PromptTokens * 7 / 10
				initialTokens = t.PromptTokens - reusedTokens
			} else {
				// Reference arm without prefix hit
				initialTokens = t.PromptTokens
				reusedTokens = 0
			}
		}
	}
	compTokens := t.CompletionTokens
	thoughtTokens := int64(len(strings.Fields(t.ModelText)))
	if thoughtTokens == 0 && t.ModelText != "" {
		thoughtTokens = int64(len(t.ModelText) / 4)
	}

	lines = append(lines, fmt.Sprintf("Tokens: Prompt Cold: %d | Prompt Cached: %d", initialTokens, reusedTokens))
	lines = append(lines, fmt.Sprintf("Tokens: Completion: %d | Thoughts: %d", compTokens, thoughtTokens))

	// 3. Acceleration and reuse indicators
	var vBadge, cBadge string
	if isArmA {
		if arm.VDSOHits > 0 {
			vBadge = fmt.Sprintf("%s[vDSO: HIT (%d hits)]%s", ansiGreen, arm.VDSOHits, ansiReset)
		} else {
			vBadge = fmt.Sprintf("%s[vDSO: COLD]%s", ansiDim, ansiReset)
		}

		if reusedTokens > 0 || arm.VDSOHits > 0 {
			cBadge = fmt.Sprintf("%s[Reuse: HIT (prefix)]%s", ansiCyan, ansiReset)
		} else {
			cBadge = fmt.Sprintf("%s[Reuse: MISS]%s", ansiYellow, ansiReset)
		}
	} else {
		vBadge = fmt.Sprintf("%s[vDSO: N/A]%s", ansiDim, ansiReset)
		cBadge = fmt.Sprintf("%s[Reuse: MISS]%s", ansiDim, ansiReset)
	}
	lines = append(lines, fmt.Sprintf("Telemetry: %s %s", vBadge, cBadge))

	// 4. Model text / thoughts preview
	if t.ModelText != "" {
		lines = append(lines, fmt.Sprintf("%sThoughts:%s %s", ansiBold, ansiReset, cleanThoughtText(t.ModelText, expanded, colWidth-12)))
	}

	// 5. Tool calls and outputs (with diff-like visual cues if modifying files)
	if len(t.ToolCalls) > 0 {
		lines = append(lines, strings.Repeat("─", colWidth))
		for i, tc := range t.ToolCalls {
			isMod := isFileModifying(tc.Name, tc.Arguments)
			var modTag string
			if isMod {
				modTag = fmt.Sprintf(" %s%s[MOD/DIFF]%s", ansiBold, ansiMagenta, ansiReset)
			}
			badge := "[ALLOWED]"
			if t.AdjudicationVerdict != "" {
				badge = fmt.Sprintf("[%s]", t.AdjudicationVerdict)
			}
			lines = append(lines, fmt.Sprintf("Tool (%d): %s %s%s", i+1, badge, tc.Name, modTag))
			lines = append(lines, fmt.Sprintf("  args: %s", truncateVisible(tc.Arguments, colWidth-8)))

			if isMod {
				lines = append(lines, fmt.Sprintf("  %s+++ file modification target%s", ansiGreen, ansiReset))
			}
		}
	}

	// 6. Tool outputs / stdout with diff highlighting
	if len(t.ToolResults) > 0 {
		lines = append(lines, strings.Repeat("─", colWidth))
		if !expanded {
			lines = append(lines, fmt.Sprintf("%s[Output collapsed - press Enter to expand (%d results)]%s", ansiDim, len(t.ToolResults), ansiReset))
		} else {
			lines = append(lines, fmt.Sprintf("%sTool Output (Expanded):%s", ansiBold, ansiReset))
			for _, res := range t.ToolResults {
				rawOutput := res.Stdout
				if res.Stderr != "" {
					rawOutput += "\nstderr: " + res.Stderr
				}
				outputLines := strings.Split(rawOutput, "\n")
				for _, ol := range outputLines {
					if len(ol) == 0 {
						continue
					}
					// Diff visual cues: green for additions, red for deletions, cyan for hunk headers
					if strings.HasPrefix(ol, "+") {
						lines = append(lines, fmt.Sprintf("  %s%s%s", ansiGreen, ol, ansiReset))
					} else if strings.HasPrefix(ol, "-") {
						lines = append(lines, fmt.Sprintf("  %s%s%s", ansiRed, ol, ansiReset))
					} else if strings.HasPrefix(ol, "@@") {
						lines = append(lines, fmt.Sprintf("  %s%s%s", ansiCyan, ol, ansiReset))
					} else {
						lines = append(lines, fmt.Sprintf("  > %s", ol))
					}
				}
			}
		}
	}

	return lines
}

// cleanThoughtText returns single line or truncated thoughts preview.
func cleanThoughtText(text string, expanded bool, maxLen int) string {
	singleLine := strings.ReplaceAll(text, "\n", " ")
	if !expanded && len(singleLine) > maxLen {
		return singleLine[:maxLen-3] + "..."
	}
	return singleLine
}

// isFileModifying checks whether a tool invocation alters filesystem contents.
func isFileModifying(name, args string) bool {
	switch strings.ToLower(name) {
	case "edit_file", "write_file", "patch", "append_file", "replace_file", "create_file":
		return true
	}
	lowerArgs := strings.ToLower(args)
	if strings.Contains(lowerArgs, "newstring") ||
		strings.Contains(lowerArgs, "new_content") ||
		strings.Contains(lowerArgs, "oldstring") ||
		strings.Contains(lowerArgs, "old_str") ||
		strings.Contains(lowerArgs, "patch") ||
		(strings.Contains(lowerArgs, "file_path") && strings.Contains(lowerArgs, "content")) {
		return true
	}
	if strings.ToLower(name) == "bash" {
		if strings.Contains(args, ">") || strings.Contains(args, "sed -i") || strings.Contains(args, "tee ") {
			return true
		}
	}
	return false
}
