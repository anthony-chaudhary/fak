package auditreason

import (
	"encoding/json"
	"path"
	"strings"
)

// SemanticExitStatus represents the closed vocabulary of semantic CLI exit qualifications.
// Rather than treating every non-zero exit code as a fatal execution failure, semantic exit
// qualification distinguishes benign negative query outcomes (e.g., grep pattern not found,
// git diff differences exist, test predicate false) from fatal crashes and configuration errors.
type SemanticExitStatus string

const (
	// StatusSuccess indicates the command completed successfully (exit code 0).
	StatusSuccess SemanticExitStatus = "STATUS_SUCCESS"

	// StatusPatternNotFound indicates a search/filter tool (e.g., grep, rg) evaluated
	// successfully but found no matching lines (exit code 1). This is a benign query outcome.
	StatusPatternNotFound SemanticExitStatus = "STATUS_PATTERN_NOT_FOUND"

	// StatusMatchNotFound indicates a query/lookup tool (e.g., which, command -v)
	// evaluated successfully but found no match (exit code 1). Benign negative query.
	StatusMatchNotFound SemanticExitStatus = "STATUS_MATCH_NOT_FOUND"

	// StatusDiffPresent indicates a comparison tool (e.g., git diff --quiet, diff -q)
	// evaluated successfully and reported that differences exist (exit code 1). Benign evaluation result.
	StatusDiffPresent SemanticExitStatus = "STATUS_DIFF_PRESENT"

	// StatusFileNotFound indicates a path or file existence predicate (e.g., test -e, [ -f ... ])
	// evaluated to false because the target was not found (exit code 1). Benign negative test.
	StatusFileNotFound SemanticExitStatus = "STATUS_FILE_NOT_FOUND"

	// StatusPredicateFalse indicates a general boolean or predicate expression (e.g., test, [, expr)
	// evaluated to false (exit code 1). Benign evaluation outcome.
	StatusPredicateFalse SemanticExitStatus = "STATUS_PREDICATE_FALSE"

	// StatusSyntaxError indicates invalid invocation syntax, unrecognized arguments, or malformed regex
	// (e.g., exit code 2 on grep/diff). Fatal error.
	StatusSyntaxError SemanticExitStatus = "STATUS_SYNTAX_ERROR"

	// StatusCommandNotFound indicates the executable was not found in PATH (exit code 127). Fatal error.
	StatusCommandNotFound SemanticExitStatus = "STATUS_COMMAND_NOT_FOUND"

	// StatusPermissionDenied indicates the executable was not permitted or lacked execution permissions (exit code 126).
	StatusPermissionDenied SemanticExitStatus = "STATUS_PERMISSION_DENIED"

	// StatusProcessCrash indicates the process was terminated abnormally by a signal (e.g., exit code 130, 137, 143).
	StatusProcessCrash SemanticExitStatus = "STATUS_PROCESS_CRASH"

	// StatusError indicates an unclassified non-benign execution failure or fatal non-zero exit.
	StatusError SemanticExitStatus = "STATUS_ERROR"
)

// IsBenign reports whether the status represents a benign, expected negative or successful outcome
// rather than an actual execution error or crash.
func (s SemanticExitStatus) IsBenign() bool {
	switch s {
	case StatusSuccess, StatusPatternNotFound, StatusMatchNotFound, StatusDiffPresent, StatusFileNotFound, StatusPredicateFalse:
		return true
	default:
		return false
	}
}

// IsFatal reports whether the status represents a fatal execution failure, crash, or syntax error.
func (s SemanticExitStatus) IsFatal() bool {
	return !s.IsBenign()
}

// Summary returns a human-readable explanation of the semantic exit status.
func (s SemanticExitStatus) Summary() string {
	switch s {
	case StatusSuccess:
		return "command succeeded"
	case StatusPatternNotFound:
		return "pattern not found (expected negative query outcome)"
	case StatusMatchNotFound:
		return "match not found (expected negative query outcome)"
	case StatusDiffPresent:
		return "differences exist between compared inputs"
	case StatusFileNotFound:
		return "file or path predicate evaluated to false (not found)"
	case StatusPredicateFalse:
		return "predicate expression evaluated to false"
	case StatusSyntaxError:
		return "command syntax or argument error"
	case StatusCommandNotFound:
		return "command not found (exit code 127)"
	case StatusPermissionDenied:
		return "permission denied or command cannot execute (exit code 126)"
	case StatusProcessCrash:
		return "process terminated abnormally by signal"
	case StatusError:
		return "command failed with non-zero exit code"
	default:
		return string(s)
	}
}

// JSONHostBoundary represents a structured JSON host envelope separating
// process exit code, working directory, status qualification, and output metadata.
// This provides literal reasoning models with deterministic execution semantics,
// preventing benign negative query outcomes (exit 1 on grep or git diff --quiet)
// from triggering panic loops or expensive failure recoveries.
type JSONHostBoundary struct {
	Command     string             `json:"command"`
	ExitCode    int                `json:"exit_code"`
	Status      SemanticExitStatus `json:"status"`
	Benign      bool               `json:"benign"`
	Fatal       bool               `json:"fatal"`
	Summary     string             `json:"summary"`
	Cwd         string             `json:"cwd,omitempty"`
	Stdout      string             `json:"stdout,omitempty"`
	Stderr      string             `json:"stderr,omitempty"`
	Description string             `json:"description,omitempty"`
}

// SemanticExitQualification is an alias for JSONHostBoundary providing the typed qualification.
type SemanticExitQualification = JSONHostBoundary

// HostBoundary projects a SemanticExitStatus into a JSONHostBoundary envelope.
func (s SemanticExitStatus) HostBoundary(command string, exitCode int) JSONHostBoundary {
	return JSONHostBoundary{
		Command:  command,
		ExitCode: exitCode,
		Status:   s,
		Benign:   s.IsBenign(),
		Fatal:    s.IsFatal(),
		Summary:  s.Summary(),
	}
}

// JSON serializes the host boundary envelope to indented JSON bytes.
func (b JSONHostBoundary) JSON() ([]byte, error) {
	return json.Marshal(b)
}

// MustJSON serializes the host boundary envelope to a JSON string, ignoring errors.
func (b JSONHostBoundary) MustJSON() string {
	bytes, _ := json.Marshal(b)
	return string(bytes)
}

// QualifyExitCode identifies whether a command exit code represents a benign query outcome,
// clean success, or a fatal crash/error, returning a typed SemanticExitStatus.
func QualifyExitCode(command string, exitCode int) SemanticExitStatus {
	if exitCode == 0 {
		return StatusSuccess
	}
	if exitCode == 127 {
		return StatusCommandNotFound
	}
	if exitCode == 126 {
		return StatusPermissionDenied
	}
	if exitCode >= 128 {
		return StatusProcessCrash
	}

	tokens := unwrapCommand(tokenizeCommand(command))
	if len(tokens) == 0 {
		return StatusError
	}

	exe := cleanExeName(tokens[0])

	switch exe {
	case "grep", "egrep", "fgrep", "rgrep", "rg", "ag", "ack":
		if exitCode == 1 {
			return StatusPatternNotFound
		}
		if exitCode == 2 {
			return StatusSyntaxError
		}
		return StatusError

	case "git":
		return qualifyGitCommand(tokens, exitCode)

	case "diff", "diff3", "cmp", "comm", "git-diff":
		if exitCode == 1 {
			return StatusDiffPresent
		}
		if exitCode >= 2 {
			return StatusSyntaxError
		}
		return StatusError

	case "test", "[", "[[":
		if exitCode == 1 {
			for _, arg := range tokens[1:] {
				switch arg {
				case "-e", "-f", "-d", "-s", "-r", "-w", "-x", "-L", "-h", "-b", "-c", "-p", "-S", "-k", "-u", "-g", "-O", "-G", "-N", "-nt", "-ot", "-ef":
					return StatusFileNotFound
				}
			}
			return StatusPredicateFalse
		}
		if exitCode >= 2 {
			return StatusSyntaxError
		}
		return StatusError

	case "which", "type":
		if exitCode == 1 {
			return StatusMatchNotFound
		}
		return StatusError

	case "command":
		hasV := false
		for _, arg := range tokens[1:] {
			if arg == "-v" || arg == "-V" {
				hasV = true
				break
			}
		}
		if hasV && exitCode == 1 {
			return StatusMatchNotFound
		}
		return StatusError

	case "expr":
		if exitCode == 1 {
			return StatusPredicateFalse
		}
		if exitCode >= 2 {
			return StatusSyntaxError
		}
		return StatusError

	case "false":
		if exitCode == 1 {
			return StatusPredicateFalse
		}
		return StatusError

	default:
		return StatusError
	}
}

func qualifyGitCommand(tokens []string, exitCode int) SemanticExitStatus {
	subcmd := ""
	var subcmdArgs []string
	skipNext := false

	for i := 1; i < len(tokens); i++ {
		tok := tokens[i]
		if skipNext {
			skipNext = false
			continue
		}
		if tok == "-C" || tok == "-c" || tok == "--git-dir" || tok == "--work-tree" {
			skipNext = true
			continue
		}
		if strings.HasPrefix(tok, "-") {
			continue
		}
		subcmd = strings.ToLower(tok)
		subcmdArgs = tokens[i+1:]
		break
	}

	switch subcmd {
	case "diff", "diff-files", "diff-index":
		if exitCode == 1 {
			return StatusDiffPresent
		}
		return StatusError

	case "grep":
		if exitCode == 1 {
			return StatusPatternNotFound
		}
		if exitCode == 2 {
			return StatusSyntaxError
		}
		return StatusError

	case "merge-base":
		hasIsAncestor := false
		for _, a := range subcmdArgs {
			if a == "--is-ancestor" {
				hasIsAncestor = true
				break
			}
		}
		if hasIsAncestor && exitCode == 1 {
			return StatusPredicateFalse
		}
		if exitCode == 1 {
			return StatusMatchNotFound
		}
		return StatusError

	case "check-ref-format":
		if exitCode == 1 {
			return StatusPredicateFalse
		}
		return StatusError

	default:
		return StatusError
	}
}

// QualifyExit returns a full JSONHostBoundary qualification for the command and exit code.
func QualifyExit(command string, exitCode int) JSONHostBoundary {
	return QualifyExitCode(command, exitCode).HostBoundary(command, exitCode)
}

// QualifyHostBoundary returns a JSONHostBoundary with complete execution metadata
// including cwd, stdout, and stderr.
func QualifyHostBoundary(command string, exitCode int, cwd, stdout, stderr string) JSONHostBoundary {
	boundary := QualifyExit(command, exitCode)
	boundary.Cwd = cwd
	boundary.Stdout = stdout
	boundary.Stderr = stderr
	return boundary
}

// IsBenignExit reports whether command's exit code represents a benign, non-fatal outcome.
func IsBenignExit(command string, exitCode int) bool {
	return QualifyExitCode(command, exitCode).IsBenign()
}

// IsFatalExit reports whether command's exit code represents a fatal process crash or error.
func IsFatalExit(command string, exitCode int) bool {
	return QualifyExitCode(command, exitCode).IsFatal()
}

// ShouldEscalateFailure reports whether an exit outcome should increment failure counters
// or trigger error recovery. Returns false for exit 0 or benign exits (e.g. exit 1 on grep/git diff).
func ShouldEscalateFailure(command string, exitCode int) bool {
	if exitCode == 0 {
		return false
	}
	return !IsBenignExit(command, exitCode)
}

// FailureCounter tracks command failure escalations, ignoring clean and benign negative query exits.
type FailureCounter struct {
	escalations int
}

// Record evaluates a command execution. If the failure is non-benign, it increments
// the failure counter and returns true. If benign (or exit 0), it returns false without incrementing.
func (fc *FailureCounter) Record(command string, exitCode int) bool {
	if ShouldEscalateFailure(command, exitCode) {
		fc.escalations++
		return true
	}
	return false
}

// Count returns the number of non-benign failure escalations.
func (fc *FailureCounter) Count() int {
	return fc.escalations
}

// Reset clears the failure counter.
func (fc *FailureCounter) Reset() {
	fc.escalations = 0
}

func cleanExeName(name string) string {
	base := path.Base(strings.ReplaceAll(name, `\`, "/"))
	base = strings.ToLower(base)
	base = strings.TrimSuffix(base, ".exe")
	return base
}

func tokenizeCommand(cmd string) []string {
	var tokens []string
	var cur strings.Builder
	inSingle := false
	inDouble := false
	escaped := false

	for i := 0; i < len(cmd); i++ {
		ch := cmd[i]
		if escaped {
			cur.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' && !inSingle {
			if i+1 < len(cmd) {
				next := cmd[i+1]
				if next == ' ' || next == '\t' || next == '"' || next == '\'' || next == '\\' {
					escaped = true
					continue
				}
			}
			cur.WriteByte(ch)
			continue
		}
		if ch == '\'' && !inDouble {
			inSingle = !inSingle
			continue
		}
		if ch == '"' && !inSingle {
			inDouble = !inDouble
			continue
		}
		if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') && !inSingle && !inDouble {
			if cur.Len() > 0 {
				tokens = append(tokens, cur.String())
				cur.Reset()
			}
			continue
		}
		cur.WriteByte(ch)
	}
	if cur.Len() > 0 {
		tokens = append(tokens, cur.String())
	}
	return tokens
}

func unwrapCommand(tokens []string) []string {
	if len(tokens) == 0 {
		return tokens
	}
	// Skip leading environment variable assignments like FOO=bar
	idx := 0
	for idx < len(tokens) && strings.Contains(tokens[idx], "=") && !strings.HasPrefix(tokens[idx], "-") {
		idx++
	}
	if idx >= len(tokens) {
		return tokens
	}
	tokens = tokens[idx:]

	exe := cleanExeName(tokens[0])
	switch exe {
	case "sh", "bash", "zsh", "dash":
		for i := 1; i < len(tokens); i++ {
			if (tokens[i] == "-c" || tokens[i] == "-lc") && i+1 < len(tokens) {
				inner := tokenizeCommand(tokens[i+1])
				if len(inner) > 0 {
					return unwrapCommand(inner)
				}
			}
		}
	case "powershell", "pwsh":
		for i := 1; i < len(tokens); i++ {
			arg := strings.ToLower(tokens[i])
			if (arg == "-command" || arg == "-c") && i+1 < len(tokens) {
				inner := tokenizeCommand(tokens[i+1])
				if len(inner) > 0 {
					return unwrapCommand(inner)
				}
			}
		}
	case "cmd":
		for i := 1; i < len(tokens); i++ {
			if strings.EqualFold(tokens[i], "/c") && i+1 < len(tokens) {
				inner := tokenizeCommand(strings.Join(tokens[i+1:], " "))
				if len(inner) > 0 {
					return unwrapCommand(inner)
				}
			}
		}
	}
	return tokens
}
