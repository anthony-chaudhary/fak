package adjudicator

import (
	"errors"
	"fmt"
	"strings"
)

var ErrCLIGrammarNotApplicable = errors.New("cli grammar not applicable")

var ghReadOnlyGrammar = map[string]map[string]bool{
	"issue":    {"list": true, "view": true, "status": true},
	"pr":       {"list": true, "view": true, "status": true, "checks": true, "diff": true},
	"repo":     {"view": true, "list": true},
	"run":      {"list": true, "view": true},
	"workflow": {"list": true, "view": true},
	"release":  {"list": true, "view": true},
	"label":    {"list": true},
	"api":      {"get": true},
}

var gitReadOnlyGrammar = map[string]bool{
	"status": true, "diff": true, "show": true, "log": true, "grep": true,
	"branch": true, "tag": true, "remote": true, "rev-parse": true,
	"merge-base": true, "ls-files": true, "ls-tree": true, "cat-file": true,
	"for-each-ref": true, "describe": true,
}

// attenuateCLIGrammar positively recognizes read-only gh/git command forms and
// removes cross-repository gh search qualifiers. It rejects compound shell
// programs: each arg rule governs exactly one CLI invocation, never a pipeline.
func attenuateCLIGrammar(command string) (rewritten string, changed bool, err error) {
	rawSegments := quoteAwareSegmentTexts(command)
	if len(rawSegments) != 1 {
		return "", false, fmt.Errorf("expected one gh/git invocation")
	}
	words := strings.Fields(rawSegments[0])
	for len(words) > 0 && (envAssignmentRE.MatchString(words[0]) || commandWrapperHeads[strings.ToLower(words[0])]) {
		words = words[1:]
	}
	if len(words) == 0 {
		return "", false, fmt.Errorf("expected one gh/git invocation")
	}
	head := strings.ToLower(baseCommand(words[0]))
	switch head {
	case "gh", "gh.exe":
		return attenuateGH(words)
	case "git", "git.exe":
		return command, false, validateGit(words)
	default:
		return command, false, ErrCLIGrammarNotApplicable
	}
}

func baseCommand(s string) string {
	s = strings.ReplaceAll(s, "\\", "/")
	if i := strings.LastIndexByte(s, '/'); i >= 0 {
		return s[i+1:]
	}
	return s
}

func attenuateGH(words []string) (string, bool, error) {
	if len(words) < 2 {
		return "", false, fmt.Errorf("gh command is missing a family")
	}
	family := strings.ToLower(words[1])
	if family == "search" {
		if len(words) < 3 {
			return "", false, fmt.Errorf("gh search is missing its kind")
		}
		kind := strings.ToLower(words[2])
		if kind != "issues" && kind != "prs" && kind != "repos" && kind != "commits" && kind != "code" {
			return "", false, fmt.Errorf("gh search kind %q is not read-only", kind)
		}
		kept := append([]string(nil), words[:3]...)
		changed := false
		for _, word := range words[3:] {
			lower := strings.ToLower(strings.Trim(word, "'\""))
			if strings.HasPrefix(lower, "repo:") || strings.HasPrefix(lower, "org:") || strings.HasPrefix(lower, "user:") || lower == "--repo" || strings.HasPrefix(lower, "--repo=") {
				changed = true
				continue
			}
			kept = append(kept, word)
		}
		return strings.Join(kept, " "), changed, nil
	}
	if len(words) < 3 || !ghReadOnlyGrammar[family][strings.ToLower(words[2])] {
		return "", false, fmt.Errorf("gh %s form is not in the read-only grammar", family)
	}
	for _, word := range words[3:] {
		if strings.HasPrefix(strings.ToLower(word), "--repo=") || strings.EqualFold(word, "--repo") {
			return "", false, fmt.Errorf("cross-repository flag is not allowed")
		}
	}
	return strings.Join(words, " "), false, nil
}

func validateGit(words []string) error {
	i := 1
	for i < len(words) && strings.HasPrefix(words[i], "-") {
		if words[i] == "-C" || words[i] == "-c" {
			i += 2
		} else {
			i++
		}
	}
	if i >= len(words) || !gitReadOnlyGrammar[strings.ToLower(words[i])] {
		return fmt.Errorf("git form is not in the read-only grammar")
	}
	return nil
}
