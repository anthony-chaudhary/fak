package repoguard

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// ReasonBuildCacheCleanRace is the structured refusal token for deleting Go's
// build cache while another process may still be compiling against it.
const ReasonBuildCacheCleanRace = "BUILD_CACHE_CLEAN_RACE"

const buildCacheCleanFix = `use the managed isolated paths: fak-dev buildcheck --vet, or fak validate --mine <paths>. For raw work, set GOCACHE to an explicitly allocated OS-temp directory and move only that exact owned directory to the platform Trash/Recycle Bin afterward. Go creates and trims caches automatically; do not clear the ambient cache opportunistically. For coordinated host maintenance only, launch the guarded session with FAK_REPO_GUARD_SEVERITY=BUILD_CACHE_CLEAN_RACE=off`

// ClassifyBuildCacheClean recognizes an executed `go clean` command carrying
// the build-cache flag. It tokenizes command segments instead of searching raw
// text, so a quoted example, grep pattern, or commit message is not mistaken for
// an invocation. An inline GOCACHE assignment is deliberately not a carve-out:
// the command text cannot prove that the named directory is process-owned.
func ClassifyBuildCacheClean(command string) []Violation {
	return classifyBuildCacheCleanDepth(command, 0)
}

func classifyBuildCacheCleanDepth(command string, depth int) []Violation {
	if depth > 4 {
		return nil
	}
	var out []Violation
	for _, segment := range executableShellSegments(stripHeredocBodies(command)) {
		tokens, ok := shlexSplit(segment)
		if !ok {
			// A shell rejects an unterminated quote before launching any command.
			continue
		}
		if invokesGoBuildCacheClean(tokens, depth) {
			out = append(out, Violation{
				Reason:   ReasonBuildCacheCleanRace,
				Op:       "go clean -cache",
				Target:   strings.TrimSpace(segment),
				Resolved: "<ambient Go build cache>",
				Why:      "deletes cache entries that concurrent build and vet processes may still reference",
				Fix:      buildCacheCleanFix,
			})
		}
	}
	return out
}

func invokesGoBuildCacheClean(tokens []string, depth int) bool {
	verb, args := unwrapCommandPrefix(tokens)
	switch strings.ToLower(strings.TrimSuffix(pathutil.Base(verb), ".exe")) {
	case "go":
		return goCleanDeletesBuildCache(args)
	case "sh", "bash", "dash", "zsh", "ksh":
		for i := 0; i+1 < len(args); i++ {
			if args[i] == "-c" || args[i] == "--command" {
				return len(classifyBuildCacheCleanDepth(args[i+1], depth+1)) > 0
			}
		}
	}
	return false
}

// unwrapCommandPrefix resolves the command word after shell assignments, env,
// and command. These wrappers do not change which executable receives argv.
func unwrapCommandPrefix(tokens []string) (string, []string) {
	i := 0
	for i < len(tokens) {
		if _, ok := envAssignName(tokens[i]); ok {
			i++
			continue
		}
		base := strings.ToLower(strings.TrimSuffix(pathutil.Base(tokens[i]), ".exe"))
		switch base {
		case "env":
			i++
			for i < len(tokens) {
				if _, ok := envAssignName(tokens[i]); ok {
					i++
					continue
				}
				if strings.HasPrefix(tokens[i], "-") {
					i++
					continue
				}
				break
			}
			continue
		case "command":
			i++
			for i < len(tokens) && strings.HasPrefix(tokens[i], "-") {
				i++
			}
			continue
		}
		break
	}
	if i >= len(tokens) {
		return "", nil
	}
	return tokens[i], tokens[i+1:]
}

func goCleanDeletesBuildCache(args []string) bool {
	i := 0
	for i < len(args) {
		tok := args[i]
		if tok == "--" {
			return false
		}
		if tok == "-C" {
			i += 2
			continue
		}
		if strings.HasPrefix(tok, "-") {
			i++
			continue
		}
		if tok != "clean" {
			return false
		}
		return cleanArgsDeleteBuildCache(args[i+1:])
	}
	return false
}

func cleanArgsDeleteBuildCache(args []string) bool {
	for _, tok := range args {
		if tok == "--" || !strings.HasPrefix(tok, "-") {
			return false
		}
		switch strings.ToLower(tok) {
		case "-cache", "--cache", "-cache=true", "--cache=true", "-cache=1", "--cache=1":
			return true
		}
	}
	return false
}

// executableShellSegments splits only on operators outside quotes. The older
// target extractor intentionally mirrors its Python predecessor's looser split;
// this safety classifier needs the stronger property because quoted prose is a
// required negative control.
func executableShellSegments(command string) []string {
	var out []string
	var current strings.Builder
	inSingle, inDouble, escaped := false, false, false
	flush := func() {
		if s := strings.TrimSpace(current.String()); s != "" {
			out = append(out, s)
		}
		current.Reset()
	}
	for i := 0; i < len(command); i++ {
		ch := command[i]
		if escaped {
			current.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' && !inSingle {
			current.WriteByte(ch)
			escaped = true
			continue
		}
		if ch == '\'' && !inDouble {
			inSingle = !inSingle
			current.WriteByte(ch)
			continue
		}
		if ch == '"' && !inSingle {
			inDouble = !inDouble
			current.WriteByte(ch)
			continue
		}
		if !inSingle && !inDouble && (ch == ';' || ch == '|' || ch == '&' || ch == '\n') {
			flush()
			if i+1 < len(command) && command[i+1] == ch && (ch == '|' || ch == '&') {
				i++
			}
			continue
		}
		current.WriteByte(ch)
	}
	flush()
	return out
}
