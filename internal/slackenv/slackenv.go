// Package slackenv is the ONE resolver for fak's Slack-surface configuration: it
// reads a key from the process environment or a gitignored .env.slack.local file
// (walking up from the working directory), so the "one gitignored file configures
// every workspace" idiom lives in a single tested place instead of eight verbatim
// copies.
//
// Every outbound Slack publisher (internal/scoreboard, blockerpost, benchpost,
// dispatchpost, dojopost, marketing, nodeusagepost) and the chatrelay bridge resolve
// their bot token and channel id through here. Before this package each one carried a
// byte-identical envFileValue walk-up; the duplication meant a fix or a test had to be
// applied eight times and usually was not. Now they delegate to FileValue.
//
// It holds NO secret and NO channel id in source: it only knows HOW to read the key an
// operator set, never WHICH value. Pure stdlib (os + path/filepath + strings); tier-1
// foundation, off the hot path.
package slackenv

import (
	"os"
	"path/filepath"
	"strings"
)

// EnvFileName is the gitignored file every Slack surface reads its token/channel from.
// It is exported so a diagnostic ("fak slack check") can name the file it consulted.
const EnvFileName = ".env.slack.local"

// maxWalkUp bounds how many directories FileValue searches for EnvFileName, starting at
// the working directory and ascending. Six levels covers a deep monorepo checkout
// (repo/cmd/foo/bar/...) while keeping the walk bounded — it never climbs to the
// filesystem root scanning unrelated parents.
const maxWalkUp = 6

// FileValue walks up from the working directory looking for .env.slack.local and returns
// the value of the first `key=...` line it finds. An optional `export ` prefix is
// tolerated and surrounding whitespace is trimmed. It returns "" when the file is absent
// at every level, the key is not present in any file found, the matched value is blank,
// or the working directory cannot be determined — callers treat "" as "unset" and fall
// back or error. A file that exists but lacks the key does NOT stop the walk; the search
// continues to the parent, matching the historical per-package behavior exactly.
func FileValue(key string) string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	return fileValueFrom(dir, key)
}

// fileValueFrom is FileValue with an explicit starting directory, the seam tests drive so
// they never have to chdir the process.
func fileValueFrom(dir, key string) string {
	for i := 0; i < maxWalkUp; i++ {
		if v, ok := scanFile(filepath.Join(dir, EnvFileName), key); ok {
			return v
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break // reached the filesystem root
		}
		dir = parent
	}
	return ""
}

// scanFile returns (value, true) for the first `key=...` line in the file at path and
// (\"\", false) when the file is unreadable or the key is absent. A present-but-blank value
// (`KEY=`) returns ("", true) so it ends the walk the same way a non-blank match would —
// an operator who deliberately blanks a key in a near checkout is not overridden by a
// value set further up the tree.
func scanFile(path, key string) (string, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	prefix := key + "="
	for _, ln := range strings.Split(string(b), "\n") {
		ln = strings.TrimSpace(ln)
		ln = strings.TrimPrefix(ln, "export ")
		ln = strings.TrimSpace(ln)
		if v, ok := strings.CutPrefix(ln, prefix); ok {
			return strings.TrimSpace(v), true
		}
	}
	return "", false
}

// Source names where a resolved value came from, for diagnostics.
type Source string

const (
	// SourceUnset means no value was found in the environment or the file.
	SourceUnset Source = "unset"
	// SourceEnv means the value came from the process environment.
	SourceEnv Source = "env"
	// SourceFile means the value came from a .env.slack.local line.
	SourceFile Source = "file"
)

// Resolved is a value plus where it was found.
type Resolved struct {
	Value  string // the resolved value ("" when unset)
	Source Source // env | file | unset
	Key    string // the env/file key consulted
}

// Set reports whether a value was found.
func (r Resolved) Set() bool { return r.Value != "" }

// Lookup resolves key from the process environment first, then a .env.slack.local line,
// recording which source supplied it. It is the env-then-file half every package's
// ResolveToken/ResolveChannel applies; surface-specific fallbacks (e.g. blockers falling
// back to the scoreboard token, or a built-in public channel default) layer on top in the
// caller. An unset key returns a zero-value-but-keyed Resolved so a diagnostic can still
// report which key it tried.
func Lookup(key string) Resolved {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return Resolved{Value: v, Source: SourceEnv, Key: key}
	}
	if v := FileValue(key); v != "" {
		return Resolved{Value: v, Source: SourceFile, Key: key}
	}
	return Resolved{Source: SourceUnset, Key: key}
}
