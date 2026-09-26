// Package stamp is the public readback facade a fak deployable uses to answer one question:
// which commit is this binary, is that build clean, and is it current?
//
// It is a facade, not a second implementation. The provenance primitives live in
// internal/binstamp (the embedded VCS stamp and the Fresh/Stale/Unknown verdict),
// internal/versionskew (the ancestry-aware, refusable skew verdict) and internal/appversion
// (the application version). Code outside this module, notably fak-private's deployables,
// cannot import internal/*, so each of them grew its own readback parser. This package
// re-exports those primitives unchanged and adds exactly one new piece: ParseVersionJSON,
// the single parser for the `version --json` document any fak binary prints.
//
// The fail-closed rule is inherited, not re-derived: a dirty or unstamped binary is never
// reported current. Explain classifies it Unknown with CauseDirty or CauseUnstamped, and
// AssessSkew classifies it VerdictDirty or VerdictUnstamped, both of which are refusable.
package stamp

import (
	"bytes"
	"context"
	"debug/buildinfo"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

// Stamp is the build provenance read out of a binary: Revision (the full VCS revision, ""
// when unstamped), Dirty (built from a tree with uncommitted changes) and HasVCS (a
// vcs.revision setting was present at all).
type Stamp = binstamp.Stamp

// Freshness is the three-state verdict of comparing a stamp to a HEAD revision.
type Freshness = binstamp.Freshness

const (
	// Unknown: cannot prove fresh or stale. Callers must not act (restart, swap) on it.
	Unknown = binstamp.Unknown
	// Fresh: the stamp's clean revision equals HEAD.
	Fresh = binstamp.Fresh
	// Stale: the stamp is a clean revision that differs from HEAD.
	Stale = binstamp.Stale
)

// Cause says why Explain reached its Freshness, separating the three Unknown cases.
type Cause = binstamp.Cause

const (
	// CauseMatched: Fresh, the revision equals HEAD.
	CauseMatched = binstamp.CauseMatched
	// CauseDiverged: Stale, a clean revision that differs from HEAD.
	CauseDiverged = binstamp.CauseDiverged
	// CauseUnstamped: Unknown, no revision is embedded, so the binary cannot attest its commit.
	CauseUnstamped = binstamp.CauseUnstamped
	// CauseDirty: Unknown, the revision is a base commit that does not describe the binary.
	CauseDirty = binstamp.CauseDirty
	// CauseNoHead: Unknown, there is no HEAD revision to compare against.
	CauseNoHead = binstamp.CauseNoHead
)

// Self reads the build stamp embedded in the running process. When the toolchain recorded no
// VCS stamp it falls back to a linker-injected source-install commit, exactly as binstamp.Self.
func Self() Stamp { return binstamp.Self() }

// FromFile reads the build stamp embedded in the binary at path without executing it. It reads
// only the Go toolchain's vcs.* build settings, so a binary stamped solely through the
// -ldflags source-install commit reads as unstamped here; execute its `version --json` and
// use ParseVersionJSON to read that commit instead.
func FromFile(path string) (Stamp, error) {
	bi, err := buildinfo.ReadFile(path)
	if err != nil {
		return Stamp{}, fmt.Errorf("stamp: read build info from %s: %w", path, err)
	}
	return binstamp.FromBuildInfo(bi), nil
}

// Compare returns the Freshness of running against headRev (a full or >=7-char revision).
// Unstamped, dirty, or HEAD-less input is Unknown, never Fresh.
func Compare(running Stamp, headRev string) Freshness { return binstamp.Compare(running, headRev) }

// Explain returns the same Freshness as Compare plus the Cause that produced it.
func Explain(running Stamp, headRev string) (Freshness, Cause) {
	return binstamp.Explain(running, headRev)
}

// Verdict is the closed, ancestry-aware skew classification AssessSkew returns. Use
// Verdict.Refusable to decide whether a gate should refuse the binary.
type Verdict = versionskew.Verdict

const (
	// VerdictUnknown: the running commit could not be located against the trunk. Not refusable.
	VerdictUnknown = versionskew.Unknown
	// VerdictFresh: the running commit is the trunk tip.
	VerdictFresh = versionskew.Fresh
	// VerdictSkewed: the running commit is a strict ancestor of the trunk tip. Refusable.
	VerdictSkewed = versionskew.Skewed
	// VerdictAhead: the trunk tip is a strict ancestor of the running commit. Not refusable.
	VerdictAhead = versionskew.Ahead
	// VerdictDiverged: neither commit is an ancestor of the other. Refusable.
	VerdictDiverged = versionskew.Diverged
	// VerdictUnstamped: the binary carries no VCS revision. Refusable.
	VerdictUnstamped = versionskew.Unstamped
	// VerdictDirty: the binary was built from a tree with uncommitted changes. Refusable.
	VerdictDirty = versionskew.Dirty
)

// Relation is the git-ancestry relationship between the running commit and the trunk tip.
type Relation = versionskew.Relation

const (
	// RelUndetermined: ancestry could not be computed.
	RelUndetermined = versionskew.RelUndetermined
	// RelEqual: the running commit is the trunk tip.
	RelEqual = versionskew.RelEqual
	// RelBehind: the running commit is a strict ancestor of the trunk tip.
	RelBehind = versionskew.RelBehind
	// RelAhead: the trunk tip is a strict ancestor of the running commit.
	RelAhead = versionskew.RelAhead
	// RelDiverged: neither commit is an ancestor of the other.
	RelDiverged = versionskew.RelDiverged
)

// Assessment is a skew Verdict plus the evidence (running rev, dirty bit, trunk tip,
// relation) that produced it.
type Assessment = versionskew.Assessment

// Runner runs name+args in dir and returns the combined output and whether it exited zero.
type Runner = versionskew.Runner

// RealRunner is the default Runner: it executes the command on this host.
func RealRunner(ctx context.Context, dir, name string, args ...string) (string, bool) {
	return versionskew.RealRunner(ctx, dir, name, args...)
}

// AssessSkew classifies running against the commit trunkRef resolves to in the repository at
// dir, using git ancestry. A nil run uses RealRunner. An unstamped or dirty stamp is decided
// without calling git at all: it is VerdictUnstamped or VerdictDirty, both refusable.
func AssessSkew(ctx context.Context, run Runner, dir, trunkRef string, running Stamp) Assessment {
	if run == nil {
		run = RealRunner
	}
	return versionskew.AssessStamp(ctx, run, dir, trunkRef, running)
}

// AppVersion returns the running binary's application version, resolved exactly as
// appversion.Current: $FAK_APP_VERSION, the release -ldflags stamp, the module release tag,
// then a VERSION marker beside the executable, else "dev".
func AppVersion() string { return appversion.Current() }

// Identity is the provenance a fak binary reports through `version --json`. Build a Stamp
// from it with Identity.Stamp before comparing it to a HEAD revision.
type Identity struct {
	AppVersion string `json:"app_version"`
	Commit     string `json:"commit"`  // full vcs revision, lower-cased; "" when unstamped
	Dirty      bool   `json:"dirty"`   // built from a tree with uncommitted changes
	Stamped    bool   `json:"stamped"` // a VCS revision is embedded at all
}

// Stamp converts the identity into a Stamp. It fails closed: an identity that is not stamped,
// or whose commit is not a full 40-hex object ID, yields an unstamped Stamp, so a short or
// forged commit can never prefix-match its way to Fresh.
func (id Identity) Stamp() Stamp {
	commit := strings.ToLower(strings.TrimSpace(id.Commit))
	if !id.Stamped || !fullCommit(commit) {
		return Stamp{Dirty: id.Dirty}
	}
	return Stamp{Revision: commit, Dirty: id.Dirty, HasVCS: true}
}

var (
	// ErrMalformedVersionJSON reports output that is not a `version --json` document: invalid
	// JSON, a non-object value, trailing data, or a missing commit, dirty, or stamped field.
	ErrMalformedVersionJSON = errors.New("stamp: malformed version --json output")
	// ErrInvalidCommit reports a document whose commit is present but is not a full 40-hex
	// object ID, or which claims stamped:true with no commit.
	ErrInvalidCommit = errors.New("stamp: version --json commit is not a full 40-hex object ID")
)

// ParseVersionJSON is the one parser for the `version --json` output of any fak binary. The
// commit, dirty and stamped fields are required; app_version is optional and unknown fields
// are ignored. Surrounding whitespace and a UTF-8 byte-order mark are tolerated.
//
// An honest unstamped document (stamped:false, commit "") and a dirty document parse without
// error: they are valid observations, classified through Identity.Stamp as CauseUnstamped or
// CauseDirty. Malformed input returns ErrMalformedVersionJSON with a zero Identity, and a bad
// commit returns ErrInvalidCommit with the parsed Identity for diagnostics. In every one of
// these cases the Identity's Stamp is never Fresh against any HEAD.
func ParseVersionJSON(data []byte) (Identity, error) {
	data = bytes.TrimSpace(bytes.TrimPrefix(bytes.TrimSpace(data), []byte("\xef\xbb\xbf")))
	var raw struct {
		AppVersion *string `json:"app_version"`
		Commit     *string `json:"commit"`
		Dirty      *bool   `json:"dirty"`
		Stamped    *bool   `json:"stamped"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return Identity{}, fmt.Errorf("%w: %v", ErrMalformedVersionJSON, err)
	}
	var missing []string
	if raw.Commit == nil {
		missing = append(missing, "commit")
	}
	if raw.Dirty == nil {
		missing = append(missing, "dirty")
	}
	if raw.Stamped == nil {
		missing = append(missing, "stamped")
	}
	if len(missing) > 0 {
		return Identity{}, fmt.Errorf("%w: missing %s", ErrMalformedVersionJSON, strings.Join(missing, ", "))
	}
	id := Identity{
		Commit:  strings.ToLower(strings.TrimSpace(*raw.Commit)),
		Dirty:   *raw.Dirty,
		Stamped: *raw.Stamped,
	}
	if raw.AppVersion != nil {
		id.AppVersion = strings.TrimSpace(*raw.AppVersion)
	}
	if id.Commit == "" && !id.Stamped {
		return id, nil
	}
	if !fullCommit(id.Commit) {
		return id, fmt.Errorf("%w: %q", ErrInvalidCommit, clip(id.Commit))
	}
	return id, nil
}

// fullCommit reports whether s is a full 40-character hexadecimal Git object ID.
func fullCommit(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, r := range s {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')) {
			return false
		}
	}
	return true
}

// clip bounds a value quoted into an error so hostile input cannot flood a log line.
func clip(s string) string {
	const max = 80
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}
