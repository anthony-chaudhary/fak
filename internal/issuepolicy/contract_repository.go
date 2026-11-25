package issuepolicy

import (
	"regexp"
	"strings"
)

// repositoryNameRE validates a canonical "owner/repo" qualifier. It rejects
// values that do not carry exactly one slash, so a malformed owner or URL
// fragment cannot be normalized into another repository's identity.
var repositoryNameRE = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?/[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$`)

// normalizeRepository canonicalizes an optional repository qualifier. An empty
// value is a valid, explicitly unqualified reference. A non-empty value must be
// a well-formed "owner/repo" (case-insensitive; normalized to lower case);
// malformed values are rejected so they cannot be mistaken for another
// repository.
func normalizeRepository(repo string) (string, bool) {
	repo = strings.TrimSpace(repo)
	if repo == "" {
		return "", true
	}
	if !repositoryNameRE.MatchString(repo) {
		return "", false
	}
	return strings.ToLower(repo), true
}

// CompanionRepo names one of the two companion repositories an issue may
// belong to. The public core repo is anthony-chaudhary/fak; fak-private is
// the private companion.
type CompanionRepo int

const (
	// RepoUnresolved means the owning companion repository could not be
	// determined from the declared paths.
	RepoUnresolved CompanionRepo = iota
	// RepoPublic is the public core repository anthony-chaudhary/fak.
	RepoPublic
	// RepoPrivate is the private companion repository anthony-chaudhary/fak-private.
	RepoPrivate
)

// Name returns the canonical "owner/repo" qualifier for a resolved companion
// repository. RepoUnresolved has no name.
func (c CompanionRepo) Name() string {
	switch c {
	case RepoPublic:
		return "anthony-chaudhary/fak"
	case RepoPrivate:
		return "anthony-chaudhary/fak-private"
	default:
		return ""
	}
}

// PathRepoClass classifies one declared path against the two companion
// repository roots.
type PathRepoClass int

const (
	// PathUnresolved means the path could not be classified at all.
	PathUnresolved PathRepoClass = iota
	// PathPublicOnly means the path exists only under the public root.
	PathPublicOnly
	// PathPrivateOnly means the path exists only under the private root.
	PathPrivateOnly
	// PathBoth means the path exists under both roots and therefore carries
	// no repository signal.
	PathBoth
	// PathNeither means the path exists under neither root.
	PathNeither
)

// OwnCompanionRepo derives the owning companion repository from the
// classification of every declared path. It is definitive only when at
// least one path resolves and every resolvable path agrees on a single
// repository. PathBoth and PathNeither carry no signal and never count; a
// split or an all-unresolved set leaves the repository unresolved.
func OwnCompanionRepo(classes []PathRepoClass) (CompanionRepo, bool) {
	owner := RepoUnresolved
	seen := false
	for _, c := range classes {
		switch c {
		case PathPublicOnly:
			if seen && owner != RepoPublic {
				return RepoUnresolved, false
			}
			owner, seen = RepoPublic, true
		case PathPrivateOnly:
			if seen && owner != RepoPrivate {
				return RepoUnresolved, false
			}
			owner, seen = RepoPrivate, true
		default:
			// PathUnresolved, PathBoth, PathNeither: no repository signal.
		}
	}
	if !seen {
		return RepoUnresolved, false
	}
	return owner, true
}
