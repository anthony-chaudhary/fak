package selfupdate

import (
	"fmt"
	"strconv"
	"strings"
)

type releaseVersion struct {
	core       []uint64
	prerelease string
}

// CompareReleaseVersions compares release-shaped application versions. It deliberately
// refuses opaque/dev versions: rollback prevention cannot be fail-closed when ordering is
// unknowable.
func CompareReleaseVersions(left, right string) (int, error) {
	a, err := parseReleaseVersion(left)
	if err != nil {
		return 0, fmt.Errorf("left version: %w", err)
	}
	b, err := parseReleaseVersion(right)
	if err != nil {
		return 0, fmt.Errorf("right version: %w", err)
	}
	n := len(a.core)
	if len(b.core) > n {
		n = len(b.core)
	}
	for i := 0; i < n; i++ {
		var av, bv uint64
		if i < len(a.core) {
			av = a.core[i]
		}
		if i < len(b.core) {
			bv = b.core[i]
		}
		switch {
		case av < bv:
			return -1, nil
		case av > bv:
			return 1, nil
		}
	}
	switch {
	case a.prerelease == b.prerelease:
		return 0, nil
	case a.prerelease == "":
		return 1, nil
	case b.prerelease == "":
		return -1, nil
	case a.prerelease < b.prerelease:
		return -1, nil
	default:
		return 1, nil
	}
}

func parseReleaseVersion(raw string) (releaseVersion, error) {
	raw = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(raw), "v"))
	if raw == "" {
		return releaseVersion{}, fmt.Errorf("empty")
	}
	if plus := strings.IndexByte(raw, '+'); plus >= 0 {
		raw = raw[:plus]
	}
	prerelease := ""
	if dash := strings.IndexByte(raw, '-'); dash >= 0 {
		prerelease = raw[dash+1:]
		raw = raw[:dash]
		if prerelease == "" {
			return releaseVersion{}, fmt.Errorf("empty prerelease")
		}
	}
	parts := strings.Split(raw, ".")
	if len(parts) == 0 || len(parts) > 4 {
		return releaseVersion{}, fmt.Errorf("%q is not a supported release version", raw)
	}
	core := make([]uint64, len(parts))
	for i, part := range parts {
		if part == "" {
			return releaseVersion{}, fmt.Errorf("%q has an empty component", raw)
		}
		n, err := strconv.ParseUint(part, 10, 64)
		if err != nil {
			return releaseVersion{}, fmt.Errorf("%q has a non-numeric component", raw)
		}
		core[i] = n
	}
	return releaseVersion{core: core, prerelease: prerelease}, nil
}
