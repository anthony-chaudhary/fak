package version

import (
	"strconv"
	"strings"
)

// Build-time variables, set via -ldflags.
var (
	Version   = "dev"     // -X github.com/anthony-chaudhary/fak/internal/l3server/version.Version=...
	Commit    = "unknown" // -X github.com/anthony-chaudhary/fak/internal/l3server/version.Commit=...
	BuildDate = "unknown" // -X github.com/anthony-chaudhary/fak/internal/l3server/version.BuildDate=...
)

// Protocol and compatibility constants.
const (
	ProtocolVersion  = 1       // wire framing version (the 0x01 byte)
	APIVersion       = 1       // handshake API version
	ServerVersion    = "1.5.7" // semver
	MinClientVersion = "0.2.0"
	MinSGLangVersion = "0.5.9"
)

// CompareVersions compares two "X.Y.Z" semver strings numerically.
// Returns -1 if a < b, 0 if a == b, +1 if a > b.
// Non-numeric or malformed segments are treated as 0.
func CompareVersions(a, b string) int {
	aParts := strings.Split(a, ".")
	bParts := strings.Split(b, ".")

	maxLen := len(aParts)
	if len(bParts) > maxLen {
		maxLen = len(bParts)
	}

	for i := 0; i < maxLen; i++ {
		av := 0
		bv := 0
		if i < len(aParts) {
			av, _ = strconv.Atoi(aParts[i])
		}
		if i < len(bParts) {
			bv, _ = strconv.Atoi(bParts[i])
		}
		if av < bv {
			return -1
		}
		if av > bv {
			return 1
		}
	}
	return 0
}
