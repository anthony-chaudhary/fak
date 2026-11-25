package sessionreg

import (
	"regexp"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

var sidRE = regexp.MustCompile(`[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}`)

// ProcessLiveSIDs is the live-driver census: the session ids on a running claude.exe's
// command line. It fails open to an empty set, because an unreadable process table must
// never strand a crashed session; the watchdog's own skip-live guard is the second belt.
// The census is Windows-only, matching the scrape it replaces.
func ProcessLiveSIDs() map[string]bool {
	out := map[string]bool{}
	if runtime.GOOS != "windows" {
		return out
	}
	procs, _ := procguard.CollectRelations()
	for _, p := range procs {
		if name := strings.ToLower(p.Name); name != "claude" && name != "claude.exe" {
			continue
		}
		for _, m := range sidRE.FindAllString(p.Cmdline, -1) {
			out[m] = true
		}
	}
	return out
}
