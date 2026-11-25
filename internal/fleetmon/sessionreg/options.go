package sessionreg

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountprobe"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/resume/rehome"
)

// DefaultWindowH is the scan window the registry writers use: every LIVE and recently
// stopped session, without tailing a crash-loop's thousands of hour-old corpses.
const DefaultWindowH = 3.0

// FromEnv resolves Options the way tools/fleet_sessions.py resolves its module globals,
// for a checkout rooted at repoRoot (whose tools/ holds the account policy and the clone
// registry fallback). now is the pass's clock.
func FromEnv(repoRoot string, now time.Time) Options {
	toolsDir := pyJoin(repoRoot, "tools")
	paths := fleetaccounts.ResolvePaths(toolsDir)
	o := Options{
		Home:          envOr("FLEET_USER_HOME", paths.Home),
		ConfigHome:    paths.ConfigHome,
		RegDir:        resolveRegDir(toolsDir),
		Policy:        fleetaccounts.LoadPolicy(paths),
		Now:           now.UTC().Truncate(time.Microsecond),
		WindowH:       envFloat("FLEET_WINDOW_H", DefaultWindowH),
		MaxAgeMin:     1e9,
		AppVersion:    appVersion(repoRoot),
		RehomeCap:     rehome.RehomeCap(),
		EscalateAfter: envInt("FAK_RESUME_ESCALATE_AFTER", 2),
		MaxAttempts:   envInt("FAK_MAX_ATTEMPTS", 8),
		LoadRehome:    true,
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv(rehome.LoadSpreadEnv))) {
	case "0", "false", "off", "no":
		o.LoadRehome = false
	}
	return o
}

// resolveRegDir is the host's one registry dir: $FLEET_REG_DIR, else the ladder
// accountprobe.ResolveRegDir walks, with its clone rung anchored at this checkout.
func resolveRegDir(toolsDir string) string {
	if v := os.Getenv("FLEET_REG_DIR"); v != "" {
		return v
	}
	c := accountprobe.ResolveRegDir()
	if c.Rung == accountprobe.RungClone {
		return pyJoin(toolsDir, "_registry")
	}
	return c.Dir
}

// appVersion is tools/appversion.app_version: $FAK_APP_VERSION, else the nearest VERSION
// file at or above repoRoot, "dev" when absent, empty, or carrying conflict markers.
func appVersion(repoRoot string) string {
	if v := strings.TrimSpace(os.Getenv("FAK_APP_VERSION")); v != "" {
		return v
	}
	dir, err := filepath.Abs(repoRoot)
	if err != nil {
		dir = repoRoot
	}
	for {
		if data, rerr := os.ReadFile(filepath.Join(dir, "VERSION")); rerr == nil {
			v := strings.TrimSpace(string(data))
			for _, line := range strings.Split(v, "\n") {
				for _, m := range []string{"<<<<<<<", "=======", ">>>>>>>"} {
					if strings.HasPrefix(line, m) {
						return "dev"
					}
				}
			}
			if v == "" {
				return "dev"
			}
			return v
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "dev"
		}
		dir = parent
	}
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envFloat(k string, def float64) float64 {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func envInt(k string, def int) int {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
