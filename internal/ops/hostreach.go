package ops

import (
	"os"
	"path/filepath"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
)

// HostReach is every effect a Tick may have outside its RepoRoot: the host Go
// build cache, the host temp dumps, the host-wide process census, and the
// authority to kill what that census flags.
//
// The zero value reaches nothing, so an Engine built by NewEngine (tests,
// library callers) is confined to its root. Only an entrypoint that is the
// operator's own maintenance daemon passes HostReachFromEnv explicitly.
type HostReach struct {
	// GoCacheRoot resolves the Go build cache to sweep; nil skips tier 2.
	GoCacheRoot func() string
	// TempRoots lists the host temp directories tier 3 prunes; nil skips tier 3.
	TempRoots func() []string
	// CollectProcesses is the process census; nil skips the process tier.
	CollectProcesses func() ([]procguard.Proc, string)
	// EnactProcessReap authorizes kills. Config.OrphanReapEnabled is policy;
	// this is the separate grant that the caller owns the host.
	EnactProcessReap bool
}

// HostReachFromEnv is the full host reach of the shipped ops daemon: the Go
// cache named by GOCACHE (or the user cache default), the platform temp dumps,
// the whole-host process census, and kill authority.
func HostReachFromEnv() HostReach {
	return HostReach{
		GoCacheRoot: func() string {
			return treedoctor.GoCacheRootFromEnv(os.Getenv, os.UserCacheDir)
		},
		TempRoots:        defaultHostTempRoots,
		CollectProcesses: procguard.CollectProcesses,
		EnactProcessReap: true,
	}
}

func defaultHostTempRoots() []string {
	if runtime.GOOS == "windows" {
		if t := os.Getenv("TEMP"); t != "" {
			return []string{filepath.Join(t, "opencode")}
		}
		return nil
	}
	return []string{"/tmp"}
}

// ScopeSkip is the closed vocabulary for a host-reaching step a tick declined
// because its engine was not granted that reach.
type ScopeSkip string

const (
	SkipGoCacheUnauthorized       ScopeSkip = "GO_CACHE_HOST_REACH_UNAUTHORIZED"
	SkipHostTempUnauthorized      ScopeSkip = "HOST_TEMP_REACH_UNAUTHORIZED"
	SkipProcessCensusUnauthorized ScopeSkip = "PROCESS_CENSUS_HOST_REACH_UNAUTHORIZED"
	SkipProcessReapUnauthorized   ScopeSkip = "PROCESS_REAP_ENACT_UNAUTHORIZED"
)
