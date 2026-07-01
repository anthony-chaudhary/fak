package rehome

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Match is one account dir holding a session's transcript, mirroring the dicts
// resume_resolver._locate_matches builds. Size is the transcript's byte length — the
// cheap content-freshness signal the duplicate-owner re-selection uses (these
// transcripts are append-only, so fewer turns == strictly smaller).
type Match struct {
	ConfigDir string
	Account   string
	Project   string
	ModTime   time.Time
	Size      int64
	IsHost    bool
}

// Owner is the selected on-disk owner of a session: the newest-mtime, host-last Match
// plus the duplicate-fork summary (DupCount / AllAccounts) resume_resolver.locate_owner
// stamps for the cross-account decision.
type Owner struct {
	Match
	DupCount    int
	AllAccounts []string
}

// isHost reports whether a config dir is the host ~/.claude login — the account `c`
// keeps OFF the rotation, chosen as owner only when it is the SOLE one. Mirrors
// resume_resolver._is_host (basename == ".claude").
func isHost(configDir string) bool {
	return filepath.Base(strings.TrimRight(configDir, `\/`)) == ".claude"
}

// LocateMatches returns every <home>/.claude* account dir holding this session's
// <sid>.jsonl, each as a Match. It is the Go port of resume_resolver._locate_matches:
// the raw input shared by the owner pick and the duplicate-owner re-selection. A dir
// or file that cannot be statted is skipped (fail-open), never fatal.
func LocateMatches(sid, home string) []Match {
	var matches []Match
	acctDirs, err := filepath.Glob(filepath.Join(home, ".claude*"))
	if err != nil {
		return nil
	}
	for _, acctDir := range acctDirs {
		if fi, err := os.Stat(acctDir); err != nil || !fi.IsDir() {
			continue
		}
		projRoot := filepath.Join(acctDir, "projects")
		if fi, err := os.Stat(projRoot); err != nil || !fi.IsDir() {
			continue
		}
		files, err := filepath.Glob(filepath.Join(projRoot, "*", sid+".jsonl"))
		if err != nil {
			continue
		}
		for _, f := range files {
			st, err := os.Stat(f)
			if err != nil {
				continue
			}
			matches = append(matches, Match{
				ConfigDir: acctDir,
				Account:   filepath.Base(acctDir),
				Project:   filepath.Base(filepath.Dir(f)),
				ModTime:   st.ModTime(),
				Size:      st.Size(),
				IsHost:    isHost(acctDir),
			})
		}
	}
	return matches
}

// LocateOwner returns the on-disk owner record for session sid, or nil when no
// ~/.claude* account holds it. It is the Go port of resume_resolver.locate_owner:
// among non-host accounts the freshest .jsonl wins; the host (~/.claude) is chosen
// only when it is the SOLE owner. Scanning every projects/* (not just the current cwd
// slug) makes the lookup robust to a session created under a different directory.
//
// Ties on ModTime resolve to the lexicographic glob order of the account dirs (Go's
// filepath.Glob is sorted, so this is deterministic); the Python leaves exact-mtime
// ties to glob order, which the duplicate-owner re-selection then re-probes anyway.
func LocateOwner(sid, home string) *Owner {
	matches := LocateMatches(sid, home)
	if len(matches) == 0 {
		return nil
	}
	pool := make([]Match, 0, len(matches))
	for _, m := range matches {
		if !m.IsHost {
			pool = append(pool, m)
		}
	}
	if len(pool) == 0 {
		pool = append(pool, matches...)
	}
	sort.SliceStable(pool, func(i, j int) bool {
		return pool[i].ModTime.After(pool[j].ModTime)
	})
	accts := make([]string, 0, len(matches))
	for _, m := range matches {
		accts = append(accts, m.Account)
	}
	sort.Strings(accts)
	return &Owner{
		Match:       pool[0],
		DupCount:    len(matches),
		AllAccounts: accts,
	}
}
