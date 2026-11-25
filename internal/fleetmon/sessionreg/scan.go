package sessionreg

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// Options is everything one registry pass reads from outside the transcript tree. The scan
// is a pure function of these plus the files under Home and RegDir, so a test pins the
// clock and the roots and gets a deterministic registry.
type Options struct {
	Home       string // the user home holding the .claude* config dirs (FLEET_USER_HOME)
	ConfigHome string // the opencode config root
	RegDir     string // the registry dir read and written
	Policy     fleetaccounts.Policy
	Now        time.Time
	WindowH    float64 // scan window in hours
	MaxAgeMin  float64
	AppVersion string
	// LiveSIDs is the process census of session ids a running driver is advancing. nil
	// means no census was taken (the liveness gate then rests on on-disk LIVE copies alone).
	LiveSIDs      func() map[string]bool
	RehomeCap     int
	EscalateAfter int
	MaxAttempts   int
	LoadRehome    bool
}

// pyJoin is os.path.join: it appends with the platform separator without cleaning, so the
// paths this port publishes are the strings the Python writer published.
func pyJoin(base string, parts ...string) string {
	sep := string(filepath.Separator)
	out := base
	for _, p := range parts {
		if filepath.IsAbs(p) {
			out = p
			continue
		}
		if out == "" || strings.HasSuffix(out, sep) || strings.HasSuffix(out, "/") ||
			(runtime.GOOS == "windows" && strings.HasSuffix(out, ":")) {
			out += p
		} else {
			out += sep + p
		}
	}
	return out
}

// listDir returns dir's entries in the filesystem's own order, the order Python's
// os.scandir (and so glob.glob) walks; os.ReadDir would sort them.
func listDir(dir string) []os.DirEntry {
	f, err := os.Open(dir)
	if err != nil {
		return nil
	}
	defer f.Close()
	ents, _ := f.ReadDir(-1)
	return ents
}

func entryIsDir(dir string, e os.DirEntry) bool {
	if e.Type()&os.ModeSymlink != 0 {
		st, err := os.Stat(filepath.Join(dir, e.Name()))
		return err == nil && st.IsDir()
	}
	return e.IsDir()
}

// globMatch is fnmatch under glob's rules: case-insensitive on Windows (normcase), and a
// non-dot pattern never matches a hidden name.
func globMatch(pattern, name string) bool {
	if !strings.HasPrefix(pattern, ".") && strings.HasPrefix(name, ".") {
		return false
	}
	if runtime.GOOS == "windows" {
		pattern, name = strings.ToLower(pattern), strings.ToLower(name)
	}
	ok, _ := filepath.Match(pattern, name)
	return ok
}

// transcriptPaths is glob.glob(<proj>/*/*.jsonl) in walk order.
func transcriptPaths(proj string) []string {
	var out []string
	for _, pd := range listDir(proj) {
		if !entryIsDir(proj, pd) || !globMatch("*", pd.Name()) {
			continue
		}
		pdir := pyJoin(proj, pd.Name())
		for _, fe := range listDir(pdir) {
			if globMatch("*.jsonl", fe.Name()) {
				out = append(out, pyJoin(pdir, fe.Name()))
			}
		}
	}
	return out
}

type scanResult struct {
	rows     []*Row
	throttle *Obj
}

// scan classifies every in-window transcript under each worker .claude* config dir and
// collects the accounts currently walled by a live usage limit.
func scan(ctx context.Context, o Options, isWorker func(string) bool, rearmed map[string]bool) (scanResult, error) {
	type candidate struct {
		acct, path, project string
		st                  os.FileInfo
		row                 *Row
	}
	var cands []*candidate
	for _, ae := range listDir(o.Home) {
		if !globMatch(".claude*", ae.Name()) {
			continue
		}
		acct := ae.Name()
		proj := pyJoin(o.Home, acct, "projects")
		if st, err := os.Stat(proj); err != nil || !st.IsDir() {
			continue
		}
		if !isWorker(acct) {
			continue
		}
		for _, path := range transcriptPaths(proj) {
			base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
			project := filepath.Base(filepath.Dir(path))
			if uuidRE.MatchString(base) && !strings.HasPrefix(project, "wf_") {
				cands = append(cands, &candidate{acct: acct, path: path, project: project})
			}
		}
	}
	// A fleet host carries tens of thousands of transcripts and each stat opens the file on
	// Windows, so the stat + classify work fans out; results are reassembled in walk order.
	err := parallelEach(ctx, len(cands), func(i int) {
		c := cands[i]
		st, statErr := os.Stat(c.path)
		if statErr != nil {
			return
		}
		base := strings.TrimSuffix(filepath.Base(c.path), filepath.Ext(c.path))
		age := minutesSince(o.Now, mtimeUTC(st.ModTime()))
		if (age > o.WindowH*60 || age > o.MaxAgeMin) && !rearmed[base] {
			return
		}
		c.row = classify(c.path, st, o.Now)
		c.row.Account, c.row.Project = c.acct, c.project
	})
	res := scanResult{throttle: NewObj()}
	if err != nil {
		return res, err
	}
	ageOf := map[string]float64{}
	for _, c := range cands {
		r := c.row
		if r == nil {
			continue
		}
		res.rows = append(res.rows, r)
		if r.ThrottleReset != "" && r.Disp == "STOPPED_LIMIT" && throttleActive(r.ThrottleReset, o.Now) {
			if _, seen := res.throttle.Get(r.Account); !seen || r.AgeMin < ageOf[r.Account] {
				entry := NewObj()
				entry.Set("reset", r.ThrottleReset)
				entry.Set("age_min", r.AgeMin)
				if r.ThrottleWeekly != "" {
					entry.Set("weekly", r.ThrottleWeekly)
				}
				res.throttle.Set(r.Account, entry)
				ageOf[r.Account] = r.AgeMin
			}
		}
	}
	sort.SliceStable(res.rows, func(i, j int) bool { return res.rows[i].AgeMin < res.rows[j].AgeMin })
	return res, nil
}

// parallelEach runs fn(0..n-1) on a bounded worker pool, stopping early when ctx ends.
func parallelEach(ctx context.Context, n int, fn func(i int)) error {
	workers := runtime.NumCPU()
	if workers > 16 {
		workers = 16
	}
	if workers < 1 {
		workers = 1
	}
	next := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	var stopped error
	for i := 0; i < n; i++ {
		if cerr := ctx.Err(); cerr != nil {
			stopped = cerr
			break
		}
		next <- i
	}
	close(next)
	wg.Wait()
	if stopped == nil {
		stopped = ctx.Err()
	}
	return stopped
}

// throttleActive is fleet_accounts.throttle_is_active over a throttle entry or bare reset.
func throttleActive(info any, now time.Time) bool {
	var m map[string]any
	switch t := info.(type) {
	case *Obj:
		m, _ = plain(t).(map[string]any)
	case string:
		m = map[string]any{"reset": t}
	default:
		m = map[string]any{"reset": info}
	}
	return fleetaccounts.DisambiguateCap(m, fleetaccounts.CapObservation{}, now, fleetaccounts.DefaultCapPolicy()).Active
}
