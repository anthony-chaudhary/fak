package workerworktree

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
	"github.com/anthony-chaudhary/fak/internal/processstart"
)

// The scorecard analyzer is the candidate tree's own Python script and walks a
// real directory, so the corpus cannot stay in memory without changing that
// tree-versioned contract. Instead every scratch tree names its owner
// (pid + process start) so any later call can prove it orphaned and reclaim it,
// even when the owning process was killed before its deferred cleanup ran.
const disambiguationScratchPrefix = "fak-disambiguation-tree-"

// Untagged scratch trees come from binaries that predate owner tagging. No call
// outlives maxDisambiguationTimeout, so one this old has no live user.
const disambiguationLegacyScratchAge = 4 * maxDisambiguationTimeout

var (
	disambiguationScratchBase = os.TempDir

	disambiguationScratchMu   sync.Mutex
	disambiguationScratchHeld = map[string]bool{}
	disambiguationScratchReap atomic.Bool

	disambiguationScratchRemoveBackoff = []time.Duration{
		50 * time.Millisecond, 200 * time.Millisecond, 800 * time.Millisecond,
	}
)

type disambiguationScratchOwner struct {
	PID int
	// Start is the owner's process start in Unix nanoseconds; 0 means the
	// identity was unavailable and only PID liveness can be judged.
	Start int64
}

func currentDisambiguationScratchOwner() disambiguationScratchOwner {
	owner := disambiguationScratchOwner{PID: os.Getpid()}
	if started, ok := processstart.Start(owner.PID); ok && !started.IsZero() {
		owner.Start = started.UnixNano()
	}
	return owner
}

func (o disambiguationScratchOwner) pattern() string {
	return fmt.Sprintf("%s%d-%d-*", disambiguationScratchPrefix, o.PID, o.Start)
}

func parseDisambiguationScratchOwner(name string) (disambiguationScratchOwner, bool) {
	rest, ok := strings.CutPrefix(name, disambiguationScratchPrefix)
	if !ok {
		return disambiguationScratchOwner{}, false
	}
	parts := strings.SplitN(rest, "-", 3)
	if len(parts) != 3 {
		return disambiguationScratchOwner{}, false
	}
	pid, err := strconv.Atoi(parts[0])
	if err != nil || pid <= 0 {
		return disambiguationScratchOwner{}, false
	}
	start, err := strconv.ParseInt(parts[1], 10, 64)
	if err != nil || start < 0 {
		return disambiguationScratchOwner{}, false
	}
	return disambiguationScratchOwner{PID: pid, Start: start}, true
}

// Linux derives process start from a boot time that moves when the wall clock
// steps, so a live owner can read back a slightly different start.
const disambiguationScratchStartTolerance = 2 * time.Second

// dead is true only on proof: the PID is gone, or it now names a process that
// started at a different instant (PID reuse). Unknown identity keeps the tree.
func (o disambiguationScratchOwner) dead() bool {
	if !processalive.Check(o.PID) {
		return true
	}
	if o.Start == 0 {
		return false
	}
	started, ok := processstart.Start(o.PID)
	if !ok || started.IsZero() {
		return false
	}
	drift := started.UnixNano() - o.Start
	if drift < 0 {
		drift = -drift
	}
	return time.Duration(drift) > disambiguationScratchStartTolerance
}

// createDisambiguationScratch starts reclaiming orphaned sibling trees, then
// creates an owner-tagged root. The reclaim runs in the background, one pass per
// process at a time, because deleting gigabytes of leaked trees must not spend
// the witness deadline. release removes the root and reports a removal that
// still fails after retries; the tree is then left for a later reclaim pass.
func createDisambiguationScratch() (dir string, release func() error, err error) {
	base := disambiguationScratchBase()
	self := currentDisambiguationScratchOwner()
	if disambiguationScratchReap.CompareAndSwap(false, true) {
		go func() {
			defer disambiguationScratchReap.Store(false)
			reapDisambiguationScratch(base, self, time.Now())
		}()
	}

	disambiguationScratchMu.Lock()
	dir, err = os.MkdirTemp(base, self.pattern())
	if err == nil {
		disambiguationScratchHeld[dir] = true
	}
	disambiguationScratchMu.Unlock()
	if err != nil {
		return "", nil, err
	}
	release = func() error {
		removeErr := removeDisambiguationScratch(dir)
		disambiguationScratchMu.Lock()
		delete(disambiguationScratchHeld, dir)
		disambiguationScratchMu.Unlock()
		return removeErr
	}
	return dir, release, nil
}

// reapDisambiguationScratch removes scratch trees under base whose owner is
// provably dead, trees this process leaked from an earlier failed release, and
// untagged legacy trees older than any call can run.
func reapDisambiguationScratch(base string, self disambiguationScratchOwner, now time.Time) {
	entries, err := os.ReadDir(base)
	if err != nil {
		return
	}
	var victims []string
	disambiguationScratchMu.Lock()
	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() || !strings.HasPrefix(name, disambiguationScratchPrefix) {
			continue
		}
		path := filepath.Join(base, name)
		if disambiguationScratchHeld[path] {
			continue
		}
		owner, tagged := parseDisambiguationScratchOwner(name)
		switch {
		case tagged && owner == self:
		case tagged:
			if !owner.dead() {
				continue
			}
		default:
			info, infoErr := entry.Info()
			if infoErr != nil || now.Sub(info.ModTime()) < disambiguationLegacyScratchAge {
				continue
			}
		}
		// Claim the victim so a concurrent caller in this process skips it.
		disambiguationScratchHeld[path] = true
		victims = append(victims, path)
	}
	disambiguationScratchMu.Unlock()

	for _, path := range victims {
		_ = removeDisambiguationScratch(path)
		disambiguationScratchMu.Lock()
		delete(disambiguationScratchHeld, path)
		disambiguationScratchMu.Unlock()
	}
}

// removeDisambiguationScratch retries because Windows refuses to delete a file
// for a short while after the analyzer process that held it has exited.
func removeDisambiguationScratch(path string) error {
	err := os.RemoveAll(path)
	for _, wait := range disambiguationScratchRemoveBackoff {
		if err == nil {
			return nil
		}
		time.Sleep(wait)
		err = os.RemoveAll(path)
	}
	if err != nil {
		return err
	}
	if _, statErr := os.Lstat(path); statErr == nil {
		return fmt.Errorf("scratch tree %s still present after removal", path)
	}
	return nil
}
