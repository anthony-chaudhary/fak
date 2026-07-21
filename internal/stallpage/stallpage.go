package stallpage

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/stallscan"
)

const (
	// Schema is the durable operator-page record written when stallscan's reboot
	// high-water is crossed. It is deliberately distinct from the sample ledger:
	// stallscan.jsonl records every observation, while this ledger records only
	// earned, human-authority pages.
	Schema = "fak.stall-reboot-page.v1"

	stateSchema = "fak.stall-reboot-page-state.v1"
	pageLogName = "stallscan-reboot-pages.jsonl"
	stateName   = "stallscan-reboot-page-state.json"
	lockName    = "stallscan-reboot-page.lock"
)

// DefaultDedupWindow bounds reminders for a sustained crossing. Six hours
// matches the read-only reboot-advisor stopgap and is long enough that a 15-20s
// monitor cannot page-storm while the operator schedules a quiet reboot window.
const DefaultDedupWindow = 6 * time.Hour

// Page is one earned operator page. Advice carries the measured, thresholded
// evidence; Triage proves that the action is a genuine human-authority residual
// rather than ordinary work the fleet should do itself.
type Page struct {
	Schema      string                 `json:"schema"`
	GeneratedAt string                 `json:"generated_at"`
	Key         string                 `json:"key"`
	Advice      stallscan.RebootAdvice `json:"advice"`
	Triage      choicetriage.Verdict   `json:"triage"`
	Question    string                 `json:"question"`
	Action      string                 `json:"action"`
}

// PublishResult distinguishes a newly emitted page from a deduped repeat.
// Page is populated for both cases so callers can render the same measured
// reason without reconstructing it; Published alone controls interruption.
type PublishResult struct {
	Page      Page
	Published bool
}

type pageState struct {
	Schema string           `json:"schema"`
	Last   map[string]int64 `json:"last_page_unix_nano"`
}

// FromAdvice adapts stallscan's pure high-water decision into the paging gate.
// A reboot is destructive to every live session and therefore requires an
// explicit operator approval/scheduling decision. Naming that authority is
// load-bearing: choicetriage must classify the page HUMAN_RESIDUAL, never route
// it to an agent as a runnable reboot command.
func FromAdvice(adv stallscan.RebootAdvice, now time.Time) (Page, bool) {
	if !adv.Advised {
		return Page{}, false
	}
	question := "approve a quiet-window host reboot before the machine freezes?"
	action := "approve and schedule a quiet-window host reboot; do not reboot or kill automatically"
	v := choicetriage.Triage(choicetriage.Signal{
		Severity:    "decision",
		Source:      "host-reboot",
		Question:    question,
		Detail:      adv.Reason,
		Action:      action,
		OptionCount: 2,
	})
	return Page{
		Schema:      Schema,
		GeneratedAt: now.UTC().Format(time.RFC3339Nano),
		Key:         DedupKey(adv),
		Advice:      adv,
		Triage:      v,
		Question:    question,
		Action:      action,
	}, true
}

// FromSample is the end-to-end pure producer: a below-line sample yields no
// page; a real 30k-handle / 2k-thread crossing yields an earned human page.
func FromSample(s stallscan.Sample, now time.Time) (Page, bool) {
	return FromAdvice(stallscan.AdviseReboot(s, stallscan.DefaultRebootThresholds()), now)
}

// DedupKey implements the issue contract: sustained crossings dedupe by
// (Axis, Process), not PID. A leaking process restart may get a fresh PID while
// representing the same operator decision, so including PID would re-page it.
func DedupKey(adv stallscan.RebootAdvice) string {
	return strings.ToLower(strings.TrimSpace(adv.Axis)) + "\x00" +
		strings.ToLower(strings.TrimSpace(adv.Process))
}

// PageLogPath and StatePath expose the durable read-back locations to operator
// surfaces and witness tests. dir is the stallscan ledger directory.
func PageLogPath(dir string) string { return filepath.Join(dir, pageLogName) }
func StatePath(dir string) string   { return filepath.Join(dir, stateName) }

// Publish writes an advised crossing to the durable operator-page ledger at
// most once per (Axis, Process) in window. It is cross-process serialized: the
// native --watch loop and the PowerShell wrapper may overlap, but only one may
// win the page. A below-line sample is a true no-op and creates no files.
func Publish(dir string, sample stallscan.Sample, now time.Time, window time.Duration) (PublishResult, error) {
	page, advised := FromSample(sample, now)
	if !advised {
		return PublishResult{}, nil
	}
	if strings.TrimSpace(dir) == "" {
		return PublishResult{Page: page}, errors.New("stallpage: empty state directory")
	}
	if window <= 0 {
		window = DefaultDedupWindow
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return PublishResult{Page: page}, fmt.Errorf("stallpage: create state directory: %w", err)
	}

	lock, err := acquire(filepath.Join(dir, lockName))
	if err != nil {
		return PublishResult{Page: page}, err
	}
	defer func() {
		_ = flock.Unlock(lock)
		_ = lock.Close()
	}()

	st := readState(StatePath(dir))
	lastNS, seen := st.Last[page.Key]
	if seen {
		last := time.Unix(0, lastNS)
		// A future timestamp (clock rollback) also suppresses: time
		// uncertainty is never a reason to page-storm.
		if last.After(now) || now.Sub(last) < window {
			return PublishResult{Page: page}, nil
		}
	}

	b, err := json.Marshal(page)
	if err != nil {
		return PublishResult{Page: page}, fmt.Errorf("stallpage: encode page: %w", err)
	}
	f, err := os.OpenFile(PageLogPath(dir), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return PublishResult{Page: page}, fmt.Errorf("stallpage: open page ledger: %w", err)
	}
	_, writeErr := f.Write(append(b, '\n'))
	closeErr := f.Close()
	if writeErr != nil {
		return PublishResult{Page: page}, fmt.Errorf("stallpage: append page ledger: %w", writeErr)
	}
	if closeErr != nil {
		return PublishResult{Page: page}, fmt.Errorf("stallpage: close page ledger: %w", closeErr)
	}

	st.Last[page.Key] = now.UnixNano()
	pruneState(&st, now, 2*window)
	if err := writeState(StatePath(dir), st); err != nil {
		return PublishResult{Page: page}, err
	}
	return PublishResult{Page: page, Published: true}, nil
}

func acquire(path string) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, fmt.Errorf("stallpage: open page lock: %w", err)
	}
	for i := 0; i < 40; i++ {
		err = flock.TryLock(f)
		if err == nil {
			return f, nil
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			_ = f.Close()
			return nil, fmt.Errorf("stallpage: lock page state: %w", err)
		}
		time.Sleep(25 * time.Millisecond)
	}
	_ = f.Close()
	return nil, fmt.Errorf("stallpage: page state remained busy: %w", flock.ErrLockBusy)
}

func readState(path string) pageState {
	st := pageState{Schema: stateSchema, Last: map[string]int64{}}
	b, err := os.ReadFile(path)
	if err != nil {
		return st
	}
	var disk pageState
	if json.Unmarshal(b, &disk) != nil || disk.Schema != stateSchema || disk.Last == nil {
		// Fail open to one extra page rather than let corrupt state suppress a
		// real reboot high-water indefinitely.
		return st
	}
	return disk
}

func writeState(path string, st pageState) error {
	st.Schema = stateSchema
	if st.Last == nil {
		st.Last = map[string]int64{}
	}
	b, err := json.Marshal(st)
	if err != nil {
		return fmt.Errorf("stallpage: encode page state: %w", err)
	}
	tmp := path + fmt.Sprintf(".%d.tmp", os.Getpid())
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		return fmt.Errorf("stallpage: write page state: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		// Windows cannot rename over an existing file. The page lock serializes
		// every writer, so remove+rename is safe there; Unix took the atomic fast
		// path above. A crash in the tiny gap fails open to one extra page rather
		// than suppressing a real high-water indefinitely.
		if removeErr := os.Remove(path); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			_ = os.Remove(tmp)
			return fmt.Errorf("stallpage: remove old page state: %w", removeErr)
		}
		if renameErr := os.Rename(tmp, path); renameErr != nil {
			_ = os.Remove(tmp)
			return fmt.Errorf("stallpage: replace page state: %w", renameErr)
		}
	}
	return nil
}

func pruneState(st *pageState, now time.Time, keep time.Duration) {
	if keep <= 0 {
		return
	}
	for key, ns := range st.Last {
		at := time.Unix(0, ns)
		if !at.After(now) && now.Sub(at) >= keep {
			delete(st.Last, key)
		}
	}
}
