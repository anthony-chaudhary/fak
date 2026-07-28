package session

// Corrupt-registry recovery observability and bounded quarantine-evidence
// retention (#4658). A corrupt descriptor index is quarantined beside the
// active file as `<path>.corrupt-<stamp>` evidence; this file measures each
// recovery in a privacy-safe sidecar ledger (counts, normalized cause class,
// sizes, times — never descriptor contents) and bounds how much evidence may
// accumulate. The event shape is deliberately registry-agnostic so other
// rebuildable state can adopt it later.

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// RecoveryCause is the normalized, privacy-safe class of descriptor-index
// corruption. It names why the index could not be trusted without echoing any
// of its contents.
type RecoveryCause string

const (
	// RecoveryCauseDecode means the index was not valid JSON.
	RecoveryCauseDecode RecoveryCause = "decode"
	// RecoveryCauseVersion means the index carried an unsupported version tag.
	RecoveryCauseVersion RecoveryCause = "version"
	// RecoveryCauseBlankID means a descriptor row had no ID.
	RecoveryCauseBlankID RecoveryCause = "blank-id"
	// RecoveryCauseUnknown covers corrupt errors with no recorded class.
	RecoveryCauseUnknown RecoveryCause = "unknown"
)

// ClassifyRecoveryCause maps a restore error to its normalized recovery cause.
// Non-corrupt errors and untagged corrupt errors classify as unknown.
func ClassifyRecoveryCause(err error) RecoveryCause {
	var target *CorruptDescriptorFileError
	if errors.As(err, &target) && target.Cause != "" {
		return target.Cause
	}
	return RecoveryCauseUnknown
}

// RecoveryEvent is one privacy-safe corrupt-registry recovery observation.
// It records outcome, cause class, sizes and paths only — never descriptor
// contents.
type RecoveryEvent struct {
	At              time.Time     `json:"at"`
	Cause           RecoveryCause `json:"cause"`
	Bytes           int64         `json:"bytes"`
	ActivePath      string        `json:"active_path"`
	Quarantined     bool          `json:"quarantined"`
	QuarantinePath  string        `json:"quarantine_path,omitempty"`
	QuarantineError string        `json:"quarantine_error,omitempty"`
}

const recoveryLedgerVersion = "fak.session-recovery.v1"

// RecoveryStats is the cumulative, privacy-safe recovery ledger persisted
// beside the active registry. Missing or corrupt ledgers reset to zero: the
// ledger is itself rebuildable observability, never load-bearing state.
type RecoveryStats struct {
	Version            string         `json:"version"`
	Total              int            `json:"total"`
	QuarantineFailures int            `json:"quarantine_failures"`
	Causes             map[string]int `json:"causes,omitempty"`
	LastAt             time.Time      `json:"last_at"`
	LastCause          RecoveryCause  `json:"last_cause,omitempty"`
	LastBytes          int64          `json:"last_bytes"`
}

// RecoveryLedgerPath returns where recovery stats for activePath live.
func RecoveryLedgerPath(activePath string) string { return activePath + ".recovery.json" }

// ReadRecoveryStats reads the recovery ledger beside activePath without
// taking any lock or creating any file: diagnostic surfaces must stay
// strictly read-only. A missing ledger returns ok=false with zero stats; a
// corrupt ledger returns ok=false so callers report "none recorded" rather
// than trusting garbage.
func ReadRecoveryStats(activePath string) (RecoveryStats, bool, error) {
	b, err := os.ReadFile(RecoveryLedgerPath(activePath))
	if errors.Is(err, os.ErrNotExist) {
		return RecoveryStats{}, false, nil
	}
	if err != nil {
		return RecoveryStats{}, false, err
	}
	var stats RecoveryStats
	if err := json.Unmarshal(b, &stats); err != nil || stats.Version != recoveryLedgerVersion {
		return RecoveryStats{}, false, nil
	}
	return stats, true, nil
}

// recordRecovery folds one event into the sidecar ledger under the same
// cross-process lock discipline as the descriptor file, so concurrent
// recoverers serialize their increments. A missing or corrupt ledger resets
// to zero rather than failing: measurement must never block recovery.
func recordRecovery(activePath string, ev RecoveryEvent) (RecoveryStats, error) {
	unlock, err := lockDescriptorFile(RecoveryLedgerPath(activePath))
	if err != nil {
		return RecoveryStats{}, err
	}
	defer unlock()
	return recordRecoveryLocked(activePath, ev)
}

// recordRecoveryLocked is recordRecovery with the ledger lock already held.
func recordRecoveryLocked(activePath string, ev RecoveryEvent) (RecoveryStats, error) {
	ledger := RecoveryLedgerPath(activePath)
	stats, _, _ := ReadRecoveryStats(activePath)
	stats.Version = recoveryLedgerVersion
	stats.Total++
	if !ev.Quarantined {
		stats.QuarantineFailures++
	}
	if stats.Causes == nil {
		stats.Causes = map[string]int{}
	}
	stats.Causes[string(ev.Cause)]++
	stats.LastAt = ev.At
	stats.LastCause = ev.Cause
	stats.LastBytes = ev.Bytes
	if err := writeRecoveryLedger(ledger, stats); err != nil {
		return stats, err
	}
	return stats, nil
}

func writeRecoveryLedger(ledger string, stats RecoveryStats) error {
	dir := filepath.Dir(ledger)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create session recovery ledger dir: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".session-recovery-*.tmp")
	if err != nil {
		return fmt.Errorf("create session recovery ledger temp file: %w", err)
	}
	tmpName := tmp.Name()
	committed := false
	defer func() {
		if !committed {
			_ = os.Remove(tmpName)
		}
	}()
	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	if err := enc.Encode(stats); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("encode session recovery ledger: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close session recovery ledger: %w", err)
	}
	if err := replaceFile(tmpName, ledger); err != nil {
		return err
	}
	committed = true
	return nil
}

// QuarantineRetention bounds how much quarantine evidence may accumulate
// beside one active registry. Zero-valued dimensions are unbounded; Off
// disables cleanup entirely (evidence is then kept forever, the pre-#4658
// behavior).
type QuarantineRetention struct {
	MaxCount int           // keep at most this many evidence files
	MaxAge   time.Duration // remove evidence older than this
	MaxBytes int64         // keep at most this many total evidence bytes
	Off      bool          // disable cleanup entirely
}

// DefaultQuarantineRetention is deliberately conservative: it keeps plenty of
// evidence for diagnosis while guaranteeing repeated corruption cannot grow
// the user profile unbounded.
func DefaultQuarantineRetention() QuarantineRetention {
	return QuarantineRetention{
		MaxCount: 8,
		MaxAge:   30 * 24 * time.Hour,
		MaxBytes: 8 << 20,
	}
}

// ParseQuarantineRetention parses an operator retention override:
// "" means the default policy, "off" disables cleanup, and a comma list of
// count=N, age=DURATION, bytes=N overrides individual dimensions (unset
// dimensions keep their defaults; 0 makes a dimension unbounded). On a parse
// error the default policy is returned so a typo can never disable retention
// or block startup.
func ParseQuarantineRetention(s string) (QuarantineRetention, error) {
	policy := DefaultQuarantineRetention()
	s = strings.TrimSpace(s)
	if s == "" {
		return policy, nil
	}
	if strings.EqualFold(s, "off") {
		return QuarantineRetention{Off: true}, nil
	}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		key, value, ok := strings.Cut(part, "=")
		if !ok {
			return DefaultQuarantineRetention(), fmt.Errorf("retention term %q is not key=value", part)
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		switch strings.ToLower(key) {
		case "count":
			n, err := strconv.Atoi(value)
			if err != nil || n < 0 {
				return DefaultQuarantineRetention(), fmt.Errorf("retention count %q is not a non-negative integer", value)
			}
			policy.MaxCount = n
		case "age":
			d, err := time.ParseDuration(value)
			if err != nil || d < 0 {
				return DefaultQuarantineRetention(), fmt.Errorf("retention age %q is not a non-negative duration", value)
			}
			policy.MaxAge = d
		case "bytes":
			n, err := strconv.ParseInt(value, 10, 64)
			if err != nil || n < 0 {
				return DefaultQuarantineRetention(), fmt.Errorf("retention bytes %q is not a non-negative integer", value)
			}
			policy.MaxBytes = n
		default:
			return DefaultQuarantineRetention(), fmt.Errorf("unknown retention dimension %q", key)
		}
	}
	return policy, nil
}

const quarantineStampFormat = "20060102T150405.000000000Z"

// QuarantineCorruptRegistry renames the corrupt index at path to a
// timestamped `.corrupt-` evidence sibling and returns the evidence path.
// Stamp collisions get a numeric suffix. If path no longer exists the rename
// reports os.ErrNotExist: a concurrent recoverer already quarantined it.
func QuarantineCorruptRegistry(path string, now time.Time) (string, error) {
	base := path + ".corrupt-" + now.UTC().Format(quarantineStampFormat)
	for attempt := 0; attempt < 100; attempt++ {
		dst := base
		if attempt > 0 {
			dst = fmt.Sprintf("%s-%d", base, attempt)
		}
		if _, err := os.Stat(dst); err == nil {
			continue
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
		if err := os.Rename(path, dst); err != nil {
			return "", err
		}
		return dst, nil
	}
	return "", errors.New("could not allocate corrupt session registry quarantine path")
}

// quarantineEvidence is one `.corrupt-*` sibling of an active registry.
type quarantineEvidence struct {
	path  string
	stamp string // parsed stamp portion, "" when unparseable
	seq   int    // collision suffix, 0 when absent
	size  int64
	mod   time.Time
}

func (e quarantineEvidence) observedAt() time.Time {
	if t, err := time.Parse(quarantineStampFormat, e.stamp); err == nil {
		return t
	}
	return e.mod
}

// listQuarantineEvidence returns evidence newest-first, ordered by embedded
// stamp then collision suffix so cleanup stays deterministic even when many
// recoveries share one timestamp.
func listQuarantineEvidence(activePath string) ([]quarantineEvidence, error) {
	matches, err := filepath.Glob(activePath + ".corrupt-*")
	if err != nil {
		return nil, err
	}
	out := make([]quarantineEvidence, 0, len(matches))
	for _, m := range matches {
		if m == activePath {
			continue // defensive: the active file is never evidence
		}
		fi, err := os.Lstat(m)
		if err != nil || !fi.Mode().IsRegular() {
			continue
		}
		ev := quarantineEvidence{path: m, size: fi.Size(), mod: fi.ModTime()}
		suffix := strings.TrimPrefix(m, activePath+".corrupt-")
		if i := strings.Index(suffix, "Z"); i >= 0 {
			ev.stamp = suffix[:i+1]
			if rest := strings.TrimPrefix(suffix[i+1:], "-"); rest != suffix[i+1:] {
				if n, err := strconv.Atoi(rest); err == nil {
					ev.seq = n
				}
			}
		}
		out = append(out, ev)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].stamp != out[j].stamp {
			return out[i].stamp > out[j].stamp
		}
		if out[i].seq != out[j].seq {
			return out[i].seq > out[j].seq
		}
		return out[i].path > out[j].path
	})
	return out, nil
}

// quarantineRemove is a test-only cleanup-failure seam. Production leaves it
// as os.Remove; tests may inject removal errors to prove cleanup failure
// stays advisory.
var quarantineRemove = os.Remove

// ReapQuarantine applies the retention policy to the `.corrupt-*` evidence
// siblings of activePath. It never touches the active file itself, always
// preserves the newest evidence file even when that file alone exceeds a
// bound, and removes files one atomic os.Remove at a time, continuing past
// individual failures. The joined error is advisory: callers must treat
// cleanup failure as a warning, never a startup blocker.
func ReapQuarantine(activePath string, policy QuarantineRetention, now time.Time) (removed []string, err error) {
	if policy.Off {
		return nil, nil
	}
	evidence, listErr := listQuarantineEvidence(activePath)
	if listErr != nil {
		return nil, listErr
	}
	var errs []error
	kept, keptBytes := 0, int64(0)
	for i, ev := range evidence {
		over := false
		if i > 0 { // the newest evidence file is always preserved
			if policy.MaxCount > 0 && kept >= policy.MaxCount {
				over = true
			}
			if policy.MaxAge > 0 && now.Sub(ev.observedAt()) > policy.MaxAge {
				over = true
			}
			if policy.MaxBytes > 0 && keptBytes+ev.size > policy.MaxBytes {
				over = true
			}
		}
		if !over {
			kept++
			keptBytes += ev.size
			continue
		}
		if rmErr := quarantineRemove(ev.path); rmErr != nil && !errors.Is(rmErr, os.ErrNotExist) {
			errs = append(errs, rmErr)
			continue
		}
		removed = append(removed, ev.path)
	}
	return removed, errors.Join(errs...)
}

// RegistryRecovery is the result of one corrupt-registry recovery pass.
type RegistryRecovery struct {
	Event  RecoveryEvent
	Stats  RecoveryStats
	Reaped []string
	// AlreadyRecovered reports that a concurrent recoverer quarantined the
	// active file first; the caller should simply retry its restore.
	AlreadyRecovered bool
	// LedgerErr and ReapErr are advisory measurement/cleanup failures. They
	// must never prevent startup; callers may warn about them.
	LedgerErr error
	ReapErr   error
}

// RecoverCorruptRegistry quarantines the corrupt descriptor index at path,
// records a privacy-safe recovery event in the sidecar ledger, and applies
// the retention policy to accumulated evidence. Only a quarantine failure is
// returned as a hard error (evidence preservation stays load-bearing, as in
// #4647); ledger and cleanup failures ride along as advisory fields.
//
// The whole pass serializes under the sidecar-ledger lock: a bare rename race
// is NOT enough to elect one winner, because Windows renames a source that a
// concurrent recoverer already renamed onward from its new location (rename
// works by open handle), so every unserialized recoverer can report success.
// Under the lock, losers observe the active file already gone and defer.
func RecoverCorruptRegistry(path string, cause error, policy QuarantineRetention, now time.Time) (RegistryRecovery, error) {
	rec := RegistryRecovery{}
	ev := RecoveryEvent{At: now.UTC(), Cause: ClassifyRecoveryCause(cause), ActivePath: path}
	unlock, lockErr := lockDescriptorFile(RecoveryLedgerPath(path))
	if lockErr == nil {
		defer unlock()
	}
	record := func(ev RecoveryEvent) (RecoveryStats, error) {
		if lockErr != nil {
			// The serializing lock was never acquired; recordRecovery retries
			// the lock itself so a transient failure still counts the event.
			return recordRecovery(path, ev)
		}
		return recordRecoveryLocked(path, ev)
	}
	// deferToWinner is how this pass reports "someone else already recovered this file":
	// no new ledger event (the corruption was counted once, by the winner), just the
	// already-recorded stats. Both places that can observe a lost race — the pre-rename
	// stat and the rename itself — must report it identically, or the two paths would
	// disagree about whether a deferral counts.
	deferToWinner := func() (RegistryRecovery, error) {
		rec.Event = ev
		rec.AlreadyRecovered = true
		rec.Stats, _, _ = ReadRecoveryStats(path)
		return rec, nil
	}
	fi, statErr := os.Stat(path)
	if errors.Is(statErr, os.ErrNotExist) {
		// A concurrent recoverer quarantined the active file before we held
		// the lock; the corruption was already counted once. Do not
		// double-count it.
		return deferToWinner()
	}
	if statErr == nil {
		ev.Bytes = fi.Size()
	}
	dst, qErr := QuarantineCorruptRegistry(path, now)
	if qErr != nil && errors.Is(qErr, os.ErrNotExist) {
		// Reachable only when the lock could not be taken and a concurrent
		// recoverer won the rename between our stat and rename.
		return deferToWinner()
	}
	if qErr != nil {
		ev.QuarantineError = qErr.Error()
		rec.Stats, rec.LedgerErr = record(ev)
		rec.Event = ev
		return rec, qErr
	}
	ev.Quarantined = true
	ev.QuarantinePath = dst
	rec.Stats, rec.LedgerErr = record(ev)
	rec.Reaped, rec.ReapErr = ReapQuarantine(path, policy, now)
	rec.Event = ev
	return rec, nil
}

// QuarantineEvidenceCount reports how many quarantine evidence files
// currently sit beside activePath, for diagnostic surfaces.
func QuarantineEvidenceCount(activePath string) (int, error) {
	evidence, err := listQuarantineEvidence(activePath)
	if err != nil {
		return 0, err
	}
	return len(evidence), nil
}
