package swap

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
	"github.com/anthony-chaudhary/fak/pkg/deploykit"
)

// SlotKind names the scheme a rollback slot follows.
type SlotKind string

const (
	// KindPrior is the one slot name this package writes: "<name>.deploy-prior-<sha12>".
	KindPrior SlotKind = "deploy-prior"

	// The kinds below are legacy names, recognized for one release so a deploy can still
	// find a rollback copy made before this package existed. Nothing here writes them.

	// KindOld is "<bin>.old-<YYYYMMDD-HHMMSS>" or "<bin>.old-<unixnano>" (the cadence
	// refresher's timestamped backup; the dashboard reads the same name).
	KindOld SlotKind = "old"
	// KindBak is "<name>.bak", a file or directory (the Halo front-door, upgrade, and push
	// backups, including the SPIR-V directory snapshot).
	KindBak SlotKind = "bak"
	// KindPredeploy is "<name>.predeploy", a file or directory (the Halo upgrade's binary and
	// SPIR-V snapshots, written in its staging directory rather than next to the target).
	KindPredeploy SlotKind = "predeploy"
	// KindSelfUpdatePrior is `fak self-update`'s selfinstall.LaunchPriorPath copy,
	// "<stem>.self-update-prior.exe" or "<bin>.self-update-prior".
	KindSelfUpdatePrior SlotKind = "self-update-prior"
	// KindRouterCommit is the Windows router's commit-pinned candidate directory
	// "<private40>-<public40>/"; Path names the binary inside it.
	KindRouterCommit SlotKind = "router-commit"
)

// Slot is one rollback copy found for a target.
type Slot struct {
	Path    string    `json:"path"`
	Kind    SlotKind  `json:"kind"`
	ModTime time.Time `json:"mod_time"`
}

const (
	priorMarker = ".deploy-prior-"
	sha12Len    = 12
)

// PriorSlotPath returns the rollback slot for target whose prior bytes hash to digest (the
// SHA-256 of those bytes): "<target>.deploy-prior-<sha12>" in target's directory, where sha12
// is the first 12 lowercase hex digits. When target ends in ".exe" the suffix stays last
// ("<stem>.deploy-prior-<sha12>.exe") so the slot is still executable on Windows; this is the
// same rule as selfinstall.LaunchPriorPath. Identical bytes always map to the same path.
func PriorSlotPath(target string, digest [sha256.Size]byte) string {
	target = filepath.Clean(target)
	tag := priorMarker + hex.EncodeToString(digest[:])[:sha12Len]
	if ext := filepath.Ext(target); strings.EqualFold(ext, ".exe") {
		return strings.TrimSuffix(target, ext) + tag + ext
	}
	return target + tag
}

// SavePrior copies target's current bytes into its content-addressed rollback slot and returns
// the slot path. The copy is staged in target's directory, hashed while it is copied, and
// swapped in under the name of the bytes actually staged, so a crash never leaves a partial slot
// and a concurrent replace of target can never file new bytes under the old digest. A slot that
// already holds identical bytes is reused and its modification time refreshed, so PruneSlots
// keeps it as the newest.
func SavePrior(target string) (string, error) {
	target = filepath.Clean(target)
	staged, digest, err := stagePrior(target)
	if err != nil {
		return "", fmt.Errorf("deploykit/swap: stage prior %q: %w", target, err)
	}
	slot := PriorSlotPath(target, digest)
	if existing, err := fileDigest(slot); err == nil && existing == digest {
		_ = os.Remove(staged)
		now := time.Now()
		if err := os.Chtimes(slot, now, now); err != nil {
			return "", fmt.Errorf("deploykit/swap: refresh slot %q: %w", slot, err)
		}
		return slot, nil
	}
	if err := Swap(staged, slot); err != nil {
		_ = os.Remove(staged)
		return "", fmt.Errorf("deploykit/swap: write slot %q: %w", slot, err)
	}
	return slot, nil
}

// PriorSlots lists target's "<name>.deploy-prior-<sha12>" slots, newest first (ties by path).
// A missing directory is an empty list.
func PriorSlots(target string) []Slot {
	target = filepath.Clean(target)
	dir, base := filepath.Dir(target), filepath.Base(target)
	var slots []Slot
	for _, e := range readDir(dir) {
		if e.IsDir() || !isPriorSlotName(base, e.Name()) {
			continue
		}
		slots = append(slots, newSlot(filepath.Join(dir, e.Name()), KindPrior, e))
	}
	sortNewestFirst(slots)
	return slots
}

// PruneSlots removes target's oldest "<name>.deploy-prior-<sha12>" slots beyond
// policy.KeepSlots and returns the paths it removed. A KeepSlots below 1 keeps one slot, so a
// committed deploy always stays one step reversible. Legacy slots are never removed. Removal
// failures are joined into the error; the remaining slots are still attempted.
func PruneSlots(target string, policy deploykit.RollbackPolicy) ([]string, error) {
	keep := policy.KeepSlots
	if keep < 1 {
		keep = 1
	}
	slots := PriorSlots(target)
	if len(slots) <= keep {
		return nil, nil
	}
	var removed []string
	var errs []error
	for _, s := range slots[keep:] {
		if err := os.Remove(s.Path); err != nil {
			errs = append(errs, err)
			continue
		}
		removed = append(removed, s.Path)
	}
	return removed, errors.Join(errs...)
}

// LegacySlots lists the pre-deploykit rollback copies of target (see the legacy SlotKind
// values), newest first (ties by path). It only reads: nothing in this package writes a legacy
// name, and the reader is removed one release after every deployable writes KindPrior only.
//
// It scans target's directory plus any extra dirs (for example the staging directory where the
// Halo upgrade keeps its "<name>.predeploy" snapshots) and classifies each entry with
// LegacyKind. When target sits inside a router commit directory it also reports the same file
// in the sibling commit directories. A router candidate whose binary is newer than the live
// target is never reported: a failed update leaves its candidate behind, and that build never
// went live, so it is not a rollback copy. A missing directory contributes nothing.
func LegacySlots(target string, dirs ...string) []Slot {
	target = filepath.Clean(target)
	base := filepath.Base(target)
	var liveMod time.Time
	if info, err := os.Stat(target); err == nil {
		liveMod = info.ModTime()
	}
	var slots []Slot
	seen := map[string]bool{}
	for _, dir := range append([]string{filepath.Dir(target)}, dirs...) {
		dir = filepath.Clean(dir)
		if seen[dir] {
			continue
		}
		seen[dir] = true
		for _, e := range readDir(dir) {
			kind, ok := LegacyKind(target, e.Name())
			if !ok {
				continue
			}
			switch kind {
			case KindRouterCommit:
				if e.IsDir() {
					slots = appendRouterSlot(slots, filepath.Join(dir, e.Name(), base), liveMod)
				}
			case KindBak, KindPredeploy:
				slots = append(slots, newSlot(filepath.Join(dir, e.Name()), kind, e))
			default:
				if !e.IsDir() {
					slots = append(slots, newSlot(filepath.Join(dir, e.Name()), kind, e))
				}
			}
		}
	}
	// A target inside a commit-pinned directory: its rollback copies are the same binary in
	// the sibling commit directories, never its own.
	if own := filepath.Base(filepath.Dir(target)); isRouterCommitDirName(own) {
		root := filepath.Dir(filepath.Dir(target))
		for _, e := range readDir(root) {
			if e.IsDir() && isRouterCommitDirName(e.Name()) && !sameName(e.Name(), own) {
				slots = appendRouterSlot(slots, filepath.Join(root, e.Name(), base), liveMod)
			}
		}
	}
	sortNewestFirst(slots)
	return slots
}

// LegacyKind classifies one directory-entry name as a legacy rollback copy of target, by name
// alone, so a remote listing can use the same rules as LegacySlots. For KindRouterCommit the
// name is a "<private40>-<public40>" directory that holds a file named like target. The live
// target, the "<bin>.old" and "<bin>.old.<pid>.<i>" swap-asides, KindPrior slots, backups of
// other names, and the ".predeploy.pending" / ".predeploy.absent" markers are not legacy slots.
func LegacyKind(target, name string) (SlotKind, bool) {
	base := filepath.Base(filepath.Clean(target))
	oldPrefix := base + ".old-"
	switch {
	case sameName(name, base):
		return "", false // the live target
	case sameName(name, filepath.Base(selfinstall.LaunchPriorPath(target))):
		return KindSelfUpdatePrior, true
	case sameName(name, base+".bak"):
		return KindBak, true
	case sameName(name, base+".predeploy"):
		return KindPredeploy, true
	case len(name) > len(oldPrefix) && sameName(name[:len(oldPrefix)], oldPrefix):
		// ".old-" (hyphen) only: the ".old" and ".old.<pid>.<i>" swap-asides use a dot.
		return KindOld, true
	case isRouterCommitDirName(name):
		return KindRouterCommit, true
	}
	return "", false
}

// isPriorSlotName reports whether name is a KindPrior slot of a target named base, in the
// exact shape PriorSlotPath produces.
func isPriorSlotName(base, name string) bool {
	stem, ext := base, ""
	if e := filepath.Ext(base); strings.EqualFold(e, ".exe") {
		stem, ext = strings.TrimSuffix(base, e), e
	}
	if len(name) != len(stem)+len(priorMarker)+sha12Len+len(ext) ||
		!sameName(name[:len(stem)+len(priorMarker)], stem+priorMarker) ||
		!strings.EqualFold(name[len(name)-len(ext):], ext) {
		return false
	}
	return isLowerHex(name[len(stem)+len(priorMarker) : len(name)-len(ext)])
}

// isRouterCommitDirName reports whether name is "<private40>-<public40>": two full hex
// commits joined by one hyphen.
func isRouterCommitDirName(name string) bool {
	private, public, ok := strings.Cut(name, "-")
	return ok && isFullCommit(private) && isFullCommit(public)
}

func isFullCommit(s string) bool {
	if len(s) != 40 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

func isLowerHex(s string) bool {
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return s != ""
}

// appendRouterSlot adds the binary inside a commit directory when it is a regular file no newer
// than the live target (liveMod; zero when the target is missing). A commit directory without
// the binary is not a usable rollback copy.
func appendRouterSlot(slots []Slot, path string, liveMod time.Time) []Slot {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() || (!liveMod.IsZero() && info.ModTime().After(liveMod)) {
		return slots
	}
	return append(slots, Slot{Path: path, Kind: KindRouterCommit, ModTime: info.ModTime()})
}

func newSlot(path string, kind SlotKind, e os.DirEntry) Slot {
	s := Slot{Path: path, Kind: kind}
	if info, err := e.Info(); err == nil {
		s.ModTime = info.ModTime()
	}
	return s
}

func sortNewestFirst(slots []Slot) {
	sort.SliceStable(slots, func(i, j int) bool {
		if !slots[i].ModTime.Equal(slots[j].ModTime) {
			return slots[i].ModTime.After(slots[j].ModTime)
		}
		return slots[i].Path < slots[j].Path
	})
}

func readDir(dir string) []os.DirEntry {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	return entries
}

// sameName compares file names the way the host file system does: case-insensitively on
// Windows, exactly elsewhere.
func sameName(a, b string) bool {
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}

func fileDigest(path string) (digest [sha256.Size]byte, err error) {
	f, err := os.Open(path)
	if err != nil {
		return digest, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return digest, err
	}
	copy(digest[:], h.Sum(nil))
	return digest, nil
}

// stagePrior copies target to a temp file beside it, keeping its permission bits, hashes the
// bytes as they are copied, and syncs the copy so it is durable before the swap names it.
func stagePrior(target string) (path string, digest [sha256.Size]byte, err error) {
	in, err := os.Open(target)
	if err != nil {
		return "", digest, err
	}
	defer in.Close()
	info, err := in.Stat()
	if err != nil {
		return "", digest, err
	}
	out, err := os.CreateTemp(filepath.Dir(target), "."+filepath.Base(target)+".deploy-prior-stage-*")
	if err != nil {
		return "", digest, err
	}
	path = out.Name()
	defer func() {
		if cerr := out.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			_ = os.Remove(path)
			path = ""
		}
	}()
	if err = out.Chmod(info.Mode().Perm()); err != nil {
		return path, digest, err
	}
	h := sha256.New()
	if _, err = io.Copy(io.MultiWriter(out, h), in); err != nil {
		return path, digest, err
	}
	copy(digest[:], h.Sum(nil))
	return path, digest, out.Sync()
}
