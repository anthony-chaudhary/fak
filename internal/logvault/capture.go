package logvault

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// Op names for manifest rows and plan lines.
const (
	OpFull    = "capture-full"
	OpAppend  = "capture-append"
	OpRewrite = "capture-rewrite"
	OpTouch   = "capture-touch" // content verified unchanged, mtime advanced
	OpSkip    = "skip-error"
)

// LockName is the vault's single-writer capture lock file.
const LockName = "vault.lock"

// SourceStats folds one source's outcome for a plan or capture pass.
type SourceStats struct {
	Source    string
	Files     int // files examined (after excludes)
	Unchanged int
	Full      int
	Append    int
	Rewrite   int
	Errors    int
	CopyBytes int64 // bytes a capture would write / did write to the vault
	Missing   bool  // source root absent on this box (valid-empty)
}

// Vault is a capture session against one vault directory.
type Vault struct {
	Dir     string
	Sources []Source
}

// mirrorPath is the current mirror of a source file inside the vault.
func (v *Vault) mirrorPath(srcID, relPath string) string {
	return filepath.Join(v.Dir, "by-source", srcID, filepath.FromSlash(relPath))
}

// historyPath is where a superseded mirror version is retired to. 16 hex chars
// of the content hash key the slot (a shorter prefix risks two different prior
// versions colliding and the earlier one being silently destroyed).
func (v *Vault) historyPath(srcID, relPath, sha string) string {
	short := sha
	if len(short) > 16 {
		short = short[:16]
	}
	return filepath.Join(v.Dir, "by-source", srcID, ".history", filepath.FromSlash(relPath)+"."+short)
}

// walkProblem records a subtree the walk could not read — silence here would
// make a permission-denied directory look successfully backed up.
type walkProblem struct {
	Rel string
	Err error
}

// pathWithin reports whether path is p itself or inside it (both cleaned abs).
func pathWithin(path, p string) bool {
	if path == p {
		return true
	}
	return strings.HasPrefix(path, p+string(os.PathSeparator))
}

// walkSource visits every non-excluded regular file under the source root and
// calls fn with the forward-slash relative path and its info. A missing root is
// a valid empty source; an unreadable subtree is returned as a problem, never
// silently skipped. Excluded directory prefixes are pruned without descent.
func (v *Vault) walkSource(src Source, fn func(relPath string, info fs.FileInfo) error) (missing bool, problems []walkProblem, err error) {
	if _, statErr := os.Stat(src.Root); statErr != nil {
		return true, nil, nil
	}
	vaultAbs, _ := filepath.Abs(v.Dir)
	srcAbs, _ := filepath.Abs(src.Root)
	// A source that IS the vault (or lives inside it) would capture the vault
	// into itself and grow without bound: a config error, refused loudly. The
	// vault living inside a source root is fine — the walk prunes it below.
	if vaultAbs != "" && pathWithin(srcAbs, vaultAbs) {
		return false, nil, fmt.Errorf("logvault: source %s root %s overlaps the vault %s", src.ID, src.Root, v.Dir)
	}
	err = filepath.WalkDir(src.Root, func(path string, d fs.DirEntry, walkErr error) error {
		relOS, relErr := filepath.Rel(src.Root, path)
		rel := ""
		if relErr == nil && relOS != "." {
			rel = filepath.ToSlash(relOS)
		}
		if walkErr != nil {
			problems = append(problems, walkProblem{Rel: rel, Err: walkErr})
			return nil
		}
		if relOS == "." {
			return nil
		}
		if d.IsDir() {
			if abs, _ := filepath.Abs(path); vaultAbs != "" && abs == vaultAbs {
				return filepath.SkipDir // never capture the vault into itself
			}
			if excluded(src, rel+"/") || !includesCouldReach(src, rel+"/") {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() || !admitted(src, rel) || excluded(src, rel) {
			return nil
		}
		info, infoErr := d.Info()
		if infoErr != nil {
			problems = append(problems, walkProblem{Rel: rel, Err: infoErr})
			return nil
		}
		return fn(rel, info)
	})
	return false, problems, err
}

// Plan diffs the live sources against the manifest replay without copying or
// hashing. Grown files are counted as appends and shrunk/touched files as
// rewrites optimistically; Capture makes the real (hash-checked) decision.
func (v *Vault) Plan() ([]SourceStats, error) {
	rows, err := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
	if err != nil {
		return nil, err
	}
	states := replayStates(rows)
	var out []SourceStats
	for _, src := range v.Sources {
		st := SourceStats{Source: src.ID}
		missing, problems, walkErr := v.walkSource(src, func(rel string, info fs.FileInfo) error {
			st.Files++
			prev, seen := states[src.ID+"\x00"+rel]
			size, mtime := info.Size(), info.ModTime().UnixNano()
			switch {
			case !seen:
				st.Full++
				st.CopyBytes += size
			case size == prev.SizeAfter && mtime == prev.MTimeNano:
				st.Unchanged++
			case size > prev.SizeAfter:
				st.Append++
				st.CopyBytes += size - prev.SizeAfter
			default:
				st.Rewrite++
				st.CopyBytes += size
			}
			return nil
		})
		if walkErr != nil {
			return nil, walkErr
		}
		st.Errors += len(problems)
		st.Missing = missing
		out = append(out, st)
	}
	return out, nil
}

// Capture copies every new/changed source file into the vault and appends one
// chained manifest row per operation. Sources are read-only: files are opened
// for read and never locked; a file that cannot be read (e.g. a Windows sharing
// violation) is recorded as a skip-error row and retried next capture. The
// vault itself is single-writer: a cross-process lock serializes captures so
// two runs cannot interleave manifest rows and fork the chain.
func (v *Vault) Capture() ([]SourceStats, error) {
	if err := os.MkdirAll(v.Dir, 0o755); err != nil {
		return nil, err
	}
	lock, err := os.OpenFile(filepath.Join(v.Dir, LockName), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		if errors.Is(err, flock.ErrLockBusy) {
			return nil, fmt.Errorf("logvault: another capture holds %s", LockName)
		}
		return nil, err
	}
	defer flock.Unlock(lock)

	man, err := OpenManifest(v.Dir)
	if err != nil {
		return nil, err
	}
	defer man.Close()
	rows, err := ReadManifestRows(man.path)
	if err != nil {
		return nil, err
	}
	states := replayStates(rows)
	var out []SourceStats
	for _, src := range v.Sources {
		st := SourceStats{Source: src.ID}
		missing, problems, walkErr := v.walkSource(src, func(rel string, info fs.FileInfo) error {
			st.Files++
			prev, seen := states[src.ID+"\x00"+rel]
			size, mtime := info.Size(), info.ModTime().UnixNano()
			if seen && size == prev.SizeAfter && mtime == prev.MTimeNano {
				st.Unchanged++
				return nil
			}
			op, written, sha, capErr := v.captureFile(src, rel, prev, seen, size)
			// A hot file can grow while it streams: record the byte position the
			// hash actually covers, not the stat-time size, so the next capture
			// classifies the continuation as a clean append.
			sizeAfter := written
			switch op {
			case OpAppend:
				sizeAfter = prev.SizeAfter + written
			case OpTouch:
				sizeAfter = prev.SizeAfter
			}
			row := ManifestRow{
				TSUnixNano: time.Now().UnixNano(),
				Op:         op,
				Source:     src.ID,
				RelPath:    rel,
				Bytes:      written,
				SizeAfter:  sizeAfter,
				MTimeNano:  mtime,
				SHA256:     sha,
			}
			if capErr != nil {
				st.Errors++
				row.Op = OpSkip
				row.Bytes = 0
				row.SHA256 = ""
				row.Note = capErr.Error()
			} else {
				switch op {
				case OpFull:
					st.Full++
				case OpAppend:
					st.Append++
				case OpRewrite:
					st.Rewrite++
				case OpTouch:
					// Content verified identical; the row just advances the recorded
					// mtime so the next capture takes the cheap unchanged path again.
					st.Unchanged++
				}
				st.CopyBytes += written
				states[src.ID+"\x00"+rel] = fileState{SizeAfter: sizeAfter, MTimeNano: mtime, SHA256: sha}
			}
			_, appendErr := man.Append(row)
			return appendErr
		})
		if walkErr != nil {
			return out, walkErr
		}
		for _, p := range problems {
			st.Errors++
			if _, appendErr := man.Append(ManifestRow{
				TSUnixNano: time.Now().UnixNano(),
				Op:         OpSkip,
				Source:     src.ID,
				RelPath:    p.Rel,
				Note:       "walk: " + p.Err.Error(),
			}); appendErr != nil {
				return out, appendErr
			}
		}
		st.Missing = missing
		out = append(out, st)
	}
	if seq, hash := man.Head(); seq > 0 {
		if err := WriteAnchor(v.Dir, seq, hash); err != nil {
			return out, err
		}
	}
	return out, nil
}

// captureFile performs the real per-file operation and returns the op taken,
// bytes written to the vault, and the full-content sha256 at capture time. An
// op of "" with a nil error means the content was verified unchanged.
func (v *Vault) captureFile(src Source, rel string, prev fileState, seen bool, size int64) (op string, written int64, sha string, err error) {
	srcPath := filepath.Join(src.Root, filepath.FromSlash(rel))
	mirror := v.mirrorPath(src.ID, rel)

	if seen && size > prev.SizeAfter {
		op, written, sha, err = v.tryAppend(srcPath, mirror, prev)
		if err != nil || op != "" {
			return op, written, sha, err
		}
		// prefix diverged: fall through to a rewrite
	}
	if seen {
		// same-size mtime touch, shrink, or diverged prefix: re-hash to decide
		curSHA, hashErr := hashFile(srcPath)
		if hashErr != nil {
			return "", 0, "", hashErr
		}
		if curSHA == prev.SHA256 {
			return OpTouch, 0, curSHA, nil // verified unchanged, advance mtime only
		}
		if _, statErr := os.Stat(mirror); statErr == nil {
			// Retire under the mirror's ACTUAL content hash: after an interrupted
			// append the mirror can differ from the last recorded state, and filing
			// it under prev.SHA256 would put wrong bytes behind that name.
			histSHA := prev.SHA256
			if mirrorSHA, mhErr := hashFile(mirror); mhErr == nil {
				histSHA = mirrorSHA
			}
			hist := v.historyPath(src.ID, rel, histSHA)
			if mkErr := os.MkdirAll(filepath.Dir(hist), 0o755); mkErr != nil {
				return "", 0, "", mkErr
			}
			os.Remove(hist) // same-hash slot: identical content re-retired, idempotent
			if mvErr := os.Rename(mirror, hist); mvErr != nil {
				return "", 0, "", mvErr
			}
		}
		written, sha, err = copyToMirror(srcPath, mirror)
		return OpRewrite, written, sha, err
	}
	written, sha, err = copyToMirror(srcPath, mirror)
	return OpFull, written, sha, err
}

// tryAppend streams the source once: it hashes the first prev.SizeAfter bytes
// and, if that prefix still matches the last captured content, appends only the
// delta to the mirror while finishing the full-content hash. op "" with nil
// error means append is not safe (prefix diverged, or the mirror is missing or
// not exactly prev.SizeAfter bytes — e.g. after an interrupted append) and the
// caller must recapture in full; appending onto a diverged mirror would
// duplicate bytes into the backup permanently.
func (v *Vault) tryAppend(srcPath, mirror string, prev fileState) (op string, written int64, sha string, err error) {
	if mi, statErr := os.Stat(mirror); statErr != nil || mi.Size() != prev.SizeAfter {
		return "", 0, "", nil
	}
	f, err := os.Open(srcPath)
	if err != nil {
		return "", 0, "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.CopyN(h, f, prev.SizeAfter); err != nil {
		return "", 0, "", err
	}
	if hex.EncodeToString(h.Sum(nil)) != prev.SHA256 {
		return "", 0, "", nil // rewritten in place, not an append
	}
	out, err := os.OpenFile(mirror, os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return "", 0, "", err
	}
	written, err = io.Copy(io.MultiWriter(out, h), f)
	if err == nil {
		err = out.Sync()
	}
	if closeErr := out.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return "", 0, "", err
	}
	return OpAppend, written, hex.EncodeToString(h.Sum(nil)), nil
}

// copyToMirror streams src to <mirror>.part while hashing, then renames it into
// place so a crashed copy never masquerades as a mirror.
func copyToMirror(srcPath, mirror string) (written int64, sha string, err error) {
	f, err := os.Open(srcPath)
	if err != nil {
		return 0, "", err
	}
	defer f.Close()
	if err := os.MkdirAll(filepath.Dir(mirror), 0o755); err != nil {
		return 0, "", err
	}
	part := mirror + ".part"
	out, err := os.Create(part)
	if err != nil {
		return 0, "", err
	}
	h := sha256.New()
	written, err = io.Copy(io.MultiWriter(out, h), f)
	if err == nil {
		err = out.Sync() // a manifest row must never outlive the bytes behind it
	}
	closeErr := out.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		os.Remove(part)
		return 0, "", err
	}
	os.Remove(mirror) // Windows: rename does not replace
	if err := os.Rename(part, mirror); err != nil {
		os.Remove(part)
		return 0, "", err
	}
	return written, hex.EncodeToString(h.Sum(nil)), nil
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// VerifyProblem is one mirror that fails re-verification against the manifest.
type VerifyProblem struct {
	Source  string
	RelPath string
	Reason  string
}

// Verify re-derives the manifest chain, cross-checks it against the head
// anchor (so a truncated or deleted manifest cannot verify clean), then
// re-hashes mirror files against the replayed state. sample bounds how many
// mirrors are re-hashed (0 = all), chosen by deterministic stride so repeated
// runs cover the same set.
//
// Honesty note: the chain + anchor catch corruption, truncation, and casual
// edits. They are not proof against an adversary with full write access to the
// vault, who can recompute hashes and rewrite the anchor — that requires an
// off-vault anchor (the off-box replication rung).
func (v *Vault) Verify(sample int) (chainRows int, checked int, problems []VerifyProblem, err error) {
	manPath := filepath.Join(v.Dir, ManifestName)
	chainRows, err = VerifyManifest(manPath)
	if err != nil {
		return chainRows, 0, nil, err
	}
	rows, err := ReadManifestRows(manPath)
	if err != nil {
		return chainRows, 0, nil, err
	}
	if a, ok, aErr := readAnchor(v.Dir); aErr != nil {
		problems = append(problems, VerifyProblem{Reason: "head anchor unreadable: " + aErr.Error()})
	} else if ok {
		switch {
		case uint64(len(rows)) < a.Seq:
			problems = append(problems, VerifyProblem{Reason: fmt.Sprintf("manifest truncated: anchor head seq %d, manifest tail seq %d", a.Seq, len(rows))})
		case rows[a.Seq-1].Hash != a.Hash:
			problems = append(problems, VerifyProblem{Reason: fmt.Sprintf("manifest row %d hash disagrees with head anchor", a.Seq)})
		}
	}
	if len(rows) == 0 {
		if ents, rdErr := os.ReadDir(filepath.Join(v.Dir, "by-source")); rdErr == nil && len(ents) > 0 {
			problems = append(problems, VerifyProblem{Reason: "manifest is empty but by-source/ holds captured content (manifest lost?)"})
		}
	}
	states := replayStates(rows)
	keys := make([]string, 0, len(states))
	for k := range states {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	stride := 1
	if sample > 0 && len(keys) > sample {
		stride = len(keys) / sample
	}
	for i := 0; i < len(keys); i += stride {
		k := keys[i]
		srcID, rel, _ := strings.Cut(k, "\x00")
		st := states[k]
		mirror := v.mirrorPath(srcID, rel)
		got, hashErr := hashFile(mirror)
		checked++
		if hashErr != nil {
			problems = append(problems, VerifyProblem{srcID, rel, "mirror unreadable: " + hashErr.Error()})
			continue
		}
		if got != st.SHA256 {
			problems = append(problems, VerifyProblem{srcID, rel, "mirror hash mismatch vs manifest"})
		}
	}
	return chainRows, checked, problems, nil
}
