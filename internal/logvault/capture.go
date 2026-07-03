package logvault

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Op names for manifest rows and plan lines.
const (
	OpFull    = "capture-full"
	OpAppend  = "capture-append"
	OpRewrite = "capture-rewrite"
	OpSkip    = "skip-error"
)

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

// historyPath is where a superseded mirror version is retired to.
func (v *Vault) historyPath(srcID, relPath, sha string) string {
	short := sha
	if len(short) > 8 {
		short = short[:8]
	}
	return filepath.Join(v.Dir, "by-source", srcID, ".history", filepath.FromSlash(relPath)+"."+short)
}

// walkSource visits every non-excluded regular file under the source root and
// calls fn with the forward-slash relative path and its info. A missing root is
// a valid empty source. Excluded directory prefixes are pruned without descent.
func (v *Vault) walkSource(src Source, fn func(relPath string, info fs.FileInfo) error) (missing bool, err error) {
	if _, statErr := os.Stat(src.Root); statErr != nil {
		return true, nil
	}
	vaultAbs, _ := filepath.Abs(v.Dir)
	err = filepath.WalkDir(src.Root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil // unreadable entry: skip, capture records per-file errors it can see
		}
		relOS, relErr := filepath.Rel(src.Root, path)
		if relErr != nil || relOS == "." {
			return nil
		}
		rel := filepath.ToSlash(relOS)
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
			return nil
		}
		return fn(rel, info)
	})
	return false, err
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
		missing, walkErr := v.walkSource(src, func(rel string, info fs.FileInfo) error {
			st.Files++
			prev, seen := states[src.ID+"\x00"+rel]
			size, mtime := info.Size(), info.ModTime().Unix()
			switch {
			case !seen:
				st.Full++
				st.CopyBytes += size
			case size == prev.SizeAfter && mtime == prev.MTimeUnix:
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
		st.Missing = missing
		out = append(out, st)
	}
	return out, nil
}

// Capture copies every new/changed source file into the vault and appends one
// chained manifest row per operation. Sources are read-only: files are opened
// for read and never locked; a file that cannot be read (e.g. a Windows sharing
// violation) is recorded as a skip-error row and retried next capture.
func (v *Vault) Capture() ([]SourceStats, error) {
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
		missing, walkErr := v.walkSource(src, func(rel string, info fs.FileInfo) error {
			st.Files++
			prev, seen := states[src.ID+"\x00"+rel]
			size, mtime := info.Size(), info.ModTime().Unix()
			if seen && size == prev.SizeAfter && mtime == prev.MTimeUnix {
				st.Unchanged++
				return nil
			}
			op, written, sha, capErr := v.captureFile(src, rel, prev, seen, size)
			// A hot file can grow while it streams: record the byte position the
			// hash actually covers, not the stat-time size, so the next capture
			// classifies the continuation as a clean append.
			sizeAfter := written
			if op == OpAppend {
				sizeAfter = prev.SizeAfter + written
			}
			row := ManifestRow{
				TSUnixNano: time.Now().UnixNano(),
				Op:         op,
				Source:     src.ID,
				RelPath:    rel,
				Bytes:      written,
				SizeAfter:  sizeAfter,
				MTimeUnix:  mtime,
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
				case "": // content identical after re-hash (mtime-only touch): no row needed
					st.Unchanged++
					return nil
				}
				st.CopyBytes += written
				states[src.ID+"\x00"+rel] = fileState{SizeAfter: sizeAfter, MTimeUnix: mtime, SHA256: sha}
			}
			_, appendErr := man.Append(row)
			return appendErr
		})
		if walkErr != nil {
			return out, walkErr
		}
		st.Missing = missing
		out = append(out, st)
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
			return "", 0, curSHA, nil // touch only — verified unchanged
		}
		if _, statErr := os.Stat(mirror); statErr == nil {
			hist := v.historyPath(src.ID, rel, prev.SHA256)
			if mkErr := os.MkdirAll(filepath.Dir(hist), 0o755); mkErr != nil {
				return "", 0, "", mkErr
			}
			os.Remove(hist) // same superseded content re-retired: idempotent
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
// error means the prefix diverged (caller rewrites).
func (v *Vault) tryAppend(srcPath, mirror string, prev fileState) (op string, written int64, sha string, err error) {
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
	if _, err := os.Stat(mirror); err != nil {
		return "", 0, "", nil // mirror lost: recapture in full
	}
	out, err := os.OpenFile(mirror, os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return "", 0, "", err
	}
	defer out.Close()
	written, err = io.Copy(io.MultiWriter(out, h), f)
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

// Verify re-derives the manifest chain, then re-hashes mirror files against the
// replayed state. sample bounds how many mirrors are re-hashed (0 = all),
// chosen by deterministic stride so repeated runs cover the same set.
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
