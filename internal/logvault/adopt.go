package logvault

// Cold-park adoption — bank a whole tree the vault does NOT continuously mirror
// (a retired experiment dir, an off-lane snapshot) as one deterministic,
// content-addressed archive. Unlike a source, a cold-parked tree is captured
// once: it is packed into a normalized tar, hashed, banked under
// by-source/cold-park/, and witnessed by a manifest row so Verify re-hashes it
// like any other mirror. The pack is byte-reproducible (entries in lexical
// order, headers normalized, mtimes pinned), so re-adopting identical content
// dedups for free. The tool NEVER deletes the original — it prints the command
// an operator may run to reclaim the space, and stops there.

import (
	"archive/tar"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// ColdParkSourceID is the synthetic source id under which cold-park archives are
// banked (by-source/cold-park/<name>). It is not a capture source — nothing
// re-walks it; the archive is a one-shot mirror.
const ColdParkSourceID = "cold-park"

// OpColdAdopt is the manifest op stamped when a tree is adopted as a cold-park
// archive. It carries the archive's own sha, so Verify re-hashes the banked
// archive exactly like a captured mirror.
const OpColdAdopt = "cold-adopt"

// coldParkEpoch is the fixed mtime stamped on every archived entry. Pinning it
// (rather than copying each file's mtime) is what makes two adoptions of the
// same content byte-identical regardless of on-disk timestamps.
var coldParkEpoch = time.Unix(0, 0).UTC()

// AdoptReport is the outcome of a cold-park adoption.
type AdoptReport struct {
	SrcDir      string // the tree that was adopted
	ArchiveRel  string // vault-relative path of the banked archive, forward slash
	ArchivePath string // absolute on-disk path of the banked archive
	SHA256      string // sha256 of the archive bytes (content address)
	Bytes       int64  // archive size
	Files       int    // regular files packed
	Deduped     bool   // an identical archive was already banked (no re-write, no new row)
	DeleteCmd   string // the command an OPERATOR may run to delete the original — never run by the tool
}

// AdoptCold packs srcDir into one deterministic archive banked in the vault and
// witnessed by a manifest row. It reads srcDir but never mutates or deletes it.
// Re-adopting byte-identical content resolves to the same content address and
// dedups: the existing archive is kept and no duplicate row is appended.
func (v *Vault) AdoptCold(srcDir string) (AdoptReport, error) {
	rep := AdoptReport{SrcDir: srcDir, DeleteCmd: deleteCmd(srcDir)}
	info, err := os.Stat(srcDir)
	if err != nil {
		return rep, err
	}
	if !info.IsDir() {
		return rep, fmt.Errorf("logvault: adopt --cold expects a directory, got %s", srcDir)
	}
	coldDir := filepath.Join(v.Dir, "by-source", ColdParkSourceID)
	if err := os.MkdirAll(coldDir, 0o755); err != nil {
		return rep, err
	}

	// Pack to a temp file while hashing, so a large tree never buffers in memory
	// and a crashed pack never masquerades as a banked archive.
	part := filepath.Join(coldDir, ".adopt.part")
	pf, err := os.Create(part)
	if err != nil {
		return rep, err
	}
	h := sha256.New()
	files, packErr := packTree(srcDir, io.MultiWriter(pf, h))
	if packErr == nil {
		packErr = pf.Sync()
	}
	if cerr := pf.Close(); packErr == nil {
		packErr = cerr
	}
	if packErr != nil {
		os.Remove(part)
		return rep, packErr
	}
	fi, err := os.Stat(part)
	if err != nil {
		os.Remove(part)
		return rep, err
	}
	sha := hex.EncodeToString(h.Sum(nil))
	rep.SHA256, rep.Bytes, rep.Files = sha, fi.Size(), files

	name := coldParkArchiveName(srcDir, sha)
	rep.ArchiveRel = "by-source/" + ColdParkSourceID + "/" + name
	target := filepath.Join(coldDir, name)
	rep.ArchivePath = target

	if _, err := os.Stat(target); err == nil {
		// Content-addressed: these exact bytes are already banked and already
		// witnessed by a prior row. Drop the temp and report the dedup.
		os.Remove(part)
		rep.Deduped = true
		return rep, nil
	}
	if err := os.Rename(part, target); err != nil {
		os.Remove(part)
		return rep, err
	}
	err = v.withManifest(func(man *Manifest) error {
		_, e := man.Append(ManifestRow{
			TSUnixNano: time.Now().UnixNano(),
			Op:         OpColdAdopt,
			Source:     ColdParkSourceID,
			RelPath:    name,
			Bytes:      fi.Size(),
			SizeAfter:  fi.Size(),
			SHA256:     sha,
			Note:       fmt.Sprintf("cold-adopt %s (%d files, deterministic tar)", srcDir, files),
		})
		return e
	})
	if err != nil {
		return rep, err
	}
	return rep, nil
}

// packTree writes the regular files under root to w as a deterministic tar:
// entries in lexical rel-path order with fully normalized headers (mode 0644,
// uid/gid 0, mtime pinned to coldParkEpoch), so identical content packs
// byte-identical. It mirrors the construction in internal/sessionimage's
// archive writer; that package's Pack is session-image-specific (it reads an
// image.json part list), so the tree walk is reproduced here rather than reused.
// A non-regular entry (symlink, device) is refused rather than followed.
func packTree(root string, w io.Writer) (int, error) {
	var rels []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, we error) error {
		if we != nil {
			return we
		}
		if d.IsDir() {
			return nil
		}
		if !d.Type().IsRegular() {
			return fmt.Errorf("logvault: refuse to cold-adopt non-regular file %s", path)
		}
		rel, rErr := filepath.Rel(root, path)
		if rErr != nil {
			return rErr
		}
		rels = append(rels, filepath.ToSlash(rel))
		return nil
	})
	if err != nil {
		return 0, err
	}
	sort.Strings(rels)
	tw := tar.NewWriter(w)
	for _, rel := range rels {
		body, rErr := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if rErr != nil {
			return 0, rErr
		}
		if err := tw.WriteHeader(&tar.Header{
			Name:     rel,
			Mode:     0o644,
			Size:     int64(len(body)),
			ModTime:  coldParkEpoch,
			Typeflag: tar.TypeReg,
			Format:   tar.FormatPAX,
		}); err != nil {
			return 0, err
		}
		if _, err := tw.Write(body); err != nil {
			return 0, err
		}
	}
	if err := tw.Close(); err != nil {
		return 0, err
	}
	return len(rels), nil
}

// coldParkArchiveName is the content-addressed archive name: a sanitized base of
// the source dir plus the sha's 16-hex prefix, so identical content lands on the
// same path (free dedup) and distinct content never collides.
func coldParkArchiveName(srcDir, sha string) string {
	base := sanitizeBase(filepath.Base(filepath.Clean(srcDir)))
	return base + "." + sha[:16] + ".tar"
}

// sanitizeBase reduces a directory base name to a filesystem-safe archive stem.
func sanitizeBase(s string) string {
	var b strings.Builder
	for _, c := range s {
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '_', c == '-', c == '.':
			b.WriteRune(c)
		default:
			b.WriteByte('-')
		}
	}
	out := strings.Trim(b.String(), ".-")
	if out == "" {
		return "cold"
	}
	return out
}

// deleteCmd renders the OS-appropriate command an operator may run to delete the
// adopted original. The tool prints it; it never runs it.
func deleteCmd(dir string) string {
	if runtime.GOOS == "windows" {
		return fmt.Sprintf("Remove-Item -Recurse -Force %q", dir)
	}
	return fmt.Sprintf("rm -rf %q", dir)
}
