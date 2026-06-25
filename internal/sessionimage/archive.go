package sessionimage

// archive.go — the single-file offload: pack a bundle directory into one .faksession tar
// you can scp, drop in object storage, attach to a job, or hand to another user, and
// unpack it on a fresh host into a fresh directory. The format is plain stdlib
// archive/tar (zero deps, consistent with the rest of the kernel) and DETERMINISTIC: the
// entries are exactly image.json + the parts image.json lists, in a fixed order, with
// normalized headers (mode 0644, uid/gid 0, ModTime pinned to the image's UpdatedUnix),
// so the same image packs to byte-identical bytes — a property a content-addressed
// offload store can rely on.
//
// Unpack is the trust boundary for an UNTRUSTED archive (it came from another machine):
// every entry name is forced through safeName (no path traversal, no absolute paths, no
// separators) and each file is size-bounded, so a malicious tar can neither escape the
// target directory nor exhaust the disk. The CONTENT trust boundary is unchanged and
// downstream: LoadDir re-verifies every part's sha256, and recall's gate re-screens
// every page-in — so even a byte-perfect-but-poisoned image is caught on resume, not
// trusted because it unpacked cleanly.

import (
	"archive/tar"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// maxPartBytes caps a single extracted entry, a backstop against a tar bomb. A real
// session image's parts are small (a page table + content-addressed bytes); 512 MiB is
// far above any honest image and far below "fill the disk".
const maxPartBytes = 512 << 20

// maxArchiveEntries and maxArchiveBytes bound an UNTRUSTED archive as a WHOLE, not just
// per entry — the per-entry cap alone does not stop a tar with thousands of near-cap
// entries from exhausting the disk. A real image is a fixed small set (image.json plus
// the ≤5 known parts), so the headroom here is generous and an over-large archive is
// refused before it is written, delivering the "cannot exhaust the disk" guarantee the
// package doc claims.
const (
	maxArchiveEntries = 16
	maxArchiveBytes   = maxPartBytes // total across all entries, not per entry
)

// Pack writes the bundle in dir to w as a deterministic .faksession tar: image.json plus
// every part image.json lists, normalized headers, fixed order. It reads image.json to
// learn the part list (so it never sweeps a stray file into the archive) but does not
// re-verify digests — the receiving LoadDir does that. A part listed but missing on disk
// is an error (a corrupt source bundle must not pack into a deceptively-whole archive).
func Pack(dir string, w io.Writer) error {
	mb, err := os.ReadFile(filepath.Join(dir, ImageFile))
	if err != nil {
		return err
	}
	var meta Meta
	if err := json.Unmarshal(mb, &meta); err != nil {
		return fmt.Errorf("sessionimage: bad %s: %w", ImageFile, err)
	}

	modTime := time.Unix(meta.UpdatedUnix, 0).UTC()
	tw := tar.NewWriter(w)

	// image.json first, then the parts in their listed (deterministic) order.
	if err := writeTarEntry(tw, dir, ImageFile, mb, modTime); err != nil {
		return err
	}
	for _, p := range meta.Parts {
		name, ok := safeName(p.Name)
		if !ok {
			return fmt.Errorf("sessionimage: refuse to pack unsafe part name %q", p.Name)
		}
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return fmt.Errorf("sessionimage: part %q listed but missing — refusing to pack a partial image: %w", name, err)
		}
		if err := writeTarEntry(tw, dir, name, b, modTime); err != nil {
			return err
		}
	}
	return tw.Close()
}

// writeTarEntry writes one normalized regular-file entry. Headers are fully pinned (no
// uid/gid/uname, fixed mode, caller-pinned ModTime) so packing is deterministic.
func writeTarEntry(tw *tar.Writer, _ string, name string, body []byte, modTime time.Time) error {
	h := &tar.Header{
		Name:     name,
		Mode:     0o644,
		Size:     int64(len(body)),
		ModTime:  modTime,
		Typeflag: tar.TypeReg,
		Format:   tar.FormatPAX,
	}
	if err := tw.WriteHeader(h); err != nil {
		return err
	}
	_, err := tw.Write(body)
	return err
}

// Unpack extracts a .faksession tar from r into dir, creating dir if needed. It is the
// untrusted-input boundary: every entry name must be a safe in-bundle base name (no
// traversal, no absolute path) and every entry is size-capped. It does NOT verify
// content integrity — call LoadDir (or LoadArchive) afterward, which re-hashes every
// part and fails closed on tampering.
func Unpack(r io.Reader, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tr := tar.NewReader(r)
	seen := map[string]bool{}
	var entries int
	var total int64
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if entries++; entries > maxArchiveEntries {
			return fmt.Errorf("sessionimage: archive has more than %d entries — refused (not a session image)", maxArchiveEntries)
		}
		if h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA { //nolint:staticcheck // TypeRegA for old writers
			// A session image is a flat set of regular files; anything else (dir,
			// symlink, device) is unexpected and refused rather than interpreted.
			return fmt.Errorf("sessionimage: archive entry %q has non-regular type %d — refused", h.Name, h.Typeflag)
		}
		name, ok := safeName(h.Name)
		if !ok {
			return fmt.Errorf("sessionimage: archive entry %q is not a safe in-bundle name (path traversal?) — refused", h.Name)
		}
		if seen[name] {
			// A duplicate name would O_TRUNC-overwrite an already-extracted file (wasted
			// write bandwidth, and a way to smuggle a second payload past a first-seen
			// check). A session image names each part once.
			return fmt.Errorf("sessionimage: archive names %q more than once — refused", name)
		}
		seen[name] = true
		if h.Size < 0 || h.Size > maxPartBytes {
			return fmt.Errorf("sessionimage: archive entry %q size %d out of bounds [0,%d]", name, h.Size, int64(maxPartBytes))
		}
		if total += h.Size; total > maxArchiveBytes {
			return fmt.Errorf("sessionimage: archive exceeds the %d-byte total budget — refused", int64(maxArchiveBytes))
		}
		f, err := os.OpenFile(filepath.Join(dir, name), os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		// Bound the copy by the declared size; a writer claiming N but streaming more
		// cannot overrun the cap.
		_, cpErr := io.CopyN(f, tr, h.Size)
		closeErr := f.Close()
		if cpErr != nil && cpErr != io.EOF {
			return fmt.Errorf("sessionimage: extracting %q: %w", name, cpErr)
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

// PackFile packs dir into the single archive file at path (overwriting it).
func PackFile(dir, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Pack(dir, f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// UnpackFile extracts the archive at path into dir.
func UnpackFile(path, dir string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return Unpack(f, dir)
}

// LoadArchive is the receiving-side convenience: unpack the archive at path into dir,
// then LoadDir it (which re-verifies every part's integrity, fail-closed). This is the
// one call a restoring host makes on an offloaded .faksession before Rehydrate.
func LoadArchive(path, dir string) (*Image, error) {
	if err := UnpackFile(path, dir); err != nil {
		return nil, err
	}
	return LoadDir(dir)
}
