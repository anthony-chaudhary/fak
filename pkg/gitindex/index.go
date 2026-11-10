// Package gitindex replaces `git status` / `git diff-files` for TRACKED-file
// dirty detection by parsing the git index (`.git/index`) directly and
// stat-comparing each entry against the working tree, hashing only the
// entries whose stat data mismatches or that are racily clean.
//
// Format reference: https://git-scm.com/docs/index-format (versions 2, 3, 4,
// including v4 path-prefix compression). Extensions are skipped by their
// declared size; the trailing SHA-1 checksum is verified (an all-zero
// checksum, written under index.skipHash, is accepted). A split index
// ("link" extension) or a SHA-256 repository is reported as ErrUnsupported so
// callers fall back to git.
//
// Scope: tracked files only. Untracked files are OUT OF SCOPE — this package
// never walks the working tree for paths absent from the index, so it is not a
// replacement for `git status --porcelain` untracked (`??`) output.
//
// Semantics match `git update-index --refresh && git diff-files --name-only`,
// i.e. content-based dirtiness (what `git status` reports for the
// worktree-vs-index side), not raw `git diff-files` which also lists entries
// that are merely stat-dirty with unchanged content.
package gitindex

import (
	"bytes"
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ErrUnsupported marks an index this parser deliberately does not handle
// (split index, SHA-256 object format, unknown version). Callers in
// FallbackAuto mode transparently shell out to git instead.
var ErrUnsupported = errors.New("gitindex: unsupported index")

// Entry flag bits (index-format "flags" and v3+ "extended flags").
const (
	flagAssumeValid  = 0x8000
	flagExtended     = 0x4000
	flagStageMask    = 0x3000
	flagNameMask     = 0x0fff
	extSkipWorktree  = 0x4000
	extIntentToAdd   = 0x2000
	modeTypeMask     = 0170000
	modeRegular      = 0100000
	modeSymlink      = 0120000
	modeGitlink      = 0160000
	hashLen          = 20
	entryFixedHeader = 62 // 10*u32 stat fields + 20-byte oid + u16 flags
)

// Entry is one parsed index entry.
type Entry struct {
	Path         string
	CtimeSec     uint32
	CtimeNsec    uint32
	MtimeSec     uint32
	MtimeNsec    uint32
	Mode         uint32
	Size         uint32
	OID          [hashLen]byte
	Stage        int
	AssumeValid  bool
	SkipWorktree bool
	IntentToAdd  bool
}

// Index is a parsed index file.
type Index struct {
	Version uint32
	Entries []Entry
	// MtimeSec/MtimeNsec is the index file's own mtime, used for the racy-git
	// check (an entry whose mtime is >= the index mtime cannot be trusted by
	// stat alone).
	Mtime time.Time
}

// ParseIndex parses raw index bytes (version 2, 3 or 4) and verifies the
// trailing checksum.
func ParseIndex(data []byte) (*Index, error) {
	if len(data) < 12+hashLen {
		return nil, fmt.Errorf("gitindex: index too short (%d bytes)", len(data))
	}
	if string(data[:4]) != "DIRC" {
		return nil, fmt.Errorf("gitindex: bad signature %q", data[:4])
	}
	body := data[:len(data)-hashLen]
	sum := data[len(data)-hashLen:]
	if !allZero(sum) {
		got := sha1.Sum(body)
		if !bytes.Equal(got[:], sum) {
			return nil, fmt.Errorf("gitindex: checksum mismatch (got %x want %x)", got, sum)
		}
	}
	ver := binary.BigEndian.Uint32(data[4:8])
	if ver < 2 || ver > 4 {
		return nil, fmt.Errorf("%w: version %d", ErrUnsupported, ver)
	}
	n := binary.BigEndian.Uint32(data[8:12])
	idx := &Index{Version: ver, Entries: make([]Entry, 0, n)}
	off := 12
	var prev []byte
	for i := uint32(0); i < n; i++ {
		start := off
		if off+entryFixedHeader > len(body) {
			return nil, fmt.Errorf("gitindex: entry %d truncated", i)
		}
		u := func(k int) uint32 { return binary.BigEndian.Uint32(body[off+4*k:]) }
		e := Entry{
			CtimeSec: u(0), CtimeNsec: u(1), MtimeSec: u(2), MtimeNsec: u(3),
			Mode: u(6), Size: u(9),
		}
		copy(e.OID[:], body[off+40:off+40+hashLen])
		flags := binary.BigEndian.Uint16(body[off+60:])
		off += entryFixedHeader
		e.AssumeValid = flags&flagAssumeValid != 0
		e.Stage = int(flags&flagStageMask) >> 12
		if flags&flagExtended != 0 {
			if ver < 3 {
				return nil, fmt.Errorf("gitindex: entry %d has extended flag in v%d index", i, ver)
			}
			if off+2 > len(body) {
				return nil, fmt.Errorf("gitindex: entry %d truncated", i)
			}
			ext := binary.BigEndian.Uint16(body[off:])
			off += 2
			e.SkipWorktree = ext&extSkipWorktree != 0
			e.IntentToAdd = ext&extIntentToAdd != 0
		}
		if ver == 4 {
			strip, w := decodeVarint(body[off:])
			if w == 0 {
				return nil, fmt.Errorf("gitindex: entry %d bad prefix varint", i)
			}
			off += w
			if strip > uint64(len(prev)) {
				return nil, fmt.Errorf("gitindex: entry %d strips %d of %d-byte previous name", i, strip, len(prev))
			}
			nul := bytes.IndexByte(body[off:], 0)
			if nul < 0 {
				return nil, fmt.Errorf("gitindex: entry %d unterminated name", i)
			}
			name := make([]byte, 0, len(prev)-int(strip)+nul)
			name = append(name, prev[:len(prev)-int(strip)]...)
			name = append(name, body[off:off+nul]...)
			off += nul + 1
			prev = name
			e.Path = string(name)
		} else {
			nameLen := int(flags & flagNameMask)
			nul := bytes.IndexByte(body[off:], 0)
			if nul < 0 {
				return nil, fmt.Errorf("gitindex: entry %d unterminated name", i)
			}
			if nameLen < flagNameMask && nul != nameLen {
				return nil, fmt.Errorf("gitindex: entry %d name length %d != %d", i, nul, nameLen)
			}
			e.Path = string(body[off : off+nul])
			// Entry is padded with 1..8 NULs to a multiple of 8 bytes.
			entLen := off + nul - start
			off = start + (entLen+8)&^7
			if off > len(body) {
				return nil, fmt.Errorf("gitindex: entry %d padding truncated", i)
			}
		}
		idx.Entries = append(idx.Entries, e)
	}
	// Extensions: 4-byte signature, u32 size, payload. Skip all; a split
	// index ("link") means entries live in a shared index we do not read.
	for off+8 <= len(body) {
		sig := string(body[off : off+4])
		size := int(binary.BigEndian.Uint32(body[off+4:]))
		if sig == "link" {
			return nil, fmt.Errorf("%w: split index (link extension)", ErrUnsupported)
		}
		off += 8 + size
		if off > len(body) {
			return nil, fmt.Errorf("gitindex: extension %q overruns index", sig)
		}
	}
	if off != len(body) {
		return nil, fmt.Errorf("gitindex: %d trailing bytes before checksum", len(body)-off)
	}
	return idx, nil
}

// decodeVarint decodes git's offset varint (varint.c decode_varint): each
// continuation adds one before shifting so encodings are unique.
func decodeVarint(b []byte) (uint64, int) {
	if len(b) == 0 {
		return 0, 0
	}
	c := b[0]
	val := uint64(c & 127)
	i := 1
	for c&128 != 0 {
		if i >= len(b) || i > 9 {
			return 0, 0
		}
		val++
		c = b[i]
		i++
		val = (val << 7) + uint64(c&127)
	}
	return val, i
}

func allZero(b []byte) bool {
	for _, c := range b {
		if c != 0 {
			return false
		}
	}
	return true
}

// ReadIndexFile reads and parses an index file, recording its mtime for the
// racy-git check.
func ReadIndexFile(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	data := make([]byte, st.Size())
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, err
	}
	idx, err := ParseIndex(data)
	if err != nil {
		return nil, err
	}
	idx.Mtime = st.ModTime()
	return idx, nil
}

// repoDirs resolves the per-worktree git dir (holding index) and the common
// git dir (holding config, info/) for a working tree root. A `.git` FILE
// ("gitdir: <path>", used by linked worktrees and submodules) is followed.
func repoDirs(root string) (gitDir, commonDir string, err error) {
	dotgit := filepath.Join(root, ".git")
	st, err := os.Stat(dotgit)
	if err != nil {
		return "", "", err
	}
	gitDir = dotgit
	if !st.IsDir() {
		b, err := os.ReadFile(dotgit)
		if err != nil {
			return "", "", err
		}
		line := strings.TrimSpace(string(b))
		if !strings.HasPrefix(line, "gitdir:") {
			return "", "", fmt.Errorf("gitindex: %s is not a gitdir file", dotgit)
		}
		gitDir = strings.TrimSpace(strings.TrimPrefix(line, "gitdir:"))
		if !filepath.IsAbs(gitDir) {
			gitDir = filepath.Join(root, gitDir)
		}
	}
	commonDir = gitDir
	if b, err := os.ReadFile(filepath.Join(gitDir, "commondir")); err == nil {
		c := strings.TrimSpace(string(b))
		if !filepath.IsAbs(c) {
			c = filepath.Join(gitDir, c)
		}
		commonDir = filepath.Clean(c)
	}
	return gitDir, commonDir, nil
}

func oidHex(o [hashLen]byte) string { return hex.EncodeToString(o[:]) }
