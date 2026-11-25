package adjudicator

import (
	"bufio"
	"bytes"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// maxAuthoredTestEditBytes bounds the on-disk read the authored-edit screen
// performs on the decide path; a larger file fails closed.
const maxAuthoredTestEditBytes = 4 << 20

// testAuthorshipLimit bounds one adjudicator's authorship ledger. A full
// ledger records nothing further, which only withholds the exemption.
const testAuthorshipLimit = 256

// maxAlternateDepth bounds how far an object-store alternates chain is followed.
const maxAlternateDepth = 4

type testAuthorshipKey struct {
	trace string
	path  string
}

// testAuthorship is what this adjudicator itself admitted for one new test
// file: the directory it was authored into and the exact content the file may
// legitimately hold. Repository state is deliberately not the anchor: the
// index, HEAD and every ref can be rewritten by git plumbing that names no
// test path, so none of them can prove a file was never committed.
type testAuthorship struct {
	dir     os.FileInfo
	digests [2][sha256.Size]byte
}

// testAuthorshipPathKey spells a resolved path for the ledger. The spelling is
// exact; it folds case only where the filesystem is known to, so on a
// case-sensitive filesystem two files differing by case never share an entry,
// and on any other case-insensitive filesystem a variant spelling simply finds
// no entry.
func testAuthorshipPathKey(resolved string) string {
	if filepath.Separator == '\\' {
		return strings.ToLower(resolved)
	}
	return resolved
}

// noteTestAuthorship records an admitted new-test-file write so later typed
// edits of that same file by the same trace can be recognized. Anything that
// cannot be pinned (no trace, a path through a link, a missing parent
// directory, a full ledger) records nothing.
func (a *Adjudicator) noteTestAuthorship(c *abi.ToolCall, args map[string]any, target string) {
	if a == nil || c == nil || c.TraceID == "" {
		return
	}
	content, ok := args["content"].(string)
	if !ok {
		return
	}
	resolved := workspaceResolvedPath(target)
	if resolved == "" {
		return
	}
	parent := filepath.Dir(resolved)
	if !plainDirectoryChain(parent) {
		return
	}
	dir, err := os.Stat(parent)
	if err != nil || !dir.IsDir() {
		return
	}
	// Directory identity is resolved lazily on some platforms; resolve it now
	// so a later comparison sees the directory as it was at authorship.
	if !os.SameFile(dir, dir) {
		return
	}
	key := testAuthorshipKey{trace: c.TraceID, path: testAuthorshipPathKey(resolved)}
	if _, exists := a.testAuthored.Load(key); !exists {
		if a.testAuthoredCount.Load() >= testAuthorshipLimit {
			return
		}
		a.testAuthoredCount.Add(1)
	}
	digest := testContentDigest(content)
	a.testAuthored.Store(key, testAuthorship{dir: dir, digests: [2][sha256.Size]byte{digest, digest}})
}

func (a *Adjudicator) retireTestAuthorship(key testAuthorshipKey) {
	if _, loaded := a.testAuthored.LoadAndDelete(key); loaded {
		a.testAuthoredCount.Add(-1)
	}
}

func (a *Adjudicator) resetTestAuthorship() {
	a.testAuthored.Range(func(k, _ any) bool {
		if key, ok := k.(testAuthorshipKey); ok {
			a.retireTestAuthorship(key)
		} else {
			a.testAuthored.Delete(k)
		}
		return true
	})
}

// sessionAuthoredTestEditExempt reports whether this call is a typed edit of a
// Go test file that this adjudicator admitted as new-file authorship earlier in
// the same trace, and that still holds exactly the content it admitted. Such a
// file is in-progress authorship, not an existing gating suite, and editing it
// grants nothing the new-file exemption did not already grant. Fail-closed on
// every uncertainty:
//   - typed edit/write tools only — never a shell, a delete, or an append;
//   - the ledger entry is host-held and keyed by trace and exact path, so a
//     file authored by another session, a prior session, under another
//     spelling, or outside this adjudicator stays gated;
//   - no path component may be a link or mount point, and the parent directory
//     must be the very directory the file was authored into, so an alias or a
//     swapped directory cannot stand a committed test in its place;
//   - the on-disk bytes must be bytes this adjudicator admitted, so a file
//     restored, copied, reformatted, or checked out over the path stays gated;
//   - no enclosing repository may list the path in its index or hold the
//     on-disk bytes as an object: either means the file was staged or
//     committed, and the entry is retired for good. Objects outlive index and
//     ref rewrites, so removing the file from the index afterwards does not
//     bring the exemption back;
//   - the post-edit content must be reconstructable from unambiguous arguments
//     and must not carry a vacuous-pass vector.
func (a *Adjudicator) sessionAuthoredTestEditExempt(c *abi.ToolCall, lowerTool string, args map[string]any, target string) bool {
	if a == nil || c == nil || c.TraceID == "" {
		return false
	}
	if _, isCmd := commandArg(args); isCmd {
		return false
	}
	if !writeShapedLower(lowerTool) {
		return false
	}
	for _, w := range []string{"delete", "exec", "run", "unlink", "truncate", "remove", "append", "insert"} {
		if strings.Contains(lowerTool, w) {
			return false
		}
	}
	resolved := workspaceResolvedPath(target)
	if resolved == "" || !strings.HasSuffix(strings.ToLower(filepath.Base(resolved)), "_test.go") {
		return false
	}
	key := testAuthorshipKey{trace: c.TraceID, path: testAuthorshipPathKey(resolved)}
	v, found := a.testAuthored.Load(key)
	if !found {
		return false
	}
	entry, typed := v.(testAuthorship)
	if !typed {
		return false
	}
	parent := filepath.Dir(resolved)
	if !plainDirectoryChain(parent) {
		return false
	}
	dir, err := os.Stat(parent)
	if err != nil || !os.SameFile(entry.dir, dir) {
		return false
	}
	info, err := os.Lstat(resolved)
	if err != nil || !info.Mode().IsRegular() || info.Size() > maxAuthoredTestEditBytes {
		return false
	}
	existing, err := os.ReadFile(resolved)
	if err != nil {
		return false
	}
	switch repositoryKnowledge(resolved, existing) {
	case repoKnowsFile:
		a.retireTestAuthorship(key)
		return false
	case repoUnknown:
		return false
	case repoNeverSawFile:
		// No repository has ever taken this file in, so the session's own recorded
		// digests alone decide whether the edit is still the authored one.
	}
	current := testContentDigest(string(existing))
	if current != entry.digests[0] && current != entry.digests[1] {
		return false
	}
	post, ok := postEditContent(string(existing), args)
	if !ok || strings.TrimSpace(post) == "" || carriesVacuousPassVector(post) {
		return false
	}
	// The edit may fail to apply, so the content it replaces stays acceptable.
	entry.digests = [2][sha256.Size]byte{current, testContentDigest(post)}
	a.testAuthored.Store(key, entry)
	return true
}

// testContentDigest hashes content exactly, apart from line-ending style.
func testContentDigest(content string) [sha256.Size]byte {
	return sha256.Sum256([]byte(strings.ReplaceAll(content, "\r\n", "\n")))
}

// plainDirectoryChain reports whether dir and every ancestor up to the volume
// root is an ordinary directory: no symlink, junction, or other reparse point.
func plainDirectoryChain(dir string) bool {
	dir = filepath.Clean(dir)
	if !filepath.IsAbs(dir) {
		return false
	}
	for {
		parent := filepath.Dir(dir)
		if parent == dir {
			return true
		}
		info, err := os.Lstat(dir)
		if err != nil || info.Mode().Type() != os.ModeDir {
			return false
		}
		dir = parent
	}
}

type repoKnowledge int

const (
	repoUnknown repoKnowledge = iota
	repoKnowsFile
	repoNeverSawFile
)

// repositoryKnowledge walks from the file's directory to the volume root and
// asks every repository on the way whether it has ever taken the file in:
// listed in its index under any letter case, or holding the file's bytes as an
// object. repoNeverSawFile requires at least one repository and a clean answer
// from each; anything unreadable or unsupported is repoUnknown.
func repositoryKnowledge(file string, content []byte) repoKnowledge {
	found := false
	dir := filepath.Dir(file)
	for {
		if _, err := os.Lstat(filepath.Join(dir, ".git")); err == nil {
			indexPath, err := repositoryIndexPath(dir)
			if err != nil {
				return repoUnknown
			}
			if st, err := os.Stat(indexPath); err != nil || !st.Mode().IsRegular() {
				return repoUnknown
			}
			listed, err := gitIndexLists(dir, file, true)
			if err != nil {
				return repoUnknown
			}
			if listed {
				return repoKnowsFile
			}
			held, err := objectStoreHoldsBlob(gitObjectsDir(filepath.Dir(indexPath)), content, 0)
			if err != nil {
				return repoUnknown
			}
			if held {
				return repoKnowsFile
			}
			found = true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			if found {
				return repoNeverSawFile
			}
			return repoUnknown
		}
		dir = parent
	}
}

// gitObjectsDir returns the object store of the repository whose per-worktree
// git directory is gitDir, following a linked worktree's commondir pointer.
func gitObjectsDir(gitDir string) string {
	common := gitDir
	if data, err := os.ReadFile(filepath.Join(gitDir, "commondir")); err == nil {
		c := strings.TrimSpace(string(data))
		if c != "" {
			if !filepath.IsAbs(c) {
				c = filepath.Join(gitDir, c)
			}
			common = filepath.Clean(c)
		}
	}
	return filepath.Join(common, "objects")
}

// objectStoreHoldsBlob reports whether the object store, or any alternate it
// borrows from, holds content as a blob, as written or with CRLF line endings
// normalized the way a checkout filter would store it. Only the SHA-1 object
// format and version-2 pack indexes are read; anything else is an error.
func objectStoreHoldsBlob(objects string, content []byte, depth int) (bool, error) {
	if depth > maxAlternateDepth {
		return false, errors.New("alternates chain too deep")
	}
	if st, err := os.Stat(objects); err != nil || !st.IsDir() {
		return false, errors.New("unreadable object store")
	}
	ids := [][sha1.Size]byte{gitBlobID(content)}
	if normalized := bytes.ReplaceAll(content, []byte("\r\n"), []byte("\n")); !bytes.Equal(normalized, content) {
		ids = append(ids, gitBlobID(normalized))
	}
	for _, id := range ids {
		name := hex.EncodeToString(id[:])
		if _, err := os.Lstat(filepath.Join(objects, name[:2], name[2:])); err == nil {
			return true, nil
		} else if !os.IsNotExist(err) {
			return false, err
		}
	}
	packDir := filepath.Join(objects, "pack")
	entries, err := os.ReadDir(packDir)
	if err != nil && !os.IsNotExist(err) {
		return false, err
	}
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".idx") {
			continue
		}
		for _, id := range ids {
			held, err := packIndexHolds(filepath.Join(packDir, e.Name()), id)
			if err != nil || held {
				return held, err
			}
		}
	}
	alternates, err := os.Open(filepath.Join(objects, "info", "alternates"))
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer alternates.Close()
	scanner := bufio.NewScanner(alternates)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "\"") {
			return false, errors.New("quoted alternate path unsupported")
		}
		if !filepath.IsAbs(line) {
			line = filepath.Join(objects, line)
		}
		held, err := objectStoreHoldsBlob(filepath.Clean(line), content, depth+1)
		if err != nil || held {
			return held, err
		}
	}
	return false, scanner.Err()
}

func gitBlobID(content []byte) [sha1.Size]byte {
	h := sha1.New()
	h.Write([]byte("blob " + strconv.Itoa(len(content)) + "\x00"))
	h.Write(content)
	var id [sha1.Size]byte
	copy(id[:], h.Sum(nil))
	return id
}

// packIndexHolds binary-searches one version-2 pack index for id.
func packIndexHolds(path string, id [sha1.Size]byte) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	const fanoutStart, namesStart = 8, 8 + 256*4
	var header [8]byte
	if _, err := f.ReadAt(header[:], 0); err != nil {
		return false, err
	}
	if !bytes.Equal(header[:4], []byte{0xff, 't', 'O', 'c'}) || binary.BigEndian.Uint32(header[4:]) != 2 {
		return false, errors.New("unsupported pack index version")
	}
	fanout := func(b int) (uint32, error) {
		if b < 0 {
			return 0, nil
		}
		var word [4]byte
		if _, err := f.ReadAt(word[:], int64(fanoutStart+4*b)); err != nil {
			return 0, err
		}
		return binary.BigEndian.Uint32(word[:]), nil
	}
	lo, err := fanout(int(id[0]) - 1)
	if err != nil {
		return false, err
	}
	hi, err := fanout(int(id[0]))
	if err != nil {
		return false, err
	}
	if lo > hi {
		return false, errors.New("corrupt pack index fanout")
	}
	var name [sha1.Size]byte
	for lo < hi {
		mid := lo + (hi-lo)/2
		if _, err := f.ReadAt(name[:], int64(namesStart)+int64(mid)*sha1.Size); err != nil {
			return false, err
		}
		switch cmp := bytes.Compare(name[:], id[:]); {
		case cmp == 0:
			return true, nil
		case cmp < 0:
			lo = mid + 1
		default:
			hi = mid
		}
	}
	return false, nil
}

// postEditContent reconstructs the file content a typed write or edit would
// leave behind. A whole-file body is taken as-is; replacement edits are applied
// to the existing content in order. An argument shape it cannot model, a field
// supplied under more than one spelling, or a replacement whose anchor is
// absent answers false: the screened content must be the written content.
func postEditContent(existing string, args map[string]any) (string, bool) {
	for _, k := range []string{"contents", "text", "file_text", "fileText", "new_content", "newContent", "data", "patch", "diff"} {
		if _, present := args[k]; present {
			return "", false
		}
	}
	cur := strings.ReplaceAll(existing, "\r\n", "\n")
	pair, pairOK := readEditPair(args)
	if !pairOK {
		return "", false
	}
	rawEdits, hasEdits := args["edits"]
	if rawContent, present := args["content"]; present {
		content, isString := rawContent.(string)
		if !isString || pair.hasOld || pair.hasNew || hasEdits {
			return "", false
		}
		return content, true
	}
	if hasEdits {
		if pair.hasOld || pair.hasNew {
			return "", false
		}
		list, ok := rawEdits.([]any)
		if !ok || len(list) == 0 {
			return "", false
		}
		for _, item := range list {
			m, ok := item.(map[string]any)
			if !ok {
				return "", false
			}
			p, ok := readEditPair(m)
			if !ok || !p.hasOld || !p.hasNew {
				return "", false
			}
			if cur, ok = applyReplacement(cur, p); !ok {
				return "", false
			}
		}
		return cur, true
	}
	if !pair.hasOld || !pair.hasNew {
		return "", false
	}
	return applyReplacement(cur, pair)
}

type editPair struct {
	old, repl      string
	hasOld, hasNew bool
	all            bool
}

// readEditPair reads one replacement. Each field may appear under exactly one
// spelling and must have its expected type.
func readEditPair(m map[string]any) (editPair, bool) {
	var p editPair
	oldSeen, newSeen, allSeen := 0, 0, 0
	for _, k := range []string{"old_string", "oldString", "old_str", "oldStr", "old_text", "oldText"} {
		if v, present := m[k]; present {
			s, isString := v.(string)
			if !isString {
				return p, false
			}
			p.old, p.hasOld = s, true
			oldSeen++
		}
	}
	for _, k := range []string{"new_string", "newString", "new_str", "newStr", "new_text", "newText"} {
		if v, present := m[k]; present {
			s, isString := v.(string)
			if !isString {
				return p, false
			}
			p.repl, p.hasNew = s, true
			newSeen++
		}
	}
	for _, k := range []string{"replace_all", "replaceAll"} {
		if v, present := m[k]; present {
			b, isBool := v.(bool)
			if !isBool {
				return p, false
			}
			p.all = b
			allSeen++
		}
	}
	return p, oldSeen <= 1 && newSeen <= 1 && allSeen <= 1
}

func applyReplacement(cur string, p editPair) (string, bool) {
	old := strings.ReplaceAll(p.old, "\r\n", "\n")
	repl := strings.ReplaceAll(p.repl, "\r\n", "\n")
	if old == "" || !strings.Contains(cur, old) {
		return "", false
	}
	if p.all {
		return strings.ReplaceAll(cur, old, repl), true
	}
	return strings.Replace(cur, old, repl, 1), true
}

// carriesVacuousPassVector reports whether Go test source carries a construct
// that can make a package report ok without running its tests: a TestMain
// definition in any whitespace spelling (one that skips m.Run() runs nothing),
// an os.Exit call, or a test-flag override that filters every test out.
func carriesVacuousPassVector(content string) bool {
	normalized := strings.Map(func(r rune) rune {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			return -1
		}
		return r
	}, content)
	for _, vector := range []string{"funcTestMain(", "os.Exit(", "flag.Set(", "CommandLine.Set(", ".Value.Set("} {
		if strings.Contains(normalized, vector) {
			return true
		}
	}
	return false
}
