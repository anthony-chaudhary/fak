package amdgpu

import (
	"bufio"
	"compress/zlib"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
)

const (
	strixGitSnapshotUnattestedToken = "UNATTESTED_VALIDATOR_BINARY"
	strixGitSnapshotConfig          = "[core]\n\trepositoryformatversion = 0\n\tbare = true\n"

	strixGitSnapshotDefaultMaxFiles      = 250_000
	strixGitSnapshotDefaultMaxEntries    = 500_000
	strixGitSnapshotDefaultMaxFileBytes  = int64(4 << 30)
	strixGitSnapshotDefaultMaxTotalBytes = int64(64 << 30)
	strixGitSnapshotDefaultCopyBuffer    = 128 << 10
	strixGitSnapshotDefaultReadDirBatch  = 128
)

// strixGitSnapshotRefusal intentionally carries no raw cause or pathname. The
// snapshot is an authority boundary, so all failure classes collapse to one
// typed, redacted, unattested outcome for its same-package consumer.
type strixGitSnapshotRefusal struct {
	token string
}

func (e *strixGitSnapshotRefusal) Error() string {
	return e.token + ": sterile Git object snapshot refused"
}

func refuseStrixGitSnapshot() error {
	return &strixGitSnapshotRefusal{token: strixGitSnapshotUnattestedToken}
}

// strixGitObjectSnapshot is an opaque, private bare-repository object view.
// The only downstream seam is its directory and deterministic cleanup; it
// carries no live repository path, revision, runner, or authority verdict.
type strixGitObjectSnapshot struct {
	root      string
	removeAll func(string) error
	closeMu   sync.Mutex
}

func (s *strixGitObjectSnapshot) gitDir() string {
	if s == nil {
		return ""
	}
	return s.root
}

func (s *strixGitObjectSnapshot) close() error {
	if s == nil {
		return nil
	}
	s.closeMu.Lock()
	defer s.closeMu.Unlock()
	if s.root == "" {
		return nil
	}
	removeAll := s.removeAll
	if removeAll == nil {
		removeAll = os.RemoveAll
	}
	if err := removeAll(s.root); err != nil {
		return &strixGitSnapshotCleanupError{}
	}
	s.root = ""
	return nil
}

type strixGitSnapshotCleanupError struct{}

func (*strixGitSnapshotCleanupError) Error() string {
	return "sterile Git object snapshot cleanup failed; retry close"
}

type strixGitSnapshotOptions struct {
	tempParent          string
	maxFiles            int
	maxEntries          int
	maxFileBytes        int64
	maxTotalBytes       int64
	copyBufferBytes     int
	readDirBatchEntries int
	removeAll           func(string) error

	// The hooks are deterministic same-package race witnesses, not production
	// inputs. They run only after all source files are open, and after each
	// copied chunk respectively.
	afterOpen         func() error
	afterCopyChunk    func(rel string) error
	afterReadDirBatch func()
}

func (o strixGitSnapshotOptions) withDefaults() (strixGitSnapshotOptions, bool) {
	if o.maxFiles < 0 || o.maxEntries < 0 || o.maxFileBytes < 0 || o.maxTotalBytes < 0 || o.copyBufferBytes < 0 || o.readDirBatchEntries < 0 {
		return strixGitSnapshotOptions{}, false
	}
	if o.maxFiles == 0 {
		o.maxFiles = strixGitSnapshotDefaultMaxFiles
	}
	if o.maxEntries == 0 {
		o.maxEntries = strixGitSnapshotDefaultMaxEntries
	}
	if o.maxFileBytes == 0 {
		o.maxFileBytes = strixGitSnapshotDefaultMaxFileBytes
	}
	if o.maxTotalBytes == 0 {
		o.maxTotalBytes = strixGitSnapshotDefaultMaxTotalBytes
	}
	if o.copyBufferBytes == 0 {
		o.copyBufferBytes = strixGitSnapshotDefaultCopyBuffer
	}
	if o.readDirBatchEntries == 0 {
		o.readDirBatchEntries = strixGitSnapshotDefaultReadDirBatch
	}
	if o.removeAll == nil {
		o.removeAll = os.RemoveAll
	}
	return o, o.maxFiles > 0 && o.maxEntries > 0 && o.maxFileBytes > 0 && o.maxTotalBytes > 0 && o.copyBufferBytes > 0 && o.readDirBatchEntries > 0
}

type strixGitObjectKind uint8

const (
	strixGitLooseObject strixGitObjectKind = iota + 1
	strixGitPackFile
	strixGitIndexFile
)

type openedStrixGitObject struct {
	destination string
	file        *os.File
	info        os.FileInfo
	kind        strixGitObjectKind
	identity    string
}

// newStrixGitObjectSnapshot is the production same-package seam. Callers can
// select only the context and already-resolved repository root; resource caps
// and snapshot construction policy are fixed here.
func newStrixGitObjectSnapshot(ctx context.Context, repoRoot string) (*strixGitObjectSnapshot, error) {
	return snapshotStrixGitObjects(ctx, repoRoot, strixGitSnapshotOptions{})
}

func (o *openedStrixGitObject) close() {
	if o != nil && o.file != nil {
		_ = o.file.Close()
		o.file = nil
	}
}

// snapshotStrixGitObjects freezes only canonical SHA-1 loose objects and
// complete pack/index pairs. All live path traversal ends before afterOpen;
// publication copies exclusively from the already-open regular-file handles.
func snapshotStrixGitObjects(ctx context.Context, repoRoot string, rawOpts strixGitSnapshotOptions) (result *strixGitObjectSnapshot, resultErr error) {
	if ctx == nil {
		return nil, refuseStrixGitSnapshot()
	}
	opts, ok := rawOpts.withDefaults()
	if !ok || ctx.Err() != nil {
		return nil, refuseStrixGitSnapshot()
	}
	repository, err := openCleanStrixRepositoryRoot(repoRoot)
	if err != nil {
		return nil, refuseStrixGitSnapshot()
	}
	defer repository.Close()

	gitRoot, err := openVerifiedStrixDirectory(repository, ".git")
	if err != nil {
		return nil, refuseStrixGitSnapshot()
	}
	defer gitRoot.Close()
	objectsRoot, err := openVerifiedStrixDirectory(gitRoot, "objects")
	if err != nil {
		return nil, refuseStrixGitSnapshot()
	}
	defer objectsRoot.Close()

	opened, err := enumerateOpenedStrixGitObjects(ctx, objectsRoot, opts)
	if err != nil {
		return nil, refuseStrixGitSnapshot()
	}
	defer func() {
		for i := range opened {
			opened[i].close()
		}
	}()
	if opts.afterOpen != nil {
		if err := opts.afterOpen(); err != nil {
			return nil, refuseStrixGitSnapshot()
		}
	}
	if ctx.Err() != nil {
		return nil, refuseStrixGitSnapshot()
	}

	snapshotPath, err := os.MkdirTemp(opts.tempParent, "fak-strix-git-objects-*")
	if err != nil {
		return nil, refuseStrixGitSnapshot()
	}
	owned := &strixGitObjectSnapshot{root: snapshotPath, removeAll: opts.removeAll}
	cleanupRequired := true
	defer func() {
		if cleanupRequired {
			if cleanupErr := owned.close(); cleanupErr != nil {
				result = owned
				resultErr = errors.Join(resultErr, cleanupErr)
			}
		}
	}()
	if err := os.Chmod(snapshotPath, 0o700); err != nil {
		return nil, refuseStrixGitSnapshot()
	}
	destination, err := os.OpenRoot(snapshotPath)
	if err != nil {
		return nil, refuseStrixGitSnapshot()
	}
	defer destination.Close()
	if err := destination.Mkdir("objects", 0o700); err != nil {
		return nil, refuseStrixGitSnapshot()
	}
	if _, err := writeExclusiveStrixSnapshotFile(destination, "config", strings.NewReader(strixGitSnapshotConfig), int64(len(strixGitSnapshotConfig)), opts, nil); err != nil {
		return nil, refuseStrixGitSnapshot()
	}

	createdDirs := map[string]bool{"objects": true}
	packProofs := make(map[string]strixGitPackProof)
	indexProofs := make(map[string]strixGitPackProof)
	for i := range opened {
		if ctx.Err() != nil {
			return nil, refuseStrixGitSnapshot()
		}
		dir := filepath.Dir(opened[i].destination)
		if !createdDirs[dir] {
			if err := destination.MkdirAll(dir, 0o700); err != nil {
				return nil, refuseStrixGitSnapshot()
			}
			createdDirs[dir] = true
		}
		proof, err := copyOpenedStrixGitObject(ctx, destination, &opened[i], opts)
		if err != nil {
			return nil, refuseStrixGitSnapshot()
		}
		switch opened[i].kind {
		case strixGitPackFile:
			packProofs[opened[i].identity] = proof
		case strixGitIndexFile:
			indexProofs[opened[i].identity] = proof
		}
	}
	for id, packProof := range packProofs {
		indexProof, ok := indexProofs[id]
		if !ok || packProof.checksum != indexProof.checksum || packProof.objectCount != indexProof.objectCount {
			return nil, refuseStrixGitSnapshot()
		}
	}
	if ctx.Err() != nil {
		return nil, refuseStrixGitSnapshot()
	}
	for i := range opened {
		opened[i].close()
	}
	cleanupRequired = false
	return owned, nil
}

func openCleanStrixRepositoryRoot(repoRoot string) (*os.Root, error) {
	if repoRoot == "" || !filepath.IsAbs(repoRoot) || filepath.Clean(repoRoot) != repoRoot {
		return nil, fmt.Errorf("unclean repository root")
	}
	before, err := os.Lstat(repoRoot)
	if err != nil || before.Mode()&os.ModeSymlink != 0 || !before.IsDir() {
		return nil, fmt.Errorf("unsupported repository root")
	}
	resolved, err := filepath.EvalSymlinks(repoRoot)
	if err != nil || !sameStrixFilesystemPath(repoRoot, resolved) {
		return nil, fmt.Errorf("redirected repository root")
	}
	root, err := os.OpenRoot(repoRoot)
	if err != nil {
		return nil, err
	}
	opened, openErr := root.Stat(".")
	after, afterErr := os.Lstat(repoRoot)
	if openErr != nil || afterErr != nil || after.Mode()&os.ModeSymlink != 0 || !after.IsDir() ||
		!os.SameFile(before, opened) || !os.SameFile(opened, after) {
		_ = root.Close()
		return nil, fmt.Errorf("repository root changed while opening")
	}
	return root, nil
}

func openVerifiedStrixDirectory(parent *os.Root, name string) (*os.Root, error) {
	before, err := parent.Lstat(name)
	if err != nil || before.Mode()&os.ModeSymlink != 0 || !before.IsDir() {
		return nil, fmt.Errorf("unsupported directory")
	}
	child, err := parent.OpenRoot(name)
	if err != nil {
		return nil, err
	}
	opened, openErr := child.Stat(".")
	after, afterErr := parent.Lstat(name)
	if openErr != nil || afterErr != nil || after.Mode()&os.ModeSymlink != 0 || !after.IsDir() ||
		!os.SameFile(before, opened) || !os.SameFile(opened, after) {
		_ = child.Close()
		return nil, fmt.Errorf("directory changed while opening")
	}
	return child, nil
}

func readStrixRootDirectory(ctx context.Context, root *os.Root, opts strixGitSnapshotOptions, inspected *int) ([]os.DirEntry, error) {
	dir, err := root.Open(".")
	if err != nil {
		return nil, err
	}
	defer dir.Close()
	var entries []os.DirEntry
	for {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		batch, readErr := dir.ReadDir(opts.readDirBatchEntries)
		for _, entry := range batch {
			*inspected++
			if *inspected > opts.maxEntries {
				return nil, fmt.Errorf("object-store enumeration cap exceeded")
			}
			entries = append(entries, entry)
		}
		if len(batch) > 0 && opts.afterReadDirBatch != nil {
			opts.afterReadDirBatch()
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, readErr
		}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	return entries, nil
}

func enumerateOpenedStrixGitObjects(ctx context.Context, objects *os.Root, opts strixGitSnapshotOptions) ([]openedStrixGitObject, error) {
	var opened []openedStrixGitObject
	inspected := 0
	fail := func(err error) ([]openedStrixGitObject, error) {
		for i := range opened {
			opened[i].close()
		}
		return nil, err
	}
	rootEntries, err := readStrixRootDirectory(ctx, objects, opts, &inspected)
	if err != nil {
		return fail(err)
	}
	packKinds := make(map[string]uint8)
	var total int64
	admit := func(parent *os.Root, sourceName, destination string, kind strixGitObjectKind, identity string) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		object, err := openVerifiedStrixObject(parent, sourceName, destination)
		if err != nil {
			return err
		}
		if object.info.Size() < 0 || len(opened) >= opts.maxFiles || object.info.Size() > opts.maxFileBytes || object.info.Size() > opts.maxTotalBytes-total {
			object.close()
			return fmt.Errorf("snapshot cap exceeded")
		}
		total += object.info.Size()
		object.kind = kind
		object.identity = identity
		opened = append(opened, object)
		return nil
	}

	for _, entry := range rootEntries {
		if ctx.Err() != nil {
			return fail(ctx.Err())
		}
		name := entry.Name()
		info, statErr := objects.Lstat(name)
		if statErr != nil {
			return fail(statErr)
		}
		if info.Mode()&os.ModeSymlink != 0 || (!info.IsDir() && !info.Mode().IsRegular()) {
			return fail(fmt.Errorf("unsupported object-store entry"))
		}
		switch {
		case name == "pack":
			if !info.IsDir() {
				return fail(fmt.Errorf("pack is not a directory"))
			}
			packRoot, openErr := openVerifiedStrixDirectory(objects, name)
			if openErr != nil {
				return fail(openErr)
			}
			packEntries, readErr := readStrixRootDirectory(ctx, packRoot, opts, &inspected)
			if readErr != nil {
				_ = packRoot.Close()
				return fail(readErr)
			}
			for _, packEntry := range packEntries {
				packName := packEntry.Name()
				packInfo, packErr := packRoot.Lstat(packName)
				if packErr != nil || packInfo.Mode()&os.ModeSymlink != 0 || !packInfo.Mode().IsRegular() {
					_ = packRoot.Close()
					if packErr != nil {
						return fail(packErr)
					}
					return fail(fmt.Errorf("unsupported pack entry"))
				}
				id, kind, canonical := canonicalStrixPackObjectName(packName)
				if !canonical {
					if isStrixSHA256PackName(packName) {
						_ = packRoot.Close()
						return fail(fmt.Errorf("unsupported SHA-256 pack"))
					}
					continue
				}
				objectKind := strixGitPackFile
				if kind == 2 {
					objectKind = strixGitIndexFile
				}
				if err := admit(packRoot, packName, filepath.Join("objects", "pack", packName), objectKind, id); err != nil {
					_ = packRoot.Close()
					return fail(err)
				}
				packKinds[id] |= kind
			}
			_ = packRoot.Close()

		case isLowerHex(name, 2):
			if !info.IsDir() {
				return fail(fmt.Errorf("loose object fanout is not a directory"))
			}
			fanout, openErr := openVerifiedStrixDirectory(objects, name)
			if openErr != nil {
				return fail(openErr)
			}
			looseEntries, readErr := readStrixRootDirectory(ctx, fanout, opts, &inspected)
			if readErr != nil {
				_ = fanout.Close()
				return fail(readErr)
			}
			for _, looseEntry := range looseEntries {
				looseName := looseEntry.Name()
				looseInfo, looseErr := fanout.Lstat(looseName)
				if looseErr != nil || looseInfo.Mode()&os.ModeSymlink != 0 || !looseInfo.Mode().IsRegular() {
					_ = fanout.Close()
					if looseErr != nil {
						return fail(looseErr)
					}
					return fail(fmt.Errorf("unsupported loose object entry"))
				}
				if isLowerHex(looseName, 62) {
					_ = fanout.Close()
					return fail(fmt.Errorf("unsupported SHA-256 loose object"))
				}
				if !isLowerHex(looseName, 38) {
					continue
				}
				if err := admit(fanout, looseName, filepath.Join("objects", name, looseName), strixGitLooseObject, name+looseName); err != nil {
					_ = fanout.Close()
					return fail(err)
				}
			}
			_ = fanout.Close()
		}
	}
	for _, kinds := range packKinds {
		if kinds != 3 {
			return fail(fmt.Errorf("pack/index pair is incomplete"))
		}
	}
	sort.Slice(opened, func(i, j int) bool { return opened[i].destination < opened[j].destination })
	return opened, nil
}

func openVerifiedStrixObject(root *os.Root, name, destination string) (openedStrixGitObject, error) {
	before, err := root.Lstat(name)
	if err != nil || before.Mode()&os.ModeSymlink != 0 || !before.Mode().IsRegular() {
		return openedStrixGitObject{}, fmt.Errorf("unsupported object")
	}
	file, err := root.Open(name)
	if err != nil {
		return openedStrixGitObject{}, err
	}
	opened, openErr := file.Stat()
	after, afterErr := root.Lstat(name)
	if openErr != nil || afterErr != nil || after.Mode()&os.ModeSymlink != 0 || !after.Mode().IsRegular() ||
		!sameStrixFileView(before, opened) || !sameStrixFileView(opened, after) {
		_ = file.Close()
		return openedStrixGitObject{}, fmt.Errorf("object changed while opening")
	}
	return openedStrixGitObject{destination: destination, file: file, info: opened}, nil
}

func sameStrixFileView(a, b os.FileInfo) bool {
	return a != nil && b != nil && os.SameFile(a, b) && a.Mode() == b.Mode() && a.Size() == b.Size() && a.ModTime().Equal(b.ModTime())
}

type strixGitPackProof struct {
	checksum    [sha1.Size]byte
	objectCount uint32
}

func copyOpenedStrixGitObject(ctx context.Context, destination *os.Root, source *openedStrixGitObject, opts strixGitSnapshotOptions) (strixGitPackProof, error) {
	if source == nil || source.file == nil || ctx.Err() != nil {
		return strixGitPackProof{}, fmt.Errorf("invalid opened object")
	}
	copyHash := sha256.New()
	written, err := writeExclusiveStrixSnapshotFile(destination, source.destination, io.TeeReader(source.file, copyHash), source.info.Size(), opts, ctx)
	if err != nil || written != source.info.Size() {
		return strixGitPackProof{}, fmt.Errorf("copy object: %w", err)
	}
	reReadHash, err := hashOpenedStrixFile(ctx, source.file, source.info.Size(), opts.copyBufferBytes)
	if err != nil || !equalStrixDigest(copyHash.Sum(nil), reReadHash[:]) {
		return strixGitPackProof{}, fmt.Errorf("object content changed during copy")
	}
	after, err := source.file.Stat()
	if err != nil || !sameStrixFileView(source.info, after) {
		return strixGitPackProof{}, fmt.Errorf("object changed during copy")
	}
	captured, err := destination.Open(source.destination)
	if err != nil {
		return strixGitPackProof{}, err
	}
	defer captured.Close()
	capturedInfo, err := captured.Stat()
	if err != nil || !capturedInfo.Mode().IsRegular() || capturedInfo.Size() != source.info.Size() {
		return strixGitPackProof{}, fmt.Errorf("invalid captured object")
	}
	capturedHash, err := hashOpenedStrixFile(ctx, captured, capturedInfo.Size(), opts.copyBufferBytes)
	if err != nil || !equalStrixDigest(copyHash.Sum(nil), capturedHash[:]) {
		return strixGitPackProof{}, fmt.Errorf("captured object differs from opened source")
	}
	switch source.kind {
	case strixGitLooseObject:
		if err := validateOpenedStrixLooseObject(ctx, captured, source.identity, opts); err != nil {
			return strixGitPackProof{}, err
		}
		return strixGitPackProof{}, nil
	case strixGitPackFile:
		return validateOpenedStrixPack(ctx, captured, capturedInfo.Size(), opts.copyBufferBytes)
	case strixGitIndexFile:
		return validateOpenedStrixIndex(ctx, captured, capturedInfo.Size(), opts.copyBufferBytes)
	default:
		return strixGitPackProof{}, fmt.Errorf("unsupported object kind")
	}
}

func equalStrixDigest(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	var different byte
	for i := range a {
		different |= a[i] ^ b[i]
	}
	return different == 0
}

func hashOpenedStrixFile(ctx context.Context, file *os.File, exactBytes int64, bufferBytes int) ([sha256.Size]byte, error) {
	var digest [sha256.Size]byte
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return digest, err
	}
	h := sha256.New()
	if err := hashExactStrixBytes(ctx, file, exactBytes, make([]byte, bufferBytes), h); err != nil {
		return digest, err
	}
	copy(digest[:], h.Sum(nil))
	return digest, nil
}

func hashExactStrixBytes(ctx context.Context, reader io.Reader, exactBytes int64, buffer []byte, h hash.Hash) error {
	var read int64
	for read < exactBytes {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		chunk := buffer
		if int64(len(chunk)) > exactBytes-read {
			chunk = chunk[:exactBytes-read]
		}
		n, err := reader.Read(chunk)
		if n > 0 {
			_, _ = h.Write(chunk[:n])
			read += int64(n)
		}
		if err == io.EOF {
			return fmt.Errorf("source shrank during verification")
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrNoProgress
		}
	}
	var probe [1]byte
	n, err := reader.Read(probe[:])
	if n != 0 || (err != nil && err != io.EOF) {
		return fmt.Errorf("source grew during verification")
	}
	return nil
}

func validateOpenedStrixLooseObject(ctx context.Context, file *os.File, expectedOID string, opts strixGitSnapshotOptions) error {
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	zr, err := zlib.NewReader(file)
	if err != nil {
		return err
	}
	defer zr.Close()
	reader := bufio.NewReaderSize(zr, 128)
	header, err := reader.ReadSlice(0)
	if err != nil || len(header) < 3 || len(header) > 96 {
		return fmt.Errorf("invalid loose object header")
	}
	parts := strings.Split(strings.TrimSuffix(string(header), "\x00"), " ")
	if len(parts) != 2 || (parts[0] != "blob" && parts[0] != "tree" && parts[0] != "commit" && parts[0] != "tag") {
		return fmt.Errorf("invalid loose object type")
	}
	declared, err := strconv.ParseInt(parts[1], 10, 64)
	if err != nil || declared < 0 || declared > opts.maxFileBytes {
		return fmt.Errorf("invalid loose object size")
	}
	h := sha1.New()
	_, _ = h.Write(header)
	buffer := make([]byte, opts.copyBufferBytes)
	var bodyBytes int64
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		n, readErr := reader.Read(buffer)
		if n > 0 {
			bodyBytes += int64(n)
			if bodyBytes > declared {
				return fmt.Errorf("loose object exceeds declared size")
			}
			_, _ = h.Write(buffer[:n])
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
		if n == 0 {
			return io.ErrNoProgress
		}
	}
	if bodyBytes != declared || hex.EncodeToString(h.Sum(nil)) != expectedOID {
		return fmt.Errorf("loose object identity mismatch")
	}
	return nil
}

func validateOpenedStrixPack(ctx context.Context, file *os.File, size int64, bufferBytes int) (strixGitPackProof, error) {
	var proof strixGitPackProof
	if size < 12+sha1.Size {
		return proof, fmt.Errorf("short pack")
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return proof, err
	}
	var header [12]byte
	if _, err := io.ReadFull(file, header[:]); err != nil || string(header[:4]) != "PACK" {
		return proof, fmt.Errorf("invalid pack header")
	}
	version := binary.BigEndian.Uint32(header[4:8])
	if version != 2 && version != 3 {
		return proof, fmt.Errorf("unsupported pack version")
	}
	proof.objectCount = binary.BigEndian.Uint32(header[8:12])
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return proof, err
	}
	h := sha1.New()
	if err := hashExactStrixBytes(ctx, io.LimitReader(file, size-sha1.Size), size-sha1.Size, make([]byte, bufferBytes), h); err != nil {
		return proof, err
	}
	if _, err := io.ReadFull(file, proof.checksum[:]); err != nil || !equalStrixDigest(h.Sum(nil), proof.checksum[:]) {
		return proof, fmt.Errorf("pack checksum mismatch")
	}
	return proof, nil
}

func validateOpenedStrixIndex(ctx context.Context, file *os.File, size int64, bufferBytes int) (strixGitPackProof, error) {
	var proof strixGitPackProof
	const indexHeaderBytes = 8 + 256*4
	if size < indexHeaderBytes+2*sha1.Size {
		return proof, fmt.Errorf("short pack index")
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return proof, err
	}
	header := make([]byte, indexHeaderBytes)
	if _, err := io.ReadFull(file, header); err != nil || !equalStrixDigest(header[:4], []byte{0xff, 0x74, 0x4f, 0x63}) || binary.BigEndian.Uint32(header[4:8]) != 2 {
		return proof, fmt.Errorf("unsupported pack index")
	}
	proof.objectCount = binary.BigEndian.Uint32(header[indexHeaderBytes-4:])
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return proof, err
	}
	h := sha1.New()
	if err := hashExactStrixBytes(ctx, io.LimitReader(file, size-sha1.Size), size-sha1.Size, make([]byte, bufferBytes), h); err != nil {
		return proof, err
	}
	var indexChecksum [sha1.Size]byte
	if _, err := io.ReadFull(file, indexChecksum[:]); err != nil || !equalStrixDigest(h.Sum(nil), indexChecksum[:]) {
		return proof, fmt.Errorf("index checksum mismatch")
	}
	if _, err := file.Seek(size-2*sha1.Size, io.SeekStart); err != nil {
		return proof, err
	}
	if _, err := io.ReadFull(file, proof.checksum[:]); err != nil {
		return proof, err
	}
	return proof, nil
}

func writeExclusiveStrixSnapshotFile(root *os.Root, name string, reader io.Reader, exactBytes int64, opts strixGitSnapshotOptions, ctx context.Context) (int64, error) {
	file, err := root.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return 0, err
	}
	keep := false
	defer func() {
		_ = file.Close()
		if !keep {
			_ = root.Remove(name)
		}
	}()
	buffer := make([]byte, opts.copyBufferBytes)
	var written int64
	for {
		if ctx != nil && ctx.Err() != nil {
			return written, ctx.Err()
		}
		remaining := exactBytes - written
		if remaining == 0 {
			var probe [1]byte
			n, readErr := reader.Read(probe[:])
			if n != 0 || (readErr != nil && readErr != io.EOF) {
				return written, fmt.Errorf("source grew during copy")
			}
			break
		}
		chunk := buffer
		if int64(len(chunk)) > remaining {
			chunk = chunk[:remaining]
		}
		n, readErr := reader.Read(chunk)
		if n > 0 {
			wn, writeErr := file.Write(chunk[:n])
			written += int64(wn)
			if writeErr != nil || wn != n {
				return written, fmt.Errorf("write snapshot object")
			}
			if opts.afterCopyChunk != nil {
				if hookErr := opts.afterCopyChunk(name); hookErr != nil {
					return written, hookErr
				}
			}
		}
		if readErr == io.EOF {
			if written != exactBytes {
				return written, fmt.Errorf("source shrank during copy")
			}
			break
		}
		if readErr != nil {
			return written, readErr
		}
		if n == 0 {
			return written, io.ErrNoProgress
		}
	}
	if err := file.Sync(); err != nil {
		return written, err
	}
	if err := file.Close(); err != nil {
		return written, err
	}
	keep = true
	return written, nil
}

func canonicalStrixPackObjectName(name string) (id string, kind uint8, ok bool) {
	if !strings.HasPrefix(name, "pack-") {
		return "", 0, false
	}
	switch {
	case strings.HasSuffix(name, ".pack"):
		id, kind = strings.TrimSuffix(strings.TrimPrefix(name, "pack-"), ".pack"), 1
	case strings.HasSuffix(name, ".idx"):
		id, kind = strings.TrimSuffix(strings.TrimPrefix(name, "pack-"), ".idx"), 2
	default:
		return "", 0, false
	}
	return id, kind, isLowerHex(id, 40)
}

func isStrixSHA256PackName(name string) bool {
	if !strings.HasPrefix(name, "pack-") || (!strings.HasSuffix(name, ".pack") && !strings.HasSuffix(name, ".idx")) {
		return false
	}
	id := strings.TrimPrefix(name, "pack-")
	id = strings.TrimSuffix(strings.TrimSuffix(id, ".pack"), ".idx")
	return isLowerHex(id, 64)
}

func isLowerHex(value string, length int) bool {
	if len(value) != length {
		return false
	}
	for i := 0; i < len(value); i++ {
		if !((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'a' && value[i] <= 'f')) {
			return false
		}
	}
	return true
}
