package codesearch

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/trigram"
)

// Result represents a match in a document, carrying the document ID, relative path,
// and 1-based line numbers where the match occurs. It is aliased to trigram.Result.
type Result = trigram.Result

// WorkspaceIndex is a shared, in-memory trigram index over workspace files designed
// for concurrent worker code search. Postings and document contents reside in memory
// so candidate filtering and line verification occur without iterating disk directories
// or issuing redundant file reads.
type WorkspaceIndex struct {
	root        string
	mu          sync.RWMutex
	buildOnce   sync.Once
	built       atomic.Bool
	index       atomic.Pointer[trigram.Index]
	fileReads   atomic.Int64
	docCount    atomic.Int32
	memoryBytes atomic.Int64

	readFile func(path string) ([]byte, error)
	filter   func(path string) bool

	docsMu sync.RWMutex
	docs   map[string]string // path -> content
}

// IndexOption configures a WorkspaceIndex.
type IndexOption func(*WorkspaceIndex)

// WithReadFile sets a custom file reader for disk scanning.
func WithReadFile(fn func(string) ([]byte, error)) IndexOption {
	return func(w *WorkspaceIndex) {
		w.readFile = fn
	}
}

// WithFilter sets a path filter for disk scanning (returning true to include).
func WithFilter(fn func(string) bool) IndexOption {
	return func(w *WorkspaceIndex) {
		w.filter = fn
	}
}

// WithDocuments initializes the index with pre-loaded in-memory documents.
func WithDocuments(docs map[string]string) IndexOption {
	return func(w *WorkspaceIndex) {
		w.docsMu.Lock()
		defer w.docsMu.Unlock()
		for k, v := range docs {
			w.docs[k] = v
		}
	}
}

var (
	sharedMu      sync.RWMutex
	sharedIndices = make(map[string]*WorkspaceIndex)
)

// SharedWorkspaceIndex returns the shared in-memory WorkspaceIndex for root,
// caching the index across concurrent workers.
func SharedWorkspaceIndex(root string, opts ...IndexOption) *WorkspaceIndex {
	clean := filepath.Clean(root)
	sharedMu.RLock()
	w, ok := sharedIndices[clean]
	sharedMu.RUnlock()
	if ok {
		return w
	}

	sharedMu.Lock()
	defer sharedMu.Unlock()
	if w, ok = sharedIndices[clean]; ok {
		return w
	}
	w = NewWorkspaceIndex(clean, opts...)
	sharedIndices[clean] = w
	return w
}

// GetWorkspaceIndex is an alias for SharedWorkspaceIndex.
func GetWorkspaceIndex(root string) *WorkspaceIndex {
	return SharedWorkspaceIndex(root)
}

// ResetSharedIndices clears the shared workspace cache (useful for tests).
func ResetSharedIndices() {
	sharedMu.Lock()
	defer sharedMu.Unlock()
	sharedIndices = make(map[string]*WorkspaceIndex)
}

// NewWorkspaceIndex creates a new WorkspaceIndex for root with optional configuration.
func NewWorkspaceIndex(root string, opts ...IndexOption) *WorkspaceIndex {
	w := &WorkspaceIndex{
		root:     filepath.Clean(root),
		readFile: os.ReadFile,
		docs:     make(map[string]string),
	}
	for _, opt := range opts {
		opt(w)
	}
	return w
}

// NewWorkspaceIndexFromDocs creates and immediately builds a WorkspaceIndex from
// an in-memory document map without disk I/O.
func NewWorkspaceIndexFromDocs(docs map[string]string) *WorkspaceIndex {
	w := NewWorkspaceIndex(".", WithDocuments(docs))
	_ = w.Build()
	return w
}

// EnsureBuilt guarantees that the workspace index is built into memory.
// It executes building lazily on first access and is safe for concurrent callers.
func (w *WorkspaceIndex) EnsureBuilt() error {
	if w.built.Load() {
		return nil
	}
	var err error
	w.buildOnce.Do(func() {
		err = w.Build()
	})
	return err
}

// Build scans the workspace files (or loads initialized documents), builds the
// in-memory trigram index, compacts postings to minimize heap footprint, and
// publishes the updated index atomically.
func (w *WorkspaceIndex) Build() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	ix := &trigram.Index{}

	w.docsMu.RLock()
	hasDocs := len(w.docs) > 0
	if hasDocs {
		paths := make([]string, 0, len(w.docs))
		for p := range w.docs {
			paths = append(paths, p)
		}
		sort.Strings(paths)
		for _, p := range paths {
			ix.Add(p, p, w.docs[p])
		}
		w.docsMu.RUnlock()
	} else {
		w.docsMu.RUnlock()
		if err := w.scanDisk(ix); err != nil {
			return err
		}
	}

	ix.Compact()
	w.index.Store(ix)
	w.docCount.Store(int32(ix.DocCount()))
	w.memoryBytes.Store(ix.SizeBytes())
	w.built.Store(true)
	return nil
}

func (w *WorkspaceIndex) scanDisk(ix *trigram.Index) error {
	if _, err := os.Stat(w.root); err != nil {
		return nil
	}

	var files []string
	contents := make(map[string]string)

	err := filepath.WalkDir(w.root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			if p != w.root && (strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") || name == "vendor" || name == "node_modules") {
				return filepath.SkipDir
			}
			return nil
		}
		if w.filter != nil {
			if !w.filter(p) {
				return nil
			}
		} else if !strings.HasSuffix(p, ".go") {
			return nil
		}

		w.fileReads.Add(1)
		b, err := w.readFile(p)
		if err != nil {
			return nil
		}
		sp := filepath.ToSlash(p)
		files = append(files, sp)
		contents[sp] = string(b)
		return nil
	})
	if err != nil {
		return err
	}

	sort.Strings(files)
	w.docsMu.Lock()
	for _, f := range files {
		w.docs[f] = contents[f]
		ix.Add(f, f, contents[f])
	}
	w.docsMu.Unlock()
	return nil
}

// UpdateFile adds or updates a file in the workspace index and atomically refreshes postings.
func (w *WorkspaceIndex) UpdateFile(path, content string) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.docsMu.Lock()
	w.docs[path] = content
	paths := make([]string, 0, len(w.docs))
	for p := range w.docs {
		paths = append(paths, p)
	}
	sort.Strings(paths)

	ix := &trigram.Index{}
	for _, p := range paths {
		ix.Add(p, p, w.docs[p])
	}
	w.docsMu.Unlock()

	ix.Compact()
	w.index.Store(ix)
	w.docCount.Store(int32(ix.DocCount()))
	w.memoryBytes.Store(ix.SizeBytes())
	w.built.Store(true)
}

// LoadDocuments replaces or loads documents into memory and rebuilds the index.
func (w *WorkspaceIndex) LoadDocuments(docs map[string]string) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.docsMu.Lock()
	for p, c := range docs {
		w.docs[p] = c
	}
	paths := make([]string, 0, len(w.docs))
	for p := range w.docs {
		paths = append(paths, p)
	}
	sort.Strings(paths)

	ix := &trigram.Index{}
	for _, p := range paths {
		ix.Add(p, p, w.docs[p])
	}
	w.docsMu.Unlock()

	ix.Compact()
	w.index.Store(ix)
	w.docCount.Store(int32(ix.DocCount()))
	w.memoryBytes.Store(ix.SizeBytes())
	w.built.Store(true)
	return nil
}

// Search performs a thread-safe substring search across indexed workspace files.
// Candidate filtering and exact line matching happen strictly in memory with 0 disk I/O.
func (w *WorkspaceIndex) Search(query string) []Result {
	if err := w.EnsureBuilt(); err != nil {
		return nil
	}
	ix := w.index.Load()
	if ix == nil {
		return nil
	}
	return ix.Search(query)
}

// SearchRegexp performs a thread-safe regex search across indexed workspace files.
// Required-literal candidate narrowing and regex verification happen strictly in memory with 0 disk I/O.
func (w *WorkspaceIndex) SearchRegexp(pattern string) ([]Result, error) {
	if err := w.EnsureBuilt(); err != nil {
		return nil, err
	}
	ix := w.index.Load()
	if ix == nil {
		if _, err := regexp.Compile(pattern); err != nil {
			return nil, err
		}
		return nil, nil
	}
	return ix.SearchRegexp(pattern)
}

// FileReads returns the total count of file read operations performed by this index.
func (w *WorkspaceIndex) FileReads() int64 {
	return w.fileReads.Load()
}

// DocCount returns the number of documents indexed.
func (w *WorkspaceIndex) DocCount() int {
	return int(w.docCount.Load())
}

// MemoryBytes returns the estimated heap memory footprint of the index postings and documents.
func (w *WorkspaceIndex) MemoryBytes() int64 {
	return w.memoryBytes.Load()
}

// Root returns the root path associated with this workspace index.
func (w *WorkspaceIndex) Root() string {
	return w.root
}
