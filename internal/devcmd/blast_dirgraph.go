package devcmd

// The blast estimator's import graph, built without a subprocess. `go list ./...`
// resolves the whole transitive dependency closure to answer a question that only
// needs each package's own import block, so on a large module it dominates the
// estimate. scanDirGraph reads the same facts straight from the tree with go/build
// and keeps a per-directory cache keyed by the directory's Go-file fingerprint: a
// moved checkout rescans only the directories whose files changed, and an unchanged
// tree is a directory walk plus lookups. goListDirGraph stays as the reference
// implementation and the fallback.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"go/build"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	blastGraphCacheSchema = "fak.blast-dirgraph-cache.v1"
	// A cache file past this size is neither read nor written: the graph of a
	// module with thousands of packages is a few hundred kilobytes, so anything
	// larger is corruption or a tree this cache was not sized for.
	blastGraphCacheMaxBytes = 8 << 20
	// One cache file per repo root; every worktree is its own root, so the
	// directory is pruned to the most recently used files on write.
	blastGraphCacheMaxFiles = 64

	// Directory reads are I/O bound and slow on a busy host, so the walk overlaps
	// them rather than reading one directory at a time.
	blastWalkWorkers = 16

	blastGraphEnvMode  = "FAK_BLAST_GRAPH"
	blastGraphEnvCache = "FAK_BLAST_GRAPH_CACHE"
	blastGraphEnvTrace = "FAK_BLAST_GRAPH_TRACE"
)

// blastScanPackage reads one package directory's module-internal imports. A seam so
// the tests can count which directories a run actually rescanned.
var blastScanPackage = scanPackageImports

type blastGraphCacheFile struct {
	Schema  string                        `json:"schema"`
	Context string                        `json:"context"`
	Dirs    map[string]blastGraphCacheDir `json:"dirs"`
}

type blastGraphCacheDir struct {
	Fingerprint string   `json:"fp"`
	Imports     []string `json:"imports,omitempty"`
	// Unlisted marks a directory whose Go files are all excluded by build
	// constraints: the go command does not list it, so it is not a graph node.
	Unlisted bool `json:"unlisted,omitempty"`
}

// blastGraphStats is what one graph build did, for the trace line and the tests.
type blastGraphStats struct {
	Dirs    int
	Scanned int
	Reused  int
	Wrote   bool
	// Visited counts every directory the walk read, package or not, and Walk is how
	// long that took: on a warm cache the walk is the whole cost of the graph.
	Visited int
	Walk    time.Duration
}

// defaultBlastDirGraph is the estimator's production graph source: the cached scan,
// falling back to `go list` when the tree cannot be scanned (no go.mod to name the
// module, an unreadable root) or when FAK_BLAST_GRAPH=golist asks for the reference.
func defaultBlastDirGraph(root string) (map[string][]string, string, int, error) {
	if strings.EqualFold(strings.TrimSpace(os.Getenv(blastGraphEnvMode)), "golist") {
		return goListDirGraph(root)
	}
	edges, modPath, total, stats, err := scanDirGraph(root, blastGraphCachePath(root))
	if err != nil {
		return goListDirGraph(root)
	}
	if strings.TrimSpace(os.Getenv(blastGraphEnvTrace)) != "" {
		fmt.Fprintf(os.Stderr, "fak blast estimate: import graph %d package dir(s), %d rescanned, %d from cache, cache written=%t; walked %d dir(s) in %s\n",
			stats.Dirs, stats.Scanned, stats.Reused, stats.Wrote, stats.Visited, stats.Walk.Round(time.Millisecond))
	}
	return edges, modPath, total, nil
}

// blastGraphCachePath names the cache file for root, or "" when caching is off.
// FAK_BLAST_GRAPH_CACHE overrides the directory; "off" disables the cache.
func blastGraphCachePath(root string) string {
	dir := strings.TrimSpace(os.Getenv(blastGraphEnvCache))
	if strings.EqualFold(dir, "off") {
		return ""
	}
	if dir == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return ""
		}
		dir = filepath.Join(base, "fak", "blast-dirgraph")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		abs = root
	}
	key := filepath.ToSlash(filepath.Clean(abs))
	if runtime.GOOS == "windows" {
		key = strings.ToLower(key)
	}
	sum := sha256.Sum256([]byte(key))
	return filepath.Join(dir, hex.EncodeToString(sum[:8])+".json")
}

// scanDirGraph builds the repo-relative-directory import graph of the module rooted
// at root, in the same shape parseGoListDirs folds out of `go list -e -json ./...`.
// cachePath "" runs uncached.
func scanDirGraph(root, cachePath string) (edges map[string][]string, modPath string, total int, stats blastGraphStats, err error) {
	modPath, modBytes, err := readModulePath(root)
	if err != nil {
		return nil, "", 0, stats, err
	}
	walkStart := time.Now()
	fingerprints, visited, err := walkPackageDirs(root)
	if err != nil {
		return nil, "", 0, stats, err
	}
	walkElapsed := time.Since(walkStart)
	if len(fingerprints) == 0 {
		return nil, "", 0, stats, errors.New("no Go package directories under " + root)
	}
	ctxt, contextKey := blastBuildContext(modPath, modBytes)

	cached := readBlastGraphCache(cachePath, contextKey)
	dirs := make([]string, 0, len(fingerprints))
	for dir := range fingerprints {
		dirs = append(dirs, dir)
	}
	sort.Strings(dirs)

	imports := make(map[string][]string, len(dirs))
	unlisted := map[string]bool{}
	var stale []string
	for _, dir := range dirs {
		if entry, ok := cached[dir]; ok && entry.Fingerprint == fingerprints[dir] {
			imports[dir] = entry.Imports
			if entry.Unlisted {
				unlisted[dir] = true
			}
			continue
		}
		stale = append(stale, dir)
	}
	uncacheable := map[string]bool{}
	for dir, scan := range scanPackagesConcurrently(&ctxt, root, modPath, stale) {
		imports[dir] = scan.imports
		if !scan.complete {
			uncacheable[dir] = true
		}
		if scan.unlisted {
			unlisted[dir] = true
		}
	}
	listed := 0
	for _, dir := range dirs {
		if !unlisted[dir] {
			listed++
		}
	}
	if listed == 0 {
		return nil, "", 0, stats, errors.New("no Go packages under " + root)
	}

	edges = make(map[string][]string, len(dirs))
	for _, dir := range dirs {
		var deps []string
		for _, imp := range imports[dir] {
			if dep, ok := moduleImportDir(imp, modPath); ok {
				if _, known := fingerprints[dep]; known && !unlisted[dep] {
					deps = append(deps, dep)
				}
			}
		}
		if len(deps) > 0 {
			edges[dir] = deps
		}
	}

	stats = blastGraphStats{Dirs: len(dirs), Scanned: len(stale), Reused: len(dirs) - len(stale), Visited: visited, Walk: walkElapsed}
	if cachePath != "" && (len(stale) > 0 || len(cached) != len(dirs)) {
		next := blastGraphCacheFile{Schema: blastGraphCacheSchema, Context: contextKey, Dirs: make(map[string]blastGraphCacheDir, len(dirs))}
		for _, dir := range dirs {
			if uncacheable[dir] {
				continue
			}
			next.Dirs[dir] = blastGraphCacheDir{Fingerprint: fingerprints[dir], Imports: imports[dir], Unlisted: unlisted[dir]}
		}
		stats.Wrote = writeBlastGraphCache(cachePath, next) == nil
	}
	return edges, modPath, listed, stats, nil
}

// readModulePath returns the module path go.mod declares, plus the file's bytes so
// the cache context can cover every other directive in it.
func readModulePath(root string) (string, []byte, error) {
	data, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		return "", nil, err
	}
	for _, line := range strings.Split(string(data), "\n") {
		if i := strings.Index(line, "//"); i >= 0 {
			line = line[:i]
		}
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "module" {
			if path := strings.Trim(fields[1], "\"`"); path != "" {
				return path, data, nil
			}
		}
	}
	return "", nil, errors.New("go.mod under " + root + " declares no module path")
}

// walkPackageDirs maps every package directory `./...` would match to the
// fingerprint of its Go files, keyed by repo-relative slash path ("." for the root).
// It prunes what the go command prunes: directories whose name starts with "." or
// "_", testdata, vendor, and any subtree that is its own module.
func walkPackageDirs(root string) (map[string]string, int, error) {
	var (
		mu      sync.Mutex
		wg      sync.WaitGroup
		out     = map[string]string{}
		visited int
		rootErr error
		slots   = make(chan struct{}, blastWalkWorkers)
	)
	var visit func(abs, rel string)
	visit = func(abs, rel string) {
		defer wg.Done()
		slots <- struct{}{}
		entries, err := os.ReadDir(abs)
		<-slots
		if err != nil {
			// An unreadable subtree contributes no packages, as with go list -e;
			// an unreadable root is the caller's error.
			if rel == "." {
				mu.Lock()
				rootErr = err
				mu.Unlock()
			}
			return
		}
		if rel != "." {
			for _, entry := range entries {
				if entry.Name() == "go.mod" && !entry.IsDir() {
					return
				}
			}
		}
		var goFiles []string
		for _, entry := range entries {
			name := entry.Name()
			if entry.IsDir() {
				if strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") || name == "testdata" || name == "vendor" {
					continue
				}
				childRel := name
				if rel != "." {
					childRel = rel + "/" + name
				}
				wg.Add(1)
				go visit(filepath.Join(abs, name), childRel)
				continue
			}
			if !strings.HasSuffix(name, ".go") || strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") {
				continue
			}
			info, err := entry.Info()
			if err != nil {
				continue
			}
			goFiles = append(goFiles, name+"\x00"+strconv.FormatInt(info.Size(), 10)+"\x00"+strconv.FormatInt(info.ModTime().UnixNano(), 10))
		}
		fingerprint := ""
		if len(goFiles) > 0 {
			sort.Strings(goFiles)
			sum := sha256.Sum256([]byte(strings.Join(goFiles, "\n")))
			fingerprint = hex.EncodeToString(sum[:16])
		}
		mu.Lock()
		visited++
		if fingerprint != "" {
			out[rel] = fingerprint
		}
		mu.Unlock()
	}
	wg.Add(1)
	go visit(root, ".")
	wg.Wait()
	if rootErr != nil {
		return nil, visited, rootErr
	}
	return out, visited, nil
}

// blastBuildContext returns the build context the scan selects files with and a key
// covering everything outside a package directory that can change its import set:
// the go.mod bytes, the target platform, cgo, extra build tags, and the toolchain
// whose release tags the context carries.
func blastBuildContext(modPath string, modBytes []byte) (build.Context, string) {
	ctxt := build.Default
	ctxt.BuildTags = append([]string(nil), ctxt.BuildTags...)
	ctxt.BuildTags = append(ctxt.BuildTags, goflagsBuildTags(os.Getenv("GOFLAGS"))...)
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00%s\x00%s\x00%s\x00%t\x00%s\x00%s\x00%s\x00%s\x00",
		blastGraphCacheSchema, modPath, ctxt.GOOS, ctxt.GOARCH, ctxt.CgoEnabled, strings.Join(ctxt.BuildTags, ","),
		strings.Join(ctxt.ToolTags, ","), strings.Join(ctxt.ReleaseTags, ","), runtime.Version())
	h.Write(modBytes)
	return ctxt, hex.EncodeToString(h.Sum(nil)[:16])
}

// goflagsBuildTags extracts the -tags value from a GOFLAGS string, so the scan and
// the go command agree on which constrained files are in a package.
func goflagsBuildTags(goflags string) []string {
	fields := strings.Fields(goflags)
	var tags []string
	for i := 0; i < len(fields); i++ {
		flagText := strings.TrimLeft(fields[i], "-")
		value := ""
		switch {
		case strings.HasPrefix(flagText, "tags="):
			value = strings.TrimPrefix(flagText, "tags=")
		case flagText == "tags" && i+1 < len(fields):
			i++
			value = fields[i]
		default:
			continue
		}
		tags = strings.Split(value, ",")
	}
	return tags
}

// scanPackagesConcurrently rescans the stale directories on a small worker pool.
func scanPackagesConcurrently(ctxt *build.Context, root, modPath string, dirs []string) map[string]blastPackageScan {
	out := make(map[string]blastPackageScan, len(dirs))
	if len(dirs) == 0 {
		return out
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > 8 {
		workers = 8
	}
	if workers > len(dirs) {
		workers = len(dirs)
	}
	var (
		mu   sync.Mutex
		wg   sync.WaitGroup
		next = make(chan string)
	)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for dir := range next {
				found := blastScanPackage(ctxt, filepath.Join(root, filepath.FromSlash(dir)), modPath)
				mu.Lock()
				out[dir] = found
				mu.Unlock()
			}
		}()
	}
	for _, dir := range dirs {
		next <- dir
	}
	close(next)
	wg.Wait()
	return out
}

// blastPackageScan is one directory's module-internal imports. complete is false
// when go/build could not read every file, so the result is used for this estimate
// but not cached: a file that was briefly unreadable must not pin a partial import
// set to a fingerprint that is otherwise correct.
type blastPackageScan struct {
	imports  []string
	complete bool
	// unlisted is true when build constraints exclude every Go file in the
	// directory, which keeps it out of ./... entirely.
	unlisted bool
}

// scanPackageImports returns the module-internal import paths of the package in
// absDir, in go list's order: Imports, then TestImports, then XTestImports. A
// directory go/build cannot fully load still yields whatever imports it could read,
// which is what `go list -e` reports for it. Having no buildable files for this
// platform, or mixed package names, is a stable property of the files and counts as
// a complete scan; any other error does not.
func scanPackageImports(ctxt *build.Context, absDir, modPath string) blastPackageScan {
	pkg, err := ctxt.ImportDir(absDir, 0)
	scan := blastPackageScan{complete: true}
	if err != nil {
		var noGo *build.NoGoError
		var mixed *build.MultiplePackageError
		scan.complete = errors.As(err, &noGo) || errors.As(err, &mixed)
	}
	if pkg == nil {
		return scan
	}
	scan.unlisted = hasNoBuildableGoFiles(pkg)
	for _, group := range [][]string{pkg.Imports, pkg.TestImports, pkg.XTestImports} {
		for _, imp := range group {
			if imp == modPath || strings.HasPrefix(imp, modPath+"/") {
				scan.imports = append(scan.imports, imp)
			}
		}
	}
	return scan
}

// hasNoBuildableGoFiles reports whether no Go file in the directory is selected for
// this platform and tag set. The go command leaves such a directory out of ./...
// (a generator script tagged ignore, a package built only for another OS), so it is
// not a graph node here either.
func hasNoBuildableGoFiles(pkg *build.Package) bool {
	return len(pkg.GoFiles)+len(pkg.CgoFiles)+len(pkg.TestGoFiles)+len(pkg.XTestGoFiles)+len(pkg.InvalidGoFiles) == 0
}

// moduleImportDir maps a module-internal import path to its repo-relative directory.
func moduleImportDir(imp, modPath string) (string, bool) {
	if imp == modPath {
		return ".", true
	}
	if rest, ok := strings.CutPrefix(imp, modPath+"/"); ok && rest != "" {
		return rest, true
	}
	return "", false
}

// readBlastGraphCache returns the cached directory entries, or nil when the file is
// absent, oversized, malformed, or was written under a different build context.
func readBlastGraphCache(path, contextKey string) map[string]blastGraphCacheDir {
	if path == "" {
		return nil
	}
	info, err := os.Stat(path)
	if err != nil || info.Size() > blastGraphCacheMaxBytes {
		return nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var file blastGraphCacheFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil
	}
	if file.Schema != blastGraphCacheSchema || file.Context != contextKey {
		return nil
	}
	return file.Dirs
}

// writeBlastGraphCache replaces the cache file atomically and prunes its directory.
// Failure is never fatal to an estimate: the next run simply rescans.
func writeBlastGraphCache(path string, file blastGraphCacheFile) error {
	data, err := json.Marshal(file)
	if err != nil {
		return err
	}
	if len(data) > blastGraphCacheMaxBytes {
		return fmt.Errorf("blast graph cache would be %d bytes, over the %d byte bound", len(data), blastGraphCacheMaxBytes)
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	_, writeErr := tmp.Write(data)
	closeErr := tmp.Close()
	if writeErr == nil {
		writeErr = closeErr
	}
	if writeErr == nil {
		writeErr = os.Rename(tmp.Name(), path)
	}
	if writeErr != nil {
		os.Remove(tmp.Name())
		return writeErr
	}
	pruneBlastGraphCache(dir, blastGraphCacheMaxFiles)
	return nil
}

// isBlastGraphCacheFile reports whether name is a file this cache wrote: the
// 16-hex-digit root key plus ".json", or the temp file of a write that was killed.
// Pruning touches nothing else, so an override directory shared with other files
// is safe.
func isBlastGraphCacheFile(name string) bool {
	const keyLen = 16
	if len(name) < keyLen+len(".json") {
		return false
	}
	for _, r := range name[:keyLen] {
		if !strings.ContainsRune("0123456789abcdef", r) {
			return false
		}
	}
	rest := name[keyLen:]
	return rest == ".json" || strings.HasPrefix(rest, ".json.tmp-")
}

// pruneBlastGraphCache keeps the newest keep cache files in dir and removes the rest.
func pruneBlastGraphCache(dir string, keep int) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	type cacheFile struct {
		name    string
		modTime int64
	}
	var files []cacheFile
	for _, entry := range entries {
		if entry.IsDir() || !isBlastGraphCacheFile(entry.Name()) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		files = append(files, cacheFile{name: entry.Name(), modTime: info.ModTime().UnixNano()})
	}
	if len(files) <= keep {
		return
	}
	sort.Slice(files, func(i, j int) bool {
		if files[i].modTime != files[j].modTime {
			return files[i].modTime > files[j].modTime
		}
		return files[i].name < files[j].name
	})
	for _, file := range files[keep:] {
		os.Remove(filepath.Join(dir, file.name))
	}
}
