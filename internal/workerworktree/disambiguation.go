package workerworktree

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog"
	"github.com/anthony-chaudhary/fak/internal/gitbroker"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	// DisambiguationTimeoutEnv is the explicit bootstrap-only recovery knob for a
	// cold whole-tree oracle. It changes only the shared deadline around the same
	// three witnesses; it never skips, retries, or weakens one.
	DisambiguationTimeoutEnv = "FAK_WORKTREE_DISAMBIGUATION_TIMEOUT_MS"

	// DisambiguationTimeoutCode is the stable machine-readable diagnosis for a
	// whole-tree witness that exceeded the single pre-CAS deadline.
	DisambiguationTimeoutCode = "DISAMBIGUATION_TIMEOUT"

	// One deadline covers all three whole-tree witnesses together. The work is
	// intentionally bounded well below the 20+ minute stalls seen in #9579 while
	// leaving ample room for the normal full-corpus scan on a busy workstation.
	defaultDisambiguationTimeout = 2 * time.Minute
	maxDisambiguationTimeout     = 15 * time.Minute
)

const (
	disambiguationRecoveryDefault  = "default"
	disambiguationRecoveryExplicit = "explicit_bounded_override"
	disambiguationRecoveryInvalid  = "invalid_explicit_override"
)

// DisambiguationDiagnostic identifies the exact witness and expensive subphase
// that stopped a land. It is nested in the existing witness receipt so callers
// gain typed detail without weakening or bypassing any invariant.
type DisambiguationDiagnostic struct {
	Code      string `json:"code"`
	Witness   string `json:"witness"`
	Subphase  string `json:"subphase"`
	TimeoutMS int64  `json:"timeout_ms,omitempty"`
}

// DisambiguationTimeoutReceipt makes the deadline authority explicit on every
// whole-tree witness result. RequestedTimeoutMS is null on the ordinary path
// and when a malformed request cannot be safely represented as an integer.
type DisambiguationTimeoutReceipt struct {
	DefaultTimeoutMS   int64  `json:"default_timeout_ms"`
	RequestedTimeoutMS *int64 `json:"requested_timeout_ms"`
	EffectiveTimeoutMS int64  `json:"effective_timeout_ms"`
	RecoveryMode       string `json:"recovery_mode"`
}

// DisambiguationWitness is one independently materialized view used by land.
type DisambiguationWitness struct {
	Tree           string                    `json:"tree"`
	Fresh          bool                      `json:"fresh"`
	SemanticValid  bool                      `json:"semantic_valid"`
	CriticalClean  bool                      `json:"critical_clean"`
	ClarityDebt    int                       `json:"clarity_debt"`
	Coverage       float64                   `json:"coverage"`
	CoverageDebt   int                       `json:"coverage_debt"`
	FamilyCoverage map[string]float64        `json:"family_coverage,omitempty"`
	Detail         string                    `json:"detail,omitempty"`
	Diagnostic     *DisambiguationDiagnostic `json:"diagnostic,omitempty"`
	CacheIdentity  string                    `json:"cache_identity,omitempty"`
	CacheState     string                    `json:"cache_state,omitempty"`
	CacheReason    string                    `json:"cache_reason,omitempty"`
}
type DisambiguationWitnesses struct {
	Timeout   DisambiguationTimeoutReceipt `json:"timeout"`
	Before    DisambiguationWitness        `json:"before"`
	Worktree  DisambiguationWitness        `json:"worktree"`
	PostApply DisambiguationWitness        `json:"post_apply"`
}

func disambiguationRelevant(paths []string) bool {
	for _, p := range paths {
		p = filepath.ToSlash(p)
		if conceptcatalog.RelevantPath(p) {
			return true
		}
		if strings.HasPrefix(p, "internal/") || strings.HasPrefix(p, "cmd/") || strings.HasPrefix(p, "docs/") || strings.HasPrefix(p, "tools/") {
			return strings.HasSuffix(p, ".go") || strings.HasSuffix(p, ".md") || strings.HasSuffix(p, ".json")
		}

	}
	return false
}

type boundedReader func(context.Context, string, string, func(string)) DisambiguationWitness

type disambiguationArchiveStream struct {
	Reader io.ReadCloser
	Wait   func() error
}

var (
	readDisambiguation = boundedReader(readDisambiguationWitness)

	newDeadline = func(timeout time.Duration) (context.Context, context.CancelFunc) {
		return context.WithTimeout(context.Background(), timeout)
	}

	// These narrow seams keep tree resolution, enumeration, and object reads
	// cancellable and let tests prove that irrelevant blobs are never fetched.
	resolveDisambiguationTree = func(ctx context.Context, repo, tree string) (string, error) {
		cmd := exec.CommandContext(ctx, "git", "rev-parse", tree+"^{tree}")
		cmd.Dir = repo
		windowgate.ConfigureBackgroundCommand(cmd)
		out, err := cmd.Output()
		return strings.TrimSpace(string(out)), err
	}
	listDisambiguationTree = func(ctx context.Context, repo, tree string) ([]byte, error) {
		cmd := exec.CommandContext(ctx, "git", "ls-tree", "-r", "-z", "-l", tree)
		cmd.Dir = repo
		windowgate.ConfigureBackgroundCommand(cmd)
		return cmd.Output()
	}
	readDisambiguationObject = func(ctx context.Context, repo, objectID string) ([]byte, error) {
		info, data, err := gitbroker.Shared().ObjectBytes(ctx, repo, objectID)
		if err != nil {
			return nil, err
		}
		if info.Type != "blob" {
			return nil, fmt.Errorf("disambiguation object %s is %s, want blob", objectID, info.Type)
		}
		return data, nil
	}
	runDisambiguationArchive = func(ctx context.Context, repo, tree string) (disambiguationArchiveStream, error) {
		cmd := exec.CommandContext(ctx, "git", "archive", "--format=tar", tree)
		cmd.Dir = repo
		windowgate.ConfigureBackgroundCommand(cmd)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			return disambiguationArchiveStream{}, err
		}
		if err := cmd.Start(); err != nil {
			_ = stdout.Close()
			return disambiguationArchiveStream{}, err
		}
		return disambiguationArchiveStream{
			Reader: stdout,
			Wait: func() error {
				err := cmd.Wait()
				if err != nil && stderr.Len() > 0 {
					return fmt.Errorf("%w: %s", err, strings.TrimSpace(stderr.String()))
				}
				return err
			},
		}, nil
	}
	runAnalyzer = func(ctx context.Context, root, generated string) ([]byte, error) {
		python := "python3"
		if runtime.GOOS == "windows" {
			python = "python"
		}
		cmd := exec.CommandContext(ctx, python,
			filepath.Join(root, "tools", "concept_disambiguation_scorecard.py"),
			"--workspace", root,
			"--json",
			"--markdown-dir", generated,
		)
		cmd.Dir = root
		windowgate.ConfigureBackgroundCommand(cmd)
		return cmd.Output()
	}
	disambiguationCacheSchema     = "disambiguation-witness-v2"
	disambiguationAnalyzerConfig  = "json+markdown;catalog-invariant-v1"
	disambiguationAnalyzerVersion = "concept-disambiguation-scorecard-v1"
	disambiguationCacheRoot       = func(repo string) (string, error) {
		cmd := exec.Command("git", "rev-parse", "--git-common-dir")
		cmd.Dir = repo
		windowgate.ConfigureBackgroundCommand(cmd)
		out, err := cmd.Output()
		if err != nil {
			return "", err
		}
		common := strings.TrimSpace(string(out))
		if !filepath.IsAbs(common) {
			common = filepath.Join(repo, common)
		}
		return filepath.Join(filepath.Clean(common), "fak-cache", "disambiguation"), nil
	}
)

func verifyAppliedDisambiguation(root, wtPath, treeSHA string) (*DisambiguationWitnesses, bool) {
	timeoutReceipt, timeout, err := resolveDisambiguationTimeout(os.LookupEnv)
	if err != nil {
		all := newDisambiguationWitnesses(treeSHA, timeoutReceipt)
		all.Before.Diagnostic = &DisambiguationDiagnostic{
			Code: DisambiguationTimeoutCode, Witness: "configuration", Subphase: "timeout-config",
		}
		all.Before.Detail = fmt.Sprintf("%s: witness=configuration subphase=timeout-config: %v", DisambiguationTimeoutCode, err)
		return all, false
	}
	ctx, cancel := newDeadline(timeout)
	defer cancel()
	return verifyWithinDeadline(ctx, root, wtPath, treeSHA, timeoutReceipt)
}

func resolveDisambiguationTimeout(lookup func(string) (string, bool)) (DisambiguationTimeoutReceipt, time.Duration, error) {
	receipt := DisambiguationTimeoutReceipt{
		DefaultTimeoutMS:   defaultDisambiguationTimeout.Milliseconds(),
		EffectiveTimeoutMS: defaultDisambiguationTimeout.Milliseconds(),
		RecoveryMode:       disambiguationRecoveryDefault,
	}
	raw, present := lookup(DisambiguationTimeoutEnv)
	raw = strings.TrimSpace(raw)
	if !present {
		return receipt, defaultDisambiguationTimeout, nil
	}
	requestedMS, err := strconv.ParseInt(raw, 10, 64)
	if err == nil {
		receipt.RequestedTimeoutMS = &requestedMS
	}
	if err != nil || requestedMS < 1 || requestedMS > maxDisambiguationTimeout.Milliseconds() {
		receipt.EffectiveTimeoutMS = 0
		receipt.RecoveryMode = disambiguationRecoveryInvalid
		return receipt, 0, fmt.Errorf("%s must be an integer in [1,%d]", DisambiguationTimeoutEnv, maxDisambiguationTimeout.Milliseconds())
	}
	receipt.EffectiveTimeoutMS = requestedMS
	receipt.RecoveryMode = disambiguationRecoveryExplicit
	return receipt, time.Duration(requestedMS) * time.Millisecond, nil
}

func newDisambiguationWitnesses(treeSHA string, timeout DisambiguationTimeoutReceipt) *DisambiguationWitnesses {
	return &DisambiguationWitnesses{
		Timeout:   timeout,
		Before:    DisambiguationWitness{Tree: "HEAD"},
		Worktree:  DisambiguationWitness{Tree: "HEAD"},
		PostApply: DisambiguationWitness{Tree: treeSHA},
	}
}

// verifyWithinDeadline keeps the entire three-witness sweep under
// one deadline. landIsolated calls this only after write-tree populated a
// throwaway GIT_INDEX_FILE and before commit-tree, recovery publication, or the
// trunk CAS, so a timeout cannot move HEAD or touch the shared index by
// construction.
func verifyWithinDeadline(ctx context.Context, root, wtPath, treeSHA string, timeoutReceipt DisambiguationTimeoutReceipt) (*DisambiguationWitnesses, bool) {
	all := newDisambiguationWitnesses(treeSHA, timeoutReceipt)
	timeout := time.Duration(timeoutReceipt.EffectiveTimeoutMS) * time.Millisecond
	var complete bool
	all.Before, complete = readOneBounded(ctx, "before", root, "HEAD", timeout)
	if !complete {
		return all, false
	}
	all.Worktree, complete = readOneBounded(ctx, "worktree", wtPath, "HEAD", timeout)
	if !complete {
		return all, false
	}
	all.PostApply, complete = readOneBounded(ctx, "post_apply", root, treeSHA, timeout)
	if !complete {
		return all, false
	}
	before, post := all.Before, all.PostApply
	// Freshness is a NON-REGRESSION check, mirroring the coverage terms beside it: a land must
	// never turn a fresh HEAD stale, but it is not required to repair a peer's pre-existing
	// staleness. Requiring absolute post.Fresh over-refused every isolated land whenever an
	// unrelated doc-regen left the tree stale at HEAD (before.Fresh already false), even for a
	// diff that regressed nothing. See #5359.
	freshNonRegress := post.Fresh || !before.Fresh
	// Clarity is also a non-regression gate. A clean HEAD must remain clean, while a HEAD
	// with pre-existing clarity debt may land an unrelated change provided the candidate
	// does not increase that debt. Semantic validity remains an absolute requirement.
	clarityNonRegress := post.CriticalClean || (!before.CriticalClean && post.ClarityDebt <= before.ClarityDebt)
	ok := freshNonRegress && post.SemanticValid && clarityNonRegress && post.Coverage+0.0001 >= before.Coverage && post.CoverageDebt <= before.CoverageDebt && !coverageFamilyRegressed(before.FamilyCoverage, post.FamilyCoverage)
	if !ok && post.Detail == "" {
		all.PostApply.Detail = fmt.Sprintf("clarity debt %d -> %d; coverage %.2f -> %.2f; coverage debt %d -> %d", before.ClarityDebt, post.ClarityDebt, before.Coverage, post.Coverage, before.CoverageDebt, post.CoverageDebt)
	}
	return all, ok
}

// readOneBounded bounds even a non-command subphase such as
// the whole-corpus invariant scan. The worker owns its scratch directory until
// it eventually returns; the land caller may stop waiting immediately at the
// deadline without racing cleanup or touching trunk state.
func readOneBounded(ctx context.Context, witness, repo, tree string, timeout time.Duration) (DisambiguationWitness, bool) {
	type result struct {
		witness DisambiguationWitness
	}
	resultCh := make(chan result, 1)
	subphase := "witness-start"
	var subphaseMu sync.Mutex
	setSubphase := func(next string) {
		subphaseMu.Lock()
		subphase = next
		subphaseMu.Unlock()
	}
	currentSubphase := func() string {
		subphaseMu.Lock()
		defer subphaseMu.Unlock()
		return subphase
	}

	go func() {
		resultCh <- result{witness: readDisambiguation(ctx, repo, tree, setSubphase)}
	}()

	select {
	case got := <-resultCh:
		if err := ctx.Err(); err != nil {
			return timeoutResult(tree, witness, currentSubphase(), timeout, err), false
		}
		return got.witness, true
	case <-ctx.Done():
		return timeoutResult(tree, witness, currentSubphase(), timeout, ctx.Err()), false
	}
}

func timeoutResult(tree, witness, subphase string, timeout time.Duration, err error) DisambiguationWitness {
	timeoutMS := timeout.Milliseconds()
	detail := fmt.Sprintf("%s: witness=%s subphase=%s timeout_ms=%d", DisambiguationTimeoutCode, witness, subphase, timeoutMS)
	if err != nil {
		detail += ": " + err.Error()
	}
	return DisambiguationWitness{
		Tree:   tree,
		Detail: detail,
		Diagnostic: &DisambiguationDiagnostic{
			Code: DisambiguationTimeoutCode, Witness: witness, Subphase: subphase, TimeoutMS: timeoutMS,
		},
	}
}

func coverageFamilyRegressed(before, after map[string]float64) bool {
	for family, b := range before {
		if a, ok := after[family]; ok && a+0.0001 < b {
			return true
		}
	}
	return false
}

func readDisambiguationWitness(ctx context.Context, repo, tree string, setSubphase func(string)) DisambiguationWitness {
	w := DisambiguationWitness{Tree: tree}
	setSubphase("git-tree")
	treeID, err := resolveDisambiguationTree(ctx, repo, tree)
	if err != nil {
		w.Detail = err.Error()
		return w
	}

	key := disambiguationCacheKey(treeID)
	w.CacheIdentity = key
	cacheRoot, cacheErr := disambiguationCacheRoot(repo)
	if cacheErr != nil {
		w.CacheState, w.CacheReason = "bypass", "git-common-dir-unavailable"
		result, _ := computeDisambiguationWitness(ctx, repo, treeID, w, setSubphase)
		return result
	}
	if err := os.MkdirAll(cacheRoot, 0755); err != nil {
		w.CacheState, w.CacheReason = "bypass", "cache-root-unavailable"
		result, _ := computeDisambiguationWitness(ctx, repo, treeID, w, setSubphase)
		return result
	}

	lock := filepath.Join(cacheRoot, key+".lock")
	setSubphase("cache-lock")
	for {
		if err := os.Mkdir(lock, 0700); err == nil {
			break
		} else if !errors.Is(err, os.ErrExist) {
			w.CacheState, w.CacheReason = "bypass", "cache-lock-unavailable"
			result, _ := computeDisambiguationWitness(ctx, repo, treeID, w, setSubphase)
			return result
		}
		select {
		case <-ctx.Done():
			w.Detail = ctx.Err().Error()
			return w
		case <-time.After(10 * time.Millisecond):
		}
	}
	defer os.Remove(lock)

	cachePath := filepath.Join(cacheRoot, key+".json")
	setSubphase("cache-read")
	if cached, reason, ok := readCachedDisambiguation(cachePath, key); ok {
		cached.Tree = tree
		cached.CacheIdentity, cached.CacheState, cached.CacheReason = key, "hit", reason
		return cached
	} else {
		w.CacheState, w.CacheReason = "miss", reason
	}

	w, complete := computeDisambiguationWitness(ctx, repo, treeID, w, setSubphase)
	if complete {
		setSubphase("cache-write")
		if err := writeCachedDisambiguation(cachePath, key, w); err != nil {
			w.CacheReason = "write-error: " + err.Error()
		}
	}
	return w
}

func materializeDisambiguationArchive(ctx context.Context, repo, tree, dst string, setSubphase func(string)) (string, error) {
	setSubphase("git-archive")
	stream, err := runDisambiguationArchive(ctx, repo, tree)
	if err != nil {
		return "", err
	}
	if stream.Reader == nil || stream.Wait == nil {
		if stream.Reader != nil {
			_ = stream.Reader.Close()
		}
		if stream.Wait != nil {
			_ = stream.Wait()
		}
		return "", errors.New("git archive returned an incomplete stream")
	}

	h := newDisambiguationArchiveHash()
	tee := io.TeeReader(stream.Reader, h)
	setSubphase("extract-archive")
	extractErr := extractTarReaderBounded(ctx, tee, dst)

	var drainErr error
	if extractErr == nil {
		// tar.Reader stops at the end marker, but #9461 keys the cache over every
		// byte emitted by git archive. Drain padding and trailing bytes through
		// the same hash to preserve the existing content-addressed identity.
		setSubphase("git-archive")
		_, drainErr = copyBounded(ctx, io.Discard, tee)
	}
	closeErr := stream.Reader.Close()
	waitErr := stream.Wait()
	if extractErr != nil {
		// Preserve the failing subphase even though joining the producer is
		// mandatory before returning.
		setSubphase("extract-archive")
	}
	if err := errors.Join(extractErr, drainErr, closeErr, waitErr); err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

type disambiguationCacheEntry struct {
	Schema  string                `json:"schema"`
	Key     string                `json:"key"`
	Witness DisambiguationWitness `json:"witness"`
}

func disambiguationArchiveCacheKey(archive []byte) string {
	h := newDisambiguationArchiveHash()
	h.Write(archive)
	return fmt.Sprintf("%x", h.Sum(nil))
}

func newDisambiguationArchiveHash() hash.Hash {
	h := sha256.New()
	io.WriteString(h, disambiguationCacheSchema)
	h.Write([]byte{0})
	io.WriteString(h, disambiguationAnalyzerConfig)
	h.Write([]byte{0})
	io.WriteString(h, disambiguationAnalyzerVersion)
	h.Write([]byte{0})
	return h
}

func disambiguationCacheKey(treeID string) string {
	h := sha256.New()
	io.WriteString(h, disambiguationCacheSchema)
	h.Write([]byte{0})
	io.WriteString(h, disambiguationAnalyzerConfig)
	h.Write([]byte{0})
	io.WriteString(h, disambiguationAnalyzerVersion)
	h.Write([]byte{0})
	io.WriteString(h, treeID)
	return fmt.Sprintf("%x", h.Sum(nil))
}

func readCachedDisambiguation(path, key string) (DisambiguationWitness, string, bool) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return DisambiguationWitness{}, "not-found", false
	}
	if err != nil {
		return DisambiguationWitness{}, "read-error", false
	}
	var entry disambiguationCacheEntry
	if json.Unmarshal(data, &entry) != nil {
		return DisambiguationWitness{}, "corrupt", false
	}
	if entry.Schema != disambiguationCacheSchema || entry.Key != key {
		return DisambiguationWitness{}, "mismatch", false
	}
	return entry.Witness, "content-addressed", true
}

func writeCachedDisambiguation(path, key string, witness DisambiguationWitness) error {
	witness.Tree, witness.CacheIdentity, witness.CacheState, witness.CacheReason = "", "", "", ""
	data, err := json.Marshal(disambiguationCacheEntry{Schema: disambiguationCacheSchema, Key: key, Witness: witness})
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), key+"-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = tmp.Write(data); err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Rename(name, path)
}

type disambiguationTreeEntry struct {
	Mode     string
	ObjectID string
	Size     int64
	Path     string
}

var disambiguationSkipDir = map[string]bool{
	".git": true, ".cache": true, "node_modules": true, "testdata": true,
	"_registry": true, "__pycache__": true, ".pytest_cache": true,
	".ruff_cache": true, "vendor": true, ".dispatch-runs": true,
	".goal-runs": true,
}

const disambiguationMaxCorpusBytes = 512 * 1024

func computeDisambiguationWitness(ctx context.Context, repo, treeID string, w DisambiguationWitness, setSubphase func(string)) (DisambiguationWitness, bool) {
	setSubphase("scratch-tree")
	tmp, err := os.MkdirTemp("", "fak-disambiguation-tree-*")
	if err != nil {
		w.Detail = err.Error()
		return w, false
	}
	defer os.RemoveAll(tmp)
	out := filepath.Join(tmp, "tree")
	// Keep the stable subphase name used by timeout diagnostics. The old
	// implementation extracted every tracked blob from a whole-tree archive;
	// this now projects only the scorecard's exact corpus from Git objects.
	setSubphase("extract-archive")
	if err = materializeDisambiguationTree(ctx, repo, treeID, out); err != nil {
		w.Detail = fmt.Sprintf("materialize candidate tree: %v", err)
		return w, false
	}
	setSubphase("concept-invariant")
	inv, err := checkInvariantBounded(ctx, out, setSubphase)
	if err != nil {
		w.Detail = err.Error()
		return w, false
	}
	w.Fresh = inv.Freshness.Fresh
	w.SemanticValid = inv.SemanticValid
	w.CriticalClean = inv.CriticalClean
	w.ClarityDebt = inv.ClarityDebt
	w.Coverage = inv.Coverage
	w.CoverageDebt = inv.CoverageDebt
	w.FamilyCoverage = inv.FamilyCoverage
	w.Detail = inv.Detail
	return w, true
}

func materializeDisambiguationTree(ctx context.Context, repo, treeID, dst string) error {
	listing, err := listDisambiguationTree(ctx, repo, treeID)
	if err != nil {
		return err
	}
	entries, err := parseDisambiguationTree(listing)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dst, 0755); err != nil {
		return err
	}
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !disambiguationCorpusPath(entry.Path, entry.Size) {
			continue
		}
		body, err := readDisambiguationObject(ctx, repo, entry.ObjectID)
		if err != nil {
			return fmt.Errorf("read %s: %w", entry.Path, err)
		}
		if int64(len(body)) != entry.Size {
			return fmt.Errorf("read %s: blob size %d, want %d", entry.Path, len(body), entry.Size)
		}
		target, err := disambiguationTarget(dst, entry.Path)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return err
		}
		mode := os.FileMode(0644)
		if entry.Mode == "100755" {
			mode = 0755
		}
		if err := os.WriteFile(target, body, mode); err != nil {
			return err
		}
	}
	return nil
}

func parseDisambiguationTree(listing []byte) ([]disambiguationTreeEntry, error) {
	records := bytes.Split(listing, []byte{0})
	entries := make([]disambiguationTreeEntry, 0, len(records))
	for _, record := range records {
		if len(record) == 0 {
			continue
		}
		parts := bytes.SplitN(record, []byte{'\t'}, 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("malformed git ls-tree record %q", record)
		}
		meta := strings.Fields(string(parts[0]))
		if len(meta) != 4 || meta[1] != "blob" {
			continue
		}
		size, err := strconv.ParseInt(meta[3], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parse git ls-tree size %q: %w", meta[3], err)
		}
		path := string(parts[1])
		if _, err := disambiguationTarget(".", path); err != nil {
			return nil, err
		}
		entries = append(entries, disambiguationTreeEntry{Mode: meta[0], ObjectID: meta[2], Size: size, Path: path})
	}
	return entries, nil
}

func disambiguationCorpusPath(path string, size int64) bool {
	path = filepath.ToSlash(path)
	freshness := path == filepath.ToSlash(conceptcatalog.GeneratedReadme) || path == filepath.ToSlash(conceptcatalog.GeneratedIndex)
	if freshness {
		return true
	}
	if size > disambiguationMaxCorpusBytes {
		return false
	}
	parts := strings.Split(path, "/")
	for _, part := range parts {
		if disambiguationSkipDir[part] {
			return false
		}
	}
	if path == "tools/concept_disambiguation_scorecard.py" {
		return true
	}
	if strings.HasPrefix(path, "tools/concept_disambiguation_scorecard.data/") && strings.HasSuffix(path, ".json") {
		return true
	}
	if strings.HasPrefix(path, "docs/") && strings.HasSuffix(path, ".md") {
		return true
	}
	switch path {
	case "CLAIMS.md", "ARCHITECTURE.md", "README.md", "AGENTS.md", "INDEX.md", "GLOSSARY.md":
		return true
	}
	return (strings.HasPrefix(path, "internal/") || strings.HasPrefix(path, "cmd/")) &&
		strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go")
}

func disambiguationTarget(root, path string) (string, error) {
	clean := filepath.Clean(filepath.FromSlash(path))
	if filepath.IsAbs(clean) || clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("unsafe tree path %q", path)
	}
	return filepath.Join(root, clean), nil
}

// checkInvariantBounded is the context-aware equivalent of
// conceptcatalog.CheckInvariant for the isolated land path. It deliberately
// drives the scorecard once with both --json and --markdown-dir: the same
// payload supplies the semantic/coverage verdict and the generated artifacts
// used for freshness, while CommandContext makes the expensive Python process
// terminate with the pre-CAS deadline.
func checkInvariantBounded(ctx context.Context, root string, setSubphase func(string)) (conceptcatalog.InvariantResult, error) {
	out, err := os.MkdirTemp("", "fak-concept-fresh-*")
	if err != nil {
		return conceptcatalog.InvariantResult{}, err
	}
	defer os.RemoveAll(out)
	generated := filepath.Join(out, "generated")

	setSubphase("scorecard-command")
	payloadBytes, runErr := runAnalyzer(ctx, root, generated)
	if err := ctx.Err(); err != nil {
		return conceptcatalog.InvariantResult{}, err
	}
	if runErr != nil {
		if _, ok := runErr.(*exec.ExitError); !ok {
			return conceptcatalog.InvariantResult{}, runErr
		}
	}

	setSubphase("generated-freshness")
	fresh := conceptcatalog.FreshnessResult{Fresh: true, Regenerate: conceptcatalog.RegenerateCommand}
	for _, artifact := range []struct {
		tracked string
		name    string
	}{
		{tracked: conceptcatalog.GeneratedReadme, name: "README.md"},
		{tracked: conceptcatalog.GeneratedIndex, name: "INDEX.md"},
	} {
		expected, readErr := os.ReadFile(filepath.Join(generated, artifact.name))
		if readErr != nil {
			return conceptcatalog.InvariantResult{}, fmt.Errorf("read generated %s: %w", artifact.name, readErr)
		}
		actual, readErr := os.ReadFile(filepath.Join(root, filepath.FromSlash(artifact.tracked)))
		if readErr != nil || !bytes.Equal(normalizeDisambiguationNewlines(actual), normalizeDisambiguationNewlines(expected)) {
			fresh.Fresh = false
			fresh.StalePaths = append(fresh.StalePaths, artifact.tracked)
		}
	}
	sort.Strings(fresh.StalePaths)

	inv := conceptcatalog.InvariantResult{
		Freshness: fresh, SemanticValid: true, CriticalClean: true, FamilyCoverage: map[string]float64{},
	}
	setSubphase("catalog-validation")
	cat, loadErr := conceptcatalog.Load(root)
	if loadErr != nil {
		inv.SemanticValid = false
		inv.Detail = loadErr.Error()
	} else if diagnostics := conceptcatalog.Validate(cat); len(diagnostics) > 0 {
		inv.SemanticValid = false
		encoded, _ := json.Marshal(diagnostics)
		inv.Detail = string(encoded)
	}

	setSubphase("scorecard-decode")
	var payload struct {
		OK     bool   `json:"ok"`
		Reason string `json:"reason"`
		Corpus struct {
			CoverageDebt int `json:"coverage_debt"`
			ClarityDebt  int `json:"clarity_defects"`
			Coverage     struct {
				CoveragePct float64 `json:"coverage_pct"`
				PerFamily   []struct {
					Family     string `json:"family"`
					Discovered int    `json:"discovered"`
					Covered    int    `json:"covered"`
				} `json:"per_family"`
			} `json:"coverage"`
		} `json:"corpus"`
	}
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return inv, fmt.Errorf("decode scorecard: %w", err)
	}
	inv.ClarityDebt = payload.Corpus.ClarityDebt
	inv.CriticalClean = inv.SemanticValid && inv.ClarityDebt == 0
	inv.Coverage = payload.Corpus.Coverage.CoveragePct
	inv.CoverageDebt = payload.Corpus.CoverageDebt
	for _, family := range payload.Corpus.Coverage.PerFamily {
		if family.Discovered > 0 {
			inv.FamilyCoverage[family.Family] = 100 * float64(family.Covered) / float64(family.Discovered)
		}
	}
	if !payload.OK && inv.Detail == "" {
		inv.Detail = payload.Reason
	}
	return inv, nil
}

func normalizeDisambiguationNewlines(b []byte) []byte {
	return bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))
}

func extractTar(archive []byte, dst string) error {
	return extractTarBounded(context.Background(), archive, dst)
}

func extractTarBounded(ctx context.Context, archive []byte, dst string) error {
	return extractTarReaderBounded(ctx, bytes.NewReader(archive), dst)
}

func extractTarReaderBounded(ctx context.Context, archive io.Reader, dst string) error {
	if err := os.MkdirAll(dst, 0755); err != nil {
		return err
	}
	tr := tar.NewReader(archive)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		h, e := tr.Next()
		if errors.Is(e, io.EOF) {
			break
		}
		if e != nil {
			return e
		}
		clean := filepath.Clean(filepath.FromSlash(h.Name))
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return fmt.Errorf("unsafe archive path %q", h.Name)
		}
		target := filepath.Join(dst, clean)
		if h.FileInfo().IsDir() {
			if e := os.MkdirAll(target, 0755); e != nil {
				return e
			}
			continue
		}
		if h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA {
			continue
		}
		if e := os.MkdirAll(filepath.Dir(target), 0755); e != nil {
			return e
		}
		f, e := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, h.FileInfo().Mode())
		if e != nil {
			return e
		}
		_, copyErr := copyBounded(ctx, f, tr)
		closeErr := f.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

func copyBounded(ctx context.Context, dst io.Writer, src io.Reader) (int64, error) {
	buf := make([]byte, 64*1024)
	var total int64
	for {
		if err := ctx.Err(); err != nil {
			return total, err
		}
		n, readErr := src.Read(buf)
		if n > 0 {
			written, writeErr := dst.Write(buf[:n])
			total += int64(written)
			if writeErr != nil {
				return total, writeErr
			}
			if written != n {
				return total, io.ErrShortWrite
			}
		}
		if errors.Is(readErr, io.EOF) {
			return total, nil
		}
		if readErr != nil {
			return total, readErr
		}
	}
}

func (w *DisambiguationWitnesses) compactDetail() string { b, _ := json.Marshal(w); return string(b) }
