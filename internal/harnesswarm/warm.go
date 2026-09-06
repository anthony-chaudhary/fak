// Package harnesswarm provides a progressive non-blocking workspace warming engine
// for agent execution environments (#10649).
package harnesswarm

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Stage identifies a progressive workspace warming phase.
type Stage string

const (
	// StageFiles discovers and indexes the workspace file inventory.
	StageFiles Stage = "files"

	// StageIgnore loads and parses ignore rules (.gitignore, .fakignore, etc.).
	StageIgnore Stage = "ignore"

	// StageManifests detects and catalogues project manifests (go.mod, package.json, etc.).
	StageManifests Stage = "manifests"

	// StageSemantic indexes symbols and semantic hints across workspace source files.
	StageSemantic Stage = "semantic"
)

// AllStages lists the canonical warming stages in pipeline order.
var AllStages = []Stage{
	StageFiles,
	StageIgnore,
	StageManifests,
	StageSemantic,
}

// Status represents the operational state of a warming stage.
type Status string

const (
	// StatusPending indicates warming has not yet started for this stage.
	StatusPending Status = "pending"

	// StatusWarming indicates the stage is actively warming in the background.
	StatusWarming Status = "warming"

	// StatusWarm indicates the stage has completed warming successfully.
	StatusWarm Status = "warm"

	// StatusStale indicates the stage was invalidated and requires re-warming.
	StatusStale Status = "stale"

	// StatusFailed indicates the stage encountered an unrecoverable error during warming.
	StatusFailed Status = "failed"
)

// Known errors returned by the warming engine.
var (
	ErrClosed       = errors.New("harnesswarm: engine closed")
	ErrUnknownStage = errors.New("harnesswarm: unknown stage")
)

var defaultManifestFiles = []string{
	"go.mod", "go.sum", "go.work", "go.work.sum",
	"package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb",
	"cargo.toml", "cargo.lock",
	"pyproject.toml", "requirements.txt", "pipfile", "pipfile.lock", "setup.py", "setup.cfg",
	"pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
	"makefile", "cmakelists.txt",
	"dos.toml",
}

var defaultSemanticExtensions = []string{
	".go", ".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".c", ".cpp", ".cc", ".h", ".hpp",
	".java", ".kt", ".scala", ".rb", ".php", ".cs", ".swift",
}

// Options configures the workspace warming engine.
type Options struct {
	// Concurrency sets the maximum concurrent scanning workers (default: unconstrained).
	Concurrency int

	// StageDelays introduces an artificial delay per stage, primarily for test pacing.
	StageDelays map[Stage]time.Duration

	// IgnoreFiles customizes the file names recognized as ignore definitions.
	IgnoreFiles []string

	// ManifestFiles customizes the file names recognized as project manifests.
	ManifestFiles []string

	// SemanticExtensions customizes the file extensions parsed for semantic symbols.
	SemanticExtensions []string

	// SemanticParser allows injecting a custom semantic symbol extractor.
	SemanticParser func(root string, files []string) (map[string][]string, error)

	// OnStageTransition is an optional hook invoked whenever a stage changes status.
	OnStageTransition func(stage Stage, from Status, to Status)
}

// StageSnapshot captures the point-in-time state of an individual warming stage.
type StageSnapshot struct {
	Stage    Stage         `json:"stage"`
	Status   Status        `json:"status"`
	Err      error         `json:"error,omitempty"`
	Duration time.Duration `json:"duration,omitempty"`
	WarmAt   time.Time     `json:"warm_at,omitempty"`
	Count    int           `json:"count,omitempty"`
}

// EngineSnapshot captures the immutable point-in-time state of the entire engine.
type EngineSnapshot struct {
	Root      string                  `json:"root"`
	Stages    map[Stage]StageSnapshot `json:"stages"`
	Files     []string                `json:"files,omitempty"`
	Ignore    []string                `json:"ignore,omitempty"`
	Manifests []string                `json:"manifests,omitempty"`
	Semantic  map[string][]string     `json:"semantic,omitempty"`
	AllWarm   bool                    `json:"all_warm"`
	Timestamp time.Time               `json:"timestamp"`
}

// Status returns the status for the specified stage in the snapshot.
func (s EngineSnapshot) Status(stage Stage) Status {
	if st, ok := s.Stages[stage]; ok {
		return st.Status
	}
	return StatusPending
}

// IsWarm reports whether the specified stage is warm in the snapshot.
func (s EngineSnapshot) IsWarm(stage Stage) bool {
	return s.Status(stage) == StatusWarm
}

// Stage returns the detailed StageSnapshot for the specified stage.
func (s EngineSnapshot) Stage(stage Stage) StageSnapshot {
	return s.Stages[stage]
}

type stageState struct {
	status        Status
	err           error
	duration      time.Duration
	warmAt        time.Time
	count         int
	waitCh        chan struct{}
	pendingRewarm bool
	running       bool
}

// Engine coordinates progressive non-blocking workspace warming.
type Engine struct {
	mu             sync.RWMutex
	root           string
	opts           Options
	started        bool
	closed         bool
	closeCh        chan struct{}
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
	stages         map[Stage]*stageState
	files          []string
	ignorePatterns []string
	manifests      []string
	semantic       map[string][]string
}

// NewEngine constructs a new workspace warming engine rooted at root.
func NewEngine(root string, opts Options) *Engine {
	absRoot, err := filepath.Abs(root)
	if err == nil {
		root = absRoot
	}

	e := &Engine{
		root:     root,
		opts:     opts,
		closeCh:  make(chan struct{}),
		stages:   make(map[Stage]*stageState, len(AllStages)),
		semantic: make(map[string][]string),
	}

	for _, s := range AllStages {
		e.stages[s] = &stageState{
			status: StatusPending,
			waitCh: make(chan struct{}),
		}
	}

	return e
}

// Start launches progressive warming across stages in background goroutines
// without blocking the caller.
func (e *Engine) Start(ctx context.Context) {
	e.mu.Lock()
	if e.closed || e.started {
		e.mu.Unlock()
		return
	}
	e.started = true
	e.ctx, e.cancel = context.WithCancel(ctx)

	for _, stage := range AllStages {
		e.stages[stage].pendingRewarm = true
		e.triggerStageWarmLocked(stage)
	}
	e.mu.Unlock()
}

// IsWarm reports whether the requested stage is currently in StatusWarm.
func (e *Engine) IsWarm(stage Stage) bool {
	e.mu.RLock()
	defer e.mu.RUnlock()

	st, ok := e.stages[stage]
	if !ok {
		return false
	}
	return st.status == StatusWarm
}

// WaitStage blocks until the requested stage is warm, fails, or ctx expires.
func (e *Engine) WaitStage(ctx context.Context, stage Stage) error {
	for {
		e.mu.RLock()
		if e.closed {
			e.mu.RUnlock()
			return ErrClosed
		}
		st, ok := e.stages[stage]
		if !ok {
			e.mu.RUnlock()
			return fmt.Errorf("%w: %s", ErrUnknownStage, stage)
		}
		if st.status == StatusWarm {
			e.mu.RUnlock()
			return nil
		}
		if st.status == StatusFailed {
			err := st.err
			e.mu.RUnlock()
			if err != nil {
				return err
			}
			return fmt.Errorf("stage %s failed", stage)
		}
		ch := st.waitCh
		e.mu.RUnlock()

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-e.closeCh:
			return ErrClosed
		case <-ch:
			// Stage state transitioned; re-check under lock.
		}
	}
}

// Snapshot returns a point-in-time view of engine status, inventories, and metrics.
func (e *Engine) Snapshot() EngineSnapshot {
	e.mu.RLock()
	defer e.mu.RUnlock()

	snap := EngineSnapshot{
		Root:      e.root,
		Stages:    make(map[Stage]StageSnapshot, len(e.stages)),
		Timestamp: time.Now(),
		AllWarm:   true,
	}

	for s, st := range e.stages {
		snap.Stages[s] = StageSnapshot{
			Stage:    s,
			Status:   st.status,
			Err:      st.err,
			Duration: st.duration,
			WarmAt:   st.warmAt,
			Count:    st.count,
		}
		if st.status != StatusWarm {
			snap.AllWarm = false
		}
	}

	snap.Files = make([]string, len(e.files))
	copy(snap.Files, e.files)

	snap.Ignore = make([]string, len(e.ignorePatterns))
	copy(snap.Ignore, e.ignorePatterns)

	snap.Manifests = make([]string, len(e.manifests))
	copy(snap.Manifests, e.manifests)

	snap.Semantic = make(map[string][]string, len(e.semantic))
	for k, v := range e.semantic {
		vCopy := make([]string, len(v))
		copy(vCopy, v)
		snap.Semantic[k] = vCopy
	}

	return snap
}

// Invalidate marks relevant stages stale based on changed paths and triggers
// an incremental background re-warm.
func (e *Engine) Invalidate(paths ...string) {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return
	}

	affected := e.classifyPathStagesLocked(paths)

	type transition struct {
		stage Stage
		from  Status
		to    Status
	}
	var transitions []transition

	for _, stage := range AllStages {
		if !affected[stage] {
			continue
		}
		st := e.stages[stage]
		if st.status == StatusWarm || st.status == StatusFailed {
			transitions = append(transitions, transition{
				stage: stage,
				from:  st.status,
				to:    StatusStale,
			})
			st.status = StatusStale
			st.waitCh = make(chan struct{})
		}
		st.pendingRewarm = true
		if e.started {
			e.triggerStageWarmLocked(stage)
		}
	}
	cb := e.opts.OnStageTransition
	e.mu.Unlock()

	if cb != nil {
		for _, tr := range transitions {
			cb(tr.stage, tr.from, tr.to)
		}
	}
}

// Close cancels active warming routines, frees resources, and unblocks pending waiters.
func (e *Engine) Close() {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return
	}
	e.closed = true
	close(e.closeCh)
	if e.cancel != nil {
		e.cancel()
	}
	e.mu.Unlock()

	e.wg.Wait()
}

func (e *Engine) triggerStageWarmLocked(stage Stage) {
	if e.stages[stage].running {
		return
	}
	e.stages[stage].running = true
	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		e.runStageLoop(stage)
	}()
}

func (e *Engine) runStageLoop(stage Stage) {
	for {
		e.mu.Lock()
		if e.closed || !e.stages[stage].pendingRewarm {
			e.stages[stage].running = false
			e.mu.Unlock()
			return
		}
		e.stages[stage].pendingRewarm = false
		fromStatus := e.stages[stage].status
		e.stages[stage].status = StatusWarming
		waitCh := e.stages[stage].waitCh
		cb := e.opts.OnStageTransition
		ctx := e.ctx
		e.mu.Unlock()

		if cb != nil && fromStatus != StatusWarming {
			cb(stage, fromStatus, StatusWarming)
		}

		// StageManifests and StageSemantic depend on StageFiles being warm
		if stage == StageManifests || stage == StageSemantic {
			if err := e.WaitStage(ctx, StageFiles); err != nil {
				e.mu.Lock()
				e.stages[stage].running = false
				if !e.closed && (ctx == nil || ctx.Err() == nil) {
					e.stages[stage].status = StatusFailed
					e.stages[stage].err = err
					close(waitCh)
				}
				e.mu.Unlock()
				return
			}
		}

		// Apply optional pacing delay if configured
		if d := e.opts.StageDelays[stage]; d > 0 {
			select {
			case <-time.After(d):
			case <-ctx.Done():
				e.mu.Lock()
				e.stages[stage].running = false
				e.mu.Unlock()
				return
			case <-e.closeCh:
				e.mu.Lock()
				e.stages[stage].running = false
				e.mu.Unlock()
				return
			}
		}

		// Check cancellation / shutdown before starting work
		if ctx != nil {
			select {
			case <-ctx.Done():
				e.mu.Lock()
				e.stages[stage].running = false
				e.mu.Unlock()
				return
			case <-e.closeCh:
				e.mu.Lock()
				e.stages[stage].running = false
				e.mu.Unlock()
				return
			default:
			}
		}

		start := time.Now()
		err := e.executeStage(stage)
		dur := time.Since(start)

		type transition struct {
			stage Stage
			from  Status
			to    Status
		}
		var postTr transition

		e.mu.Lock()
		st := e.stages[stage]
		st.duration = dur
		oldStatus := st.status

		if err != nil {
			st.status = StatusFailed
			st.err = err
			postTr = transition{stage: stage, from: oldStatus, to: StatusFailed}
		} else {
			st.status = StatusWarm
			st.err = nil
			st.warmAt = time.Now()
			postTr = transition{stage: stage, from: oldStatus, to: StatusWarm}
		}

		// Broadcast stage completion to any waiting callers
		close(waitCh)

		// If another invalidation arrived while executing, transition to stale and re-warm
		if st.pendingRewarm && !e.closed {
			st.waitCh = make(chan struct{})
			st.status = StatusStale
			e.mu.Unlock()
			if cb != nil {
				cb(postTr.stage, postTr.from, postTr.to)
				cb(stage, postTr.to, StatusStale)
			}
			continue
		}

		st.running = false
		e.mu.Unlock()

		if cb != nil {
			cb(postTr.stage, postTr.from, postTr.to)
		}
		return
	}
}

func (e *Engine) executeStage(stage Stage) error {
	switch stage {
	case StageFiles:
		return e.executeFilesStage()
	case StageIgnore:
		return e.executeIgnoreStage()
	case StageManifests:
		return e.executeManifestsStage()
	case StageSemantic:
		return e.executeSemanticStage()
	default:
		return fmt.Errorf("%w: %s", ErrUnknownStage, stage)
	}
}

func (e *Engine) executeFilesStage() error {
	fi, err := os.Stat(e.root)
	if err != nil {
		return err
	}
	if !fi.IsDir() {
		return fmt.Errorf("root %q is not a directory", e.root)
	}

	var files []string
	err = filepath.WalkDir(e.root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			if name == ".git" || name == ".hg" || name == ".svn" {
				return filepath.SkipDir
			}
			return nil
		}
		rel, relErr := filepath.Rel(e.root, path)
		if relErr != nil {
			return nil
		}
		files = append(files, filepath.ToSlash(rel))
		return nil
	})
	if err != nil {
		return err
	}

	sort.Strings(files)

	e.mu.Lock()
	e.files = files
	e.stages[StageFiles].count = len(files)
	e.mu.Unlock()
	return nil
}

func (e *Engine) executeIgnoreStage() error {
	var patterns []string
	seen := make(map[string]bool)

	ignoreFiles := []string{".gitignore", ".fakignore", ".ignore"}
	if len(e.opts.IgnoreFiles) > 0 {
		ignoreFiles = e.opts.IgnoreFiles
	}

	// Read root ignore files
	for _, name := range ignoreFiles {
		fullPath := filepath.Join(e.root, name)
		data, err := os.ReadFile(fullPath)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			if !seen[line] {
				seen[line] = true
				patterns = append(patterns, line)
			}
		}
	}

	// Also inspect nested ignore files if files inventory is populated
	e.mu.RLock()
	filesSnapshot := make([]string, len(e.files))
	copy(filesSnapshot, e.files)
	e.mu.RUnlock()

	for _, f := range filesSnapshot {
		base := filepath.Base(f)
		for _, name := range ignoreFiles {
			if strings.EqualFold(base, name) && filepath.Dir(f) != "." {
				fullPath := filepath.Join(e.root, filepath.FromSlash(f))
				data, err := os.ReadFile(fullPath)
				if err != nil {
					continue
				}
				dir := filepath.ToSlash(filepath.Dir(f))
				for _, line := range strings.Split(string(data), "\n") {
					line = strings.TrimSpace(line)
					if line == "" || strings.HasPrefix(line, "#") {
						continue
					}
					pat := dir + "/" + line
					if !seen[pat] {
						seen[pat] = true
						patterns = append(patterns, pat)
					}
				}
			}
		}
	}

	sort.Strings(patterns)

	e.mu.Lock()
	e.ignorePatterns = patterns
	e.stages[StageIgnore].count = len(patterns)
	e.mu.Unlock()
	return nil
}

func (e *Engine) executeManifestsStage() error {
	e.mu.RLock()
	files := make([]string, len(e.files))
	copy(files, e.files)
	e.mu.RUnlock()

	var manifests []string
	for _, f := range files {
		base := filepath.Base(f)
		if isManifestFile(base, e.opts.ManifestFiles) {
			manifests = append(manifests, f)
		}
	}

	sort.Strings(manifests)

	e.mu.Lock()
	e.manifests = manifests
	e.stages[StageManifests].count = len(manifests)
	e.mu.Unlock()
	return nil
}

func (e *Engine) executeSemanticStage() error {
	e.mu.RLock()
	files := make([]string, len(e.files))
	copy(files, e.files)
	e.mu.RUnlock()

	if e.opts.SemanticParser != nil {
		res, err := e.opts.SemanticParser(e.root, files)
		if err != nil {
			return err
		}
		total := 0
		for _, syms := range res {
			total += len(syms)
		}
		e.mu.Lock()
		e.semantic = res
		e.stages[StageSemantic].count = total
		e.mu.Unlock()
		return nil
	}

	semantic := make(map[string][]string)
	total := 0

	for _, f := range files {
		ext := strings.ToLower(filepath.Ext(f))
		if !isSourceFile(ext, e.opts.SemanticExtensions) {
			continue
		}
		fullPath := filepath.Join(e.root, filepath.FromSlash(f))
		syms := extractFileSymbols(fullPath, ext)
		if len(syms) > 0 {
			semantic[f] = syms
			total += len(syms)
		}
	}

	e.mu.Lock()
	e.semantic = semantic
	e.stages[StageSemantic].count = total
	e.mu.Unlock()
	return nil
}

func (e *Engine) classifyPathStagesLocked(paths []string) map[Stage]bool {
	if len(paths) == 0 {
		return map[Stage]bool{
			StageFiles:     true,
			StageIgnore:    true,
			StageManifests: true,
			StageSemantic:  true,
		}
	}

	affected := make(map[Stage]bool)
	for _, p := range paths {
		clean := filepath.ToSlash(filepath.Clean(p))
		if clean == "." || clean == "/" || clean == "" {
			return map[Stage]bool{
				StageFiles:     true,
				StageIgnore:    true,
				StageManifests: true,
				StageSemantic:  true,
			}
		}

		base := filepath.Base(clean)
		ext := strings.ToLower(filepath.Ext(clean))

		if isIgnoreFile(base, e.opts.IgnoreFiles) {
			affected[StageIgnore] = true
			continue
		}

		if isManifestFile(base, e.opts.ManifestFiles) {
			affected[StageManifests] = true
			continue
		}

		if isSourceFile(ext, e.opts.SemanticExtensions) {
			affected[StageSemantic] = true
			continue
		}

		// Unknown file extension or directory path
		affected[StageFiles] = true
		affected[StageManifests] = true
		affected[StageSemantic] = true
	}

	if affected[StageFiles] {
		affected[StageManifests] = true
		affected[StageSemantic] = true
	}

	return affected
}

func isIgnoreFile(base string, custom []string) bool {
	for _, c := range custom {
		if strings.EqualFold(base, c) {
			return true
		}
	}
	switch strings.ToLower(base) {
	case ".gitignore", ".fakignore", ".ignore", ".dockerignore", ".npmignore":
		return true
	}
	return false
}

func isManifestFile(base string, custom []string) bool {
	for _, c := range custom {
		if strings.EqualFold(base, c) {
			return true
		}
	}
	switch strings.ToLower(base) {
	case "go.mod", "go.sum", "go.work", "go.work.sum",
		"package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb",
		"cargo.toml", "cargo.lock",
		"pyproject.toml", "requirements.txt", "pipfile", "pipfile.lock", "setup.py", "setup.cfg",
		"pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
		"makefile", "cmakelists.txt",
		"dos.toml":
		return true
	}
	return false
}

func isSourceFile(ext string, custom []string) bool {
	for _, c := range custom {
		if strings.EqualFold(ext, c) {
			return true
		}
	}
	switch ext {
	case ".go", ".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".c", ".cpp", ".cc", ".h", ".hpp",
		".java", ".kt", ".scala", ".rb", ".php", ".cs", ".swift":
		return true
	}
	return false
}

func extractFileSymbols(fullPath, ext string) []string {
	if ext == ".go" {
		return extractGoSymbols(fullPath)
	}
	return extractLineSymbols(fullPath, ext)
}

func extractGoSymbols(fullPath string) []string {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, fullPath, nil, parser.SkipObjectResolution)
	if err != nil {
		return nil
	}

	var syms []string
	if node.Name != nil {
		syms = append(syms, "package:"+node.Name.Name)
	}
	for _, decl := range node.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			name := d.Name.Name
			if d.Recv != nil && len(d.Recv.List) > 0 {
				recv := formatReceiverType(d.Recv.List[0].Type)
				syms = append(syms, "method:("+recv+")."+name)
			} else {
				syms = append(syms, "func:"+name)
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					syms = append(syms, "type:"+s.Name.Name)
				case *ast.ValueSpec:
					for _, name := range s.Names {
						syms = append(syms, "var:"+name.Name)
					}
				}
			}
		}
	}
	return syms
}

func formatReceiverType(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + formatReceiverType(t.X)
	case *ast.SelectorExpr:
		return formatReceiverType(t.X) + "." + t.Sel.Name
	default:
		return "T"
	}
}

func extractLineSymbols(fullPath, ext string) []string {
	f, err := os.Open(fullPath)
	if err != nil {
		return nil
	}
	defer f.Close()

	var syms []string
	scanner := bufio.NewScanner(f)
	lines := 0
	for scanner.Scan() && lines < 1000 {
		lines++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "//") || strings.HasPrefix(line, "#") {
			continue
		}
		switch ext {
		case ".py":
			if strings.HasPrefix(line, "def ") {
				rest := strings.TrimPrefix(line, "def ")
				if idx := strings.IndexAny(rest, "( :"); idx > 0 {
					syms = append(syms, "def:"+rest[:idx])
				}
			} else if strings.HasPrefix(line, "class ") {
				rest := strings.TrimPrefix(line, "class ")
				if idx := strings.IndexAny(rest, "( :"); idx > 0 {
					syms = append(syms, "class:"+rest[:idx])
				}
			}
		case ".ts", ".tsx", ".js", ".jsx":
			line = strings.TrimPrefix(line, "export ")
			line = strings.TrimPrefix(line, "default ")
			if strings.HasPrefix(line, "function ") {
				rest := strings.TrimPrefix(line, "function ")
				if idx := strings.IndexAny(rest, "( "); idx > 0 {
					syms = append(syms, "function:"+rest[:idx])
				}
			} else if strings.HasPrefix(line, "class ") {
				rest := strings.TrimPrefix(line, "class ")
				if idx := strings.IndexAny(rest, "{ "); idx > 0 {
					syms = append(syms, "class:"+rest[:idx])
				}
			}
		case ".rs":
			line = strings.TrimPrefix(line, "pub ")
			if strings.HasPrefix(line, "fn ") {
				rest := strings.TrimPrefix(line, "fn ")
				if idx := strings.IndexAny(rest, "(< "); idx > 0 {
					syms = append(syms, "fn:"+rest[:idx])
				}
			} else if strings.HasPrefix(line, "struct ") {
				rest := strings.TrimPrefix(line, "struct ")
				if idx := strings.IndexAny(rest, "{<; "); idx > 0 {
					syms = append(syms, "struct:"+rest[:idx])
				}
			}
		}
	}
	return syms
}
