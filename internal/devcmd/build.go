package devcmd

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	buildReceiptSchema      = "fak.build-receipt.v1"
	defaultBuildReceiptPath = ".fak/build-receipt.json"
)

type buildPhase struct {
	Name      string `json:"name"`
	Outcome   string `json:"outcome"`
	ElapsedMS int64  `json:"elapsed_ms"`
	ExitCode  *int   `json:"exit_code,omitempty"`
	Error     string `json:"error,omitempty"`
}

type buildSource struct {
	Commit            string `json:"commit"`
	CommittedTree     string `json:"committed_tree"`
	WorkingTreeSHA256 string `json:"working_tree_sha256"`
}

type buildToolchain struct {
	GoVersion   string `json:"go_version"`
	GOOS        string `json:"goos"`
	GOARCH      string `json:"goarch"`
	GOCache     string `json:"go_cache"`
	CGOEnabled  string `json:"cgo_enabled"`
	GoToolchain string `json:"go_toolchain"`
	GoFlags     string `json:"go_flags"`
}

type buildCommand struct {
	Argv        []string          `json:"argv"`
	Directory   string            `json:"directory"`
	Environment map[string]string `json:"environment"`
	Profile     string            `json:"profile"`
	Output      string            `json:"output"`
}

type buildPGO struct {
	Mode      string `json:"mode"`
	Identity  string `json:"identity,omitempty"`
	SHA256    string `json:"sha256,omitempty"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
}

// buildExecution carries values needed only by the child process. In particular,
// an explicit PGO profile's private snapshot path must never enter buildCommand,
// because buildCommand is serialized in the receipt and on JSON stdout.
type buildExecution struct {
	command     buildCommand
	environment map[string]string
}

type buildSettings struct {
	Version string
	Tags    string
	GCFlags string
}

type buildArtifact struct {
	Path      string `json:"path"`
	SizeBytes int64  `json:"size_bytes"`
	SHA256    string `json:"sha256"`
}

type buildSmokeResult struct {
	Command  []string `json:"command"`
	Outcome  string   `json:"outcome"`
	ExitCode int      `json:"exit_code"`
	Output   string   `json:"output,omitempty"`
	Error    string   `json:"error,omitempty"`
}

type buildReceipt struct {
	Schema       string            `json:"schema"`
	Outcome      string            `json:"outcome"`
	ExitCode     int               `json:"exit_code"`
	Error        string            `json:"error,omitempty"`
	StartedAt    string            `json:"started_at"`
	FinishedAt   string            `json:"finished_at"`
	ElapsedMS    int64             `json:"elapsed_ms"`
	ReceiptPath  string            `json:"receipt_path"`
	Command      buildCommand      `json:"command"`
	PGO          buildPGO          `json:"pgo"`
	Source       buildSource       `json:"source"`
	Toolchain    buildToolchain    `json:"toolchain"`
	CacheState   string            `json:"cache_state"`
	PackageCount int               `json:"package_count"`
	Phases       []buildPhase      `json:"phases"`
	Artifact     *buildArtifact    `json:"artifact,omitempty"`
	Smoke        *buildSmokeResult `json:"smoke,omitempty"`
}

type buildProvenance struct {
	Source       buildSource
	Toolchain    buildToolchain
	PackageCount int
}

var (
	buildNow              = time.Now
	buildGatherProvenance = gatherBuildProvenance
	buildPrepareOutput    = prepareBuildOutput
	buildExecute          = executeCanonicalBuild
	buildInspectArtifact  = inspectBuildArtifact
	buildSmoke            = smokeBuildArtifact
	buildWriteReceipt     = writeBuildReceiptAtomic
	buildCaptureOutput    = buildCommandOutput
)

// RunBuild creates the runnable fak product through scripts/build.sh, the repository's
// sole build-flag recipe. It records the whole path rather than presenting compile time
// alone as build time: source/toolchain provenance, compile+link, artifact inspection,
// and a real version smoke are separate ordered phases in the durable receipt.
func RunBuild(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak build", flag.ContinueOnError)
	fs.SetOutput(stderr)
	profile := fs.String("profile", "dev", "scripts/build.sh profile (release|dev|race)")
	outFlag := fs.String("out", defaultBuildOutput(), "output binary path (default: .fak/bin/fak[.exe]; relative paths are resolved from the repository root)")
	receiptFlag := fs.String("receipt", defaultBuildReceiptPath, "durable JSON receipt path (relative paths are resolved from the repository root)")
	versionFlag := fs.String("version", "", "version stamped into the binary (default: repository VERSION file)")
	tagsFlag := fs.String("tags", "", "space-separated Go build tags")
	gcflagsFlag := fs.String("gcflags", "", "extra Go compiler flags for dev or race profiles")
	pgoFlag := fs.String("pgo", "off", "release PGO input: off or an explicit non-empty profile path")
	asJSON := fs.Bool("json", false, "write the receipt as one JSON object to stdout; progress and child output stay on stderr")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak build: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	if *profile != "release" && *profile != "dev" && *profile != "race" {
		fmt.Fprintf(stderr, "fak build: --profile must be release, dev, or race (got %q)\n", *profile)
		return 2
	}
	if strings.TrimSpace(*outFlag) == "" || strings.TrimSpace(*receiptFlag) == "" {
		fmt.Fprintln(stderr, "fak build: --out and --receipt must be non-empty paths")
		return 2
	}

	root := repoRoot()
	output := rootedBuildPath(root, *outFlag)
	receiptPath := rootedBuildPath(root, *receiptFlag)
	if output == receiptPath {
		fmt.Fprintf(stderr, "fak build: --out and --receipt resolve to the same path %s; choose distinct paths\n", output)
		return 2
	}
	pgo, pgoSnapshot, err := prepareBuildPGO(root, output, receiptPath, *profile, *pgoFlag)
	if err != nil {
		fmt.Fprintf(stderr, "fak build: %v\n", err)
		return 2
	}
	if pgoSnapshot != "" {
		defer removeBuildPGOSnapshot(pgoSnapshot)
	}
	overallStart := buildNow()
	settings := buildSettings{Version: *versionFlag, Tags: *tagsFlag, GCFlags: *gcflagsFlag}
	receipt := buildReceipt{
		Schema:      buildReceiptSchema,
		Outcome:     "running",
		StartedAt:   overallStart.UTC().Format(time.RFC3339Nano),
		ReceiptPath: receiptPath,
		Command:     canonicalBuildCommand(root, output, *profile, buildToolchain{}, settings, pgo),
		PGO:         pgo,
		CacheState:  "inherited_uncontrolled",
	}

	if !*asJSON {
		fmt.Fprintf(stderr, "fak build: recording %s build timings (cache: inherited/uncontrolled)\n", *profile)
	}

	provenance, err := runBuildPhase(&receipt, "provenance", func() (buildProvenance, error) {
		return buildGatherProvenance(root)
	})
	if err != nil {
		return finishBuild(stdout, stderr, *asJSON, overallStart, receiptPath, &receipt, 1,
			fmt.Errorf("provenance failed: %w", err))
	}
	receipt.Source = provenance.Source
	receipt.Toolchain = provenance.Toolchain
	receipt.PackageCount = provenance.PackageCount
	receipt.Command = canonicalBuildCommand(root, output, *profile, provenance.Toolchain, settings, pgo)

	_, err = runBuildPhase(&receipt, "output_prepare", func() (struct{}, error) {
		return struct{}{}, buildPrepareOutput(output)
	})
	if err != nil {
		return finishBuild(stdout, stderr, *asJSON, overallStart, receiptPath, &receipt, 1,
			fmt.Errorf("prepare output directory for %s: %w", output, err))
	}

	if !*asJSON {
		fmt.Fprintf(stderr, "fak build: %s\n", renderBuildCommand(receipt.Command))
	}
	phaseStart := buildNow()
	buildCode, runErr := buildExecute(canonicalBuildExecution(receipt.Command, pgoSnapshot), stderr)
	compileErr := runErr
	if compileErr == nil && buildCode != 0 {
		compileErr = fmt.Errorf("canonical build command exited %d", buildCode)
	}
	receipt.Phases = append(receipt.Phases, finishedBuildPhase("compile_link", phaseStart, buildNow(), buildCode, compileErr))
	if compileErr != nil {
		code := buildCode
		if code == 0 {
			code = 1
		}
		return finishBuild(stdout, stderr, *asJSON, overallStart, receiptPath, &receipt, code,
			fmt.Errorf("compile/link failed; inspect the child output above: %w", compileErr))
	}

	artifact, err := runBuildPhase(&receipt, "artifact", func() (buildArtifact, error) {
		return buildInspectArtifact(output)
	})
	if err != nil {
		return finishBuild(stdout, stderr, *asJSON, overallStart, receiptPath, &receipt, 1,
			fmt.Errorf("artifact inspection failed for %s: %w", output, err))
	}
	receipt.Artifact = &artifact

	phaseStart = buildNow()
	smoke := buildSmoke(output)
	if smoke.Output != "" {
		fmt.Fprintln(stderr, smoke.Output)
	}
	smokeErr := error(nil)
	if smoke.Error != "" {
		smokeErr = errors.New(smoke.Error)
	} else if smoke.ExitCode != 0 || smoke.Outcome != "success" {
		smokeErr = fmt.Errorf("artifact smoke exited %d", smoke.ExitCode)
	}
	receipt.Smoke = &smoke
	receipt.Phases = append(receipt.Phases, finishedBuildPhase("smoke", phaseStart, buildNow(), smoke.ExitCode, smokeErr))
	if smokeErr != nil {
		code := smoke.ExitCode
		if code == 0 {
			code = 1
		}
		return finishBuild(stdout, stderr, *asJSON, overallStart, receiptPath, &receipt, code,
			fmt.Errorf("artifact smoke failed; run `%s version --json`: %w", output, smokeErr))
	}

	return finishBuild(stdout, stderr, *asJSON, overallStart, receiptPath, &receipt, 0, nil)
}

func rootedBuildPath(root, path string) string {
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	return filepath.Join(root, filepath.Clean(path))
}

func defaultBuildOutput() string {
	// GOOS is Go toolchain configuration, so ask the toolchain for its effective
	// target instead of creating a second implicit environment-config surface.
	goos := runtime.GOOS
	if raw, err := buildCaptureOutput(repoRoot(), "go", "env", "GOOS"); err == nil {
		if configured := strings.TrimSpace(string(raw)); configured != "" {
			goos = configured
		}
	}
	name := "fak"
	if goos == "windows" {
		name += ".exe"
	}
	return filepath.Join(".fak", "bin", name)
}

func prepareBuildOutput(path string) error {
	return os.MkdirAll(filepath.Dir(path), 0o755)
}

func runBuildPhase[T any](receipt *buildReceipt, name string, operation func() (T, error)) (T, error) {
	start := buildNow()
	result, err := operation()
	receipt.Phases = append(receipt.Phases, finishedBuildPhase(name, start, buildNow(), 0, err))
	return result, err
}

func finishedBuildPhase(name string, start, end time.Time, exitCode int, err error) buildPhase {
	phase := buildPhase{Name: name, Outcome: "success", ElapsedMS: elapsedMilliseconds(start, end)}
	if name == "compile_link" || name == "smoke" {
		code := exitCode
		phase.ExitCode = &code
	}
	if err != nil {
		phase.Outcome = "failed"
		phase.Error = err.Error()
	}
	return phase
}

func elapsedMilliseconds(start, end time.Time) int64 {
	if end.Before(start) {
		return 0
	}
	return end.Sub(start).Milliseconds()
}

func finishBuild(stdout, stderr io.Writer, asJSON bool, overallStart time.Time, receiptPath string, receipt *buildReceipt, code int, terminalErr error) int {
	finished := buildNow()
	receipt.FinishedAt = finished.UTC().Format(time.RFC3339Nano)
	receipt.ElapsedMS = elapsedMilliseconds(overallStart, finished)
	receipt.ExitCode = code
	if terminalErr == nil {
		receipt.Outcome = "success"
	} else {
		receipt.Outcome = "failed"
		receipt.Error = terminalErr.Error()
	}
	receipt.Phases = append(receipt.Phases, buildPhase{Name: "total", Outcome: receipt.Outcome, ElapsedMS: receipt.ElapsedMS, Error: receipt.Error})

	if err := buildWriteReceipt(receiptPath, *receipt); err != nil {
		writeErr := fmt.Errorf("write durable receipt %s: %w", receiptPath, err)
		if receipt.Error == "" {
			receipt.Error = writeErr.Error()
		} else {
			receipt.Error += "; " + writeErr.Error()
		}
		receipt.Outcome = "failed"
		receipt.ExitCode = 1
		receipt.Phases[len(receipt.Phases)-1].Outcome = "failed"
		receipt.Phases[len(receipt.Phases)-1].Error = receipt.Error
		code = 1
		fmt.Fprintf(stderr, "fak build: %v\n", writeErr)
	}

	if asJSON {
		if err := writeIndentedJSONNoEscape(stdout, *receipt); err != nil {
			fmt.Fprintf(stderr, "fak build: write JSON result: %v\n", err)
			return 1
		}
	} else {
		renderBuildReceipt(stdout, *receipt)
	}
	if terminalErr != nil {
		fmt.Fprintf(stderr, "fak build: %v\n", terminalErr)
	}
	return code
}

func renderBuildReceipt(w io.Writer, receipt buildReceipt) {
	status := "OK"
	if receipt.Outcome != "success" {
		status = "FAILED"
	}
	fmt.Fprintf(w, "fak build: %s in %s\n", status, formatBuildDuration(receipt.ElapsedMS))
	for _, phase := range receipt.Phases {
		fmt.Fprintf(w, "  %-19s %9s  %s\n", phase.Name, formatBuildDuration(phase.ElapsedMS), phase.Outcome)
	}
	if receipt.Artifact != nil {
		fmt.Fprintf(w, "  artifact            %s  %d bytes  sha256:%s\n", receipt.Artifact.Path, receipt.Artifact.SizeBytes, receipt.Artifact.SHA256)
	}
	fmt.Fprintf(w, "  packages            %d\n", receipt.PackageCount)
	fmt.Fprintf(w, "  receipt             %s\n", receipt.ReceiptPath)
	if receipt.Error != "" {
		fmt.Fprintf(w, "  error               %s\n", receipt.Error)
	}
}

func formatBuildDuration(ms int64) string {
	return (time.Duration(ms) * time.Millisecond).String()
}

func canonicalBuildCommand(root, output, profile string, toolchain buildToolchain, settings buildSettings, pgo buildPGO) buildCommand {
	version := settings.Version
	if version == "" {
		if raw, err := os.ReadFile(filepath.Join(root, "VERSION")); err == nil {
			version = strings.TrimSpace(string(raw))
		}
	}
	return buildCommand{
		Argv:      []string{"sh", "scripts/build.sh"},
		Directory: root,
		Profile:   profile,
		Output:    output,
		Environment: map[string]string{
			"OUT":         output,
			"PROFILE":     profile,
			"VERSION":     version,
			"TAGS":        settings.Tags,
			"GCFLAGS":     settings.GCFlags,
			"GOFLAGS":     toolchain.GoFlags,
			"CGO_ENABLED": toolchain.CGOEnabled,
			"GOTOOLCHAIN": toolchain.GoToolchain,
			"PGO":         pgo.Mode,
		},
	}
}

func canonicalBuildExecution(command buildCommand, pgoSnapshot string) buildExecution {
	environment := make(map[string]string, len(command.Environment))
	for key, value := range command.Environment {
		environment[key] = value
	}
	if pgoSnapshot != "" {
		environment["PGO"] = pgoSnapshot
	}
	return buildExecution{command: command, environment: environment}
}

func prepareBuildPGO(root, output, receiptPath, profile, rawPGO string) (buildPGO, string, error) {
	if rawPGO == "off" {
		return buildPGO{Mode: "off"}, "", nil
	}
	if strings.TrimSpace(rawPGO) == "" {
		return buildPGO{}, "", errors.New("--pgo must be 'off' or an explicit non-empty profile path")
	}
	if rawPGO == "auto" {
		return buildPGO{}, "", errors.New("--pgo auto is not pinned; use 'off' or an explicit profile path")
	}
	if profile != "release" {
		return buildPGO{}, "", fmt.Errorf("--pgo profiles are release-only; --profile %s requires --pgo off", profile)
	}

	sourcePath := rootedBuildPath(root, rawPGO)
	if buildPathsCollide(sourcePath, output) || buildPathsCollide(sourcePath, receiptPath) {
		return buildPGO{}, "", errors.New("--pgo profile must not resolve to the output or receipt path")
	}
	source, err := os.Open(sourcePath)
	if err != nil {
		return buildPGO{}, "", errors.New("--pgo profile must be a readable non-empty regular file")
	}
	defer source.Close()
	info, err := source.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() == 0 {
		return buildPGO{}, "", errors.New("--pgo profile must be a readable non-empty regular file")
	}

	snapshotDir, err := os.MkdirTemp("", "fak-build-pgo-*")
	if err != nil {
		return buildPGO{}, "", errors.New("create private --pgo snapshot: temporary storage unavailable")
	}
	removeSnapshot := true
	defer func() {
		if removeSnapshot {
			removeBuildPGOSnapshot(filepath.Join(snapshotDir, "profile.pprof"))
		}
	}()
	if err := os.Chmod(snapshotDir, 0o700); err != nil {
		return buildPGO{}, "", errors.New("secure private --pgo snapshot: temporary storage unavailable")
	}
	snapshotPath := filepath.Join(snapshotDir, "profile.pprof")
	snapshot, err := os.OpenFile(snapshotPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return buildPGO{}, "", errors.New("create private --pgo snapshot: temporary storage unavailable")
	}
	h := sha256.New()
	size, copyErr := io.Copy(io.MultiWriter(snapshot, h), source)
	syncErr := snapshot.Sync()
	closeErr := snapshot.Close()
	if copyErr != nil || syncErr != nil || closeErr != nil || size == 0 {
		return buildPGO{}, "", errors.New("copy --pgo profile into private snapshot: profile is unreadable or empty")
	}
	digest := hex.EncodeToString(h.Sum(nil))
	removeSnapshot = false
	return buildPGO{
		Mode:      "profile",
		Identity:  "sha256:" + digest,
		SHA256:    digest,
		SizeBytes: size,
	}, snapshotPath, nil
}

func removeBuildPGOSnapshot(snapshotPath string) {
	_ = os.Remove(snapshotPath)
	_ = os.Remove(filepath.Dir(snapshotPath))
}

func buildPathsCollide(left, right string) bool {
	left = filepath.Clean(left)
	right = filepath.Clean(right)
	if left == right || (runtime.GOOS == "windows" && strings.EqualFold(left, right)) {
		return true
	}
	leftInfo, leftErr := os.Stat(left)
	rightInfo, rightErr := os.Stat(right)
	return leftErr == nil && rightErr == nil && os.SameFile(leftInfo, rightInfo)
}

func renderBuildCommand(command buildCommand) string {
	return fmt.Sprintf("PROFILE=%s OUT=%s sh scripts/build.sh", command.Profile, command.Output)
}

func gatherBuildProvenance(root string) (buildProvenance, error) {
	commitRaw, err := buildCaptureOutput(root, "git", "rev-parse", "HEAD")
	if err != nil {
		return buildProvenance{}, fmt.Errorf("resolve commit: %w", err)
	}
	commit := strings.TrimSpace(string(commitRaw))
	treeRaw, err := buildCaptureOutput(root, "git", "rev-parse", "HEAD^{tree}")
	if err != nil {
		return buildProvenance{}, fmt.Errorf("resolve committed tree: %w", err)
	}
	committedTree := strings.TrimSpace(string(treeRaw))
	trackedDelta, err := buildCaptureOutput(root, "git", "diff", "--binary", "--no-ext-diff", "HEAD", "--")
	if err != nil {
		return buildProvenance{}, fmt.Errorf("read tracked working delta: %w", err)
	}
	untrackedRaw, err := buildCaptureOutput(root, "git", "ls-files", "--others", "--exclude-standard", "-z")
	if err != nil {
		return buildProvenance{}, fmt.Errorf("list untracked source files: %w", err)
	}
	digest, err := digestBuildTree(root, committedTree, trackedDelta, splitNUL(untrackedRaw))
	if err != nil {
		return buildProvenance{}, fmt.Errorf("digest working tree: %w", err)
	}

	goVersionRaw, err := buildCaptureOutput(root, "go", "version")
	if err != nil {
		return buildProvenance{}, fmt.Errorf("read Go version: %w", err)
	}
	goEnvRaw, err := buildCaptureOutput(root, "go", "env", "GOOS", "GOARCH", "GOCACHE", "CGO_ENABLED", "GOTOOLCHAIN", "GOFLAGS")
	if err != nil {
		return buildProvenance{}, fmt.Errorf("read effective Go environment: %w", err)
	}
	envLines := strings.Split(strings.TrimSuffix(string(goEnvRaw), "\n"), "\n")
	if len(envLines) != 6 {
		return buildProvenance{}, fmt.Errorf("go env returned %d fields, want 6", len(envLines))
	}

	packagesRaw, err := buildCaptureOutput(root, "go", "list", "-deps", "./cmd/fak")
	if err != nil {
		return buildProvenance{}, fmt.Errorf("count build packages: %w", err)
	}
	packageCount := countNonemptyLines(packagesRaw)
	if packageCount == 0 {
		return buildProvenance{}, errors.New("count build packages: go list returned no packages")
	}
	return buildProvenance{
		Source: buildSource{Commit: commit, CommittedTree: committedTree, WorkingTreeSHA256: digest},
		Toolchain: buildToolchain{
			GoVersion: strings.TrimSpace(string(goVersionRaw)), GOOS: envLines[0], GOARCH: envLines[1],
			GOCache: envLines[2], CGOEnabled: envLines[3], GoToolchain: envLines[4], GoFlags: envLines[5],
		},
		PackageCount: packageCount,
	}, nil
}

func buildCommandOutput(root, name string, args ...string) ([]byte, error) {
	cmd := windowgate.Command(name, args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s: %w: %s", strings.Join(append([]string{name}, args...), " "), err, strings.TrimSpace(string(out)))
	}
	return out, nil
}

func splitNUL(raw []byte) []string {
	parts := strings.Split(string(raw), "\x00")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if part != "" {
			out = append(out, part)
		}
	}
	sort.Strings(out)
	return out
}

func digestBuildTree(root, committedTree string, trackedDelta []byte, untrackedPaths []string) (string, error) {
	h := sha256.New()
	writeDigestField(h, "committed_tree", []byte(committedTree))
	writeDigestField(h, "tracked_delta", trackedDelta)
	untrackedPaths = append([]string(nil), untrackedPaths...)
	sort.Strings(untrackedPaths)
	for _, rel := range untrackedPaths {
		path := filepath.Join(root, filepath.FromSlash(rel))
		info, err := os.Lstat(path)
		if errors.Is(err, os.ErrNotExist) {
			writeDigestField(h, "absent:"+rel, nil)
			continue
		}
		if err != nil {
			return "", fmt.Errorf("stat %s: %w", rel, err)
		}
		writeDigestField(h, "mode:"+rel, []byte(info.Mode().String()))
		if info.Mode()&os.ModeSymlink != 0 {
			target, err := os.Readlink(path)
			if err != nil {
				return "", fmt.Errorf("read symlink %s: %w", rel, err)
			}
			writeDigestField(h, "link:"+rel, []byte(target))
			continue
		}
		if !info.Mode().IsRegular() {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			return "", fmt.Errorf("open %s: %w", rel, err)
		}
		_, copyErr := io.Copy(h, f)
		closeErr := f.Close()
		if copyErr != nil {
			return "", fmt.Errorf("hash %s: %w", rel, copyErr)
		}
		if closeErr != nil {
			return "", fmt.Errorf("close %s: %w", rel, closeErr)
		}
		writeDigestField(h, "end:"+rel, nil)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeDigestField(w io.Writer, label string, value []byte) {
	fmt.Fprintf(w, "%d:%s:%d:", len(label), label, len(value))
	_, _ = w.Write(value)
}

func countNonemptyLines(raw []byte) int {
	count := 0
	s := bufio.NewScanner(strings.NewReader(string(raw)))
	for s.Scan() {
		if strings.TrimSpace(s.Text()) != "" {
			count++
		}
	}
	return count
}

func executeCanonicalBuild(execution buildExecution, stderr io.Writer) (int, error) {
	command := execution.command
	cmd := windowgate.Command(command.Argv[0], command.Argv[1:]...)
	cmd.Dir = command.Directory
	cmd.Env = overriddenBuildEnv(os.Environ(), execution.environment)
	cmd.Stdout = stderr
	cmd.Stderr = stderr
	windowgate.ConfigureBackgroundCommand(cmd)
	err := cmd.Run()
	if err == nil {
		return 0, nil
	}
	var exitErr *os.PathError
	if errors.As(err, &exitErr) {
		return 1, err
	}
	if cmd.ProcessState != nil {
		return cmd.ProcessState.ExitCode(), nil
	}
	return 1, err
}

func overriddenBuildEnv(base []string, overrides map[string]string) []string {
	result := make([]string, 0, len(base)+len(overrides))
	for _, entry := range base {
		key := entry
		if i := strings.IndexByte(entry, '='); i >= 0 {
			key = entry[:i]
		}
		if _, replaced := overrides[key]; !replaced {
			result = append(result, entry)
		}
	}
	keys := make([]string, 0, len(overrides))
	for key := range overrides {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		result = append(result, key+"="+overrides[key])
	}
	return result
}

func inspectBuildArtifact(path string) (buildArtifact, error) {
	info, err := os.Stat(path)
	if err != nil {
		return buildArtifact{}, err
	}
	if !info.Mode().IsRegular() {
		return buildArtifact{}, fmt.Errorf("not a regular file")
	}
	f, err := os.Open(path)
	if err != nil {
		return buildArtifact{}, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return buildArtifact{}, err
	}
	return buildArtifact{Path: path, SizeBytes: info.Size(), SHA256: hex.EncodeToString(h.Sum(nil))}, nil
}

func smokeBuildArtifact(path string) buildSmokeResult {
	result := buildSmokeResult{Command: []string{path, "version", "--json"}, Outcome: "failed"}
	cmd := windowgate.Command(path, "version", "--json")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	result.Output = strings.TrimSpace(string(out))
	if cmd.ProcessState != nil {
		result.ExitCode = cmd.ProcessState.ExitCode()
	} else if err != nil {
		result.ExitCode = 1
	}
	if err != nil {
		result.Error = err.Error()
		return result
	}
	var payload any
	if err := json.Unmarshal(out, &payload); err != nil {
		result.Error = fmt.Sprintf("version --json returned invalid JSON: %v", err)
		result.ExitCode = 1
		return result
	}
	result.Outcome = "success"
	return result
}

func writeBuildReceiptAtomic(path string, receipt buildReceipt) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".build-receipt-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(receipt); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}
