package validate

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// This opt-in receipt contains private raw output and paths. The validator only
// returns it to its caller; it never publishes it or writes an external artifact.
type validateTestWitness struct {
	Status             string                        `json:"status"`
	Complete           bool                          `json:"complete"`
	Reason             string                        `json:"reason,omitempty"`
	ExecutionError     string                        `json:"execution_error,omitempty"`
	DecodeError        string                        `json:"decode_error,omitempty"`
	RawEncoding        string                        `json:"raw_encoding"`
	WaitDelayMS        int64                         `json:"wait_delay_ms"`
	WaitDelayExceeded  bool                          `json:"wait_delay_exceeded"`
	DrainStatus        string                        `json:"drain_status"`
	DescendantTeardown string                        `json:"descendant_teardown"`
	Started            bool                          `json:"started"`
	ExitCode           int                           `json:"exit_code"`
	TimedOut           bool                          `json:"timed_out"`
	Candidate          *validateCandidateFingerprint `json:"candidate,omitempty"`
	Argv               []string                      `json:"argv,omitempty"`
	Stdout             []byte                        `json:"stdout_base64"`
	Stderr             []byte                        `json:"stderr_base64"`
	StdoutTruncated    bool                          `json:"stdout_truncated"`
	StderrTruncated    bool                          `json:"stderr_truncated"`
	Events             []json.RawMessage             `json:"events,omitempty"`
	// Entries retain exact package/test identities, including subtests and helper
	// entries. A helper-return PASS is not evidence that it ran assertions.
	Tests []validateTestTerminal `json:"tests,omitempty"`
}

type validateTestTerminal struct {
	Package string `json:"package"`
	Test    string `json:"test"`
	Action  string `json:"action"`
}

type validateCandidateFile struct {
	Path   string `json:"path"`
	Mode   uint32 `json:"mode"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

type validateCandidateFingerprint struct {
	Root               string                  `json:"root"`
	Tip                string                  `json:"tip"`
	SHA256             string                  `json:"sha256"`
	Files              []validateCandidateFile `json:"files"`
	GoExecutable       string                  `json:"go_executable"`
	GoExecutableSHA256 string                  `json:"go_executable_sha256"`
	GoEnvironment      map[string]string       `json:"go_environment"`
	Limits             []string                `json:"limits"`
}

func validateEventConfiguration() error {
	if os.Getenv("GOWORK") != "off" || os.Getenv("GOTOOLCHAIN") != "local" || os.Getenv("GOENV") != "off" {
		return fmt.Errorf("TEST_EVENTS_ENV_REFUSED: requires GOWORK=off, GOTOOLCHAIN=local and GOENV=off")
	}
	if os.Getenv("FAK_WORKSPACE_ROOT") != "" {
		return fmt.Errorf("TEST_EVENTS_ENV_REFUSED: clear FAK_WORKSPACE_ROOT before launcher")
	}
	// Extra overlays/modfiles/tags can alter the candidate after fingerprinting.
	// First-version support is intentionally limited to the serial build flag.
	for _, flag := range strings.Fields(os.Getenv("GOFLAGS")) {
		if !strings.HasPrefix(flag, "-p=") {
			return fmt.Errorf("TEST_EVENTS_ENV_REFUSED: unsupported GOFLAGS")
		}
		if n, err := strconv.Atoi(strings.TrimPrefix(flag, "-p=")); err != nil || n < 1 {
			return fmt.Errorf("TEST_EVENTS_ENV_REFUSED: invalid parallelism")
		}
	}
	return nil
}

// Refuse external source scope before graph/build/test work. Walking does not
// follow links, and go mod edit -json only reads the owned module file.
func validateEventCandidateScope(ctx context.Context, root string) error {
	if err := validateEventConfiguration(); err != nil {
		return err
	}
	if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.Name() == ".git" && path == filepath.Join(root, ".git") {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("TEST_EVENTS_SOURCE_REFUSED: symlink %s", path)
		}
		return nil
	}); err != nil {
		return err
	}
	goPath, err := exec.LookPath("go")
	if err != nil {
		return err
	}
	moduleCommand := windowgate.CommandContext(ctx, goPath, "mod", "edit", "-json")
	moduleCommand.Dir = root
	windowgate.ConfigureBackgroundCommand(moduleCommand)
	moduleJSON, err := moduleCommand.Output()
	if err != nil {
		return fmt.Errorf("TEST_EVENTS_MODULE_REFUSED: %w", err)
	}
	var module struct {
		Replace []struct {
			New struct{ Path, Version string }
		}
	}
	if err := json.Unmarshal(moduleJSON, &module); err != nil {
		return err
	}
	for _, replacement := range module.Replace {
		if replacement.New.Version == "" {
			return fmt.Errorf("TEST_EVENTS_EXTERNAL_SOURCE_REFUSED: local go.mod replacement")
		}
	}
	return nil
}

func validateHashFile(ctx context.Context, path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	hash := sha256.New()
	buf := make([]byte, 128*1024)
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, readErr := f.Read(buf)
		if n > 0 {
			_, _ = hash.Write(buf[:n])
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", readErr
		}
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// Fingerprint actual copied bytes before tests, never the live source root or a
// post-test/truncated audit head. External workspace/local replacements refuse.
func fingerprintValidateCandidate(ctx context.Context, root, tip string) (*validateCandidateFingerprint, error) {
	if err := validateEventConfiguration(); err != nil {
		return nil, err
	}
	root, err := filepath.EvalSymlinks(root)
	if err != nil {
		return nil, err
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	temporary, err := filepath.Abs(os.TempDir())
	if err != nil {
		return nil, err
	}
	temporary, err = filepath.EvalSymlinks(temporary)
	if err != nil {
		return nil, err
	}
	relative, err := filepath.Rel(temporary, root)
	if err != nil || relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return nil, fmt.Errorf("TEST_EVENTS_ROOT_REFUSED: candidate outside temporary namespace")
	}
	_, err = os.Stat(filepath.Join(root, "go.mod"))
	if err != nil {
		return nil, err
	}
	goPath, err := exec.LookPath("go")
	if err != nil {
		return nil, err
	}
	goPath, err = filepath.EvalSymlinks(goPath)
	if err != nil {
		return nil, err
	}
	goPath, err = filepath.Abs(goPath)
	if err != nil {
		return nil, err
	}
	if err := validateEventCandidateScope(ctx, root); err != nil {
		return nil, err
	}
	goHash, err := validateHashFile(ctx, goPath)
	if err != nil {
		return nil, err
	}
	command := windowgate.CommandContext(ctx, goPath, "env", "-json", "GOWORK", "GOMOD", "GOFLAGS", "GOTOOLCHAIN", "GOOS", "GOARCH", "GOCACHE", "GOMODCACHE", "GOROOT", "GOENV", "CGO_ENABLED", "GOEXPERIMENT", "CC", "CXX", "CGO_CFLAGS", "CGO_CPPFLAGS", "CGO_CXXFLAGS", "CGO_LDFLAGS")
	command.Dir = root
	windowgate.ConfigureBackgroundCommand(command)
	out, err := command.Output()
	if err != nil {
		return nil, fmt.Errorf("TEST_EVENTS_GO_ENV_REFUSED: %w", err)
	}
	var environment map[string]string
	if err := json.Unmarshal(out, &environment); err != nil {
		return nil, err
	}
	if environment["GOWORK"] != "off" || filepath.Clean(environment["GOMOD"]) != filepath.Join(root, "go.mod") {
		return nil, fmt.Errorf("TEST_EVENTS_EXTERNAL_SOURCE_REFUSED: active workfile/module mismatch")
	}
	manifest := &validateCandidateFingerprint{Root: root, Tip: tip, GoExecutable: goPath, GoExecutableSHA256: goHash, GoEnvironment: environment, Limits: []string{
		"Native isolated candidate only; no active workfile, local module replacements, symlinks or extra Go overlays/modfiles/tags.",
		"Every regular candidate file is hashed except native-generated .git identity metadata; requested full tip and copied source bytes bind that identity.",
		"Downloaded modules, GOROOT and compiler cache are not byte-fingerprinted; Go module go.mod/go.sum and configured Go checksum policy remain dependency authority. Go executable is SHA-256 bound; external C compiler and system library bytes are outside this receipt, with their selected Go environment recorded.",
		"Tests are not filesystem-sandboxed by this receipt; only audited fixture/runtime-read scopes are admissible. Fingerprint binds pre-test bytes, not post-test mutations.",
	}}
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if rel == ".git" {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("TEST_EVENTS_SOURCE_REFUSED: symlink %s", rel)
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("TEST_EVENTS_SOURCE_REFUSED: nonregular file %s", rel)
		}
		hash, err := validateHashFile(ctx, path)
		if err != nil {
			return err
		}
		manifest.Files = append(manifest.Files, validateCandidateFile{Path: filepath.ToSlash(rel), Mode: uint32(info.Mode().Perm()), Size: info.Size(), SHA256: hash})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(manifest.Files, func(i, j int) bool { return manifest.Files[i].Path < manifest.Files[j].Path })
	encoded, err := json.Marshal(manifest.Files)
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(encoded)
	manifest.SHA256 = hex.EncodeToString(digest[:])
	return manifest, nil
}

// A cap never becomes silent loss: both truncated flags invalidate completeness.
// The writer continues draining the child so the deadline/exit semantics remain.
const validateEventStreamLimit = 16 * 1024 * 1024
const validateEventWaitDelay = 250 * time.Millisecond

type validateEventBuffer struct {
	bytes.Buffer
	truncated bool
	limit     int
}

func (b *validateEventBuffer) Write(p []byte) (int, error) {
	n := len(p)
	remaining := max(b.limit-b.Len(), 0)
	if len(p) > remaining {
		b.truncated = true
		p = p[:remaining]
	}
	_, _ = b.Buffer.Write(p)
	return n, nil
}

type validateEventIdentity struct {
	Action  string `json:"Action"`
	Package string `json:"Package"`
	Test    string `json:"Test"`
}

func decodeValidateTestEvents(stdout []byte, packages []string) ([]json.RawMessage, []validateTestTerminal, error) {
	wanted := map[string]bool{}
	for _, pkg := range packages {
		wanted[pkg] = true
	}
	ended := map[string]bool{}
	running := map[string]bool{}
	terminals := map[string]bool{}
	var events []json.RawMessage
	var tests []validateTestTerminal
	if !utf8.Valid(stdout) {
		return nil, nil, fmt.Errorf("Go JSON stream contains invalid UTF-8; raw bytes retained")
	}
	decoder := json.NewDecoder(bytes.NewReader(stdout))
	for {
		var raw json.RawMessage
		err := decoder.Decode(&raw)
		if err == io.EOF {
			break
		}
		if err != nil {
			return events, tests, fmt.Errorf("malformed Go JSON: %w", err)
		}
		var event validateEventIdentity
		if err := json.Unmarshal(raw, &event); err != nil || event.Action == "" {
			return events, tests, fmt.Errorf("invalid Go event object")
		}
		events = append(events, append(json.RawMessage(nil), raw...))
		if !wanted[event.Package] {
			continue
		}
		if ended[event.Package] {
			return events, tests, fmt.Errorf("event after package terminal: %s", event.Package)
		}
		key := event.Package + "\x00" + event.Test
		if event.Test != "" && terminals[key] {
			return events, tests, fmt.Errorf("event after test terminal: %s", event.Test)
		}
		if event.Action == "run" && event.Test != "" {
			if running[key] || terminals[key] {
				return events, tests, fmt.Errorf("duplicate test run: %s", event.Test)
			}
			if strings.Contains(event.Test, "/") {
				ancestor := event.Test
				found := false
				for strings.Contains(ancestor, "/") {
					ancestor = ancestor[:strings.LastIndex(ancestor, "/")]
					parentKey := event.Package + "\x00" + ancestor
					if running[parentKey] {
						if terminals[parentKey] {
							return events, tests, fmt.Errorf("subtest after parent terminal: %s", event.Test)
						}
						found = true
						break
					}
				}
				if !found {
					return events, tests, fmt.Errorf("subtest without running parent: %s", event.Test)
				}
			}
			running[key] = true
		}
		if event.Action == "pass" || event.Action == "skip" || event.Action == "fail" {
			if event.Test == "" {
				if ended[event.Package] {
					return events, tests, fmt.Errorf("duplicate package terminal")
				}
				for active := range running {
					if strings.HasPrefix(active, event.Package+"\x00") && !terminals[active] {
						return events, tests, fmt.Errorf("package terminal before test terminal: %s", active)
					}
				}
				ended[event.Package] = true
				continue
			}
			if !running[key] || terminals[key] {
				return events, tests, fmt.Errorf("test terminal without unique run: %s", event.Test)
			}
			for active := range running {
				if strings.HasPrefix(active, key+"/") && !terminals[active] {
					return events, tests, fmt.Errorf("parent terminal before subtest terminal: %s", active)
				}
			}
			terminals[key] = true
			tests = append(tests, validateTestTerminal{Package: event.Package, Test: event.Test, Action: event.Action})
		}
	}
	for _, pkg := range packages {
		if !ended[pkg] {
			return events, tests, fmt.Errorf("missing package terminal: %s", pkg)
		}
	}
	for key := range running {
		if !terminals[key] {
			return events, tests, fmt.Errorf("missing test terminal: %s", key)
		}
	}
	if len(running) == 0 {
		return events, tests, fmt.Errorf("no individual tests ran")
	}
	return events, tests, nil
}

func executeValidateEventTests(ctx context.Context, witness *validateTestWitness, packages []string) bool {
	stdout := &validateEventBuffer{limit: validateEventStreamLimit}
	stderr := &validateEventBuffer{limit: validateEventStreamLimit}
	cmd := windowgate.CommandContext(ctx, witness.Argv[0], witness.Argv[1:]...)
	// WaitDelay bounds inherited-pipe drain after cancellation or direct-child
	// exit. It does not prove that all descendants were terminated.
	cmd.WaitDelay = validateEventWaitDelay
	witness.WaitDelayMS = validateEventWaitDelay.Milliseconds()
	witness.DescendantTeardown = "unproven"
	witness.RawEncoding = "base64"
	witness.DrainStatus = "unknown"
	cmd.Dir = witness.Candidate.Root
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	windowgate.ConfigureBackgroundCommand(cmd)
	err := cmd.Start()
	if err == nil {
		witness.Started = true
		err = cmd.Wait()
		if err == nil {
			witness.DrainStatus = "complete"
		}
	}
	witness.Stdout = append([]byte(nil), stdout.Bytes()...)
	witness.Stderr = append([]byte(nil), stderr.Bytes()...)
	witness.StdoutTruncated = stdout.truncated
	witness.StderrTruncated = stderr.truncated
	witness.ExitCode = -1
	if cmd.ProcessState != nil {
		witness.ExitCode = cmd.ProcessState.ExitCode()
	}
	if err != nil {
		witness.ExecutionError = err.Error()
	}
	var errDecode error
	witness.Events, witness.Tests, errDecode = decodeValidateTestEvents(witness.Stdout, packages)
	if errDecode != nil {
		witness.DecodeError = errDecode.Error()
	}
	witness.WaitDelayExceeded = errors.Is(err, exec.ErrWaitDelay)
	reportedFailure := false
	for _, raw := range witness.Events {
		var event validateEventIdentity
		if json.Unmarshal(raw, &event) == nil && event.Action == "fail" {
			reportedFailure = true
		}
	}
	switch {
	case ctx.Err() != nil:
		witness.Status = "timeout"
		witness.TimedOut = true
		witness.Reason = ctx.Err().Error()
	case !witness.Started:
		witness.Status = "unrun"
		witness.Reason = witness.ExecutionError
	case witness.WaitDelayExceeded:
		witness.Status = "incomplete"
		witness.Reason = witness.ExecutionError
	case stdout.truncated || stderr.truncated:
		witness.Status = "incomplete"
		witness.Reason = "test output exceeded private stream limit"
	case errDecode != nil:
		witness.Status = "incomplete"
		if witness.ExecutionError != "" {
			witness.Reason = witness.ExecutionError
		} else {
			witness.Reason = witness.DecodeError
		}
	default:
		// Cmd.Wait can suppress ErrWaitDelay behind a nonzero child exit.
		// Valid terminals alone cannot establish EOF on either captured pipe.
		witness.Complete = err == nil
		if err != nil || reportedFailure {
			witness.Status = "failed"
			if err != nil {
				witness.Reason = err.Error()
			} else {
				witness.Reason = "Go events report failure despite exit 0"
			}
		} else {
			witness.Status = "complete"
		}
	}
	return err == nil && witness.Complete && witness.Status == "complete"
}

func runValidateEventTestsPhase(ctx context.Context, stdout io.Writer, res *validateResult, recorder *validateRecorder, dir, tip, testRun string, fileToPkg map[string]string, asJSON bool) (affectedtests.TestObservation, int, bool) {
	witness := &validateTestWitness{Status: "unrun", ExitCode: -1}
	res.TestEvents = witness
	phase := recorder.start("test")
	fail := func(reason string) {
		witness.Reason = reason
		res.OK = false
		recordValidateFailure(res, phase, "test-events", reason, errors.New(reason))
	}
	if len(res.Tested) == 0 {
		fail("TEST_EVENTS_UNRUN: no affected test-bearing packages")
		return affectedtests.TestObservation{}, 0, false
	}
	candidate, err := fingerprintValidateCandidate(ctx, dir, tip)
	if err != nil {
		fail(err.Error())
		if ctx.Err() != nil {
			witness.TimedOut = true
			witness.Status = "timeout"
			return affectedtests.TestObservation{}, finishValidateTimeout(stdout, res, recorder, "test", asJSON), true
		}
		return affectedtests.TestObservation{}, 0, false
	}
	witness.Candidate = candidate
	res.Runner = "go test"
	args := validateJSONTestArgs(validateTestArgs(testRun, validateGoTestTimeout(ctx, validateNow()), packagePatternsForRoot(dir, res.Tested, fileToPkg)))
	witness.Argv = append([]string{candidate.GoExecutable}, args...)
	ok := executeValidateEventTests(ctx, witness, res.Tested)
	if code, timedOut := finishValidateContextPhase(stdout, res, recorder, phase, "test", asJSON); timedOut {
		return affectedtests.TestObservation{}, code, true
	}
	if ok {
		phase.finish(nil)
	} else {
		fail(witness.Reason)
	}
	return affectedtests.TestObservation{}, 0, false
}
