//go:build linux

package llamacppinterop

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

const (
	strixAuthorityStartupTimeout = 2 * time.Second
	strixAuthorityStableWindow   = 100 * time.Millisecond
)

var strixAuthorityEnvironment = []string{
	"LANG=C",
	"LC_ALL=C",
	"TZ=UTC",
}

type strixPinnedIdentity struct {
	role   string
	path   string
	file   *os.File
	digest string
	device uint64
	inode  uint64
	mapIt  bool
}

type linuxStrixComparatorAuthority struct {
	mu           sync.Mutex
	cmd          *exec.Cmd
	done         chan error
	files        []*os.File
	pinned       []*strixPinnedIdentity
	server       *strixPinnedIdentity
	allowed      map[strixFileIdentity]struct{}
	required     map[strixFileIdentity]struct{}
	rootFD       int
	pidfd        int
	validateHook func(int)
	startupHook  func(int)
	signal       func(int, int) error
	closed       bool
}

func init() {
	openStrixComparatorAuthorityPlatform = openStrixComparatorAuthorityLinux
}

func openStrixComparatorAuthorityLinux(options StrixComparatorAuthorityOptions) (_ *StrixComparatorAuthority, retErr error) {
	plan := ValidateStrixComparatorPlan(options.Manifest)
	if !plan.PlanValid {
		return nil, fmt.Errorf("open Strix comparator authority: invalid manifest: %s", strings.Join(plan.Reasons, ","))
	}
	if len(options.LoaderICD) == 0 {
		return nil, errors.New("open Strix comparator authority: at least one loader or ICD mapping is required")
	}
	for _, arg := range options.Arguments {
		if strings.IndexByte(arg, 0) >= 0 {
			return nil, errors.New("open Strix comparator authority: argument contains NUL")
		}
	}

	rootFD, err := unix.Open("/", unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open Strix comparator authority root: %w", err)
	}
	rootOwned := true
	defer func() {
		if rootOwned {
			_ = unix.Close(rootFD)
		}
	}()

	pinned := make([]*strixPinnedIdentity, 0, 4+len(options.LoaderICD)+len(options.Dependencies))
	defer func() {
		if retErr != nil {
			for _, item := range pinned {
				_ = item.file.Close()
			}
		}
	}()
	open := func(role, path, want string, executable, mapIt bool) error {
		item, err := openAnchoredStrixFile(rootFD, role, path, want, executable, mapIt)
		if err != nil {
			return err
		}
		for _, prior := range pinned {
			if prior.device == item.device && prior.inode == item.inode {
				_ = item.file.Close()
				return fmt.Errorf("open Strix comparator authority: %s aliases %s", role, prior.role)
			}
		}
		pinned = append(pinned, item)
		return nil
	}

	if err := open("source archive", options.SourceArchivePath, options.Manifest.SourceArchiveSHA256, false, false); err != nil {
		return nil, err
	}
	if err := open("build manifest", options.BuildManifestPath, options.Manifest.BuildManifestSHA256, false, false); err != nil {
		return nil, err
	}
	if err := verifyCanonicalStrixBuildManifest(pinned[len(pinned)-1].file); err != nil {
		return nil, err
	}
	if err := open("server binary", options.ServerBinaryPath, options.Manifest.ServerBinarySHA256, true, false); err != nil {
		return nil, err
	}
	server := pinned[len(pinned)-1]
	modelSHA256 := options.Manifest.ModelSHA256
	if options.testModelSHA256 != "" {
		modelSHA256 = options.testModelSHA256
	}
	if err := open("model", options.ModelPath, modelSHA256, false, false); err != nil {
		return nil, err
	}
	model := pinned[len(pinned)-1]
	for i, declaration := range options.LoaderICD {
		if err := open(fmt.Sprintf("loader/ICD[%d]", i), declaration.Path, declaration.SHA256, false, true); err != nil {
			return nil, err
		}
	}
	for i, declaration := range options.Dependencies {
		if err := open(fmt.Sprintf("dependency[%d]", i), declaration.Path, declaration.SHA256, false, true); err != nil {
			return nil, err
		}
	}

	if options.testAfterOpen != nil {
		if err := options.testAfterOpen(); err != nil {
			return nil, fmt.Errorf("open Strix comparator authority test boundary: %w", err)
		}
	}
	for _, item := range pinned {
		if err := revalidateAnchoredStrixFile(rootFD, item); err != nil {
			return nil, err
		}
	}

	extraFiles := make([]*os.File, 0, 2+len(options.LoaderICD)+len(options.Dependencies))
	extraFiles = append(extraFiles, server.file, model.file)
	allowedMaps := map[strixFileIdentity]struct{}{
		{device: server.device, inode: server.inode}: {},
		{device: model.device, inode: model.inode}:   {},
	}
	requiredMaps := map[strixFileIdentity]struct{}{
		{device: model.device, inode: model.inode}: {},
	}
	for _, item := range pinned {
		if !item.mapIt {
			continue
		}
		extraFiles = append(extraFiles, item.file)
		identity := strixFileIdentity{device: item.device, inode: item.inode}
		allowedMaps[identity] = struct{}{}
		requiredMaps[identity] = struct{}{}
	}

	cmd := exec.Command("/proc/self/fd/3", append([]string(nil), options.Arguments...)...)
	cmd.Dir = "/"
	cmd.Env = append([]string(nil), strixAuthorityEnvironment...)
	cmd.ExtraFiles = extraFiles
	cmd.Stdin = nil
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start held Strix server inode: %w", err)
	}
	pidfd, err := unix.PidfdOpen(cmd.Process.Pid, 0)
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, fmt.Errorf("open pidfd for held Strix server child: %w", err)
	}

	state := &linuxStrixComparatorAuthority{
		cmd:         cmd,
		done:        make(chan error, 1),
		files:       make([]*os.File, 0, len(pinned)),
		pinned:      pinned,
		server:      server,
		allowed:     allowedMaps,
		required:    requiredMaps,
		rootFD:      rootFD,
		pidfd:       pidfd,
		startupHook: options.testDuringStartupValidation,
		signal:      sendStrixPidfdSignal,
	}
	if options.testPidfdSignal != nil {
		state.signal = options.testPidfdSignal
	}
	rootOwned = false
	for _, item := range pinned {
		state.files = append(state.files, item.file)
	}
	go func() { state.done <- cmd.Wait() }()
	// From this point state owns the handles, including every error path.
	ownedPinned := pinned
	pinned = nil
	if err := waitForStrixChildAuthority(state, ownedPinned, rootFD); err != nil {
		terminal, closeErr := state.close()
		if !terminal {
			return nil, &StrixComparatorAuthorityCleanupError{
				cause:   errors.Join(err, closeErr),
				cleanup: state,
			}
		}
		if closeErr != nil {
			return nil, errors.Join(err, closeErr)
		}
		return nil, err
	}
	state.validateHook = options.testDuringValidation

	authority := &StrixComparatorAuthority{mu: new(sync.Mutex), platform: state}
	authority.self = authority
	return authority, nil
}

func openAnchoredStrixFile(rootFD int, role, path, want string, executable, mapIt bool) (*strixPinnedIdentity, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || path == "/" {
		return nil, fmt.Errorf("open Strix comparator authority: %s path must be clean and absolute", role)
	}
	if !validStrixSHA256(want) {
		return nil, fmt.Errorf("open Strix comparator authority: %s digest is invalid", role)
	}
	relative := strings.TrimPrefix(path, "/")
	fd, err := unix.Openat2(rootFD, relative, &unix.OpenHow{
		Flags:   unix.O_RDONLY | unix.O_CLOEXEC | unix.O_NOFOLLOW | unix.O_NONBLOCK,
		Resolve: unix.RESOLVE_BENEATH | unix.RESOLVE_NO_SYMLINKS | unix.RESOLVE_NO_MAGICLINKS,
	})
	if err != nil {
		return nil, fmt.Errorf("open Strix comparator authority %s: %w", role, err)
	}
	file := os.NewFile(uintptr(fd), path)
	closeOnError := true
	defer func() {
		if closeOnError {
			_ = file.Close()
		}
	}()
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat held Strix %s: %w", role, err)
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("open Strix comparator authority: %s is not a regular file", role)
	}
	if executable && info.Mode().Perm()&0o111 == 0 {
		return nil, fmt.Errorf("open Strix comparator authority: %s is not executable", role)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return nil, fmt.Errorf("open Strix comparator authority: %s identity unavailable", role)
	}
	digest, err := hashHeldStrixFile(file)
	if err != nil {
		return nil, fmt.Errorf("hash held Strix %s: %w", role, err)
	}
	if digest != want {
		return nil, fmt.Errorf("open Strix comparator authority: %s digest mismatch", role)
	}
	closeOnError = false
	return &strixPinnedIdentity{role: role, path: path, file: file, digest: digest, device: uint64(stat.Dev), inode: stat.Ino, mapIt: mapIt}, nil
}

func hashHeldStrixFile(file *os.File) (string, error) {
	info, err := file.Stat()
	if err != nil {
		return "", err
	}
	hash := sha256.New()
	if _, err := io.Copy(hash, io.NewSectionReader(file, 0, info.Size())); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func verifyCanonicalStrixBuildManifest(file *os.File) error {
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("read held Strix build manifest: %w", err)
	}
	data, err := io.ReadAll(file)
	if err != nil {
		return fmt.Errorf("read held Strix build manifest: %w", err)
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("rewind held Strix build manifest: %w", err)
	}
	if !canonicalJSON(data) {
		return errors.New("open Strix comparator authority: build manifest is not canonical JSON")
	}
	return nil
}

func revalidateAnchoredStrixFile(rootFD int, held *strixPinnedIdentity) error {
	reopened, err := openAnchoredStrixFile(rootFD, held.role, held.path, held.digest, held.role == "server binary", held.mapIt)
	if err != nil {
		return fmt.Errorf("revalidate held Strix %s: %w", held.role, err)
	}
	defer reopened.file.Close() //nolint:errcheck // read-only verification handle
	if reopened.device != held.device || reopened.inode != held.inode {
		return fmt.Errorf("revalidate held Strix %s: path inode changed", held.role)
	}
	return nil
}

type strixFileIdentity struct {
	device uint64
	inode  uint64
}

func waitForStrixChildAuthority(state *linuxStrixComparatorAuthority, pinned []*strixPinnedIdentity, rootFD int) error {
	deadline := time.Now().Add(strixAuthorityStartupTimeout)
	var stableSince time.Time
	var lastErr error
	for time.Now().Before(deadline) {
		select {
		case err := <-state.done:
			state.done <- err
			return fmt.Errorf("verify Strix comparator child: exited during startup: %v", err)
		default:
		}
		if err := verifyStrixAuthorityProcess(state); err != nil {
			lastErr = err
			stableSince = time.Time{}
		} else {
			if stableSince.IsZero() {
				stableSince = time.Now()
			}
			if time.Since(stableSince) >= strixAuthorityStableWindow {
				if state.startupHook != nil {
					state.startupHook(state.cmd.Process.Pid)
				}
				for _, item := range pinned {
					if err := revalidateAnchoredStrixFile(rootFD, item); err != nil {
						return err
					}
				}
				if err := verifyStrixAuthorityProcess(state); err != nil {
					return fmt.Errorf("final Strix comparator process verification: %w", err)
				}
				return nil
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	return fmt.Errorf("verify Strix comparator child: startup did not stabilize: %w", lastErr)
}

// verifyStrixAuthorityProcess brackets every /proc identity read with the
// pidfd, so an exit or PID reuse during executable/environment/map inspection
// cannot satisfy either startup admission or later capability use.
func verifyStrixAuthorityProcess(state *linuxStrixComparatorAuthority) error {
	if exited, err := strixPidfdExited(state.pidfd); err != nil {
		return fmt.Errorf("poll child pidfd before identity verification: %w", err)
	} else if exited {
		return errors.New("child exited before identity verification")
	}
	if err := verifyStrixChild(state.cmd.Process.Pid, state.server, state.allowed, state.required); err != nil {
		return err
	}
	if exited, err := strixPidfdExited(state.pidfd); err != nil {
		return fmt.Errorf("poll child pidfd after identity verification: %w", err)
	} else if exited {
		return errors.New("child exited during identity verification")
	}
	return nil
}

func verifyStrixChild(pid int, server *strixPinnedIdentity, allowed, required map[strixFileIdentity]struct{}) error {
	info, err := os.Stat(fmt.Sprintf("/proc/%d/exe", pid))
	if err != nil {
		return fmt.Errorf("stat child executable: %w", err)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || uint64(stat.Dev) != server.device || stat.Ino != server.inode {
		return errors.New("child executable is not the held server inode")
	}
	cwdInfo, err := os.Stat(fmt.Sprintf("/proc/%d/cwd", pid))
	if err != nil {
		return fmt.Errorf("stat child cwd: %w", err)
	}
	rootInfo, err := os.Stat("/")
	if err != nil {
		return fmt.Errorf("stat authority cwd: %w", err)
	}
	if !os.SameFile(cwdInfo, rootInfo) {
		return errors.New("child cwd escaped frozen root")
	}
	environ, err := os.ReadFile(fmt.Sprintf("/proc/%d/environ", pid))
	if err != nil {
		return fmt.Errorf("read child environment: %w", err)
	}
	wantEnvironment := []byte(strings.Join(strixAuthorityEnvironment, "\x00") + "\x00")
	if !bytes.Equal(environ, wantEnvironment) {
		return errors.New("child environment differs from frozen literal set")
	}

	file, err := os.Open(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		return fmt.Errorf("open child maps: %w", err)
	}
	defer file.Close() //nolint:errcheck // read-only proc handle
	seen := make(map[strixFileIdentity]struct{}, len(required))
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 6 || !strings.HasPrefix(fields[5], "/") {
			continue
		}
		deviceParts := strings.Split(fields[3], ":")
		if len(deviceParts) != 2 {
			return fmt.Errorf("malformed mapped device %q", fields[3])
		}
		major, majorErr := strconv.ParseUint(deviceParts[0], 16, 32)
		minor, minorErr := strconv.ParseUint(deviceParts[1], 16, 32)
		inode, inodeErr := strconv.ParseUint(fields[4], 10, 64)
		if majorErr != nil || minorErr != nil || inodeErr != nil || inode == 0 {
			return fmt.Errorf("malformed file-backed child map %q", scanner.Text())
		}
		identity := strixFileIdentity{device: unix.Mkdev(uint32(major), uint32(minor)), inode: inode}
		if _, ok := allowed[identity]; !ok {
			return fmt.Errorf("child mapped undeclared file %q", strings.Join(fields[5:], " "))
		}
		if _, ok := required[identity]; ok {
			seen[identity] = struct{}{}
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scan child maps: %w", err)
	}
	if len(seen) != len(required) {
		return fmt.Errorf("child mapped %d of %d declared dependencies", len(seen), len(required))
	}
	return nil
}

func (s *linuxStrixComparatorAuthority) pid() int {
	if s == nil || s.cmd == nil || s.cmd.Process == nil {
		return 0
	}
	return s.cmd.Process.Pid
}

func (s *linuxStrixComparatorAuthority) valid() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return false
	}
	select {
	case err := <-s.done:
		s.done <- err
		return false
	default:
	}
	if err := verifyStrixAuthorityProcess(s); err != nil {
		return false
	}
	if s.validateHook != nil {
		s.validateHook(s.cmd.Process.Pid)
	}
	for _, item := range s.pinned {
		digest, err := hashHeldStrixFile(item.file)
		if err != nil || digest != item.digest {
			return false
		}
		if err := revalidateAnchoredStrixFile(s.rootFD, item); err != nil {
			return false
		}
	}
	return verifyStrixAuthorityProcess(s) == nil
}

func (s *linuxStrixComparatorAuthority) close() (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return true, nil
	}
	finish := func(waitErr, signalErr error) (bool, error) {
		s.closed = true
		for _, file := range s.files {
			_ = file.Close()
		}
		_ = unix.Close(s.rootFD)
		_ = unix.Close(s.pidfd)
		if signalErr != nil {
			return true, fmt.Errorf("signal owned Strix comparator child: %w", signalErr)
		}
		if waitErr != nil {
			var exitErr *exec.ExitError
			if !errors.As(waitErr, &exitErr) {
				return true, waitErr
			}
		}
		return true, nil
	}
	waitDone := func(timeout time.Duration) (error, bool) {
		select {
		case err := <-s.done:
			return err, true
		case <-time.After(timeout):
			return nil, false
		}
	}

	select {
	case waitErr := <-s.done:
		return finish(waitErr, nil)
	default:
	}
	// The pidfd identifies exactly the direct child started above. Descendant
	// lifecycle is deliberately outside this leaf; no process group is ever
	// signaled, so an unrelated or inherited service cannot be torn down.
	if err := s.signal(s.pidfd, int(syscall.SIGTERM)); err != nil && !errors.Is(err, unix.ESRCH) {
		if waitErr, done := waitDone(25 * time.Millisecond); done {
			return finish(waitErr, err)
		}
		return false, fmt.Errorf("%w: pidfd SIGTERM: %v", ErrStrixComparatorAuthorityCleanupPending, err)
	}
	if waitErr, done := waitDone(500 * time.Millisecond); done {
		return finish(waitErr, nil)
	}
	if err := s.signal(s.pidfd, int(syscall.SIGKILL)); err != nil && !errors.Is(err, unix.ESRCH) {
		if waitErr, done := waitDone(25 * time.Millisecond); done {
			return finish(waitErr, err)
		}
		return false, fmt.Errorf("%w: pidfd SIGKILL: %v", ErrStrixComparatorAuthorityCleanupPending, err)
	}
	if waitErr, done := waitDone(2 * time.Second); done {
		return finish(waitErr, nil)
	}
	return false, fmt.Errorf("%w: child did not exit after pidfd SIGKILL", ErrStrixComparatorAuthorityCleanupPending)
}

func sendStrixPidfdSignal(pidfd, signal int) error {
	return unix.PidfdSendSignal(pidfd, syscall.Signal(signal), nil, 0)
}

func strixPidfdExited(pidfd int) (bool, error) {
	poll := []unix.PollFd{{Fd: int32(pidfd), Events: unix.POLLIN}}
	n, err := unix.Poll(poll, 0)
	if err != nil {
		return false, err
	}
	return n > 0 && poll[0].Revents&(unix.POLLIN|unix.POLLHUP|unix.POLLERR) != 0, nil
}
