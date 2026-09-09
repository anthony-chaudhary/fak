//go:build linux

package llamacppinterop

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
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
	mu             sync.Mutex
	cmd            *exec.Cmd
	done           chan error
	files          []*os.File
	pinned         []*strixPinnedIdentity
	server         *strixPinnedIdentity
	allowed        map[strixFileIdentity]struct{}
	required       map[strixFileIdentity]struct{}
	rootFD         int
	pidfd          int
	validateHook   func(int)
	startupHook    func(int)
	exchangeHook   func(int, *net.TCPConn)
	signal         func(int, int) error
	modelClaim     string
	radvDigest     string
	resourceActive bool
	closed         bool
}

type strixTCPAuthoritySnapshot struct {
	clientInode   uint64
	acceptedInode uint64
	listenerInode uint64
	clientLocal   strixIPv4Endpoint
	clientRemote  strixIPv4Endpoint
}

type strixIPv4Endpoint struct {
	address [4]byte
	port    uint16
}

type linuxStrixComparatorExchangeAuthority struct {
	state      *linuxStrixComparatorAuthority
	reference  strixComparatorApprovedReference
	connection *net.TCPConn
	snapshot   strixTCPAuthoritySnapshot
}

type strixComparatorResourceSnapshot struct {
	processPeakBytes uint64
	deviceBytes      uint64
	engineActiveNS   uint64
	engineCounters   map[string]uint64
	drmFD            int
	drmClientID      string
	drmPDev          string
	renderTarget     string
	renderDevice     uint64
	radvIdentity     string
}

type linuxStrixComparatorResourceWindow struct {
	exchange *linuxStrixComparatorExchangeAuthority
	initial  strixComparatorResourceSnapshot

	mu             sync.Mutex
	latest         strixComparatorResourceSnapshot
	processPeak    uint64
	devicePeak     uint64
	sampleErr      error
	stop           chan struct{}
	done           chan struct{}
	ctx            context.Context
	releaseOnce    sync.Once
	stopOnce       sync.Once
	testBeforeMint func()
}

type linuxStrixComparatorResourceAuthority struct {
	value StrixComparatorResourceObservation
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
		cmd:          cmd,
		done:         make(chan error, 1),
		files:        make([]*os.File, 0, len(pinned)),
		pinned:       pinned,
		server:       server,
		allowed:      allowedMaps,
		required:     requiredMaps,
		rootFD:       rootFD,
		pidfd:        pidfd,
		startupHook:  options.testDuringStartupValidation,
		exchangeHook: options.testDuringExchangeFinalValidation,
		signal:       sendStrixPidfdSignal,
		modelClaim:   options.Manifest.ModelSHA256,
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

// detectPinnedStrixRADVImplementation recognizes the executed Mesa RADV ICD
// from held ELF bytes, never from a caller-provided path or environment name.
// Returning an empty digest keeps ordinary software authority usable while
// making resource authority impossible until an unambiguous RADV image is
// present in the already-required mapped dependency set.
func detectPinnedStrixRADVImplementation(pinned []*strixPinnedIdentity) string {
	found := ""
	for _, item := range pinned {
		if item == nil || !item.mapIt || item.file == nil || item.digest == "" {
			continue
		}
		info, err := item.file.Stat()
		if err != nil || info.Size() < 4 {
			continue
		}
		header := make([]byte, 4)
		if _, err := item.file.ReadAt(header, 0); err != nil || !bytes.Equal(header, []byte{0x7f, 'E', 'L', 'F'}) {
			continue
		}
		matched, err := heldStrixFileContainsAll(item.file, info.Size(), []byte("radv_"), []byte("vk_icdGetInstanceProcAddr"))
		if err != nil || !matched {
			continue
		}
		if found != "" {
			return ""
		}
		found = item.digest
	}
	return found
}

func heldStrixFileContainsAll(file *os.File, size int64, markers ...[]byte) (bool, error) {
	found := make([]bool, len(markers))
	maxMarker := 0
	for _, marker := range markers {
		if len(marker) > maxMarker {
			maxMarker = len(marker)
		}
	}
	reader := io.NewSectionReader(file, 0, size)
	buffer := make([]byte, 64<<10)
	tail := make([]byte, 0, maxMarker)
	for {
		n, err := reader.Read(buffer)
		if n > 0 {
			window := append(append([]byte(nil), tail...), buffer[:n]...)
			all := true
			for i, marker := range markers {
				if !found[i] && bytes.Contains(window, marker) {
					found[i] = true
				}
				all = all && found[i]
			}
			if all {
				return true, nil
			}
			keep := maxMarker - 1
			if keep > len(window) {
				keep = len(window)
			}
			tail = append(tail[:0], window[len(window)-keep:]...)
		}
		if errors.Is(err, io.EOF) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
	}
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

func (s *linuxStrixComparatorAuthority) bindApprovedReferenceAndAcceptedConnection(reference strixComparatorApprovedReference, connection *net.TCPConn) (strixComparatorExchangeAuthorityPlatform, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	snapshot, err := verifyStrixExchangeLocked(s, reference, connection)
	if err != nil {
		return nil, fmt.Errorf("bind Strix comparator exchange: %w", err)
	}
	return &linuxStrixComparatorExchangeAuthority{state: s, reference: reference, connection: connection, snapshot: snapshot}, nil
}

func (e *linuxStrixComparatorExchangeAuthority) valid() bool {
	if e == nil || e.state == nil || e.connection == nil {
		return false
	}
	e.state.mu.Lock()
	defer e.state.mu.Unlock()
	snapshot, err := verifyStrixExchangeLocked(e.state, e.reference, e.connection)
	return err == nil && snapshot == e.snapshot
}

func (e *linuxStrixComparatorExchangeAuthority) beginResourceWindow(ctx context.Context) (strixComparatorResourceWindowPlatform, error) {
	if e == nil || e.state == nil || e.connection == nil {
		return nil, ErrInvalidStrixComparatorAuthority
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("begin Strix comparator resource window: %w", err)
	}
	e.state.mu.Lock()
	if err := claimStrixResourceWindowLocked(e.state); err != nil {
		e.state.mu.Unlock()
		return nil, err
	}
	snapshot, err := verifyStrixExchangeLocked(e.state, e.reference, e.connection)
	if err == nil && snapshot != e.snapshot {
		err = errors.New("accepted connection identity changed before resource window")
	}
	var initial strixComparatorResourceSnapshot
	if err == nil {
		e.state.radvDigest = detectPinnedStrixRADVImplementation(e.state.pinned)
		if e.state.radvDigest == "" {
			err = errors.New("executed child has no unambiguous observed RADV ELF implementation")
		}
	}
	if err == nil {
		initial, err = inspectStrixComparatorResources(e.state, e.reference, true)
	}
	if err == nil {
		trailing, trailingErr := verifyStrixExchangeLocked(e.state, e.reference, e.connection)
		if trailingErr != nil {
			err = trailingErr
		} else if trailing != e.snapshot {
			err = errors.New("accepted connection identity changed across resource-window begin")
		}
	}
	if err != nil {
		e.state.resourceActive = false
	}
	e.state.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("begin Strix comparator resource window: %w", err)
	}
	w := &linuxStrixComparatorResourceWindow{
		exchange:    e,
		initial:     initial,
		latest:      initial,
		processPeak: initial.processPeakBytes,
		devicePeak:  initial.deviceBytes,
		stop:        make(chan struct{}),
		done:        make(chan struct{}),
		ctx:         ctx,
	}
	go w.sample()
	return w, nil
}

func claimStrixResourceWindowLocked(state *linuxStrixComparatorAuthority) error {
	if state == nil || state.resourceActive {
		return errors.New("begin Strix comparator resource window: comparator child already has an active resource window")
	}
	state.resourceActive = true
	return nil
}

func (w *linuxStrixComparatorResourceWindow) sample() {
	defer close(w.done)
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	sampleCount := 0
	for {
		select {
		case <-w.stop:
			return
		case <-w.ctx.Done():
			w.record(strixComparatorResourceSnapshot{}, w.ctx.Err())
			w.releaseActive()
			return
		case <-ticker.C:
			sampleCount++
			sample, err := inspectStrixComparatorResources(w.exchange.state, w.exchange.reference, sampleCount%20 == 0)
			w.record(sample, err)
			if err != nil {
				select {
				case <-w.stop:
					return
				case <-w.ctx.Done():
					w.releaseActive()
					return
				}
			}
		}
	}
}

func (w *linuxStrixComparatorResourceWindow) record(sample strixComparatorResourceSnapshot, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.sampleErr != nil {
		return
	}
	if err != nil {
		w.sampleErr = err
		return
	}
	if err := validateStrixResourceProgress(w.initial, w.latest, sample); err != nil {
		w.sampleErr = err
		return
	}
	w.latest = sample
	if sample.processPeakBytes > w.processPeak {
		w.processPeak = sample.processPeakBytes
	}
	if sample.deviceBytes > w.devicePeak {
		w.devicePeak = sample.deviceBytes
	}
}

func (w *linuxStrixComparatorResourceWindow) finalize() (strixComparatorResourceAuthorityPlatform, error) {
	if w == nil || w.exchange == nil {
		return nil, ErrInvalidStrixComparatorAuthority
	}
	w.stopOnce.Do(func() { close(w.stop) })
	<-w.done
	if err := w.ctx.Err(); err != nil {
		w.releaseActive()
		return nil, fmt.Errorf("finalize Strix comparator resource window after sampler join: %w", err)
	}
	w.mu.Lock()
	sampleErr := w.sampleErr
	w.mu.Unlock()
	if sampleErr != nil {
		w.releaseActive()
		return nil, fmt.Errorf("finalize Strix comparator resource window: %w", sampleErr)
	}

	w.exchange.state.mu.Lock()
	socket, err := verifyStrixExchangeLocked(w.exchange.state, w.exchange.reference, w.exchange.connection)
	if err == nil && socket != w.exchange.snapshot {
		err = errors.New("accepted connection identity changed after resource window")
	}
	var final strixComparatorResourceSnapshot
	if err == nil {
		final, err = inspectStrixComparatorResources(w.exchange.state, w.exchange.reference, true)
	}
	if err == nil {
		trailing, trailingErr := verifyStrixExchangeLocked(w.exchange.state, w.exchange.reference, w.exchange.connection)
		if trailingErr != nil {
			err = trailingErr
		} else if trailing != w.exchange.snapshot {
			err = errors.New("accepted connection identity changed across resource-window finalize")
		}
	}
	w.exchange.state.resourceActive = false
	w.exchange.state.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("finalize Strix comparator resource window: %w", err)
	}
	w.record(final, nil)

	w.mu.Lock()
	defer w.mu.Unlock()
	if w.sampleErr != nil {
		return nil, fmt.Errorf("finalize Strix comparator resource window: %w", w.sampleErr)
	}
	value, err := finalizeStrixResourceObservation(w.initial, w.latest, w.processPeak, w.devicePeak)
	if err != nil {
		return nil, fmt.Errorf("finalize Strix comparator resource window: %w", err)
	}
	return w.mintResourceAuthority(value)
}

func (w *linuxStrixComparatorResourceWindow) mintResourceAuthority(value StrixComparatorResourceObservation) (strixComparatorResourceAuthorityPlatform, error) {
	if w.testBeforeMint != nil {
		w.testBeforeMint()
	}
	if err := w.ctx.Err(); err != nil {
		return nil, fmt.Errorf("finalize Strix comparator resource window before authority mint: %w", err)
	}
	return &linuxStrixComparatorResourceAuthority{value: value}, nil
}

func (w *linuxStrixComparatorResourceWindow) cancel() error {
	if w == nil || w.exchange == nil {
		return ErrInvalidStrixComparatorAuthority
	}
	w.stopOnce.Do(func() { close(w.stop) })
	<-w.done
	w.releaseActive()
	return nil
}

func (w *linuxStrixComparatorResourceWindow) releaseActive() {
	w.releaseOnce.Do(func() {
		w.exchange.state.mu.Lock()
		w.exchange.state.resourceActive = false
		w.exchange.state.mu.Unlock()
	})
}

func (a *linuxStrixComparatorResourceAuthority) valid() bool {
	return a != nil && a.value.ProcessPeakBytes > 0 && a.value.DevicePeakBytes > 0 && a.value.GPUEngineActiveNS > 0 && a.value.RADVDeviceIdentity != ""
}

func (a *linuxStrixComparatorResourceAuthority) observation() StrixComparatorResourceObservation {
	if !a.valid() {
		return StrixComparatorResourceObservation{}
	}
	return a.value
}

func validateStrixResourceProgress(initial, prior, next strixComparatorResourceSnapshot) error {
	if initial.drmFD != next.drmFD || initial.drmClientID != next.drmClientID || initial.drmPDev != next.drmPDev ||
		initial.renderTarget != next.renderTarget || initial.renderDevice != next.renderDevice || initial.radvIdentity != next.radvIdentity {
		return errors.New("owned RADV resource identity changed during request window")
	}
	if next.processPeakBytes < prior.processPeakBytes || len(initial.engineCounters) == 0 || len(initial.engineCounters) != len(next.engineCounters) || len(prior.engineCounters) != len(next.engineCounters) {
		return errors.New("resource counter reset or became stale during request window")
	}
	for engine, initialValue := range initial.engineCounters {
		priorValue, priorOK := prior.engineCounters[engine]
		nextValue, nextOK := next.engineCounters[engine]
		if !priorOK || !nextOK || priorValue < initialValue || nextValue < priorValue {
			return fmt.Errorf("GPU-engine %s counter reset or changed identity during request window", engine)
		}
	}
	return nil
}

func finalizeStrixResourceObservation(initial, latest strixComparatorResourceSnapshot, processPeak, devicePeak uint64) (StrixComparatorResourceObservation, error) {
	if latest.engineActiveNS <= initial.engineActiveNS {
		return StrixComparatorResourceObservation{}, errors.New("no owned GPU-engine activity observed")
	}
	value := StrixComparatorResourceObservation{
		ProcessPeakBytes:   processPeak,
		DevicePeakBytes:    devicePeak,
		GPUEngineActiveNS:  latest.engineActiveNS - initial.engineActiveNS,
		RADVDeviceIdentity: initial.radvIdentity,
	}
	if value.ProcessPeakBytes == 0 || value.DevicePeakBytes == 0 || value.GPUEngineActiveNS == 0 || value.RADVDeviceIdentity == "" {
		return StrixComparatorResourceObservation{}, errors.New("incomplete resource evidence")
	}
	return value, nil
}

func inspectStrixComparatorResources(state *linuxStrixComparatorAuthority, reference strixComparatorApprovedReference, checkShared bool) (strixComparatorResourceSnapshot, error) {
	if state == nil || state.cmd == nil || state.cmd.Process == nil {
		return strixComparatorResourceSnapshot{}, ErrInvalidStrixComparatorAuthority
	}
	pid := state.cmd.Process.Pid
	if exited, err := strixPidfdExited(state.pidfd); err != nil {
		return strixComparatorResourceSnapshot{}, fmt.Errorf("poll child pidfd for resource sample: %w", err)
	} else if exited {
		return strixComparatorResourceSnapshot{}, errors.New("comparator child exited during resource sample")
	}
	if !validStrixSHA256(reference.loaderSetSHA256) || !validStrixSHA256(reference.dependencySetSHA256) {
		return strixComparatorResourceSnapshot{}, errors.New("approved RADV implementation identity is unavailable")
	}
	if !validStrixSHA256(state.radvDigest) {
		return strixComparatorResourceSnapshot{}, errors.New("executed child has no unambiguous observed RADV ELF implementation")
	}

	processPeak, err := readStrixProcessPeak(pid)
	if err != nil {
		return strixComparatorResourceSnapshot{}, err
	}
	fdDir := fmt.Sprintf("/proc/%d/fd", pid)
	entries, err := os.ReadDir(fdDir)
	if err != nil {
		return strixComparatorResourceSnapshot{}, fmt.Errorf("read child fd directory for DRM ownership: %w", err)
	}
	candidates := make([]strixComparatorResourceSnapshot, 0, 1)
	for _, entry := range entries {
		fd, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		linkPath := filepath.Join(fdDir, entry.Name())
		target, err := os.Readlink(linkPath)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return strixComparatorResourceSnapshot{}, fmt.Errorf("read child fd %d target: %w", fd, err)
		}
		if !strings.HasPrefix(target, "/dev/dri/renderD") {
			continue
		}
		if strings.HasSuffix(target, " (deleted)") {
			return strixComparatorResourceSnapshot{}, fmt.Errorf("child render fd %d refers to a deleted device", fd)
		}
		info, err := os.Stat(linkPath)
		if err != nil {
			return strixComparatorResourceSnapshot{}, fmt.Errorf("stat child render fd %d: %w", fd, err)
		}
		stat, ok := info.Sys().(*syscall.Stat_t)
		if !ok || info.Mode()&os.ModeDevice == 0 || info.Mode()&os.ModeCharDevice == 0 {
			return strixComparatorResourceSnapshot{}, fmt.Errorf("child render fd %d has no device identity", fd)
		}
		fdinfo, err := os.ReadFile(fmt.Sprintf("/proc/%d/fdinfo/%d", pid, fd))
		if err != nil {
			return strixComparatorResourceSnapshot{}, fmt.Errorf("read child render fdinfo %d: %w", fd, err)
		}
		fields, err := parseStrixDRMFDInfo(fdinfo)
		if err != nil {
			return strixComparatorResourceSnapshot{}, fmt.Errorf("parse child render fdinfo %d: %w", fd, err)
		}
		afterTarget, err := os.Readlink(linkPath)
		if err != nil {
			return strixComparatorResourceSnapshot{}, fmt.Errorf("re-read child render fd %d target: %w", fd, err)
		}
		afterInfo, err := os.Stat(linkPath)
		if err != nil {
			return strixComparatorResourceSnapshot{}, fmt.Errorf("re-stat child render fd %d: %w", fd, err)
		}
		afterStat, ok := afterInfo.Sys().(*syscall.Stat_t)
		if !ok || afterTarget != target || afterStat.Rdev != stat.Rdev {
			return strixComparatorResourceSnapshot{}, fmt.Errorf("child render fd %d identity changed during sample", fd)
		}
		if fields.driver != "amdgpu" {
			continue
		}
		candidates = append(candidates, strixComparatorResourceSnapshot{
			processPeakBytes: processPeak,
			deviceBytes:      fields.deviceBytes,
			engineActiveNS:   fields.engineActiveNS,
			engineCounters:   fields.engineCounters,
			drmFD:            fd,
			drmClientID:      fields.clientID,
			drmPDev:          fields.pdev,
			renderTarget:     target,
			renderDevice:     uint64(stat.Rdev),
			radvIdentity:     fmt.Sprintf("radv-%s-amdgpu-pci-%s-render-%d", state.radvDigest, fields.pdev, uint64(stat.Rdev)),
		})
	}
	if len(candidates) != 1 {
		return strixComparatorResourceSnapshot{}, fmt.Errorf("child owns %d unambiguous AMDGPU render clients", len(candidates))
	}
	result := candidates[0]
	if result.deviceBytes == 0 {
		return strixComparatorResourceSnapshot{}, errors.New("child AMDGPU client reports zero device memory")
	}
	if checkShared {
		owners, err := strixDRMClientOwners(pid, result.drmPDev, result.drmClientID)
		if err != nil {
			return strixComparatorResourceSnapshot{}, err
		}
		if err := validateStrixDRMOwners(owners, pid); err != nil {
			return strixComparatorResourceSnapshot{}, err
		}
	}
	if exited, err := strixPidfdExited(state.pidfd); err != nil {
		return strixComparatorResourceSnapshot{}, fmt.Errorf("poll child pidfd after resource sample: %w", err)
	} else if exited {
		return strixComparatorResourceSnapshot{}, errors.New("comparator child exited after resource sample")
	}
	return result, nil
}

type strixDRMFDInfo struct {
	driver         string
	pdev           string
	clientID       string
	deviceBytes    uint64
	engineActiveNS uint64
	engineCounters map[string]uint64
}

func parseStrixDRMFDInfo(data []byte) (strixDRMFDInfo, error) {
	result := strixDRMFDInfo{engineCounters: make(map[string]uint64)}
	seen := make(map[string]struct{})
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		relevant := key == "drm-driver" || key == "drm-pdev" || key == "drm-client-id" || key == "drm-memory-vram" || key == "drm-memory-gtt" || strings.HasPrefix(key, "drm-engine-")
		if relevant {
			if _, duplicate := seen[key]; duplicate {
				return strixDRMFDInfo{}, fmt.Errorf("duplicate %s field", key)
			}
			seen[key] = struct{}{}
		}
		switch {
		case key == "drm-driver":
			result.driver = value
		case key == "drm-pdev":
			result.pdev = value
		case key == "drm-client-id":
			result.clientID = value
		case key == "drm-memory-vram" || key == "drm-memory-gtt":
			amount, err := parseStrixResourceAmount(value, "KiB")
			if err != nil {
				return strixDRMFDInfo{}, fmt.Errorf("%s: %w", key, err)
			}
			if ^uint64(0)-result.deviceBytes < amount {
				return strixDRMFDInfo{}, errors.New("device-memory counter overflow")
			}
			result.deviceBytes += amount
		case strings.HasPrefix(key, "drm-engine-"):
			amount, err := parseStrixResourceAmount(value, "ns")
			if err != nil {
				return strixDRMFDInfo{}, fmt.Errorf("%s: %w", key, err)
			}
			if ^uint64(0)-result.engineActiveNS < amount {
				return strixDRMFDInfo{}, errors.New("GPU-engine counter overflow")
			}
			result.engineActiveNS += amount
			result.engineCounters[key] = amount
		}
	}
	if err := scanner.Err(); err != nil {
		return strixDRMFDInfo{}, err
	}
	if result.driver == "" || result.pdev == "" || result.clientID == "" || len(result.engineCounters) == 0 {
		return strixDRMFDInfo{}, errors.New("incomplete DRM ownership identity")
	}
	return result, nil
}

func parseStrixResourceAmount(value, unit string) (uint64, error) {
	fields := strings.Fields(value)
	if len(fields) != 2 || fields[1] != unit {
		return 0, errors.New("unexpected resource counter unit")
	}
	amount, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil {
		return 0, err
	}
	if unit == "KiB" || unit == "kB" {
		if amount > ^uint64(0)/1024 {
			return 0, errors.New("resource counter overflow")
		}
		amount *= 1024
	}
	return amount, nil
}

func validateStrixDRMOwners(owners map[int]int, childPID int) error {
	if len(owners) != 1 || owners[childPID] != 1 {
		return errors.New("AMDGPU client identity is shared or ambiguous")
	}
	return nil
}

func readStrixProcessPeak(pid int) (uint64, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return 0, fmt.Errorf("read child status for process peak: %w", err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "VmHWM:") {
			continue
		}
		amount, err := parseStrixResourceAmount(strings.TrimSpace(strings.TrimPrefix(line, "VmHWM:")), "kB")
		if err != nil {
			return 0, fmt.Errorf("parse child process peak: %w", err)
		}
		if amount == 0 {
			return 0, errors.New("child process peak is zero")
		}
		return amount, nil
	}
	return 0, errors.New("child status omitted process peak")
}

func strixDRMClientOwners(childPID int, pdev, clientID string) (map[int]int, error) {
	mountInfo, err := os.ReadFile("/proc/self/mountinfo")
	if err != nil {
		return nil, fmt.Errorf("read proc mount authority: %w", err)
	}
	return strixDRMClientOwnersWithMountInfo(childPID, pdev, clientID, mountInfo)
}

func strixDRMClientOwnersWithMountInfo(childPID int, pdev, clientID string, mountInfo []byte) (map[int]int, error) {
	if err := verifyStrixGlobalProcMount(mountInfo); err != nil {
		return nil, err
	}
	procEntries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, fmt.Errorf("read proc for DRM ownership: %w", err)
	}
	owners := make(map[int]int)
	for _, procEntry := range procEntries {
		pid, err := strconv.Atoi(procEntry.Name())
		if err != nil {
			continue
		}
		fdinfos, err := os.ReadDir(filepath.Join("/proc", procEntry.Name(), "fdinfo"))
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, fmt.Errorf("read process %d fdinfo for global DRM ownership: %w", pid, err)
		}
		for _, fdinfo := range fdinfos {
			data, err := os.ReadFile(filepath.Join("/proc", procEntry.Name(), "fdinfo", fdinfo.Name()))
			if err != nil {
				if errors.Is(err, os.ErrNotExist) {
					continue
				}
				return nil, fmt.Errorf("read process %d fdinfo %s for global DRM ownership: %w", pid, fdinfo.Name(), err)
			}
			fields, err := parseStrixDRMFDInfo(data)
			if err != nil || fields.driver != "amdgpu" || fields.pdev != pdev || fields.clientID != clientID {
				continue
			}
			owners[pid]++
		}
	}
	if owners[childPID] == 0 {
		return nil, errors.New("child DRM client disappeared during ownership scan")
	}
	return owners, nil
}

func verifyStrixGlobalProcMount(data []byte) error {
	matches := 0
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 5 {
			continue
		}
		if mountPoint := fields[4]; strings.HasPrefix(mountPoint, "/proc/") {
			pidSegment := strings.TrimPrefix(mountPoint, "/proc/")
			if slash := strings.IndexByte(pidSegment, '/'); slash >= 0 {
				pidSegment = pidSegment[:slash]
			}
			allDigits := pidSegment != ""
			for _, character := range pidSegment {
				if character < '0' || character > '9' {
					allDigits = false
					break
				}
			}
			if allDigits {
				return fmt.Errorf("global process visibility unavailable: nested mount masks %s", mountPoint)
			}
		}
		if len(fields) < 10 || fields[4] != "/proc" {
			continue
		}
		separator := -1
		for i := 6; i < len(fields); i++ {
			if fields[i] == "-" {
				separator = i
				break
			}
		}
		if separator < 0 || separator+3 >= len(fields) || fields[separator+1] != "proc" {
			return errors.New("proc mount authority is malformed or not procfs")
		}
		matches++
		options := append(strings.Split(fields[5], ","), strings.Split(fields[separator+3], ",")...)
		for _, option := range options {
			if strings.HasPrefix(option, "hidepid=") && option != "hidepid=0" {
				return fmt.Errorf("global process visibility unavailable: %s", option)
			}
		}
	}
	if matches != 1 {
		return fmt.Errorf("global process visibility unproved: found %d authoritative /proc mounts", matches)
	}
	return nil
}

func verifyStrixExchangeLocked(state *linuxStrixComparatorAuthority, reference strixComparatorApprovedReference, connection *net.TCPConn) (strixTCPAuthoritySnapshot, error) {
	if state.closed {
		return strixTCPAuthoritySnapshot{}, errors.New("comparator authority is closed")
	}
	select {
	case err := <-state.done:
		state.done <- err
		return strixTCPAuthoritySnapshot{}, errors.New("comparator child exited")
	default:
	}
	if err := verifyStrixAuthorityProcess(state); err != nil {
		return strixTCPAuthoritySnapshot{}, err
	}
	before, err := inspectStrixAcceptedConnection(state.cmd.Process.Pid, connection)
	if err != nil {
		return strixTCPAuthoritySnapshot{}, err
	}
	if err := matchStrixApprovedReference(state, reference); err != nil {
		return strixTCPAuthoritySnapshot{}, err
	}
	for _, item := range state.pinned {
		digest, err := hashHeldStrixFile(item.file)
		if err != nil || digest != item.digest {
			return strixTCPAuthoritySnapshot{}, fmt.Errorf("held %s bytes changed", item.role)
		}
		if err := revalidateAnchoredStrixFile(state.rootFD, item); err != nil {
			return strixTCPAuthoritySnapshot{}, err
		}
	}
	if err := verifyStrixAuthorityProcess(state); err != nil {
		return strixTCPAuthoritySnapshot{}, err
	}
	after, err := inspectStrixAcceptedConnection(state.cmd.Process.Pid, connection)
	if err != nil {
		return strixTCPAuthoritySnapshot{}, err
	}
	if before != after {
		return strixTCPAuthoritySnapshot{}, errors.New("accepted connection identity changed during inspection")
	}
	if state.exchangeHook != nil {
		state.exchangeHook(state.cmd.Process.Pid, connection)
	}
	clientInode, clientLocal, clientRemote, err := inspectStrixClientSocket(connection)
	if err != nil {
		return strixTCPAuthoritySnapshot{}, fmt.Errorf("revalidate client socket after final owner scan: %w", err)
	}
	if clientInode != after.clientInode || clientLocal != after.clientLocal || clientRemote != after.clientRemote {
		return strixTCPAuthoritySnapshot{}, errors.New("client socket identity changed after final owner scan")
	}
	if exited, err := strixPidfdExited(state.pidfd); err != nil {
		return strixTCPAuthoritySnapshot{}, fmt.Errorf("poll child pidfd after final owner scan: %w", err)
	} else if exited {
		return strixTCPAuthoritySnapshot{}, errors.New("comparator child exited after final owner scan")
	}
	return after, nil
}

func matchStrixApprovedReference(state *linuxStrixComparatorAuthority, reference strixComparatorApprovedReference) error {
	var source, build, server, model string
	loaders := make([]StrixComparatorPinnedFile, 0)
	dependencies := make([]StrixComparatorPinnedFile, 0)
	seenDigest := make(map[string]struct{}, len(state.pinned))
	for _, item := range state.pinned {
		if _, duplicate := seenDigest[item.digest]; duplicate {
			return errors.New("duplicate held comparator identity")
		}
		seenDigest[item.digest] = struct{}{}
		switch {
		case item.role == "source archive":
			source = item.digest
		case item.role == "build manifest":
			build = item.digest
		case item.role == "server binary":
			server = item.digest
		case item.role == "model":
			model = item.digest
		case strings.HasPrefix(item.role, "loader/ICD["):
			loaders = append(loaders, StrixComparatorPinnedFile{SHA256: item.digest})
		case strings.HasPrefix(item.role, "dependency["):
			dependencies = append(dependencies, StrixComparatorPinnedFile{SHA256: item.digest})
		default:
			return fmt.Errorf("unknown held comparator role %q", item.role)
		}
	}
	loaderSet, err := strixComparatorCompleteSetDigest("loader/icd", loaders)
	if err != nil {
		return err
	}
	dependencySet, err := strixComparatorCompleteSetDigest("dependency", dependencies)
	if err != nil {
		return err
	}
	if source != reference.sourceArchiveSHA256 || build != reference.buildManifestSHA256 || server != reference.serverBinarySHA256 ||
		state.modelClaim != reference.modelSHA256 || model == "" || loaderSet != reference.loaderSetSHA256 || dependencySet != reference.dependencySetSHA256 {
		return errors.New("held comparator identities do not match approved reference")
	}
	return nil
}

func inspectStrixAcceptedConnection(pid int, connection *net.TCPConn) (strixTCPAuthoritySnapshot, error) {
	clientInode, local, remote, err := inspectStrixClientSocket(connection)
	if err != nil {
		return strixTCPAuthoritySnapshot{}, err
	}
	owned, err := strixProcessSocketInodes(pid)
	if err != nil {
		return strixTCPAuthoritySnapshot{}, err
	}
	rows, err := readStrixTCPRows(pid)
	if err != nil {
		return strixTCPAuthoritySnapshot{}, err
	}
	var accepted, listener []uint64
	for _, row := range rows {
		if owned[row.inode] == 0 {
			continue
		}
		if row.state == "01" && row.local == remote && row.remote == local {
			accepted = append(accepted, row.inode)
		}
		if row.state == "0A" && row.local == remote && row.remote.port == 0 && row.remote.address == [4]byte{} {
			listener = append(listener, row.inode)
		}
	}
	if len(accepted) != 1 || len(listener) != 1 {
		return strixTCPAuthoritySnapshot{}, errors.New("child does not uniquely own the accepted socket and listener")
	}
	for _, inode := range []uint64{accepted[0], listener[0]} {
		owners, err := strixSameUserSocketOwners(inode)
		if err != nil {
			return strixTCPAuthoritySnapshot{}, err
		}
		if len(owners) != 1 || owners[pid] == 0 {
			return strixTCPAuthoritySnapshot{}, errors.New("accepted socket or listener is shared with another process")
		}
	}
	return strixTCPAuthoritySnapshot{clientInode: clientInode, acceptedInode: accepted[0], listenerInode: listener[0], clientLocal: local, clientRemote: remote}, nil
}

func inspectStrixClientSocket(connection *net.TCPConn) (uint64, strixIPv4Endpoint, strixIPv4Endpoint, error) {
	if connection == nil {
		return 0, strixIPv4Endpoint{}, strixIPv4Endpoint{}, errors.New("TCP connection is required")
	}
	raw, err := connection.SyscallConn()
	if err != nil {
		return 0, strixIPv4Endpoint{}, strixIPv4Endpoint{}, err
	}
	var inode uint64
	var local, remote strixIPv4Endpoint
	var inspectErr error
	err = raw.Control(func(fd uintptr) {
		var stat unix.Stat_t
		if err := unix.Fstat(int(fd), &stat); err != nil || stat.Mode&unix.S_IFMT != unix.S_IFSOCK {
			inspectErr = errors.New("connection is not a live socket")
			return
		}
		localAddr, err := unix.Getsockname(int(fd))
		if err != nil {
			inspectErr = err
			return
		}
		remoteAddr, err := unix.Getpeername(int(fd))
		if err != nil {
			inspectErr = err
			return
		}
		local4, localOK := localAddr.(*unix.SockaddrInet4)
		remote4, remoteOK := remoteAddr.(*unix.SockaddrInet4)
		loopback := [4]byte{127, 0, 0, 1}
		if !localOK || !remoteOK || local4.Addr != loopback || remote4.Addr != loopback || local4.Port <= 0 || remote4.Port <= 0 {
			inspectErr = errors.New("connection is not literal IPv4 loopback")
			return
		}
		inode = stat.Ino
		local = strixIPv4Endpoint{address: local4.Addr, port: uint16(local4.Port)}
		remote = strixIPv4Endpoint{address: remote4.Addr, port: uint16(remote4.Port)}
	})
	if err != nil {
		return 0, strixIPv4Endpoint{}, strixIPv4Endpoint{}, err
	}
	if inspectErr != nil || inode == 0 {
		return 0, strixIPv4Endpoint{}, strixIPv4Endpoint{}, inspectErr
	}
	return inode, local, remote, nil
}

type strixTCPRow struct {
	local, remote strixIPv4Endpoint
	state         string
	inode         uint64
}

func readStrixTCPRows(pid int) ([]strixTCPRow, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/net/tcp", pid))
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(data), "\n")
	rows := make([]strixTCPRow, 0, len(lines))
	for _, line := range lines[1:] {
		fields := strings.Fields(line)
		if len(fields) < 10 {
			continue
		}
		local, err := parseStrixProcIPv4Endpoint(fields[1])
		if err != nil {
			return nil, err
		}
		remote, err := parseStrixProcIPv4Endpoint(fields[2])
		if err != nil {
			return nil, err
		}
		inode, err := strconv.ParseUint(fields[9], 10, 64)
		if err != nil || inode == 0 {
			continue
		}
		rows = append(rows, strixTCPRow{local: local, remote: remote, state: fields[3], inode: inode})
	}
	return rows, nil
}

func parseStrixProcIPv4Endpoint(value string) (strixIPv4Endpoint, error) {
	rawAddress, rawPort, ok := strings.Cut(value, ":")
	if !ok || len(rawAddress) != 8 || len(rawPort) != 4 {
		return strixIPv4Endpoint{}, errors.New("malformed /proc IPv4 endpoint")
	}
	addressValue, err := strconv.ParseUint(rawAddress, 16, 32)
	if err != nil {
		return strixIPv4Endpoint{}, err
	}
	port, err := strconv.ParseUint(rawPort, 16, 16)
	if err != nil {
		return strixIPv4Endpoint{}, err
	}
	return strixIPv4Endpoint{address: [4]byte{byte(addressValue), byte(addressValue >> 8), byte(addressValue >> 16), byte(addressValue >> 24)}, port: uint16(port)}, nil
}

func strixProcessSocketInodes(pid int) (map[uint64]int, error) {
	entries, err := os.ReadDir(fmt.Sprintf("/proc/%d/fd", pid))
	if err != nil {
		return nil, err
	}
	return strixSocketInodesInFDDir(fmt.Sprintf("/proc/%d/fd", pid), entries)
}

func strixSocketInodesInFDDir(dir string, entries []os.DirEntry) (map[uint64]int, error) {
	result := make(map[uint64]int)
	for _, entry := range entries {
		target, err := os.Readlink(filepath.Join(dir, entry.Name()))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("inspect socket owner fd %s: %w", filepath.Join(dir, entry.Name()), err)
		}
		if !strings.HasPrefix(target, "socket:[") || !strings.HasSuffix(target, "]") {
			continue
		}
		inode, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(target, "socket:["), "]"), 10, 64)
		if err == nil {
			result[inode]++
		}
	}
	return result, nil
}

func strixSameUserSocketOwners(inode uint64) (map[int]int, error) {
	processes, err := os.ReadDir("/proc")
	if err != nil {
		return nil, err
	}
	owners := make(map[int]int)
	for _, process := range processes {
		pid, err := strconv.Atoi(process.Name())
		if err != nil {
			continue
		}
		processDir := filepath.Join("/proc", process.Name())
		status, err := os.ReadFile(filepath.Join(processDir, "status"))
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("inspect process owner %d: %w", pid, err)
		}
		effectiveUID := -1
		for _, line := range strings.Split(string(status), "\n") {
			if !strings.HasPrefix(line, "Uid:") {
				continue
			}
			fields := strings.Fields(line)
			if len(fields) >= 3 {
				effectiveUID, err = strconv.Atoi(fields[2])
			}
			break
		}
		if err != nil || effectiveUID < 0 {
			return nil, fmt.Errorf("inspect process owner %d: malformed effective uid", pid)
		}
		if effectiveUID != os.Geteuid() {
			continue
		}
		statData, err := os.ReadFile(filepath.Join(processDir, "stat"))
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("inspect same-user process state %d: %w", pid, err)
		}
		closingParen := bytes.LastIndexByte(statData, ')')
		if closingParen < 0 {
			return nil, fmt.Errorf("inspect same-user process state %d: malformed stat", pid)
		}
		stateFields := strings.Fields(string(statData[closingParen+1:]))
		if len(stateFields) == 0 {
			return nil, fmt.Errorf("inspect same-user process state %d: missing state", pid)
		}
		if stateFields[0] == "Z" || stateFields[0] == "X" {
			tasks, err := os.ReadDir(filepath.Join(processDir, "task"))
			if err != nil {
				if os.IsNotExist(err) {
					continue
				}
				return nil, fmt.Errorf("inspect tasks behind terminal process leader %d: %w", pid, err)
			}
			for _, task := range tasks {
				taskStat, err := os.ReadFile(filepath.Join(processDir, "task", task.Name(), "stat"))
				if os.IsNotExist(err) {
					continue
				}
				if err != nil {
					return nil, fmt.Errorf("inspect task %s behind terminal process leader %d: %w", task.Name(), pid, err)
				}
				taskParen := bytes.LastIndexByte(taskStat, ')')
				if taskParen < 0 {
					return nil, fmt.Errorf("inspect task %s behind terminal process leader %d: malformed stat", task.Name(), pid)
				}
				taskFields := strings.Fields(string(taskStat[taskParen+1:]))
				if len(taskFields) == 0 {
					return nil, fmt.Errorf("inspect task %s behind terminal process leader %d: missing state", task.Name(), pid)
				}
				if taskFields[0] != "Z" && taskFields[0] != "X" {
					return nil, fmt.Errorf("terminal process leader %d retains live task %s", pid, task.Name())
				}
			}
			continue
		}
		fdDir := filepath.Join(processDir, "fd")
		entries, err := os.ReadDir(fdDir)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("inspect same-user socket owners in %s: %w", fdDir, err)
		}
		inodes, err := strixSocketInodesInFDDir(fdDir, entries)
		if err != nil {
			return nil, err
		}
		if count := inodes[inode]; count > 0 {
			owners[pid] = count
		}
	}
	return owners, nil
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
