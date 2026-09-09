//go:build linux

package llamacppinterop

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestOpenStrixComparatorAuthorityPinsExecutedFiles(t *testing.T) {
	if len(os.Args) >= 2 && os.Args[len(os.Args)-1] == "strix-authority-exit" {
		return
	}
	if len(os.Args) >= 7 && os.Args[len(os.Args)-6] == "strix-authority-late-map" {
		for _, rawFD := range os.Args[len(os.Args)-5 : len(os.Args)-2] {
			fd, err := strconv.Atoi(rawFD)
			if err != nil {
				os.Exit(100)
			}
			mapped, err := syscall.Mmap(fd, 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				os.Exit(101)
			}
			defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		}
		trigger := os.Args[len(os.Args)-1]
		for {
			if _, err := os.Stat(trigger); err == nil {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
		rogue, err := os.Open(os.Args[len(os.Args)-2])
		if err != nil {
			os.Exit(102)
		}
		defer rogue.Close() //nolint:errcheck // helper exits immediately after the test parent closes it
		mapped, err := syscall.Mmap(int(rogue.Fd()), 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
		if err != nil {
			os.Exit(103)
		}
		defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		for {
			time.Sleep(time.Second)
		}
	}
	if len(os.Args) >= 6 && os.Args[len(os.Args)-5] == "strix-authority-rogue" {
		if wd, err := os.Getwd(); err != nil || wd != "/" || os.Getenv("PATH") != "" || os.Getenv("STRIX_INJECTED") != "" {
			os.Exit(91)
		}
		for _, rawFD := range os.Args[len(os.Args)-4 : len(os.Args)-1] {
			fd, err := strconv.Atoi(rawFD)
			if err != nil {
				os.Exit(92)
			}
			mapped, err := syscall.Mmap(fd, 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				os.Exit(93)
			}
			defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		}
		rogue, err := os.Open(os.Args[len(os.Args)-1])
		if err != nil {
			os.Exit(94)
		}
		defer rogue.Close() //nolint:errcheck // helper exits immediately after the test parent closes it
		mapped, err := syscall.Mmap(int(rogue.Fd()), 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
		if err != nil {
			os.Exit(95)
		}
		defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		for {
			time.Sleep(time.Second)
		}
	}
	if len(os.Args) >= 4 && os.Args[len(os.Args)-3] == "strix-authority-no-model" {
		for _, rawFD := range os.Args[len(os.Args)-2:] {
			fd, err := strconv.Atoi(rawFD)
			if err != nil {
				os.Exit(96)
			}
			mapped, err := syscall.Mmap(fd, 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				os.Exit(97)
			}
			defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		}
		for {
			time.Sleep(time.Second)
		}
	}
	if len(os.Args) >= 5 && os.Args[len(os.Args)-4] == "strix-authority-helper" {
		if wd, err := os.Getwd(); err != nil || wd != "/" || os.Getenv("PATH") != "" || os.Getenv("STRIX_INJECTED") != "" {
			os.Exit(91)
		}
		for _, rawFD := range os.Args[len(os.Args)-3:] {
			fd, err := strconv.Atoi(rawFD)
			if err != nil {
				os.Exit(92)
			}
			mapped, err := syscall.Mmap(fd, 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				os.Exit(93)
			}
			defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		}
		if _, err := syscall.Seek(4, 2, 0); err != nil {
			os.Exit(98)
		}
		for {
			if offset, err := syscall.Seek(4, 0, 1); err != nil || offset != 2 {
				os.Exit(99)
			}
			time.Sleep(10 * time.Millisecond)
		}
	}

	dir := t.TempDir()
	write := func(name string, data []byte, mode os.FileMode) string {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, data, mode); err != nil {
			t.Fatal(err)
		}
		return path
	}
	hashFile := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		return hex.EncodeToString(sum[:])
	}

	server, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	server, err = filepath.EvalSymlinks(server)
	if err != nil {
		t.Fatal(err)
	}
	source := write("source.tar", []byte("source archive"), 0o600)
	build := write("build.json", []byte(`{"build":"b10588","vulkan":true}`), 0o600)
	model := write("model.gguf", []byte("model"), 0o600)
	loader := write("loader.icd", make([]byte, 4096), 0o600)
	dependency := write("dependency.so", make([]byte, 4096), 0o600)
	serverInfo, err := os.Stat(server)
	if err != nil {
		t.Fatal(err)
	}
	dependencies := []StrixComparatorPinnedFile{{Path: dependency, SHA256: hashFile(dependency)}}
	maps, err := os.Open("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	seenMappedPath := make(map[string]struct{})
	scanner := bufio.NewScanner(maps)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 6 || !strings.HasPrefix(fields[5], "/") {
			continue
		}
		path := strings.TrimSuffix(strings.Join(fields[5:], " "), " (deleted)")
		path, err = filepath.EvalSymlinks(path)
		if err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if os.SameFile(info, serverInfo) {
			continue
		}
		if _, ok := seenMappedPath[path]; ok {
			continue
		}
		seenMappedPath[path] = struct{}{}
		dependencies = append(dependencies, StrixComparatorPinnedFile{Path: path, SHA256: hashFile(path)})
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if err := maps.Close(); err != nil {
		t.Fatal(err)
	}

	manifest := validStrixComparatorManifest()
	manifest.SourceArchiveSHA256 = hashFile(source)
	manifest.BuildManifestSHA256 = hashFile(build)
	manifest.ServerBinarySHA256 = hashFile(server)

	options := StrixComparatorAuthorityOptions{
		Manifest:          manifest,
		SourceArchivePath: source,
		BuildManifestPath: build,
		ServerBinaryPath:  server,
		ModelPath:         model,
		LoaderICD:         []StrixComparatorPinnedFile{{Path: loader, SHA256: hashFile(loader)}},
		Dependencies:      dependencies,
		Arguments:         []string{"-test.run=^TestOpenStrixComparatorAuthorityPinsExecutedFiles$", "--", "strix-authority-helper", "4", "5", "6"},
		testModelSHA256:   hashFile(model),
	}

	t.Setenv("PATH", filepath.Join(dir, "attacker-bin"))
	t.Setenv("STRIX_INJECTED", "must-not-cross-authority")
	authority, err := OpenStrixComparatorAuthority(options)
	if err != nil {
		t.Fatalf("open authority: %v", err)
	}
	pid := authority.PID()
	if pid <= 0 || !authority.Valid() {
		t.Fatalf("authority is not live: pid=%d valid=%v", pid, authority.Valid())
	}
	readFDPosition := func(pid, fd int) int64 {
		t.Helper()
		data, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/fdinfo/" + strconv.Itoa(fd))
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range strings.Split(string(data), "\n") {
			if !strings.HasPrefix(line, "pos:") {
				continue
			}
			position, err := strconv.ParseInt(strings.TrimSpace(strings.TrimPrefix(line, "pos:")), 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			return position
		}
		t.Fatal("child fdinfo omitted position")
		return -1
	}
	if before := readFDPosition(pid, 4); before != 2 {
		t.Fatalf("inherited model fd position before validation = %d", before)
	}
	if !authority.Valid() {
		t.Fatal("authority invalid during offset-independent revalidation")
	}
	if after := readFDPosition(pid, 4); after != 2 {
		t.Fatalf("retained-handle hashing changed inherited model fd position to %d", after)
	}
	if _, err := json.Marshal(authority); err == nil {
		t.Fatal("opaque authority unexpectedly marshaled")
	}
	constructed := new(StrixComparatorAuthority)
	if constructed.Valid() || !errors.Is(constructed.Close(), ErrInvalidStrixComparatorAuthority) {
		t.Fatal("constructed lookalike retained capability")
	}
	if err := json.Unmarshal([]byte(`{}`), constructed); err == nil {
		t.Fatal("opaque authority unexpectedly unmarshaled")
	}
	lookalike := *authority
	if lookalike.Valid() {
		t.Fatal("copied authority retained capability")
	}
	if err := lookalike.Close(); !errors.Is(err, ErrInvalidStrixComparatorAuthority) {
		t.Fatalf("copied authority close error = %v", err)
	}
	if !authority.Valid() {
		t.Fatal("closing copied lookalike affected minted authority")
	}
	if err := authority.Close(); err != nil {
		t.Fatalf("close authority: %v", err)
	}
	if authority.Valid() {
		t.Fatal("closed authority remained valid")
	}

	t.Run("rejects symlink and special file", func(t *testing.T) {
		symlink := filepath.Join(dir, "source-link")
		if err := os.Symlink(source, symlink); err != nil {
			t.Fatal(err)
		}
		bad := options
		bad.SourceArchivePath = symlink
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("symlink result = (%v, %v)", got, err)
		}
		bad = options
		bad.ModelPath = "/dev/null"
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("special-file result = (%v, %v)", got, err)
		}
		fifo := filepath.Join(dir, "model.fifo")
		if err := syscall.Mkfifo(fifo, 0o600); err != nil {
			t.Fatal(err)
		}
		bad.ModelPath = fifo
		started := time.Now()
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("FIFO result = (%v, %v)", got, err)
		}
		if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
			t.Fatalf("FIFO rejection blocked for %v", elapsed)
		}
	})

	t.Run("rejects digest mismatch", func(t *testing.T) {
		bad := options
		bad.testModelSHA256 = strings.Repeat("a", 64)
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("hash mismatch result = (%v, %v)", got, err)
		}
	})

	t.Run("rejects undeclared mapped file", func(t *testing.T) {
		rogue := write("rogue.so", make([]byte, 4096), 0o600)
		bad := options
		bad.Arguments = []string{"-test.run=^TestOpenStrixComparatorAuthorityPinsExecutedFiles$", "--", "strix-authority-rogue", "4", "5", "6", rogue}
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("undeclared map result = (%v, %v)", got, err)
		}
	})

	t.Run("rejects partial startup", func(t *testing.T) {
		bad := options
		bad.Arguments = []string{"-test.run=^TestOpenStrixComparatorAuthorityPinsExecutedFiles$", "--", "strix-authority-exit"}
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("partial-start result = (%v, %v)", got, err)
		}
	})

	t.Run("requires the pinned model mapping", func(t *testing.T) {
		bad := options
		bad.Arguments = []string{"-test.run=^TestOpenStrixComparatorAuthorityPinsExecutedFiles$", "--", "strix-authority-no-model", "5", "6"}
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("unmapped model result = (%v, %v)", got, err)
		}
	})

	t.Run("rejects replaced executable path", func(t *testing.T) {
		copyPath := filepath.Join(dir, "server-copy")
		serverBytes, err := os.ReadFile(server)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(copyPath, serverBytes, 0o700); err != nil {
			t.Fatal(err)
		}
		bad := options
		bad.ServerBinaryPath = copyPath
		bad.testAfterOpen = func() error {
			return os.Rename(write("replacement", serverBytes, 0o700), copyPath)
		}
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("replaced executable result = (%v, %v)", got, err)
		}
	})

	t.Run("rejects child exit during final startup verification", func(t *testing.T) {
		candidate := options
		candidate.testDuringStartupValidation = func(pid int) {
			process, err := os.FindProcess(pid)
			if err != nil {
				t.Error(err)
				return
			}
			if err := process.Kill(); err != nil {
				t.Error(err)
			}
			deadline := time.Now().Add(time.Second)
			for time.Now().Before(deadline) {
				if _, err := os.Stat("/proc/" + strconv.Itoa(pid)); os.IsNotExist(err) {
					return
				}
				time.Sleep(5 * time.Millisecond)
			}
			t.Error("child remained present after injected final-startup exit")
		}
		if got, err := OpenStrixComparatorAuthority(candidate); err == nil || got != nil {
			t.Fatalf("final-startup exit result = (%v, %v)", got, err)
		}
	})

	t.Run("retains rejected-startup cleanup after signal failure", func(t *testing.T) {
		rogue := write("late-startup-rogue.so", make([]byte, 4096), 0o600)
		trigger := filepath.Join(dir, "late-startup.trigger")
		candidate := options
		candidate.Arguments = []string{"-test.run=^TestOpenStrixComparatorAuthorityPinsExecutedFiles$", "--", "strix-authority-late-map", "4", "5", "6", rogue, trigger}
		childPID := 0
		candidate.testDuringStartupValidation = func(pid int) {
			childPID = pid
			if err := os.WriteFile(trigger, []byte("map-now"), 0o600); err != nil {
				t.Error(err)
				return
			}
			deadline := time.Now().Add(time.Second)
			for time.Now().Before(deadline) {
				maps, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/maps")
				if err == nil && strings.Contains(string(maps), rogue) {
					return
				}
				time.Sleep(5 * time.Millisecond)
			}
			t.Error("child did not install injected late mapping")
		}
		injected := errors.New("injected rejected-startup signal failure")
		signalCalls := 0
		candidate.testPidfdSignal = func(pidfd, signal int) error {
			signalCalls++
			if signalCalls == 1 {
				return injected
			}
			return sendStrixPidfdSignal(pidfd, signal)
		}
		fdsBefore, err := os.ReadDir("/proc/self/fd")
		if err != nil {
			t.Fatal(err)
		}
		got, openErr := OpenStrixComparatorAuthority(candidate)
		if got != nil || openErr == nil {
			t.Fatalf("late-map startup result = (%v, %v)", got, openErr)
		}
		var cleanupErr *StrixComparatorAuthorityCleanupError
		if !errors.As(openErr, &cleanupErr) || !cleanupErr.CleanupPending() || !errors.Is(openErr, ErrStrixComparatorAuthorityCleanupPending) {
			t.Fatalf("startup error did not retain cleanup authority: %T %v", openErr, openErr)
		}
		if childPID <= 0 {
			t.Fatal("startup hook did not capture child PID")
		}
		if err := cleanupErr.RetryCleanup(); err != nil {
			t.Fatalf("retry rejected-startup cleanup: %v", err)
		}
		if cleanupErr.CleanupPending() {
			t.Fatal("cleanup owner remained pending after successful retry")
		}
		if err := cleanupErr.RetryCleanup(); err != nil {
			t.Fatalf("idempotent cleanup retry: %v", err)
		}
		if _, err := os.Stat("/proc/" + strconv.Itoa(childPID)); !os.IsNotExist(err) {
			t.Fatalf("rejected startup child still exists: %v", err)
		}
		fdsAfter, err := os.ReadDir("/proc/self/fd")
		if err != nil {
			t.Fatal(err)
		}
		if len(fdsAfter) != len(fdsBefore) {
			t.Fatalf("rejected startup leaked parent fds: before=%d after=%d", len(fdsBefore), len(fdsAfter))
		}
	})

	t.Run("brackets retained-handle validation with child identity", func(t *testing.T) {
		candidate := options
		candidate.testDuringValidation = func(pid int) {
			process, err := os.FindProcess(pid)
			if err != nil {
				t.Error(err)
				return
			}
			if err := process.Kill(); err != nil {
				t.Error(err)
			}
		}
		minted, err := OpenStrixComparatorAuthority(candidate)
		if err != nil {
			t.Fatalf("mint bracket authority: %v", err)
		}
		if minted.Valid() {
			t.Fatal("authority survived child exit during retained-handle validation")
		}
		if err := minted.Close(); err != nil {
			t.Fatalf("close bracket authority: %v", err)
		}
	})

	t.Run("retains cleanup authority after bounded signal failure", func(t *testing.T) {
		candidate := options
		injected := errors.New("injected pidfd signal failure")
		calls := 0
		candidate.testPidfdSignal = func(pidfd, signal int) error {
			calls++
			if calls == 1 {
				return injected
			}
			return sendStrixPidfdSignal(pidfd, signal)
		}
		minted, err := OpenStrixComparatorAuthority(candidate)
		if err != nil {
			t.Fatalf("mint cleanup authority: %v", err)
		}
		started := time.Now()
		err = minted.Close()
		if !errors.Is(err, ErrStrixComparatorAuthorityCleanupPending) {
			t.Fatalf("first close error = %v", err)
		}
		if elapsed := time.Since(started); elapsed > 250*time.Millisecond {
			t.Fatalf("failed pidfd signal blocked close for %v", elapsed)
		}
		if !minted.Valid() {
			t.Fatal("signal failure discarded the live cleanup capability")
		}
		if err := minted.Close(); err != nil {
			t.Fatalf("retry close: %v", err)
		}
		if minted.Valid() {
			t.Fatal("retried close retained authority")
		}
	})

	t.Run("invalidates post-mint artifact drift", func(t *testing.T) {
		cloneOptions := func() StrixComparatorAuthorityOptions {
			cloned := options
			cloned.Manifest.BuildFlags = append([]string(nil), options.Manifest.BuildFlags...)
			cloned.Manifest.BuildTargets = append([]string(nil), options.Manifest.BuildTargets...)
			cloned.LoaderICD = append([]StrixComparatorPinnedFile(nil), options.LoaderICD...)
			cloned.Dependencies = append([]StrixComparatorPinnedFile(nil), options.Dependencies...)
			cloned.Arguments = append([]string(nil), options.Arguments...)
			return cloned
		}
		copyFile := func(name, sourcePath string, mode os.FileMode) string {
			t.Helper()
			data, err := os.ReadFile(sourcePath)
			if err != nil {
				t.Fatal(err)
			}
			return write(name, data, mode)
		}
		replacePath := func(path, name string, mode os.FileMode) {
			t.Helper()
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Rename(write(name, data, mode), path); err != nil {
				t.Fatal(err)
			}
		}
		cases := []struct {
			name  string
			setup func(*StrixComparatorAuthorityOptions) func()
		}{
			{
				name: "source held bytes",
				setup: func(candidate *StrixComparatorAuthorityOptions) func() {
					path := copyFile("post-mint-source", source, 0o600)
					candidate.SourceArchivePath = path
					candidate.Manifest.SourceArchiveSHA256 = hashFile(path)
					return func() {
						if err := os.WriteFile(path, []byte("mutated source archive"), 0o600); err != nil {
							t.Fatal(err)
						}
					}
				},
			},
			{
				name: "build manifest path",
				setup: func(candidate *StrixComparatorAuthorityOptions) func() {
					path := copyFile("post-mint-build", build, 0o600)
					candidate.BuildManifestPath = path
					candidate.Manifest.BuildManifestSHA256 = hashFile(path)
					return func() { replacePath(path, "post-mint-build-replacement", 0o600) }
				},
			},
			{
				name: "model path",
				setup: func(candidate *StrixComparatorAuthorityOptions) func() {
					path := copyFile("post-mint-model", model, 0o600)
					candidate.ModelPath = path
					candidate.testModelSHA256 = hashFile(path)
					return func() { replacePath(path, "post-mint-model-replacement", 0o600) }
				},
			},
			{
				name: "dependency path",
				setup: func(candidate *StrixComparatorAuthorityOptions) func() {
					path := copyFile("post-mint-dependency", dependency, 0o600)
					candidate.Dependencies[0] = StrixComparatorPinnedFile{Path: path, SHA256: hashFile(path)}
					return func() { replacePath(path, "post-mint-dependency-replacement", 0o600) }
				},
			},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				candidate := cloneOptions()
				mutate := tc.setup(&candidate)
				minted, err := OpenStrixComparatorAuthority(candidate)
				if err != nil {
					t.Fatalf("mint authority: %v", err)
				}
				mutate()
				if minted.Valid() {
					t.Fatal("authority remained valid after pinned artifact drift")
				}
				if err := minted.Close(); err != nil {
					t.Fatalf("close invalidated authority: %v", err)
				}
			})
		}
	})
}
