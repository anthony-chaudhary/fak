package amdgpu

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestStrixValidationRejectsUntrustedSSH(t *testing.T) {
	shimDir := t.TempDir()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	sshName := "ssh"
	if runtime.GOOS == "windows" {
		sshName += ".exe"
	}
	selfBytes, err := os.ReadFile(self)
	if err != nil {
		t.Fatal(err)
	}
	shimPath := filepath.Join(shimDir, sshName)
	if err := os.WriteFile(shimPath, selfBytes, 0o700); err != nil {
		t.Fatalf("create inert ssh test executable: %v", err)
	}
	t.Setenv("PATH", shimDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	if resolved, err := exec.LookPath("ssh"); err != nil || resolved != shimPath {
		t.Fatalf("isolated ssh resolved to %q, %v; want %q", resolved, err, shimPath)
	}
	t.Setenv(StrixKnownHostsFileEnv, "")

	out, err := runStrixTargetCommand(context.Background(), &StrixTarget{Mode: "ssh", Host: "safe-host"}, "true", nil)
	if !errors.Is(err, ErrStrixHostTrustRefused) {
		t.Fatalf("runStrixTargetCommand() output=%q err=%v, want typed trust refusal before SSH start", out, err)
	}
	if len(out) != 0 {
		t.Fatalf("untrusted SSH returned output %q", out)
	}
}

func TestStrixSSHTrustConstructor(t *testing.T) {
	assertRefused := func(t *testing.T, err error, secrets ...string) {
		t.Helper()
		if !errors.Is(err, ErrStrixHostTrustRefused) {
			t.Fatalf("err = %v, want typed trust refusal", err)
		}
		for _, secret := range secrets {
			if secret != "" && strings.Contains(err.Error(), secret) {
				t.Fatalf("error leaked sensitive value %q: %v", secret, err)
			}
		}
	}
	assertRemoteCommand := func(t *testing.T, fixture *strixSSHTransportFixture, host string, timeout time.Duration) {
		t.Helper()
		if fixture.brokerStarts != 1 || fixture.broker.commandCalls != 1 || fixture.broker.closeCalls != 1 {
			t.Fatalf("broker start/command/close = %d/%d/%d, want 1/1/1", fixture.brokerStarts, fixture.broker.commandCalls, fixture.broker.closeCalls)
		}
		if fixture.processStarts != 1 || len(fixture.commands) != 1 {
			t.Fatalf("process starts/commands = %d/%d, want 1/1", fixture.processStarts, len(fixture.commands))
		}
		cmd := fixture.commands[0]
		seconds := int64(timeout / time.Second)
		wantArgs := []string{
			fixture.sshPath,
			"-F", "none",
			"-o", "BatchMode=yes",
			"-o", "ConnectTimeout=" + strconv.FormatInt(seconds, 10),
			"-o", "StrictHostKeyChecking=yes",
			"-o", "UserKnownHostsFile=none",
			"-o", "GlobalKnownHostsFile=none",
			"-o", "VerifyHostKeyDNS=no",
			"-o", "HostKeyAlias=" + StrixKnownHostsAlias,
			"-o", "UpdateHostKeys=no",
			"-o", "KnownHostsCommand=" + fixture.knownHostsCommand,
			"--", host, cmd.Args[len(cmd.Args)-1],
		}
		if !reflect.DeepEqual(cmd.Args, wantArgs) {
			t.Fatalf("ssh argv = %#v, want %#v", cmd.Args, wantArgs)
		}
		if !filepath.IsAbs(cmd.Path) || cmd.Path != fixture.sshPath {
			t.Fatalf("ssh path = %q, want absolute %q", cmd.Path, fixture.sshPath)
		}
		if !strings.Contains(fixture.knownHostsCommand, quoteOpenSSHArg(fixture.selfPath)) || !filepath.IsAbs(fixture.selfPath) {
			t.Fatalf("KnownHostsCommand does not contain absolute self path: %q", fixture.knownHostsCommand)
		}
		wantEnv := []string{"SystemRoot=C:\\Windows", "temp=C:\\Temp"}
		if !reflect.DeepEqual(cmd.Env, wantEnv) {
			t.Fatalf("ssh environment = %#v, want allowlist %#v", cmd.Env, wantEnv)
		}
		serialized := strings.Join(append(append([]string{}, cmd.Args...), cmd.Env...), "\x00")
		for _, absent := range []string{fixture.rawKey, fixture.trustPath, "PROCESS_SECRET", StrixKnownHostsFileEnv} {
			if strings.Contains(serialized, absent) {
				t.Fatalf("ssh process leaked %q in argv/environment", absent)
			}
		}
		if strings.Contains(serialized, "accept-new") || strings.Contains(serialized, host+"="+StrixKnownHostsAlias) {
			t.Fatalf("ssh process used fallback trust or conflated destination and alias: %#v", cmd.Args)
		}
	}

	for _, tc := range []struct {
		name    string
		kind    string
		timeout time.Duration
	}{
		{name: "discovery path", kind: "discovery", timeout: 2 * time.Second},
		{name: "validation path", kind: "validation", timeout: 10 * time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newStrixSSHTransportFixture(t)
			installStrixSSHTransportFixture(t, fixture)
			host := "strix-1.example"
			fixture.output = []byte("remote-ok")
			if tc.kind == "discovery" {
				fixture.output = []byte(`{"reachable":true,"target_isa":"gfx1151","compute_units":40}`)
				target, err := probeRemoteStrix(context.Background(), host)
				if err != nil || target == nil || !target.Reachable || target.Host != host {
					t.Fatalf("probeRemoteStrix() = %#v, %v", target, err)
				}
			} else {
				stdin := []byte("candidate archive")
				out, err := runStrixTargetCommand(context.Background(), &StrixTarget{Mode: "ssh", Host: host}, "remote command", stdin)
				if err != nil || string(out) != "remote-ok" {
					t.Fatalf("runStrixTargetCommand() = %q, %v", out, err)
				}
				gotStdin, err := io.ReadAll(fixture.commands[0].Stdin)
				if err != nil || !reflect.DeepEqual(gotStdin, stdin) {
					t.Fatalf("ssh stdin = %q, %v; want %q", gotStdin, err, stdin)
				}
			}
			assertRemoteCommand(t, fixture, host, tc.timeout)
		})
	}

	t.Run("trust precedes cached presence", func(t *testing.T) {
		fixture := newStrixSSHTransportFixture(t)
		fixture.broker = nil
		fixture.brokerErr = errors.New("missing trust at " + fixture.trustPath)
		installStrixSSHTransportFixture(t, fixture)
		root := t.TempDir()
		if err := os.MkdirAll(root+"/_scratch", 0o755); err != nil {
			t.Fatal(err)
		}
		t.Chdir(root)
		t.Setenv(StrixKnownHostsFileEnv, "")

		savePresenceCache(&StrixTarget{
			Mode:      "ssh",
			Host:      "cached-strix.example",
			Reachable: true,
		})
		got, err := DiscoverStrixTarget(context.Background(), "cached-strix.example")
		assertRefused(t, err, fixture.trustPath, fixture.rawKey, fixture.capability)
		if got != nil && got.Reachable {
			t.Fatalf("DiscoverStrixTarget() returned trusted cached target: %#v", got)
		}
		if fixture.brokerStarts != 1 || fixture.processStarts != 0 || len(fixture.commands) != 0 {
			t.Fatalf("broker/process/command counts = %d/%d/%d, want 1/0/0", fixture.brokerStarts, fixture.processStarts, len(fixture.commands))
		}
	})

	t.Run("remote availability alone permits fixed fallback", func(t *testing.T) {
		fixture := newStrixSSHTransportFixture(t)
		fixture.combined = func(attempt int) ([]byte, error) {
			if attempt == 0 {
				return nil, &exec.ExitError{}
			}
			return []byte(`{"reachable":true,"target_isa":"gfx1151","compute_units":40}`), nil
		}
		installStrixSSHTransportFixture(t, fixture)
		root := t.TempDir()
		if err := os.MkdirAll(filepath.Join(root, "_scratch"), 0o755); err != nil {
			t.Fatal(err)
		}
		t.Chdir(root)
		t.Setenv("FAK_STRIX_HOST", "")
		target, err := DiscoverStrixTarget(context.Background(), "")
		if err != nil || target == nil || !target.Reachable || target.Host != FallbackStrixMDNS {
			t.Fatalf("fallback discovery = %#v, %v", target, err)
		}
		if fixture.brokerStarts != 2 || fixture.processStarts != 2 || fixture.broker.closeCalls != 2 || len(fixture.commands) != 2 {
			t.Fatalf("fallback broker/process/close/commands = %d/%d/%d/%d, want 2/2/2/2", fixture.brokerStarts, fixture.processStarts, fixture.broker.closeCalls, len(fixture.commands))
		}
		if got := fixture.commands[0].Args[len(fixture.commands[0].Args)-2]; got != DefaultStrixHost {
			t.Fatalf("first destination = %q, want %q", got, DefaultStrixHost)
		}
		if got := fixture.commands[1].Args[len(fixture.commands[1].Args)-2]; got != FallbackStrixMDNS {
			t.Fatalf("fallback destination = %q, want %q", got, FallbackStrixMDNS)
		}
	})

	t.Run("cleanup refusal cannot fall back", func(t *testing.T) {
		fixture := newStrixSSHTransportFixture(t)
		fixture.output = []byte(`{"reachable":true}`)
		fixture.broker.closeErr = errors.New("cleanup " + fixture.capability)
		installStrixSSHTransportFixture(t, fixture)
		t.Setenv("FAK_STRIX_HOST", "")
		target, err := DiscoverStrixTarget(context.Background(), "")
		assertRefused(t, err, fixture.capability, fixture.trustPath)
		if target != nil && target.Reachable {
			t.Fatalf("cleanup refusal returned reachable target: %#v", target)
		}
		if fixture.brokerStarts != 1 || fixture.processStarts != 1 || fixture.broker.closeCalls != 1 || len(fixture.commands) != 1 {
			t.Fatalf("cleanup refusal broker/process/close/commands = %d/%d/%d/%d, want 1/1/1/1", fixture.brokerStarts, fixture.processStarts, fixture.broker.closeCalls, len(fixture.commands))
		}
	})

	for _, tc := range []struct {
		name       string
		kind       string
		wantRefuse bool
		closeErr   bool
	}{
		{name: "success cleanup", kind: "success"},
		{name: "start failure cleanup", kind: "start", wantRefuse: true},
		{name: "cancellation cleanup", kind: "cancel", wantRefuse: true},
		{name: "parse failure cleanup", kind: "parse"},
		{name: "cleanup failure invalidates success", kind: "success", wantRefuse: true, closeErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newStrixSSHTransportFixture(t)
			installStrixSSHTransportFixture(t, fixture)
			ctx := context.Background()
			fixture.output = []byte("ok")
			if tc.kind == "start" {
				fixture.outputErr = errors.New("process detail " + fixture.capability)
			}
			if tc.kind == "cancel" {
				cancelled, cancel := context.WithCancel(context.Background())
				cancel()
				ctx = cancelled
				fixture.outputErr = context.Canceled
			}
			if tc.kind == "parse" {
				fixture.output = []byte("not-json " + fixture.rawKey)
			}
			if tc.closeErr {
				fixture.broker.closeErr = errors.New("cleanup detail " + fixture.trustPath)
			}
			var err error
			if tc.kind == "parse" {
				_, err = probeRemoteStrix(ctx, "safe-host")
			} else {
				_, err = runStrixTargetCommand(ctx, &StrixTarget{Mode: "ssh", Host: "safe-host"}, "command", nil)
			}
			if tc.wantRefuse {
				assertRefused(t, err, fixture.trustPath, fixture.rawKey, fixture.capability)
			} else if tc.kind == "parse" {
				if err == nil {
					t.Fatal("malformed probe output succeeded")
				}
				for _, secret := range []string{fixture.rawKey, fixture.trustPath, fixture.capability} {
					if strings.Contains(err.Error(), secret) {
						t.Fatalf("parse error leaked sensitive value: %v", err)
					}
				}
			} else if err != nil {
				t.Fatal(err)
			}
			if fixture.broker.closeCalls != 1 {
				t.Fatalf("broker close calls = %d, want 1", fixture.broker.closeCalls)
			}
		})
	}

	t.Run("trust source refusal precedes process start", func(t *testing.T) {
		fixture := newStrixSSHTransportFixture(t)
		fixture.broker = nil
		secret := "trust source detail " + fixture.trustPath + " " + fixture.rawKey
		fixture.brokerErr = errors.New(secret)
		installStrixSSHTransportFixture(t, fixture)
		_, err := runStrixTargetCommand(context.Background(), &StrixTarget{Mode: "ssh", Host: "safe-host"}, "command", nil)
		assertRefused(t, err, secret, fixture.trustPath, fixture.rawKey)
		if fixture.processStarts != 0 || len(fixture.commands) != 0 {
			t.Fatalf("refusal constructed/started process: %d/%d", len(fixture.commands), fixture.processStarts)
		}
	})

	t.Run("broker returned with error is cleaned", func(t *testing.T) {
		fixture := newStrixSSHTransportFixture(t)
		fixture.brokerErr = errors.New("broker start detail " + fixture.capability)
		installStrixSSHTransportFixture(t, fixture)
		_, err := runStrixTargetCommand(context.Background(), &StrixTarget{Mode: "ssh", Host: "safe-host"}, "command", nil)
		assertRefused(t, err, fixture.capability)
		if fixture.broker.closeCalls != 1 || fixture.processStarts != 0 {
			t.Fatalf("broker close/process starts = %d/%d, want 1/0", fixture.broker.closeCalls, fixture.processStarts)
		}
	})

	t.Run("malformed destinations are redacted by both entrypoints", func(t *testing.T) {
		for _, host := range []string{"-oSecret=C:\\private\\known_hosts", "host;cat-secret", "host\ncontrol"} {
			t.Run(strings.ReplaceAll(host, "\n", "newline"), func(t *testing.T) {
				fixture := newStrixSSHTransportFixture(t)
				installStrixSSHTransportFixture(t, fixture)
				target, discoverErr := DiscoverStrixTarget(context.Background(), host)
				assertRefused(t, discoverErr, host, fixture.trustPath, fixture.capability)
				if target != nil {
					t.Fatalf("malformed destination survived in discovery target: %#v", target)
				}
				receipt, validationErr := RunStrixValidation(context.Background(), StrixValidationOpts{
					Host:          host,
					RunSubkernels: true,
					Subkernels:    []string{"argmax"},
				})
				assertRefused(t, validationErr, host, fixture.trustPath, fixture.capability)
				encodedReceipt, marshalErr := json.Marshal(receipt)
				if marshalErr != nil {
					t.Fatal(marshalErr)
				}
				if strings.Contains(string(encodedReceipt), host) {
					t.Fatalf("validation receipt leaked malformed destination: %s", encodedReceipt)
				}
				_, directErr := runStrixTargetCommand(context.Background(), &StrixTarget{Mode: "ssh", Host: host}, "command", nil)
				assertRefused(t, directErr, host, fixture.trustPath, fixture.capability)
				if fixture.brokerStarts != 0 || fixture.processStarts != 0 || len(fixture.commands) != 0 {
					t.Fatalf("malformed destination reached broker/process: %d/%d/%d", fixture.brokerStarts, fixture.processStarts, len(fixture.commands))
				}
			})
		}
	})

	for _, tc := range []struct {
		name   string
		mutate func(*strixSSHTransportFixture)
	}{
		{name: "relative ssh executable", mutate: func(f *strixSSHTransportFixture) { f.lookPath = "ssh" }},
		{name: "relative self executable", mutate: func(f *strixSSHTransportFixture) { f.executablePath = "fak" }},
		{name: "missing ssh executable", mutate: func(f *strixSSHTransportFixture) { f.lookPathErr = errors.New("missing " + f.trustPath) }},
		{name: "broker command failure", mutate: func(f *strixSSHTransportFixture) { f.broker.commandErr = errors.New("command " + f.capability) }},
		{name: "ssh executable replacement", mutate: func(f *strixSSHTransportFixture) {
			f.broker.onCommand = func() { _ = os.Remove(f.sshPath); _ = os.WriteFile(f.sshPath, []byte("replacement payload"), 0o700) }
		}},
		{name: "ssh executable in-place mutation", mutate: func(f *strixSSHTransportFixture) {
			f.broker.onCommand = func() { _ = os.WriteFile(f.sshPath, []byte("in-place payload with changed length"), 0o700) }
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newStrixSSHTransportFixture(t)
			tc.mutate(fixture)
			installStrixSSHTransportFixture(t, fixture)
			_, err := runStrixTargetCommand(context.Background(), &StrixTarget{Mode: "ssh", Host: "safe-host"}, "command", nil)
			assertRefused(t, err, fixture.trustPath, fixture.rawKey, fixture.capability)
			if fixture.processStarts != 0 {
				t.Fatalf("unsafe executable/broker started process %d times", fixture.processStarts)
			}
			if fixture.brokerStarts > 0 && fixture.broker.closeCalls != 1 {
				t.Fatalf("broker close calls = %d after constructor refusal, want 1", fixture.broker.closeCalls)
			}
		})
	}

	t.Run("local mode does not acquire remote trust", func(t *testing.T) {
		fixture := newStrixSSHTransportFixture(t)
		installStrixSSHTransportFixture(t, fixture)
		out, err := runStrixTargetCommand(context.Background(), &StrixTarget{Mode: "local", Host: "localhost"}, "printf local-mode", nil)
		if err != nil || string(out) != "local-mode" {
			t.Fatalf("local command = %q, %v", out, err)
		}
		if fixture.brokerStarts != 0 || fixture.processStarts != 0 || len(fixture.commands) != 0 {
			t.Fatalf("local mode used remote transport: %d/%d/%d", fixture.brokerStarts, fixture.processStarts, len(fixture.commands))
		}
	})
}

type strixSSHTestBroker struct {
	command      string
	commandErr   error
	closeErr     error
	onCommand    func()
	commandCalls int
	closeCalls   int
}

func (b *strixSSHTestBroker) KnownHostsCommand() (string, error) {
	b.commandCalls++
	if b.onCommand != nil {
		b.onCommand()
	}
	return b.command, b.commandErr
}

func (b *strixSSHTestBroker) Close() error {
	b.closeCalls++
	return b.closeErr
}

type strixSSHTransportFixture struct {
	sshPath           string
	selfPath          string
	lookPath          string
	executablePath    string
	trustPath         string
	rawKey            string
	capability        string
	knownHostsCommand string
	broker            *strixSSHTestBroker
	brokerErr         error
	brokerStarts      int
	processStarts     int
	commands          []*exec.Cmd
	output            []byte
	outputErr         error
	combined          func(int) ([]byte, error)
	lookPathErr       error
}

func newStrixSSHTransportFixture(t *testing.T) *strixSSHTransportFixture {
	t.Helper()
	binDir := filepath.Join(t.TempDir(), "bin with spaces")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	sshName := "ssh"
	if runtime.GOOS == "windows" {
		sshName += ".exe"
	}
	sshPath := filepath.Join(binDir, sshName)
	selfPath := filepath.Join(binDir, "fak-test.exe")
	for _, path := range []string{sshPath, selfPath} {
		if err := os.WriteFile(path, []byte("stable executable fixture"), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	capability := strings.Repeat("A", 43)
	knownHostsCommand := strings.Join([]string{
		quoteOpenSSHArg(selfPath),
		quoteOpenSSHArg(StrixKnownHostsOperand),
		quoteOpenSSHArg("127.0.0.1:43123"),
		quoteOpenSSHArg(capability),
	}, " ")
	fixture := &strixSSHTransportFixture{
		sshPath:           sshPath,
		selfPath:          selfPath,
		lookPath:          sshPath,
		executablePath:    selfPath,
		trustPath:         filepath.Join(t.TempDir(), "private-known-hosts"),
		rawKey:            "AAAA_PRIVATE_HOST_KEY_MATERIAL",
		capability:        capability,
		knownHostsCommand: knownHostsCommand,
	}
	fixture.broker = &strixSSHTestBroker{command: knownHostsCommand}
	return fixture
}

func installStrixSSHTransportFixture(t *testing.T, fixture *strixSSHTransportFixture) {
	t.Helper()
	previous := strixSSHDeps
	strixSSHDeps = fixture.deps()
	t.Cleanup(func() { strixSSHDeps = previous })
	t.Setenv("PATH", filepath.Dir(fixture.sshPath)+string(os.PathListSeparator)+os.Getenv("PATH"))
	if resolved, err := exec.LookPath("ssh"); err != nil || resolved != fixture.sshPath {
		t.Fatalf("fixture ssh resolved to %q, %v; want %q", resolved, err, fixture.sshPath)
	}
}

func (f *strixSSHTransportFixture) deps() strixSSHTransportDeps {
	return strixSSHTransportDeps{
		startBroker: func() (strixSSHBroker, error) {
			f.brokerStarts++
			if f.broker == nil {
				return nil, f.brokerErr
			}
			return f.broker, f.brokerErr
		},
		lookPath:   func(string) (string, error) { return f.lookPath, f.lookPathErr },
		executable: func() (string, error) { return f.executablePath, nil },
		lstat:      os.Lstat,
		sameFile:   os.SameFile,
		fileDigest: strixSSHTestFileDigest,
		commandContext: func(ctx context.Context, path string, args ...string) *exec.Cmd {
			cmd := exec.CommandContext(ctx, path, args...)
			f.commands = append(f.commands, cmd)
			return cmd
		},
		combinedOutput: func(*exec.Cmd) ([]byte, error) {
			attempt := f.processStarts
			f.processStarts++
			if f.combined != nil {
				return f.combined(attempt)
			}
			return f.output, f.outputErr
		},
		environ: func() []string {
			return []string{
				"SystemRoot=C:\\Windows",
				"PATH=C:\\untrusted",
				"temp=C:\\Temp",
				StrixKnownHostsFileEnv + "=" + f.trustPath,
				"PROCESS_SECRET=do-not-forward",
			}
		},
	}
}

func strixSSHTestFileDigest(path string) ([32]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return [32]byte{}, err
	}
	return sha256.Sum256(data), nil
}

func TestStrixPresenceCacheRoundtrip(t *testing.T) {
	defer func() {
		// restore
		_ = os.Remove(StrixPresenceFile)
	}()

	target := &StrixTarget{
		Mode:           "ssh",
		Host:           "test-strix",
		Reachable:      true,
		CPUModel:       "AMD Ryzen AI MAX+ 395",
		GPUName:        "AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
		TargetISA:      "gfx1151",
		ComputeUnits:   40,
		TotalRAMBytes:  68719476736,
		UMABufferBytes: 60129542144,
		DPMLevel:       "high",
		LockupTimeout:  -1,
		LatencyMS:      1.5,
		DiscoveredAt:   time.Now().UTC().Format(time.RFC3339),
	}

	savePresenceCache(target)
	loaded, ok := loadPresenceCache("test-strix")
	if !ok || loaded == nil {
		t.Fatal("expected loadPresenceCache to find saved cache")
	}

	if loaded.Host != "test-strix" {
		t.Errorf("loaded host = %q, want 'test-strix'", loaded.Host)
	}
	if loaded.ComputeUnits != 40 {
		t.Errorf("loaded CUs = %d, want 40", loaded.ComputeUnits)
	}
	if loaded.TargetISA != "gfx1151" {
		t.Errorf("loaded ISA = %q, want 'gfx1151'", loaded.TargetISA)
	}
}

func TestDiscoverStrixTargetSimulatedUnreachable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// An unreachable non-existent host should return reachable=false without crashing
	target, err := DiscoverStrixTarget(ctx, "nonexistent-strix-host-xyz-12345.invalid")
	if err == nil && target != nil && target.Reachable {
		t.Errorf("expected unreachable target for invalid host, got %v", target)
	}
}
