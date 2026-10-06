package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestWorktreePreservingCLIRejectsImplicitOrEmptyLeaseAuthority(t *testing.T) {
	t.Setenv("FAK_LEASE_ID", "ambient-is-not-explicit-authority")
	for _, tc := range []struct {
		name     string
		explicit map[string]bool
		lease    string
		pid      int
		sandbox  bool
	}{
		{"empty", map[string]bool{"owner-pid": true, "lease-id": true}, "", os.Getpid() + 1, false},
		{"whitespace", map[string]bool{"owner-pid": true, "lease-id": true}, " ", os.Getpid() + 1, false},
		{"ambient", map[string]bool{"owner-pid": true}, "ambient-is-not-explicit-authority", os.Getpid() + 1, false},
		{"transient-owner", map[string]bool{"owner-pid": true, "lease-id": true}, "admitted", os.Getpid(), false},
		{"localization", map[string]bool{"owner-pid": true, "lease-id": true}, "admitted", os.Getpid() + 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := worktreePreservingCLIIdentity(tc.explicit, tc.pid, tc.lease, tc.sandbox); err == nil {
				t.Fatal("invalid authority accepted")
			}
		})
	}
}

func TestWorktreePreservingPostprocessRetainsIncludesAndFailureEvidence(t *testing.T) {
	root := t.TempDir()
	include := filepath.Join(root, ".worktreeinclude")
	if err := os.WriteFile(include, []byte("source-scope must stay untouched"), 0o600); err != nil {
		t.Fatal(err)
	}
	capacity := workerworktree.CapacityAdvisory{}
	out := worktreePreservingResultOut(workerworktree.Result{OK: true, Path: root, BaseSHA: "exact-base"}, capacity, func(context.Context) error { return nil }, workerworktree.OwnerStamp{PID: os.Getpid()})
	if !out.PreserveExisting || out.Env[workerworktree.WorktreeDirEnv] != root {
		t.Fatalf("output=%+v", out)
	}
	b, err := os.ReadFile(include)
	if err != nil || string(b) != "source-scope must stay untouched" {
		t.Fatal("include postprocessing occurred")
	}
	failed := worktreePreservingResultOut(workerworktree.Result{OK: false, Code: "PREPARE_NOT_READY", Path: root, Preserved: true}, capacity, func(context.Context) error { return nil }, workerworktree.OwnerStamp{PID: os.Getpid()})
	if failed.Env != nil || !failed.Preserved || failed.Code != "PREPARE_NOT_READY" {
		t.Fatalf("failure evidence changed=%+v", failed)
	}
}

func TestWorktreePreservingPublicationRefusesNewFenceLoss(t *testing.T) {
	out := worktreePreservingResultOut(workerworktree.Result{OK: true, Path: t.TempDir()}, workerworktree.CapacityAdvisory{}, func(context.Context) error { return fmt.Errorf("STALE_LEASE") }, workerworktree.OwnerStamp{PID: os.Getpid()})
	if out.OK || out.Env != nil || out.Code != "PRESERVATION_ADMISSION_REFUSED" {
		t.Fatalf("publication output=%+v", out)
	}
}

func TestWorktreePreservingCLIHelperProcess(t *testing.T) {
	if os.Getenv("FAK_PRESERVING_CLI_FIXTURE") != "1" {
		return
	}
	for i, arg := range os.Args {
		if arg == "--" {
			worktreeWorkerPrepare(os.Args[i+1:])
			os.Exit(0)
		}
	}
	t.Fatal("fixture arguments absent")
}

func preservingCLIGit(t *testing.T, root string, input string, args ...string) string {
	t.Helper()
	git, err := exec.LookPath("git")
	if err != nil {
		t.Fatal(err)
	}
	git, err = filepath.Abs(git)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, git, args...)
	cmd.Env = preservingNativeFixtureEnv(os.Environ(), git, t.TempDir(), "", false)
	cmd.Dir = root
	if input != "" {
		cmd.Stdin = strings.NewReader(input)
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("fixture Git %v: %v %s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

// Native CLI rows that reach checkout policy require its exact qualified Git.
func requireQualifiedPreservingCLIGit(t *testing.T, cfg preservingNativeFixtureConfig) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, cfg.RealGit, "--version")
	cmd.Dir = cfg.RepoRoot
	cmd.Env = preservingNativeFixtureEnv(os.Environ(), cfg.RealGit, cfg.TemplateDir, "", false)
	output, err := cmd.CombinedOutput()
	if err != nil || ctx.Err() != nil {
		t.Fatalf("qualified Git version probe %q: %v (context %v); output=%q", cfg.RealGit, err, ctx.Err(), output)
	}
	if string(output) != "git version 2.45.0\n" {
		t.Skipf("qualified-git-2.45.0 prerequisite: native preserving CLI scenario requires exact Git 2.45.0; got %q", output)
	}
}

// Exercise the actual CLI branch above50 through the test executable. Every
// peer has an fsmonitor hook that would leave a sentinel if status were invoked.
// No ordinary Prepare, pool or sweep setup is involved.
func TestWorktreePreservingFullCLILeaseDiskAndAdvisoryIsolation(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("native executable Git/fsmonitor integration fixture; Windows qualification separate")
	}
	for _, kind := range []string{"lease-refused", "disk-refused", "admitted-leading", "admitted-trailing", "intent-leading", "intent-trailing", "publication-revoked", "success", "hostile-environment-success"} {
		t.Run(kind, func(t *testing.T) {
			config := filepath.Join(t.TempDir(), "gitconfig")
			if err := os.WriteFile(config, nil, 0o600); err != nil {
				t.Fatal(err)
			}
			t.Setenv("GIT_CONFIG_COUNT", "0")
			t.Setenv("GIT_CONFIG_PARAMETERS", "")
			t.Setenv("GIT_CONFIG_GLOBAL", config)
			t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
			t.Setenv("GIT_OPTIONAL_LOCKS", "0")
			repo := t.TempDir()
			preservingCLIGit(t, repo, "", "init", "-q", "-b", "main")
			preservingCLIGit(t, repo, "", "config", "user.name", "CLI fixture")
			preservingCLIGit(t, repo, "", "config", "user.email", "fixture@test")
			preservingCLIGit(t, repo, "", "config", "commit.gpgsign", "false")
			if err := os.WriteFile(filepath.Join(repo, "target.txt"), []byte("base\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			preservingCLIGit(t, repo, "", "add", "target.txt")
			preservingCLIGit(t, repo, "", "commit", "-q", "-m", "fixture")
			base := preservingCLIGit(t, repo, "", "rev-parse", "HEAD")
			preservingCLIGit(t, repo, "", "config", "extensions.worktreeConfig", "true")
			peers := filepath.Join(t.TempDir(), "peers")
			marker := filepath.Join(t.TempDir(), "status-ran")
			realGit, err := exec.LookPath("git")
			if err != nil {
				t.Fatal(err)
			}
			realGit, err = filepath.Abs(realGit)
			if err != nil {
				t.Fatal(err)
			}
			cfg := preservingNativeFixtureConfig{RealGit: realGit, RepoRoot: repo, FsmonitorMarker: marker, TemplateDir: t.TempDir()}
			switch kind {
			case "publication-revoked", "success", "hostile-environment-success":
				requireQualifiedPreservingCLIGit(t, cfg)
			}
			cfgPath := filepath.Join(t.TempDir(), "native-helper.json")
			writePreservingNativeFixtureConfig(t, cfgPath, cfg)
			t.Setenv("FAK_PRESERVING_NATIVE_HELPER_CONFIG", cfgPath)
			hook := preservingNativeFixtureBinary(t, t.TempDir(), "fak-preserving-fsmonitor")
			for i := 0; i < 51; i++ {
				name := fmt.Sprintf("fak-worker-wt-cmd-peer%d", i)
				admin := filepath.Join(repo, ".git", "worktrees", name)
				peer := filepath.Join(peers, name)
				if err := os.MkdirAll(admin, 0o700); err != nil {
					t.Fatal(err)
				}
				if err := os.MkdirAll(peer, 0o700); err != nil {
					t.Fatal(err)
				}
				for path, data := range map[string]string{filepath.Join(admin, "HEAD"): base + "\n", filepath.Join(admin, "commondir"): "../..\n", filepath.Join(admin, "gitdir"): filepath.Join(peer, ".git") + "\n", filepath.Join(peer, ".git"): "gitdir: " + admin + "\n", filepath.Join(admin, "config.worktree"): "[core]\n fsmonitor = " + hook + "\n"} {
					if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
						t.Fatal(err)
					}
				}
			}
			// Give the positive-control peer a real index/source so a status probe
			// cannot short-circuit on an empty synthetic registration.
			peerZero := filepath.Join(peers, "fak-worker-wt-cmd-peer0")
			if err := os.WriteFile(filepath.Join(peerZero, "target.txt"), []byte("base\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			indexCtx, indexCancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer indexCancel()
			indexCmd := exec.CommandContext(indexCtx, realGit, "read-tree", base)
			indexCmd.Dir = repo
			indexCmd.Env = append(preservingNativeFixtureEnv(os.Environ(), realGit, cfg.TemplateDir, "", false), "GIT_INDEX_FILE="+filepath.Join(repo, ".git", "worktrees", "fak-worker-wt-cmd-peer0", "index"))
			if output, err := indexCmd.CombinedOutput(); err != nil {
				t.Fatalf("peer index: %v %s", err, output)
			}
			lease := leaseref.Record{ID: "fixture-lease", Holder: "fixture-holder", Generation: 1, AcquiredAt: time.Now().Unix(), TTLSeconds: 300, TreeGlobs: []string{"target.txt"}}
			initialLeaseOID := ""
			if kind != "lease-refused" {
				data, err := json.Marshal(lease)
				if err != nil {
					t.Fatal(err)
				}
				oid := preservingCLIGit(t, repo, string(data), "hash-object", "-w", "--stdin")
				initialLeaseOID = oid
				preservingCLIGit(t, repo, "", "update-ref", lease.Ref(), oid)
			}
			reserve := "1"
			if kind == "disk-refused" {
				reserve = strconv.FormatInt(1<<62, 10)
			}
			args := []string{"-test.run=^TestWorktreePreservingCLIHelperProcess$", "--", "--preserve-existing", "--root", repo, "--wt-root", "workers", "--lane", "cmd", "--key", kind, "--base-sha", base, "--owner-pid", strconv.Itoa(os.Getpid()), "--lease-id", lease.ID, "--lease-holder", lease.Holder, "--lease-generation", "1", "--admitted-path", "target.txt", "--reserve-bytes", reserve}
			switch kind {
			case "admitted-leading":
				for i, a := range args {
					if a == "--admitted-path" {
						args[i+1] = " target.txt"
					}
				}
			case "admitted-trailing":
				for i, a := range args {
					if a == "--admitted-path" {
						args[i+1] = "target.txt "
					}
				}
			case "intent-leading":
				args = append(args, "--message", "immutable fixture intent", "--path", " target.txt")
			case "intent-trailing":
				args = append(args, "--message", "immutable fixture intent", "--path", "target.txt ")
			}
			var shimDir, revocationMarker string
			if kind == "publication-revoked" {
				newLease := lease
				newLease.Generation = 2
				newLease.Holder = "new-fixture-holder"
				data, err := json.Marshal(newLease)
				if err != nil {
					t.Fatal(err)
				}
				revokedOID := preservingCLIGit(t, repo, string(data), "hash-object", "-w", "--stdin")
				physicalRepo, _, err := workerworktree.CanonicalPreservingRoots(repo, "workers")
				if err != nil {
					t.Fatal(err)
				}
				target := workerworktree.Path("cmd", kind, filepath.Join(physicalRepo, "workers"))
				ownerPath := workerworktree.OwnerStampPath(target)
				shimDir = t.TempDir()
				queryMarker := filepath.Join(shimDir, "final-query")
				resourceMarker := filepath.Join(shimDir, "native-final-resource")
				revocationMarker = filepath.Join(shimDir, "revoked")
				// Native helper proxies real Git and advances the actual isolated
				// lease only during CLI publication's resource query, after the
				// primitive's final registration/resource/fence sequence.
				cfg.OwnerPath = ownerPath
				cfg.QueryMarker = queryMarker
				cfg.ResourceMarker = resourceMarker
				cfg.RevocationMarker = revocationMarker
				cfg.LeaseRef = lease.Ref()
				cfg.InitialOID = initialLeaseOID
				cfg.ReplacementOID = revokedOID
				writePreservingNativeFixtureConfig(t, cfgPath, cfg)
				preservingNativeFixtureBinary(t, shimDir, "git")
			}
			var hostileConfig, hostileBefore string
			if kind == "hostile-environment-success" {
				hostileConfig, hostileBefore = preservingCLIHostileEnvironment(t)
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], args...)
			cmd.Env = preservingNativeFixtureEnv(os.Environ(), realGit, cfg.TemplateDir, cfgPath, true)
			if shimDir != "" {
				cmd.Env = append(cmd.Env, "PATH="+shimDir+string(os.PathListSeparator)+preservingNativeFixturePath(realGit))
			}
			stdout, err := cmd.Output()
			var out worktreePrepareOut
			if decodeErr := json.Unmarshal(stdout, &out); decodeErr != nil {
				t.Fatalf("CLI output %s: %v (%v)", stdout, decodeErr, err)
			}
			if kind == "success" || kind == "hostile-environment-success" {
				if err != nil || !out.OK || !out.PreserveExisting || !filepath.IsAbs(out.Path) || out.Env == nil {
					t.Fatalf("CLI success=%+v err=%v", out, err)
				}
			} else {
				if err == nil || out.OK || out.Env != nil || out.Code != "PRESERVATION_ADMISSION_REFUSED" {
					t.Fatalf("CLI refusal=%+v err=%v", out, err)
				}
			}
			if kind == "publication-revoked" {
				if _, err := os.Lstat(revocationMarker); err != nil {
					t.Fatalf("actual CLI publication revocation never fired: %v", err)
				}
				if !strings.HasPrefix(out.Reason, "publication refused:") {
					t.Fatalf("refusal did not reach CLI publication: %+v", out)
				}
				var current leaseref.Record
				raw := preservingCLIGit(t, repo, "", "cat-file", "blob", lease.Ref())
				if err := json.Unmarshal([]byte(raw), &current); err != nil || current.Generation != 2 || current.Holder != "new-fixture-holder" {
					t.Fatalf("actual authority unchanged: %+v %v", current, err)
				}
			}
			if out.Capacity.CurrentCount != 51 || len(out.Capacity.ContractionRecommendations) != 0 {
				t.Fatalf("capacity inventory=%+v", out.Capacity)
			}
			if _, err := os.Lstat(marker); !os.IsNotExist(err) {
				t.Fatal("preservation CLI ran status on an existing worker")
			}
			if hostileConfig != "" {
				if data, err := os.ReadFile(hostileConfig); err != nil || string(data) != hostileBefore {
					t.Fatalf("ambient Git config modified: %v", err)
				}
				if _, err := os.Lstat(filepath.Join(repo, ".git", "hooks", "post-checkout")); !os.IsNotExist(err) {
					t.Fatal("ambient template imported")
				}
			}
			// Positive control runs only after the CLI no-peer-status assertion.
			preservingNativeFixtureGitWithConfig(t, peerZero, realGit, cfg.TemplateDir, cfgPath, "status", "--porcelain")
			if data, err := os.ReadFile(marker); err != nil || len(data) == 0 {
				t.Fatalf("peer fsmonitor sentinel positive control did not fire: %v", err)
			}
		})
	}
}

// Named aliases of the already built test executable run native fixture roles
// in init, before testing flag parsing. No shell, go run or nested compilation.
func init() {
	role := strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe")
	if role != "git" && role != "fak-preserving-fsmonitor" {
		return
	}
	path := os.Getenv("FAK_PRESERVING_NATIVE_HELPER_CONFIG")
	if path == "" {
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		preservingNativeFixtureFatal(err)
	}
	var cfg preservingNativeFixtureConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		preservingNativeFixtureFatal(err)
	}
	if role == "fak-preserving-fsmonitor" {
		file, err := os.OpenFile(cfg.FsmonitorMarker, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			preservingNativeFixtureFatal(err)
		}
		if _, err := file.WriteString("touched"); err != nil {
			preservingNativeFixtureFatal(err)
		}
		if err := file.Close(); err != nil {
			preservingNativeFixtureFatal(err)
		}
		// Git fsmonitor protocol2 uses a NUL-terminated token, with no changed paths.
		if _, err := os.Stdout.Write([]byte("fixture-token\x00")); err != nil {
			preservingNativeFixtureFatal(err)
		}
		os.Exit(0)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, cfg.RealGit, os.Args[1:]...)
	cmd.Env = preservingNativeFixtureEnv(os.Environ(), cfg.RealGit, cfg.TemplateDir, path, false)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			os.Exit(exit.ExitCode())
		}
		preservingNativeFixtureFatal(err)
	}
	args := strings.Join(os.Args[1:], " ")
	if strings.Contains(args, "worktree list --porcelain") && cfg.OwnerPath != "" {
		if _, err := os.Lstat(cfg.OwnerPath); err == nil {
			if err := os.WriteFile(cfg.QueryMarker, []byte("query"), 0o600); err != nil {
				preservingNativeFixtureFatal(err)
			}
		}
	}
	if strings.Contains(args, "ls-tree -rl ") && cfg.QueryMarker != "" {
		if _, err := os.Lstat(cfg.QueryMarker); err == nil {
			file, err := os.OpenFile(cfg.ResourceMarker, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
			if err == nil {
				if _, err := file.WriteString("native final resource"); err != nil {
					preservingNativeFixtureFatal(err)
				}
				if err := file.Close(); err != nil {
					preservingNativeFixtureFatal(err)
				}
			} else if os.IsExist(err) {
				cas := exec.CommandContext(ctx, cfg.RealGit, "-C", cfg.RepoRoot, "update-ref", cfg.LeaseRef, cfg.ReplacementOID, cfg.InitialOID)
				cas.Env = preservingNativeFixtureEnv(os.Environ(), cfg.RealGit, cfg.TemplateDir, "", false)
				if output, err := cas.CombinedOutput(); err != nil {
					preservingNativeFixtureFatal(fmt.Errorf("fixture authority CAS: %w %s", err, output))
				}
				if err := os.WriteFile(cfg.RevocationMarker, []byte("revoked"), 0o600); err != nil {
					preservingNativeFixtureFatal(err)
				}
			} else {
				preservingNativeFixtureFatal(err)
			}
		}
	}
	os.Exit(0)
}

type preservingNativeFixtureConfig struct {
	TemplateDir      string
	RealGit          string
	RepoRoot         string
	FsmonitorMarker  string
	OwnerPath        string
	QueryMarker      string
	ResourceMarker   string
	RevocationMarker string
	LeaseRef         string
	InitialOID       string
	ReplacementOID   string
}

func preservingNativeFixtureFatal(err error) {
	fmt.Fprintln(os.Stderr, "native preserving fixture:", err)
	os.Exit(97)
}
func writePreservingNativeFixtureConfig(t *testing.T, path string, cfg preservingNativeFixtureConfig) {
	t.Helper()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
func preservingNativeFixtureBinary(t *testing.T, dir, name string) string {
	t.Helper()
	source, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	target := filepath.Join(dir, name)
	if err := os.Link(source, target); err == nil {
		return target
	}
	in, err := os.Open(source)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o700)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
	return target
}

// Allow OS process essentials only; never inherit Git, SSH, loader, proxy,
// credential, helper-dispatch or workspace-root injection settings. Explicit
// fixture overrides are added after filtering, including fsmonitor's native role.
func preservingNativeFixtureEnv(base []string, git, template, helperConfig string, cli bool) []string {
	var env []string
	for _, entry := range base {
		key, _, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		switch strings.ToUpper(key) {
		case "SYSTEMROOT", "WINDIR", "TMPDIR", "TMP", "TEMP":
			env = append(env, entry)
		}
	}
	env = append(env, "PATH="+preservingNativeFixturePath(git), "HOME="+template, "USERPROFILE="+template, "XDG_CONFIG_HOME="+template, "GOMAXPROCS=2", "GIT_CONFIG_GLOBAL="+os.DevNull, "GIT_CONFIG_SYSTEM="+os.DevNull, "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_COUNT=0", "GIT_TEMPLATE_DIR="+template, "GIT_TERMINAL_PROMPT=0", "GIT_OPTIONAL_LOCKS=0", "GIT_ALLOW_PROTOCOL=file")
	if helperConfig != "" {
		env = append(env, "FAK_PRESERVING_NATIVE_HELPER_CONFIG="+helperConfig)
	}
	if cli {
		env = append(env, "FAK_PRESERVING_CLI_FIXTURE=1")
	}
	return env
}
func preservingNativeFixturePath(git string) string {
	if runtime.GOOS == "windows" {
		return filepath.Dir(git) + string(os.PathListSeparator) + filepath.Join(os.Getenv("SystemRoot"), "System32")
	}
	return filepath.Dir(git) + string(os.PathListSeparator) + "/usr/bin" + string(os.PathListSeparator) + "/bin"
}
func preservingNativeFixtureGitWithConfig(t *testing.T, root, git, template, config string, args ...string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, git, args...)
	cmd.Dir = root
	cmd.Env = preservingNativeFixtureEnv(os.Environ(), git, template, config, false)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("explicit fixture Git: %v %s", err, out)
	}
}

func TestWorktreePreservingFixtureEnvironmentConfinement(t *testing.T) {
	git := filepath.Join(t.TempDir(), "git")
	template := t.TempDir()
	hostile := []string{"GIT_CONFIG", "GIT_TEMPLATE_DIR", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_EXEC_PATH", "GIT_NAMESPACE", "GIT_CEILING_DIRECTORIES", "GIT_DISCOVERY_ACROSS_FILESYSTEM", "GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS", "GIT_TRACE", "GIT_TRACE2_EVENT", "GIT_TRACE_PACK_ACCESS", "GIT_SSH", "GIT_SSH_COMMAND", "GIT_SSH_VARIANT", "GIT_ASKPASS", "SSH_ASKPASS", "SSH_AUTH_SOCK", "SSH_AGENT_PID", "FAK_WORKSPACE_ROOT", "FAK_LEASE_ID", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "HOME", "USERPROFILE", "XDG_CONFIG_HOME", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "BASH_ENV", "ENV", "FAK_PRESERVING_NATIVE_HELPER_CONFIG", "FAK_PRESERVING_CLI_FIXTURE", "PATH"}
	var base []string
	for _, key := range hostile {
		base = append(base, key+"=ambient-poison")
	}
	base = append(base, "TMPDIR="+template)
	got := map[string]string{}
	for _, entry := range preservingNativeFixtureEnv(base, git, template, "", false) {
		key, value, ok := strings.Cut(entry, "=")
		if !ok {
			t.Fatal("malformed fixture environment")
		}
		if _, exists := got[key]; exists {
			t.Fatalf("duplicate environment key %s", key)
		}
		got[key] = value
	}
	for _, key := range hostile {
		if got[key] == "ambient-poison" {
			t.Fatalf("inherited injection %s", key)
		}
	}
	for key, value := range map[string]string{"GIT_CONFIG_GLOBAL": os.DevNull, "GIT_CONFIG_SYSTEM": os.DevNull, "GIT_CONFIG_COUNT": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_TEMPLATE_DIR": template, "GIT_ALLOW_PROTOCOL": "file", "HOME": template, "TMPDIR": template} {
		if got[key] != value {
			t.Fatalf("%s=%q, want %q", key, got[key], value)
		}
	}
	if !strings.HasPrefix(got["PATH"], filepath.Dir(git)+string(os.PathListSeparator)) {
		t.Fatal("unbound Git path")
	}
}

func preservingCLIHostileEnvironment(t *testing.T) (string, string) {
	t.Helper()
	root := t.TempDir()
	template := filepath.Join(root, "template")
	hooks := filepath.Join(template, "hooks")
	if err := os.MkdirAll(hooks, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(hooks, "post-checkout"), []byte("nonexecutable hostile-template sentinel"), 0600); err != nil {
		t.Fatal(err)
	}
	config := filepath.Join(root, "redirected-config")
	before := "[core]\n hooksPath = " + hooks + "\n"
	if err := os.WriteFile(config, []byte(before), 0600); err != nil {
		t.Fatal(err)
	}
	for key, value := range map[string]string{"GIT_CONFIG": config, "GIT_TEMPLATE_DIR": template, "GIT_CONFIG_PARAMETERS": "invalid hostile config parameters", "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": hooks, "GIT_DIR": filepath.Join(root, "absent-git"), "GIT_COMMON_DIR": filepath.Join(root, "absent-common"), "GIT_WORK_TREE": root, "GIT_INDEX_FILE": filepath.Join(root, "redirected-index"), "GIT_OBJECT_DIRECTORY": filepath.Join(root, "redirected-objects"), "GIT_ALTERNATE_OBJECT_DIRECTORIES": filepath.Join(root, "absent-alternates"), "GIT_EXEC_PATH": root, "GIT_SSH_COMMAND": "nonexistent-fixture-ssh", "GIT_SSH": filepath.Join(root, "absent-ssh"), "GIT_ASKPASS": filepath.Join(root, "absent-askpass"), "SSH_AUTH_SOCK": filepath.Join(root, "absent-agent"), "GIT_TRACE": filepath.Join(root, "redirected-trace"), "FAK_WORKSPACE_ROOT": root} {
		t.Setenv(key, value)
	}
	t.Cleanup(func() {
		for _, name := range []string{"redirected-index", "redirected-objects", "redirected-trace"} {
			if _, err := os.Lstat(filepath.Join(root, name)); !os.IsNotExist(err) {
				t.Errorf("ambient redirect written: %s (%v)", name, err)
			}
		}
	})
	return config, before
}

func TestWorktreePreservingFixtureHostileGitInit(t *testing.T) {
	config, before := preservingCLIHostileEnvironment(t)
	repo := t.TempDir()
	preservingCLIGit(t, repo, "", "init", "-q")
	preservingCLIGit(t, repo, "", "config", "user.name", "confined fixture")
	if data, err := os.ReadFile(config); err != nil || string(data) != before {
		t.Fatalf("ambient config changed: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(repo, ".git", "hooks", "post-checkout")); !os.IsNotExist(err) {
		t.Fatal("ambient template imported")
	}
}

// Fail qualification if the documentation gates read an ambient checkout.
// The native validator runs package tests from the extracted candidate tree.
func TestWorktreePreservingDocumentationCandidateBinding(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	candidate := cwd
	for {
		if _, err := os.Stat(filepath.Join(candidate, "go.mod")); err == nil {
			break
		}
		parent := filepath.Dir(candidate)
		if parent == candidate {
			t.Fatal("test working directory is not inside the extracted candidate")
		}
		candidate = parent
	}
	if ambient := os.Getenv("FAK_WORKSPACE_ROOT"); ambient != "" {
		t.Fatalf("qualification must clear FAK_WORKSPACE_ROOT before launcher, got %q", ambient)
	}
	if got := repoRoot(); got != candidate {
		t.Fatalf("documentation root=%q, candidate cwd root=%q", got, candidate)
	}
	const rel = "docs/managed-worker-worktrees.md"
	expected, err := os.ReadFile(filepath.Join(candidate, rel))
	if err != nil {
		t.Fatal(err)
	}
	if got := readRepoFile(t, rel); got != string(expected) || !strings.Contains(got, "--preserve-existing") {
		t.Fatal("documentation gate did not read preservation candidate guide")
	}
}
