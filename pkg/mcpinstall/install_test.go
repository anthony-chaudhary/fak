package mcpinstall

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func fixture(t *testing.T, config string) Spec {
	t.Helper()
	dir := t.TempDir()
	source := filepath.Join(dir, "reviewed-server")
	write(t, source, []byte("reviewed executable bytes; never executed"))
	path := filepath.Join(dir, "config.toml")
	write(t, path, []byte(config))
	if err := os.Mkdir(filepath.Join(dir, "installed"), 0o700); err != nil {
		t.Fatal(err)
	}
	return Spec{Server: "reviewed_sidecar", SourcePath: source, SourceSHA256: hash(read(t, source)),
		InstallDir: filepath.Join(dir, "installed"), ConfigPath: path,
		ExpectedConfigSHA256: hash([]byte(config)), WorkingDir: dir, Args: []string{"mcp", "--stdio"}}
}

// Test-only software guard exercises the append algorithm on Unix without
// exposing an advisory installation mode to SDK callers. Windows uses the real
// native guard for every lifecycle fixture.
func installFixture(p *Plan) (Receipt, error) {
	if runtime.GOOS == "windows" {
		return p.Install()
	}
	return p.install(func(path string) (*os.File, error) { return os.OpenFile(path, os.O_RDWR, 0) })
}

func rollbackFixture(r Receipt) error {
	if runtime.GOOS == "windows" {
		return Rollback(r)
	}
	return rollback(r, func(path string) (*os.File, error) { return os.OpenFile(path, os.O_RDWR, 0) })
}

func write(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
func read(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// fak-test:runtime fast est=30ms
func TestInstallAndDurableRollbackPreserveConfig(t *testing.T) {
	t.Parallel()
	prior := "# exact comment\r\napproval_policy = 'never'\r\nsandbox_mode = 'read-only'\r\n\r\n[mcp_servers.existing]\r\ncommand = 'keep'\r\nargs = []\r\n[profiles.work]\r\nmodel = 'chosen'\r\n # trailing space  "
	spec := fixture(t, prior)
	plan, err := Prepare(spec)
	if err != nil {
		t.Fatal(err)
	}
	if entries, err := os.ReadDir(spec.InstallDir); err != nil || len(entries) != 0 {
		t.Fatalf("Prepare wrote artifacts: %v %v", entries, err)
	}
	preview := plan.Preview()
	preview.Args[0], spec.Args[0] = "mutated", "mutated"
	beforeInfo, err := os.Stat(spec.ConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := installFixture(plan)
	if err != nil {
		t.Fatal(err)
	}
	wantBlock := RenderCodexSection(receipt.Server, receipt.Command, []string{"mcp", "--stdio"}, spec.WorkingDir)
	next := read(t, spec.ConfigPath)
	if !bytes.HasPrefix(next, []byte(prior)) || !bytes.HasSuffix(next, []byte(wantBlock)) || receipt.ConfigSHA256 != hash(next) {
		t.Fatalf("config was not an exact append: %q", next)
	}
	if strings.Contains(string(next), "mutated") || !strings.Contains(string(next), "cwd = ") {
		t.Fatalf("launch fields changed: %q", next)
	}
	if hash(read(t, receipt.Command)) != spec.SourceSHA256 {
		t.Fatal("installed executable differs")
	}
	afterInfo, err := os.Stat(spec.ConfigPath)
	if err != nil || !os.SameFile(beforeInfo, afterInfo) {
		t.Fatal("installation replaced config identity")
	}
	if receipt.Guard != platformGuard || receipt.Durability != platformDurability || receipt.ConfigIdentity == "" || receipt.UpdateMethod != "append-owned-suffix;rollback-truncate-owned-suffix" {
		t.Fatalf("missing platform guarantees: %+v", receipt)
	}
	info, _ := os.Stat(receipt.ReceiptPath)
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		t.Fatalf("rollback record exposes original config: %v", info.Mode())
	}
	// Serialize the handle to prove rollback survives loss of the in-memory Plan.
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	var restored Receipt
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatal(err)
	}
	if err := rollbackFixture(restored); err != nil {
		t.Fatal(err)
	}
	if got := read(t, spec.ConfigPath); string(got) != prior {
		t.Fatalf("rollback changed prior bytes: %q", got)
	}
	if err := rollbackFixture(restored); err == nil {
		t.Fatal("second rollback accepted stale installed hash")
	}
	if runtime.GOOS == "windows" {
		// Existing bytes do not establish a fresh WRITE_THROUGH publication.
		if _, err := plan.Install(); err == nil {
			t.Fatal("reinstall asserted durability of existing publication")
		}
	} else if _, err := installFixture(plan); err != nil {
		t.Fatalf("software fixture immutable reuse: %v", err)
	}
}

// fak-test:runtime fast est=30ms
func TestInstallRefusesChangedInputsAndCollisions(t *testing.T) {
	t.Parallel()
	for _, change := range []string{"config", "source", "destination", "guard"} {
		t.Run(change, func(t *testing.T) {
			spec := fixture(t, "approval_policy = 'never'\n")
			plan, err := Prepare(spec)
			if err != nil {
				t.Fatal(err)
			}
			before := read(t, spec.ConfigPath)
			switch change {
			case "config":
				before = append(before, []byte("# concurrent edit\n")...)
				write(t, spec.ConfigPath, before)
			case "source":
				write(t, spec.SourcePath, []byte("changed source"))
			case "destination":
				if err := os.MkdirAll(spec.InstallDir, 0o700); err != nil {
					t.Fatal(err)
				}
				write(t, plan.Preview().Command, []byte("foreign destination"))
			}
			var installErr error
			if change == "guard" {
				_, installErr = plan.install(func(string) (*os.File, error) { return nil, errors.New("native guard refused") })
			} else {
				_, installErr = installFixture(plan)
			}
			if installErr == nil {
				t.Fatal("accepted changed state")
			}
			if !bytes.Equal(read(t, spec.ConfigPath), before) {
				t.Fatal("refusal changed config")
			}
			if change == "destination" && string(read(t, plan.Preview().Command)) != "foreign destination" {
				t.Fatal("collision destroyed destination")
			}
		})
	}
	for _, config := range []string{
		"[mcp_servers.reviewed_sidecar]\ncommand = 'foreign'\n",
		"[mcp_servers.reviewed_sidecar.env]\nSECRET = 'keep'\n",
		"[[mcp_servers.reviewed_sidecar]]\n",
		"[[mcp_servers]]\n",
		`["mcp\u005fservers".'reviewed_sidecar']` + "\n",
		"mcp_servers = { reviewed_sidecar = {} }\n",
		"[mcp_servers]\nreviewed_sidecar.command = 'foreign'\n",
		"instructions = '''\nmultiline\n'''\n",
	} {
		spec := fixture(t, config)
		if _, err := Prepare(spec); err == nil {
			t.Fatalf("accepted namespace ambiguity: %q", config)
		}
		if string(read(t, spec.ConfigPath)) != config {
			t.Fatal("Prepare mutated collision")
		}
	}
}

// fak-test:runtime fast est=20ms
func TestRollbackRefusesChangedRecordOrConfig(t *testing.T) {
	t.Parallel()
	for _, change := range []string{"record", "config", "handle", "partial-suffix", "legacy-v1"} {
		t.Run(change, func(t *testing.T) {
			spec := fixture(t, "approval_policy = 'never'\n")
			plan, err := Prepare(spec)
			if err != nil {
				t.Fatal(err)
			}
			receipt, err := installFixture(plan)
			if err != nil {
				t.Fatal(err)
			}
			switch change {
			case "record":
				write(t, receipt.ReceiptPath, []byte("{}"))
			case "config":
				write(t, spec.ConfigPath, append(read(t, spec.ConfigPath), []byte("# peer edit\n")...))
			case "handle":
				receipt.Args = []string{"other"}
			case "partial-suffix":
				if err := os.Truncate(spec.ConfigPath, int64(len(plan.prior)+3)); err != nil {
					t.Fatal(err)
				}
			case "legacy-v1":
				bound := receipt
				bound.ReceiptSHA256 = ""
				legacy, err := json.Marshal(map[string]any{"schema": "fak.mcpinstall.v1", "receipt": bound, "prior": plan.prior, "mode": 0o600})
				if err != nil {
					t.Fatal(err)
				}
				write(t, receipt.ReceiptPath, legacy)
				receipt.ReceiptSHA256 = hash(legacy) // valid record integrity must not admit the old protocol
			}
			before := read(t, spec.ConfigPath)
			if err := rollbackFixture(receipt); err == nil {
				t.Fatal("accepted changed rollback state")
			}
			if !bytes.Equal(read(t, spec.ConfigPath), before) {
				t.Fatal("rollback refusal overwrote config")
			}
		})
	}
}

// fak-test:runtime fast est=10ms
func TestGuardedAppendPreservesChangedPrefix(t *testing.T) {
	t.Parallel()
	spec := fixture(t, "# original config\n")
	before := read(t, spec.ConfigPath)
	info, err := os.Stat(spec.ConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(spec.ConfigPath, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	identity, err := configIdentity(f)
	if err != nil {
		t.Fatal(err)
	}
	changed := append(append([]byte{}, before...), []byte("# peer edit\n")...)
	write(t, spec.ConfigPath, changed)
	err = appendConfig(f, identity, before, append(before, []byte("candidate")...))
	if err == nil || !bytes.Equal(read(t, spec.ConfigPath), changed) {
		t.Fatalf("prefix refusal destroyed destination: %v", err)
	}
	after, err := os.Stat(spec.ConfigPath)
	if err != nil || !os.SameFile(info, after) || info.Mode() != after.Mode() {
		t.Fatal("append refusal replaced destination identity/mode")
	}
	matches, err := filepath.Glob(filepath.Join(filepath.Dir(spec.ConfigPath), ".mcpinstall-*"))
	if err != nil || len(matches) != 0 {
		t.Fatalf("staging leftovers: %v %v", matches, err)
	}
}

// fak-test:runtime fast est=20ms
func TestStrongNativeGuard(t *testing.T) {
	t.Parallel()
	spec := fixture(t, "approval_policy = 'never'\n")
	plan, err := Prepare(spec)
	if err != nil {
		t.Fatal(err)
	}
	before := read(t, spec.ConfigPath)
	if runtime.GOOS != "windows" {
		if _, err := plan.Install(); err == nil {
			t.Fatal("Unix public Install silently selected an advisory guard")
		}
		entries, err := os.ReadDir(spec.InstallDir)
		if err != nil || len(entries) != 0 || !bytes.Equal(read(t, spec.ConfigPath), before) {
			t.Fatal("unsupported-platform refusal had effects")
		}
		return
	}
	f, err := openGuardedConfig(spec.ConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if peer, err := os.OpenFile(spec.ConfigPath, os.O_WRONLY, 0); err == nil {
		peer.Close()
		t.Fatal("ordinary writer opened while native guard held")
	}
	replacement := filepath.Join(spec.WorkingDir, "peer-replacement")
	write(t, replacement, []byte("foreign replacement"))
	if err := os.Rename(replacement, spec.ConfigPath); err == nil {
		t.Fatal("ordinary replacer bypassed native guard")
	}
	if err := os.Remove(spec.ConfigPath); err == nil {
		t.Fatal("ordinary delete bypassed native guard")
	}
	if !bytes.Equal(read(t, spec.ConfigPath), before) {
		t.Fatal("guard contention changed prefix")
	}
	if _, err := plan.Install(); err == nil {
		t.Fatal("second installer opened concurrently")
	}
	// An existing ordinary writer must also prevent acquisition of our guard.
	f.Close()
	peer, err := os.OpenFile(spec.ConfigPath, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer peer.Close()
	if guarded, err := openGuardedConfig(spec.ConfigPath); err == nil {
		guarded.Close()
		t.Fatal("native guard ignored an existing ordinary writer")
	}
}

// fak-test:runtime fast est=20ms
func TestInstallRevalidatesWorkingDirectoryUnderGuard(t *testing.T) {
	t.Parallel()
	for _, change := range []string{"file", "symlink"} {
		t.Run(change, func(t *testing.T) {
			spec := fixture(t, "approval_policy = 'never'\n")
			cwd := filepath.Join(spec.WorkingDir, "launch-root")
			if err := os.Mkdir(cwd, 0o700); err != nil {
				t.Fatal(err)
			}
			spec.WorkingDir = cwd
			plan, err := Prepare(spec)
			if err != nil {
				t.Fatal(err)
			}
			before := read(t, spec.ConfigPath)
			open := func(path string) (*os.File, error) {
				var f *os.File
				var err error
				if runtime.GOOS == "windows" {
					f, err = openGuardedConfig(path)
				} else {
					f, err = os.OpenFile(path, os.O_RDWR, 0)
				}
				if err != nil {
					return nil, err
				}
				if err := os.Remove(cwd); err != nil {
					f.Close()
					t.Fatal(err)
				}
				if change == "file" {
					write(t, cwd, []byte("changed working directory"))
				} else if err := os.Symlink(filepath.Dir(cwd), cwd); err != nil {
					f.Close()
					t.Skipf("symlink unavailable: %v", err)
				}
				return f, nil
			}
			if _, err := plan.install(open); err == nil {
				t.Fatal("accepted working directory changed after native guard acquisition")
			}
			entries, err := os.ReadDir(spec.InstallDir)
			if err != nil || len(entries) != 0 || !bytes.Equal(read(t, spec.ConfigPath), before) {
				t.Fatal("working directory refusal had publication/config effects")
			}
		})
	}
}

// fak-test:runtime fast est=20ms
func TestRollbackRecordDoesNotCopyOtherConfigSecrets(t *testing.T) {
	t.Parallel()
	const secret = "unrelated-mcp-secret-must-stay-in-original-config"
	prior := "approval_policy = 'never'\n[mcp_servers.other.env]\nTOKEN = '" + secret + "'\n"
	spec := fixture(t, prior)
	plan, err := Prepare(spec)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := installFixture(plan)
	if err != nil {
		t.Fatal(err)
	}
	data := read(t, receipt.ReceiptPath)
	if bytes.Contains(data, []byte(secret)) || bytes.Contains(data, []byte(base64.StdEncoding.EncodeToString([]byte(prior)))) {
		t.Fatal("rollback record copied another server's config or secrets")
	}
	var record rollbackRecord
	if err := json.Unmarshal(data, &record); err != nil {
		t.Fatal(err)
	}
	if record.Schema != "fak.mcpinstall.v2" || record.PrefixLength != int64(len(prior)) ||
		!bytes.Equal(record.OwnedSuffix, plan.next[len(prior):]) || receipt.PriorConfigSHA256 != hash([]byte(prior)) {
		t.Fatal("rollback record does not bind exact prefix hash/length and owned suffix")
	}
	if err := rollbackFixture(receipt); err != nil {
		t.Fatal(err)
	}
	if string(read(t, spec.ConfigPath)) != prior {
		t.Fatal("rollback did not preserve unrelated secret bytes")
	}
}

// fak-test:runtime fast est=10ms
func TestPrepareRejectsUnboundHashAndSymlinks(t *testing.T) {
	t.Parallel()
	spec := fixture(t, "# original\n")
	bad := spec
	bad.SourceSHA256 = strings.Repeat("0", 64)
	if _, err := Prepare(bad); err == nil {
		t.Fatal("accepted unbound source")
	}
	bad = spec
	bad.ExpectedConfigSHA256 = strings.Repeat("0", 64)
	if _, err := Prepare(bad); err == nil {
		t.Fatal("accepted unbound config")
	}
	bad = spec
	bad.Server = "injected]\napproval_policy"
	if _, err := Prepare(bad); err == nil {
		t.Fatal("accepted injected namespace")
	}
	link := filepath.Join(spec.WorkingDir, "link")
	if err := os.Symlink(spec.ConfigPath, link); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	bad = spec
	bad.ConfigPath = link
	if _, err := Prepare(bad); err == nil {
		t.Fatal("accepted symlink config")
	}
}

// fak-test:runtime fast est=1ms
func TestLegacyRendererBytes(t *testing.T) {
	t.Parallel()
	want := "[mcp_servers.sample]\ncommand = 'C:\\bin\\server.exe'\nargs = ['serve', \"has'quote\", 'back\\slash']\n"
	if got := RenderCodexSection("sample", `C:\bin\server.exe`, []string{"serve", "has'quote", `back\slash`}, ""); got != want {
		t.Fatalf("legacy renderer changed:\ngot %q\nwant %q", got, want)
	}
}
