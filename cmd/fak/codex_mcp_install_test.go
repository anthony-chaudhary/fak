package main

import (
	"encoding/binary"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// ---- pure config-patching logic --------------------------------------------

func TestReplaceTOMLSectionPreservesUnrelated(t *testing.T) {
	cfg := `# top comment
model = "gpt-5"

[mcp_servers.other]
command = 'C:\tools\other.exe'

[mcp_servers.fak]
command = 'C:\stale\fak.exe'
args = ["serve", "--stdio"]

[profiles.dev]
approval = "never"
`
	block := renderCodexMCPSection("fak", `C:\install\fak-1-abcd.exe`, []string{"serve", "--stdio", "--policy", `C:\install\policy.json`})
	out, prior, existed := replaceTOMLSection(cfg, "mcp_servers.fak", block)
	if !existed {
		t.Fatal("expected existing section")
	}
	if !strings.Contains(prior, `C:\stale\fak.exe`) {
		t.Fatalf("prior did not capture old command: %q", prior)
	}
	for _, must := range []string{"# top comment", `model = "gpt-5"`, "[mcp_servers.other]", `C:\tools\other.exe`, "[profiles.dev]", `approval = "never"`} {
		if !strings.Contains(out, must) {
			t.Fatalf("unrelated content lost: missing %q\n%s", must, out)
		}
	}
	if strings.Contains(out, `C:\stale\fak.exe`) {
		t.Fatalf("stale command survived replace:\n%s", out)
	}
	if !strings.Contains(out, `C:\install\fak-1-abcd.exe`) {
		t.Fatalf("new command not written:\n%s", out)
	}
	// The replacement must round-trip through the reader the doctor/health uses.
	p := filepath.Join(t.TempDir(), "config.toml")
	if err := os.WriteFile(p, []byte(out), 0o600); err != nil {
		t.Fatal(err)
	}
	cmd, args, err := readCodexMCPEntry(p, "fak")
	if err != nil {
		t.Fatalf("re-read failed: %v", err)
	}
	if cmd != `C:\install\fak-1-abcd.exe` || strings.Join(args, "|") != `serve|--stdio|--policy|C:\install\policy.json` {
		t.Fatalf("round-trip mismatch cmd=%q args=%v", cmd, args)
	}
}

func TestReplaceTOMLSectionAppendsWhenAbsent(t *testing.T) {
	cfg := "model = \"gpt-5\"\n"
	block := renderCodexMCPSection("fak", "/opt/fak", []string{"serve", "--stdio"})
	out, _, existed := replaceTOMLSection(cfg, "mcp_servers.fak", block)
	if existed {
		t.Fatal("expected absent section")
	}
	if !strings.Contains(out, "model = \"gpt-5\"") || !strings.Contains(out, "[mcp_servers.fak]") {
		t.Fatalf("append lost content:\n%s", out)
	}
	// Idempotent second apply must not duplicate the header.
	out2, _, _ := replaceTOMLSection(out, "mcp_servers.fak", block)
	if strings.Count(out2, "[mcp_servers.fak]") != 1 {
		t.Fatalf("duplicate header after re-apply:\n%s", out2)
	}
}

func TestReplaceTOMLSectionConsumesChildSubtables(t *testing.T) {
	cfg := `[mcp_servers.fak]
command = 'old'
[mcp_servers.fak.env]
API_KEY = "secret"
[other]
x = 1
`
	block := renderCodexMCPSection("fak", "new", []string{"serve"})
	out, _, _ := replaceTOMLSection(cfg, "mcp_servers.fak", block)
	if strings.Contains(out, "mcp_servers.fak.env") || strings.Contains(out, "secret") {
		t.Fatalf("child sub-table survived replace:\n%s", out)
	}
	if !strings.Contains(out, "[other]") || !strings.Contains(out, "x = 1") {
		t.Fatalf("unrelated table lost:\n%s", out)
	}
}

func TestRemoveTOMLSection(t *testing.T) {
	cfg := `[a]
x = 1

[mcp_servers.fak]
command = 'gone'

[b]
y = 2
`
	out := removeTOMLSection(cfg, "mcp_servers.fak")
	if strings.Contains(out, "mcp_servers.fak") || strings.Contains(out, "'gone'") {
		t.Fatalf("section not removed:\n%s", out)
	}
	if !strings.Contains(out, "[a]") || !strings.Contains(out, "[b]") {
		t.Fatalf("neighbors lost:\n%s", out)
	}
	// Removing an absent section is a no-op.
	if got := removeTOMLSection(out, "mcp_servers.fak"); got != out {
		t.Fatal("removing absent section changed the document")
	}
}

func TestTomlLiteralOrBasicRoundTrips(t *testing.T) {
	cases := []string{`C:\install\fak.exe`, "/opt/fak", "plain", `has "quote"`, "with'apostrophe"}
	for _, in := range cases {
		enc := tomlLiteralOrBasic(in)
		got, err := parseTOMLString(enc)
		if err != nil {
			t.Fatalf("parse %q (enc %q): %v", in, enc, err)
		}
		if got != in {
			t.Fatalf("round-trip: in=%q enc=%q got=%q", in, enc, got)
		}
	}
}

func TestBinaryArchDetectsHeaders(t *testing.T) {
	dir := t.TempDir()
	// Minimal PE amd64: MZ, e_lfanew at 0x3C -> 0x80, "PE\0\0" + machine 0x8664.
	pe := make([]byte, 0x88)
	pe[0], pe[1] = 'M', 'Z'
	binary.LittleEndian.PutUint32(pe[0x3C:], 0x80)
	copy(pe[0x80:], []byte{'P', 'E', 0, 0})
	binary.LittleEndian.PutUint16(pe[0x84:], 0x8664)
	pePath := filepath.Join(dir, "amd64.exe")
	if err := os.WriteFile(pePath, pe, 0o644); err != nil {
		t.Fatal(err)
	}
	if got := binaryArch(pePath); got != "amd64" {
		t.Fatalf("PE amd64 -> %q", got)
	}
	// Minimal ELF arm64.
	elf := make([]byte, 64)
	elf[0], elf[1], elf[2], elf[3] = 0x7F, 'E', 'L', 'F'
	binary.LittleEndian.PutUint16(elf[18:], 0xB7)
	elfPath := filepath.Join(dir, "arm64.elf")
	if err := os.WriteFile(elfPath, elf, 0o644); err != nil {
		t.Fatal(err)
	}
	if got := binaryArch(elfPath); got != "arm64" {
		t.Fatalf("ELF arm64 -> %q", got)
	}
	// Unknown magic -> "" (no false wrong-arch).
	junk := filepath.Join(dir, "junk")
	if err := os.WriteFile(junk, []byte("not a binary at all really"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := binaryArch(junk); got != "" {
		t.Fatalf("junk -> %q, want empty", got)
	}
}

// ---- hermetic install/status/uninstall lifecycle (probe stubbed) -----------

// codexMCPFixture builds an isolated CODEX_HOME with a fake source binary and a
// valid policy, and stubs the initialize probe so no real process is spawned.
type codexMCPFixture struct {
	home    string
	config  string
	source  string
	policy  string
	restore func()
}

func newCodexMCPFixture(t *testing.T, probeOK bool) *codexMCPFixture {
	t.Helper()
	home := t.TempDir()
	config := filepath.Join(home, "config.toml")
	source := filepath.Join(t.TempDir(), "src-fak"+exeSuffix())
	// Arbitrary bytes -> binaryArch returns "" so the arch gate is skipped.
	if err := os.WriteFile(source, []byte("fake-fak-binary-v1-payload"), 0o755); err != nil {
		t.Fatal(err)
	}
	policy := filepath.Join(t.TempDir(), "policy.json")
	if err := os.WriteFile(policy, []byte(`{"capabilities":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	prev := codexMCPInitializeProbe
	codexMCPInitializeProbe = func(command string, args []string, timeout time.Duration) (bool, string) {
		if probeOK {
			return true, ""
		}
		return false, "INITIALIZE_TIMEOUT"
	}
	return &codexMCPFixture{home: home, config: config, source: source, policy: policy, restore: func() { codexMCPInitializeProbe = prev }}
}

func (f *codexMCPFixture) install(t *testing.T, extra ...string) int {
	t.Helper()
	argv := append([]string{"--config", f.config, "--source", f.source, "--policy", f.policy}, extra...)
	var out, errb strings.Builder
	return runCodexMCPInstall(&out, &errb, argv)
}

func TestCmdCodexRoutesMCPLifecycleBeforeLaunchGate(t *testing.T) {
	old := codexMCPExit
	called := false
	codexMCPExit = func(code int) {
		called = true
		if code != 0 {
			t.Fatalf("codex mcp help exit=%d", code)
		}
	}
	t.Cleanup(func() { codexMCPExit = old })

	cmdCodex([]string{"mcp", "help"})
	if !called {
		t.Fatal("fak codex mcp did not route to the MCP lifecycle command")
	}
}
func TestCodexMCPLifecycle(t *testing.T) {
	f := newCodexMCPFixture(t, true)
	defer f.restore()

	if code := f.install(t); code != 0 {
		t.Fatalf("install exit=%d", code)
	}
	// Config now points at an absolute path inside the install dir.
	cmd, args, err := readCodexMCPEntry(f.config, "fak")
	if err != nil {
		t.Fatalf("read entry: %v", err)
	}
	if !filepath.IsAbs(cmd) || !withinDir(codexMCPInstallDir(f.config), cmd) {
		t.Fatalf("installed command not in install dir: %q", cmd)
	}
	if len(args) < 4 || args[0] != "serve" || args[1] != "--stdio" || args[2] != "--policy" || !filepath.IsAbs(args[3]) {
		t.Fatalf("unexpected args: %v", args)
	}
	man, err := readCodexMCPManifest(f.config)
	if err != nil {
		t.Fatalf("manifest: %v", err)
	}
	if man.SHA256 == "" || man.InstalledCommand != cmd {
		t.Fatalf("manifest inconsistent: %+v", man)
	}
	if strings.TrimSpace(man.ModuleRevision) == "" {
		t.Fatal("installed manifest omitted source/module revision")
	}

	// status: the installed fake differs from the running test binary, so the
	// artifact is healthy-but-stale (advisory OK).
	st := classifyCodexMCPStatus(f.config, "fak", time.Second, true)
	if !st.OK || (st.State != "stale" && st.State != "healthy") {
		t.Fatalf("status after install: state=%q ok=%v detail=%q", st.State, st.OK, st.Detail)
	}
	if !st.Managed {
		t.Fatal("status should report managed")
	}

	// hand-edited: point config elsewhere.
	if err := patchCodexMCPConfig(f.config, "fak", "/somewhere/else/fak", args); err != nil {
		t.Fatal(err)
	}
	if st := classifyCodexMCPStatus(f.config, "fak", time.Second, false); st.State != "hand_edited" {
		t.Fatalf("expected hand_edited, got %q", st.State)
	}
	// Re-install restores the managed entry.
	if code := f.install(t); code != 0 {
		t.Fatalf("re-install exit=%d", code)
	}

	// checksum mismatch: tamper the installed binary (clear read-only first).
	_ = os.Chmod(cmd, 0o644)
	if err := os.WriteFile(cmd, []byte("tampered-different-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if st := classifyCodexMCPStatus(f.config, "fak", time.Second, false); st.State != "checksum_mismatch" {
		t.Fatalf("expected checksum_mismatch, got %q (%s)", st.State, st.Detail)
	}

	// missing binary.
	_ = os.Chmod(cmd, 0o644)
	if err := os.Remove(cmd); err != nil {
		t.Fatal(err)
	}
	if st := classifyCodexMCPStatus(f.config, "fak", time.Second, false); st.State != "missing_binary" {
		t.Fatalf("expected missing_binary, got %q", st.State)
	}

	// uninstall removes the entry and owned artifacts; a second call is a no-op.
	var o, e strings.Builder
	if code := runCodexMCPUninstall(&o, &e, []string{"--config", f.config}); code != 0 {
		t.Fatalf("uninstall exit=%d err=%s", code, e.String())
	}
	if st := classifyCodexMCPStatus(f.config, "fak", time.Second, false); st.State != "absent" {
		t.Fatalf("expected absent after uninstall, got %q", st.State)
	}
	o.Reset()
	e.Reset()
	if code := runCodexMCPUninstall(&o, &e, []string{"--config", f.config}); code != 0 {
		t.Fatalf("idempotent uninstall exit=%d", code)
	}
	if !strings.Contains(o.String(), "already clean") {
		t.Fatalf("second uninstall not idempotent: %q", o.String())
	}
}

func TestCodexMCPRollbackRestoresPriorEntry(t *testing.T) {
	f := newCodexMCPFixture(t, true)
	defer f.restore()

	// Seed a prior known-good entry.
	priorCmd := `C:\prior\fak.exe`
	if runtime.GOOS != "windows" {
		priorCmd = "/prior/fak"
	}
	if err := patchCodexMCPConfig(f.config, "fak", priorCmd, []string{"serve", "--stdio"}); err != nil {
		t.Fatal(err)
	}
	// Install over it (manifest captures the prior section).
	if code := f.install(t); code != 0 {
		t.Fatalf("install exit=%d", code)
	}
	if cmd, _, _ := readCodexMCPEntry(f.config, "fak"); cmd == priorCmd {
		t.Fatal("install did not repoint the config")
	}
	// Roll back.
	var o, e strings.Builder
	if code := runCodexMCPInstall(&o, &e, []string{"--config", f.config, "--rollback"}); code != 0 {
		t.Fatalf("rollback exit=%d err=%s", code, e.String())
	}
	cmd, _, err := readCodexMCPEntry(f.config, "fak")
	if err != nil {
		t.Fatalf("read after rollback: %v", err)
	}
	if cmd != priorCmd {
		t.Fatalf("rollback did not restore prior command: got %q want %q", cmd, priorCmd)
	}
}

// ---- regression: config-preservation & rollback/uninstall integrity --------

// A `[[array-of-tables]]` block following the fak section must survive a rewrite.
// Before the boundary fix, the section span ran through it to the next plain
// table, silently deleting the operator's unrelated config.
func TestReplaceTOMLSectionPreservesArrayOfTables(t *testing.T) {
	cfg := `[mcp_servers.fak]
command = 'old'
args = ["serve"]

# rules the operator cares about
[[shell_environment_policy.rules]]
name = "keep-path"
pattern = "PATH"

[[shell_environment_policy.rules]]
name = "keep-home"
pattern = "HOME"

[profiles.dev]
approval = "never"
`
	block := renderCodexMCPSection("fak", "new", []string{"serve", "--stdio"})
	out, _, existed := replaceTOMLSection(cfg, "mcp_servers.fak", block)
	if !existed {
		t.Fatal("expected existing section")
	}
	for _, must := range []string{
		"[[shell_environment_policy.rules]]", "keep-path", `pattern = "PATH"`,
		"keep-home", `pattern = "HOME"`, "# rules the operator cares about",
		"[profiles.dev]", `approval = "never"`,
	} {
		if !strings.Contains(out, must) {
			t.Fatalf("array-of-tables / neighbor content lost: missing %q\n%s", must, out)
		}
	}
	if strings.Contains(out, "'old'") {
		t.Fatalf("stale fak command survived:\n%s", out)
	}
	if strings.Count(out, "[[shell_environment_policy.rules]]") != 2 {
		t.Fatalf("expected both array-of-tables blocks intact:\n%s", out)
	}
}

// A comment line directly above the next table documents that table, so removing
// the fak block must leave it attached to what follows (not swallow it).
func TestRemoveTOMLSectionKeepsTrailingCommentWithNextTable(t *testing.T) {
	cfg := `[mcp_servers.fak]
command = 'gone'
args = ["serve"]

# this comment introduces the profile below
[profiles.dev]
approval = "never"
`
	out := removeTOMLSection(cfg, "mcp_servers.fak")
	if strings.Contains(out, "mcp_servers.fak") || strings.Contains(out, "'gone'") {
		t.Fatalf("section not removed:\n%s", out)
	}
	for _, must := range []string{"# this comment introduces the profile below", "[profiles.dev]", `approval = "never"`} {
		if !strings.Contains(out, must) {
			t.Fatalf("trailing comment/next table lost: missing %q\n%s", must, out)
		}
	}
}

// Installing a plain [section] on top of an operator's [[section]] array-of-tables
// would produce a TOML duplicate key; patch must refuse instead.
func TestPatchCodexMCPConfigRefusesArrayOfTablesCollision(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "config.toml")
	if err := os.WriteFile(cfg, []byte("[[mcp_servers.fak]]\ncommand = 'x'\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	err := patchCodexMCPConfig(cfg, "fak", "/opt/fak", []string{"serve"})
	if err == nil {
		t.Fatal("expected refusal to collide with an existing [[mcp_servers.fak]] array-of-tables")
	}
	if !strings.Contains(err.Error(), "array-of-tables") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestSanitizeVersionStripsReservedChars(t *testing.T) {
	cases := map[string]string{
		"1.2.3":            "1.2.3",
		"1.0.0+build.7":    "1.0.0+build.7",
		"v2:beta":          "v2-beta",     // colon -> '-' (no ADS/invalid path)
		`a\b/c`:            "a-b-c",       // separators
		"has space":        "has-space",   // space
		`we"ird<>|?*`:      "we-ird-----", // Windows-reserved run
		"trailing.":        "trailing",    // trailing dot trimmed
		"  spaced  ":       "spaced",      // outer trim
		"":                 "dev",         // empty fallback
		string(rune(0x07)): "dev",         // control -> '-', then trim -> "" -> dev
	}
	for in, want := range cases {
		if got := sanitizeVersion(in); got != want {
			t.Errorf("sanitizeVersion(%q) = %q, want %q", in, got, want)
		}
		// Whatever comes out must be usable as a single path segment.
		if got := sanitizeVersion(in); strings.ContainsAny(got, `\/:*?"<>|`) {
			t.Errorf("sanitizeVersion(%q) = %q still holds a reserved char", in, got)
		}
	}
}

func TestValidCodexServerName(t *testing.T) {
	ok := []string{"fak", "fak-dev", "fak_2", "A1"}
	bad := []string{"", "fak dev", "fak]", "a.b", "a\nb", `fak"`, "mcp_servers.evil"}
	for _, s := range ok {
		if !validCodexServerName(s) {
			t.Errorf("validCodexServerName(%q) = false, want true", s)
		}
	}
	for _, s := range bad {
		if validCodexServerName(s) {
			t.Errorf("validCodexServerName(%q) = true, want false", s)
		}
	}
}

func TestCodexMCPInstallRejectsBadServerName(t *testing.T) {
	f := newCodexMCPFixture(t, true)
	defer f.restore()
	var out, errb strings.Builder
	code := runCodexMCPInstall(&out, &errb, []string{"--config", f.config, "--source", f.source, "--policy", f.policy, "--server", "evil name"})
	if code == 0 {
		t.Fatal("install accepted an invalid --server name")
	}
	if _, _, err := readCodexMCPEntry(f.config, "evil name"); err == nil {
		t.Fatal("install wrote a config entry for an invalid server name")
	}
}

// removeCodexOwnedArtifact distinguishes removed / absent and clears the
// installed read-only bit so the delete is permitted.
func TestRemoveCodexOwnedArtifact(t *testing.T) {
	dir := t.TempDir()
	// Read-only file (as installed binaries are) removes cleanly.
	ro := filepath.Join(dir, "ro.bin")
	if err := os.WriteFile(ro, []byte("x"), 0o555); err != nil {
		t.Fatal(err)
	}
	if removed, locked, err := removeCodexOwnedArtifact(ro); err != nil || !removed || locked {
		t.Fatalf("read-only remove: removed=%v locked=%v err=%v", removed, locked, err)
	}
	if _, err := os.Stat(ro); !os.IsNotExist(err) {
		t.Fatalf("artifact not deleted: %v", err)
	}
	// Absent artifact is an idempotent no-op, not an error.
	if removed, locked, err := removeCodexOwnedArtifact(filepath.Join(dir, "nope")); err != nil || removed || locked {
		t.Fatalf("absent remove: removed=%v locked=%v err=%v", removed, locked, err)
	}
	// Empty path is a no-op.
	if removed, locked, err := removeCodexOwnedArtifact(""); err != nil || removed || locked {
		t.Fatalf("empty path: removed=%v locked=%v err=%v", removed, locked, err)
	}
}

// A re-install (upgrade) must carry the OPERATOR's original prior forward so a
// later rollback restores it — not the fak entry a naive capture would record.
func TestCodexMCPReinstallPreservesOperatorPriorForRollback(t *testing.T) {
	f := newCodexMCPFixture(t, true)
	defer f.restore()

	priorCmd := "/prior/fak"
	if runtime.GOOS == "windows" {
		priorCmd = `C:\prior\fak.exe`
	}
	if err := patchCodexMCPConfig(f.config, "fak", priorCmd, []string{"serve", "--stdio"}); err != nil {
		t.Fatal(err)
	}
	if code := f.install(t); code != 0 {
		t.Fatalf("first install exit=%d", code)
	}
	// Upgrade with a different source binary (new bytes -> new versioned artifact).
	src2 := filepath.Join(t.TempDir(), "src-fak2"+exeSuffix())
	if err := os.WriteFile(src2, []byte("fake-fak-binary-v2-DIFFERENT-payload-xyz"), 0o755); err != nil {
		t.Fatal(err)
	}
	var o, e strings.Builder
	if code := runCodexMCPInstall(&o, &e, []string{"--config", f.config, "--source", src2, "--policy", f.policy}); code != 0 {
		t.Fatalf("upgrade install exit=%d err=%s", code, e.String())
	}
	// Rollback must restore the operator's original, not the first fak binary.
	o.Reset()
	e.Reset()
	if code := runCodexMCPInstall(&o, &e, []string{"--config", f.config, "--rollback"}); code != 0 {
		t.Fatalf("rollback exit=%d err=%s", code, e.String())
	}
	cmd, _, err := readCodexMCPEntry(f.config, "fak")
	if err != nil {
		t.Fatalf("read after rollback: %v", err)
	}
	if cmd != priorCmd {
		t.Fatalf("rollback restored %q, want operator prior %q", cmd, priorCmd)
	}
}

// Uninstall must reap artifacts orphaned by prior upgrades, not just the current
// pair, and empty the install dir.
func TestCodexMCPUninstallReapsSupersededArtifacts(t *testing.T) {
	f := newCodexMCPFixture(t, true)
	defer f.restore()

	if code := f.install(t); code != 0 {
		t.Fatalf("first install exit=%d", code)
	}
	src2 := filepath.Join(t.TempDir(), "src-fak2"+exeSuffix())
	if err := os.WriteFile(src2, []byte("fake-fak-binary-v2-DIFFERENT-payload-xyz"), 0o755); err != nil {
		t.Fatal(err)
	}
	var o, e strings.Builder
	if code := runCodexMCPInstall(&o, &e, []string{"--config", f.config, "--source", src2, "--policy", f.policy}); code != 0 {
		t.Fatalf("upgrade exit=%d err=%s", code, e.String())
	}
	dir := codexMCPInstallDir(f.config)
	before, _ := filepath.Glob(filepath.Join(dir, "fak-*"))
	if len(before) < 2 {
		t.Fatalf("expected the upgrade to leave a superseded binary; found %v", before)
	}
	o.Reset()
	e.Reset()
	if code := runCodexMCPUninstall(&o, &e, []string{"--config", f.config}); code != 0 {
		t.Fatalf("uninstall exit=%d err=%s", code, e.String())
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		leftover, _ := filepath.Glob(filepath.Join(dir, "*"))
		t.Fatalf("install dir not emptied after uninstall: stat err=%v leftover=%v", err, leftover)
	}
}

// The backup manifest must be persisted before the destructive config patch, so a
// crash between the two still leaves the prior entry recoverable.
func TestCodexMCPManifestWrittenBeforeConfigPatch(t *testing.T) {
	f := newCodexMCPFixture(t, true)
	defer f.restore()
	priorCmd := "/prior/fak"
	if runtime.GOOS == "windows" {
		priorCmd = `C:\prior\fak.exe`
	}
	if err := patchCodexMCPConfig(f.config, "fak", priorCmd, []string{"serve"}); err != nil {
		t.Fatal(err)
	}
	if code := f.install(t); code != 0 {
		t.Fatalf("install exit=%d", code)
	}
	man, err := readCodexMCPManifest(f.config)
	if err != nil {
		t.Fatalf("manifest: %v", err)
	}
	if man.PriorCommand != priorCmd {
		t.Fatalf("manifest prior command = %q, want %q", man.PriorCommand, priorCmd)
	}
	if !strings.Contains(man.PriorConfigSection, priorCmd) {
		t.Fatalf("manifest did not record the prior section: %q", man.PriorConfigSection)
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(codexMCPManifestPath(f.config))
		if err != nil {
			t.Fatal(err)
		}
		if perm := info.Mode().Perm(); perm != 0o600 {
			t.Fatalf("manifest perm = %o, want 0600 (it may carry prior-entry secrets)", perm)
		}
	}
}

// ---- live initialize handshake (real binary) -------------------------------

func TestCodexMCPInstallLiveInitialize(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a real fak binary")
	}
	root := repoRootForDoctorTest(t)
	built := filepath.Join(t.TempDir(), "fak-live"+exeSuffix())
	build := exec.Command("go", "build", "-o", built, "./cmd/fak")
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build fak: %v\n%s", err, out)
	}
	home := t.TempDir()
	config := filepath.Join(home, "config.toml")
	policy := filepath.Join(root, "examples", "dev-agent-policy.json")
	// Isolate the served child's registry from the operator's production store.
	t.Setenv(sessionRegistryEnv, filepath.Join(t.TempDir(), "session-registry.json"))

	var out, errb strings.Builder
	code := runCodexMCPInstall(&out, &errb, []string{"--config", config, "--source", built, "--policy", policy, "--timeout", "30s"})
	if code != 0 {
		t.Fatalf("install exit=%d\nstdout=%s\nstderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "initialize handshake OK") {
		t.Fatalf("install did not verify the handshake:\n%s", out.String())
	}
	// status with a real probe must pass (healthy or advisory-stale).
	st := classifyCodexMCPStatus(config, "fak", 30*time.Second, true)
	if !st.OK || st.Probe != "ok" {
		t.Fatalf("live status: state=%q ok=%v probe=%q detail=%q", st.State, st.OK, st.Probe, st.Detail)
	}
}
