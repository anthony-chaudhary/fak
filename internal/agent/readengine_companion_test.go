package agent

// readengine_companion_test.go — coverage for the WIDENED read-root set: `fak_read` is
// confined to a root SET (the working tree plus, when the canonical ladder proves one
// exists, the companion public fak checkout), not to a single root. Everything here pins a
// property that the single-root era could not express, and the widening is only safe because
// each added root is structurally PROVED (cmd/fak/main.go on disk) and the confinement is
// applied to the UNION of the roots — never to any one of them in isolation.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

// The two refusal messages the confinement emits. They are wire contract, not prose:
// the gateway surfaces `error` to the model verbatim.
const (
	wantEscapeMessage        = "fak_read: path escapes the read root"
	wantSymlinkEscapeMessage = "fak_read: path escapes the read root via symlink"
)

// companionFixture is the three-directory world every widened-root test runs against:
//
//	temp/primary    roots[0] — the working tree; relative paths resolve here
//	temp/companion  roots[1] — the declared companion public checkout (has cmd/fak/main.go)
//	temp/outsider   NEITHER root — must stay unreachable by every spelling of a path
type companionFixture struct {
	temp      string
	primary   string
	companion string
	outsider  string
}

// newCompanionFixture builds the tree and its files. primaryPayload/companionPayload land in
// a file of the SAME name in each root, so a test can prove which root a relative path
// resolved against.
func newCompanionFixture(t *testing.T) companionFixture {
	t.Helper()
	temp := t.TempDir()
	f := companionFixture{
		temp:      temp,
		primary:   filepath.Join(temp, "primary"),
		companion: filepath.Join(temp, "companion"),
		outsider:  filepath.Join(temp, "outsider"),
	}
	for _, dir := range []string{f.primary, f.companion, f.outsider} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	write := func(dir, name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(f.primary, "shared.txt", "primary-payload")
	write(f.companion, "shared.txt", "companion-payload")
	write(f.primary, "primary-only.txt", "primary only \u2603 payload")
	write(f.companion, "companion-only.txt", "companion only \u2603 payload")
	write(f.outsider, "secret.txt", "do-not-read")
	// The structural proof that makes companion a declared root upstream. Kept here so a
	// test can assert the fixture still satisfies fakroot.IsRepoRoot.
	if err := os.MkdirAll(filepath.Join(f.companion, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(f.companion, "cmd", "fak", "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return f
}

// sameDir compares two paths after resolving symlinks: t.TempDir() returns the unresolved
// form (/var/folders/... on macOS, where /var -> /private/var) while the discovery ladder
// hands back whatever filepath.Abs produced. Same convention as internal/selfupdate/cmd.
func sameDir(t *testing.T, got, want string) bool {
	t.Helper()
	resolve := func(p string) string {
		if r, err := filepath.EvalSymlinks(p); err == nil {
			return r
		}
		return filepath.Clean(p)
	}
	return resolve(got) == resolve(want)
}

// assertRefused asserts the full typed refusal: code, source, and the exact message.
func assertRefused(t *testing.T, body map[string]any, raw []byte, code, source, message string) {
	t.Helper()
	if body["error_code"] != code {
		t.Fatalf("error_code = %v, want %q (body = %v)", body["error_code"], code, body)
	}
	if body["error_source"] != source {
		t.Fatalf("error_source = %v, want %q (body = %v)", body["error_source"], source, body)
	}
	if body["error"] != message {
		t.Fatalf("error = %v, want %q (body = %v)", body["error"], message, body)
	}
}

// TestReadEngineCompanionRootMatrix pins, over one widened engine, exactly which trees the
// union admits and which it refuses. The refusal half is the security half: widening must
// add the companion tree and NOTHING else.
func TestReadEngineCompanionRootMatrix(t *testing.T) {
	f := newCompanionFixture(t)
	e := readEngine{roots: []string{f.primary, f.companion}}

	cases := []struct {
		name     string
		path     string
		want     string
		code     string
		source   string
		message  string
		noLeaked bool // assert the refusal result never echoes the rejected path
	}{
		{
			name: "absolute path in the COMPANION root is admitted",
			path: filepath.Join(f.companion, "companion-only.txt"),
			want: "companion only \u2603 payload",
		},
		{
			name: "absolute path in the PRIMARY root still works (no single-root regression)",
			path: filepath.Join(f.primary, "primary-only.txt"),
			want: "primary only \u2603 payload",
		},
		{
			name: "relative path in the PRIMARY root resolves against roots[0]",
			path: "primary-only.txt",
			want: "primary only \u2603 payload",
		},
		{
			name: "absolute path in an UNDECLARED sibling outside both roots is refused",
			path: filepath.Join(f.outsider, "secret.txt"),
			code: "path_escape", source: "confinement", message: wantEscapeMessage, noLeaked: true,
		},
		{
			name: "relative dotdot hop into the undeclared sibling is refused",
			path: filepath.Join("..", "outsider", "secret.txt"),
			code: "path_escape", source: "confinement", message: wantEscapeMessage, noLeaked: true,
		},
		{
			name: "dotdot out of the companion and into the undeclared sibling is refused",
			path: filepath.Join(f.companion, "..", "outsider", "secret.txt"),
			code: "path_escape", source: "confinement", message: wantEscapeMessage, noLeaked: true,
		},
		{
			// Proves the companion root is genuinely searched (the widened root is live,
			// not just lexically tolerated) and that widening does not change the error
			// taxonomy inside an admitted root.
			name: "missing file inside the COMPANION root is not_found",
			path: filepath.Join(f.companion, "absent.txt"),
			code: "not_found", source: "filesystem", message: "fak_read: file not found",
		},
		{
			name: "directory inside the COMPANION root is is_directory",
			path: filepath.Join(f.companion, "cmd"),
			code: "is_directory", source: "filesystem", message: "fak_read: path is a directory",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, isErr := e.read(tc.path)
			body := decodeReadBody(t, raw)
			if tc.code != "" {
				if !isErr {
					t.Fatalf("expected refusal, got success: %v", body)
				}
				assertRefused(t, body, raw, tc.code, tc.source, tc.message)
				if tc.noLeaked && strings.Contains(string(raw), f.temp) {
					t.Fatalf("refusal leaked a filesystem path: %s", raw)
				}
				return
			}
			if isErr {
				t.Fatalf("unexpected refusal for %q: %s", tc.path, raw)
			}
			if body["content"] != tc.want {
				t.Fatalf("content = %v, want %q", body["content"], tc.want)
			}
			if body["file_path"] != tc.path {
				t.Fatalf("file_path = %v, want the argument echoed back %q", body["file_path"], tc.path)
			}
		})
	}
}

// TestReadEngineCompanionBinaryBytesRoundTrip proves the widening changes nothing about the
// wire shape: a non-UTF8 file under the COMPANION root comes back as base64 with the exact
// bytes, not truncated to the single-root behaviour.
func TestReadEngineCompanionBinaryBytesRoundTrip(t *testing.T) {
	f := newCompanionFixture(t)
	want := []byte{0x00, 0xff, 0x01, 0xfe, 0x7f}
	asset := filepath.Join(f.companion, "asset.bin")
	if err := os.WriteFile(asset, want, 0o644); err != nil {
		t.Fatal(err)
	}
	e := readEngine{roots: []string{f.primary, f.companion}}

	raw, isErr := e.read(asset)
	if isErr {
		t.Fatalf("companion binary read refused: %s", raw)
	}
	body := decodeReadBody(t, raw)
	if body["encoding"] != "base64" {
		t.Fatalf("encoding = %v, want base64", body["encoding"])
	}
	encoded, ok := body["content_base64"].(string)
	if !ok {
		t.Fatalf("content_base64 missing for a binary companion read: %v", body)
	}
	got, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatalf("content_base64 is not decodable: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("bytes = %x, want %x", got, want)
	}
}

// TestReadEngineCompanionSymlinkLaunderingGuard pins the laundering guard against the WIDENED
// root set. In the single-root era one EvalSymlinks comparison was enough; with N roots the
// comparison must run once PER root and only refuse when the resolved path lands outside ALL
// of them. A guard that compared against roots[0] alone would silently admit
// primary/leak -> companion/secret, and a guard that compared against the LAST root would
// silently admit primary/leak -> outsider/secret. Both spellings are pinned here.
func TestReadEngineCompanionSymlinkLaunderingGuard(t *testing.T) {
	f := newCompanionFixture(t)
	e := readEngine{roots: []string{f.primary, f.companion}}

	outsiderSecret := filepath.Join(f.outsider, "secret.txt")
	companionSecret := filepath.Join(f.companion, "companion-only.txt")

	cases := []struct {
		name   string
		link   string
		target string
		admit  bool // a resolved path INSIDE a declared root is not an escape
	}{
		{
			// The widening-specific laundering: resolved path lands in a DECLARED root
			// (the companion), so this read is ADMITTED — but only because the companion
			// is a root, not because the symlink check was skipped.
			name:   "primary symlink into the declared companion is admitted",
			link:   filepath.Join(f.primary, "hop.txt"),
			target: companionSecret,
			admit:  true,
		},
		{
			name:   "primary symlink OUTSIDE every root is refused",
			link:   filepath.Join(f.primary, "leak-outside.txt"),
			target: outsiderSecret,
		},
		{
			// The mirror image: a symlink sitting in the newly added root must not be a
			// softer boundary than one sitting in the working tree.
			name:   "companion symlink OUTSIDE every root is refused",
			link:   filepath.Join(f.companion, "leak-outside.txt"),
			target: outsiderSecret,
		},
	}

	ran := 0
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := os.Symlink(tc.target, tc.link); err != nil {
				t.Skipf("skipping symlink subtest: symlink creation not permitted: %v", err)
			}
			ran++
			raw, isErr := e.read(tc.link)
			body := decodeReadBody(t, raw)
			if tc.admit {
				if isErr {
					t.Fatalf("symlink into a declared root must be admitted: %s", raw)
				}
				if body["content"] != "companion only \u2603 payload" {
					t.Fatalf("content = %v, want the companion file's bytes", body["content"])
				}
				return
			}
			if !isErr {
				t.Fatalf("expected the symlink escape to be refused, got success: %v", body)
			}
			assertRefused(t, body, raw, "path_escape", "confinement", wantSymlinkEscapeMessage)
			if strings.Contains(string(raw), f.temp) {
				t.Fatalf("refusal leaked a filesystem path: %s", raw)
			}
		})
	}
	if ran == 0 {
		t.Fatal("no symlink subtest ran: symlink creation was refused on this host, so the laundering guard is UNVERIFIED")
	}
}

// TestReadEngineCompanionRelativePathResolution pins the base a RELATIVE path resolves
// against: roots[0], the working tree — never the process cwd and never a widened root.
func TestReadEngineCompanionRelativePathResolution(t *testing.T) {
	f := newCompanionFixture(t)
	e := readEngine{roots: []string{f.primary, f.companion}}

	t.Run("a name present in both roots resolves to roots[0]", func(t *testing.T) {
		raw, isErr := e.read("shared.txt")
		if isErr {
			t.Fatalf("read refused: %s", raw)
		}
		if got := decodeReadBody(t, raw)["content"]; got != "primary-payload" {
			t.Fatalf("content = %v, want primary-payload: a relative path resolves against roots[0]", got)
		}
	})

	t.Run("the process cwd is NOT the base", func(t *testing.T) {
		// Park the process cwd in the undeclared sibling, which holds its own shared.txt.
		// If a relative path resolved against the cwd instead of roots[0], this read would
		// return the outsider's bytes (or be refused) instead of the primary's.
		hopped := filepath.Join(f.outsider, "shared.txt")
		if err := os.WriteFile(hopped, []byte("outsider-payload"), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Chdir(f.outsider)
		raw, isErr := e.read("shared.txt")
		if isErr {
			t.Fatalf("read refused: %s", raw)
		}
		if got := decodeReadBody(t, raw)["content"]; got != "primary-payload" {
			t.Fatalf("content = %v, want primary-payload: roots[0], not the process cwd", got)
		}
	})

	t.Run("a relative path leaving every root is refused", func(t *testing.T) {
		raw, isErr := e.read(filepath.Join("..", "outsider", "secret.txt"))
		if !isErr {
			t.Fatalf("expected refusal, got: %s", raw)
		}
		assertRefused(t, decodeReadBody(t, raw), raw, "path_escape", "confinement", wantEscapeMessage)
	})

	// KNOWN WRITTEN BEHAVIOUR, pinned deliberately so a future change to it is a conscious
	// diff. The readEngine doc comment says "only an ABSOLUTE path may land in a widened
	// secondary root", but lexicallyConfined tests the union, so a RELATIVE dotdot hop into
	// the companion is admitted today. That is not a confinement escape — the target is
	// inside a declared root either way, so the reachable set is identical to what an
	// absolute path admits — but the doc sentence overstates the mechanism. If the doc is
	// the intended contract, THIS is the subtest that must change when the behaviour does.
	t.Run("a relative dotdot hop into the DECLARED companion is admitted (doc overstates)", func(t *testing.T) {
		raw, isErr := e.read(filepath.Join("..", "companion", "companion-only.txt"))
		if isErr {
			t.Fatalf("unexpected refusal: %s", raw)
		}
		if got := decodeReadBody(t, raw)["content"]; got != "companion only \u2603 payload" {
			t.Fatalf("content = %v, want the companion file's bytes", got)
		}
		t.Log("documented invariant says only an ABSOLUTE path may land in a widened root; " +
			"the union test admits the relative hop too. Reachability is unchanged.")
	})
}

// TestReadEngineCompanionZeroEngineRefusesEverything pins the deny-by-default floor: the zero
// value has no roots, so the union is empty and nothing is admissible — including paths that
// would be admitted under any non-empty root set. A zero-value readEngine is therefore safe to
// register as a fallback: it fails closed.
func TestReadEngineCompanionZeroEngineRefusesEverything(t *testing.T) {
	f := newCompanionFixture(t)
	e := readEngine{}

	if got := e.primaryRoot(); got != "" {
		t.Fatalf("primaryRoot() = %q, want \"\" on the zero engine", got)
	}

	paths := []string{
		filepath.Join(f.primary, "primary-only.txt"),
		filepath.Join(f.companion, "companion-only.txt"),
		filepath.Join(f.outsider, "secret.txt"),
		"primary-only.txt",
		filepath.Join("..", "companion", "companion-only.txt"),
		"/etc/hosts",
	}
	for _, p := range paths {
		raw, isErr := e.read(p)
		if !isErr {
			t.Fatalf("zero engine admitted %q: %s", p, raw)
		}
		assertRefused(t, decodeReadBody(t, raw), raw, "path_escape", "confinement", wantEscapeMessage)
		if strings.Contains(string(raw), f.temp) || strings.Contains(string(raw), "/etc/hosts") {
			t.Fatalf("refusal leaked a filesystem path: %s", raw)
		}
	}

	// An empty argument is an INPUT error, not a confinement refusal; pinning it here keeps
	// "refuses everything with path_escape" honest about its own boundary.
	raw, isErr := e.read("")
	if !isErr {
		t.Fatalf("empty path accepted: %s", raw)
	}
	assertRefused(t, decodeReadBody(t, raw), raw, "missing_path", "input", "fak_read: missing required field: file_path")
}

// readViaRegisteredEngine drives the fak_read miss path the way production does: through the
// driver bound to FakReadEngineID in the kernel registry (a vDSO hit would have served a
// cached result before ever reaching this), then decodes the result payload.
func readViaRegisteredEngine(t *testing.T, path string) (abi.Status, map[string]any, string) {
	t.Helper()
	eng := abi.Engine(FakReadEngineID)
	if eng == nil {
		t.Fatalf("no driver registered under %q", FakReadEngineID)
	}
	ctx := context.Background()
	args, err := json.Marshal(map[string]any{"file_path": path})
	if err != nil {
		t.Fatal(err)
	}
	res, err := eng.Complete(ctx, &abi.ToolCall{Tool: "fak_read", Args: putBytes(ctx, args)})
	if err != nil {
		t.Fatalf("Complete(%q) failed: %v", path, err)
	}
	if got := res.Meta["engine"]; got != FakReadEngineID {
		t.Fatalf("result came from engine %q, want %q", got, FakReadEngineID)
	}
	raw := string(refutil.Bytes(ctx, res.Payload))
	return res.Status, decodeReadBody(t, []byte(raw)), raw
}

// registerReadEngineInCwd chdirs into dir and installs the engine the way Configure does
// (empty root => cwd, plus whatever the canonical ladder proves). Registration is
// process-global (abi.RegisterEngine), so the prior state is restored on cleanup; the restore
// is registered BEFORE the chdir so that LIFO cleanup runs the chdir restore first and the
// re-registration happens in the ORIGINAL cwd, not in the temp fixture.
func registerReadEngineInCwd(t *testing.T, dir string) readEngine {
	t.Helper()
	// Neutralise FAK_ROOT: RegisterReadEngine passes Getenv=nil, so ladder rung 2 would
	// otherwise read the developer's real environment and answer the companion question on
	// a machine-specific fixture.
	t.Setenv("FAK_ROOT", "")
	t.Cleanup(func() { RegisterReadEngine("") })
	t.Chdir(dir)
	RegisterReadEngine("")

	eng := abi.Engine(FakReadEngineID)
	if eng == nil {
		t.Fatalf("no driver registered under %q after RegisterReadEngine", FakReadEngineID)
	}
	re, ok := eng.(readEngine)
	if !ok {
		t.Fatalf("registered driver is %T, want agent.readEngine", eng)
	}
	return re
}

// fakRootDir builds dir/cmd/fak/main.go — the structural proof that makes the discovery
// ladder call it a public checkout.
func fakRootDir(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(dir, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cmd", "fak", "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestReadEngineCompanionRootDiscoveryViaRegisterReadEngine pins the DEFAULT wiring: run from
// a private working tree whose go.work joins the sibling public checkout, fak_read can read
// the companion. This is the regression the widening exists for — before it, `fak serve`
// from fak-private default-denied fak_read of the workspace go.work had literally joined it
// to.
func TestReadEngineCompanionRootDiscoveryViaRegisterReadEngine(t *testing.T) {
	parent := t.TempDir()
	priv := filepath.Join(parent, "priv")
	pub := filepath.Join(parent, "pub")
	if err := os.MkdirAll(priv, 0o755); err != nil {
		t.Fatal(err)
	}
	fakRootDir(t, pub)
	if err := os.WriteFile(filepath.Join(pub, "hello.txt"), []byte("companion payload"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(priv, "go.work"), []byte("go 1.26.0\n\nuse (\n\t.\n\t../pub\n)\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	re := registerReadEngineInCwd(t, priv)

	if len(re.roots) != 2 {
		t.Fatalf("roots = %v, want exactly [cwd, companion]: the widening must add the one proved companion and nothing else", re.roots)
	}
	if !sameDir(t, re.roots[0], priv) {
		t.Fatalf("roots[0] = %q, want the cwd %q: roots[0] is what relative paths resolve against", re.roots[0], priv)
	}
	if !sameDir(t, re.roots[1], pub) {
		t.Fatalf("roots[1] = %q, want the go.work-declared companion %q", re.roots[1], pub)
	}

	status, body, raw := readViaRegisteredEngine(t, filepath.Join(pub, "hello.txt"))
	if status != abi.StatusOK {
		t.Fatalf("companion read through the registered engine: status %v, want StatusOK: %s", status, raw)
	}
	if body["content"] != "companion payload" {
		t.Fatalf("companion read content = %v, want %q", body["content"], "companion payload")
	}

	// The companion being a root must not make the whole parent a root.
	status, body, raw = readViaRegisteredEngine(t, filepath.Join(parent, "elsewhere.txt"))
	if status != abi.StatusError {
		t.Fatalf("read of the parent directory: status %v, want StatusError", status)
	}
	assertRefused(t, body, []byte(raw), "path_escape", "confinement", wantEscapeMessage)
}

// TestReadEngineCompanionRootDiscoveryAbsent pins the other half of the default: with no
// discoverable companion the engine stays confined to the cwd, so the widening is opt-in by
// proof and not opt-in by hope.
func TestReadEngineCompanionRootDiscoveryAbsent(t *testing.T) {
	parent := t.TempDir()
	alone := filepath.Join(parent, "alone")
	if err := os.MkdirAll(alone, 0o755); err != nil {
		t.Fatal(err)
	}
	// A public-looking sibling that the ladder has no way to reach: no go.work, no `fak`
	// child, no `../fak` sibling.
	pub := filepath.Join(parent, "pub")
	fakRootDir(t, pub)

	re := registerReadEngineInCwd(t, alone)

	if len(re.roots) != 1 || !sameDir(t, re.roots[0], alone) {
		t.Fatalf("roots = %v, want exactly the cwd [%s] when nothing is discoverable", re.roots, alone)
	}
	status, body, raw := readViaRegisteredEngine(t, filepath.Join(pub, "cmd", "fak", "main.go"))
	if status != abi.StatusError {
		t.Fatalf("sibling read: status %v, want StatusError", status)
	}
	assertRefused(t, body, []byte(raw), "path_escape", "confinement", wantEscapeMessage)
}

// TestReadEngineCompanionRootDiscoveryAsymmetry pins the invariant that makes every widening
// in this ticket safe: a directory that lacks cmd/fak/main.go is never admitted, even when a
// go.work names it directly. That is the private-companion shape, so a public working tree
// can never reach a private one through the read engine.
func TestReadEngineCompanionRootDiscoveryAsymmetry(t *testing.T) {
	parent := t.TempDir()
	pubCwd := filepath.Join(parent, "pub")
	priv := filepath.Join(parent, "priv")
	fakRootDir(t, pubCwd) // the cwd is a PUBLIC checkout
	if err := os.MkdirAll(priv, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(priv, "secret.txt"), []byte("private payload"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(priv, "cmd", "fak-server"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(priv, "cmd", "fak-server", "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// The public cwd's go.work names the private sibling BY NAME.
	if err := os.WriteFile(filepath.Join(pubCwd, "go.work"), []byte("go 1.26.0\n\nuse ../priv\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	re := registerReadEngineInCwd(t, pubCwd)

	for _, root := range re.roots {
		if sameDir(t, root, priv) {
			t.Fatalf("roots = %v, must not contain the private companion %q: it fails the cmd/fak/main.go proof", re.roots, priv)
		}
	}
	status, body, raw := readViaRegisteredEngine(t, filepath.Join(priv, "secret.txt"))
	if status != abi.StatusError {
		t.Fatalf("private sibling read: status %v, want StatusError", status)
	}
	assertRefused(t, body, []byte(raw), "path_escape", "confinement", wantEscapeMessage)
	if strings.Contains(raw, "private payload") {
		t.Fatalf("private file bytes leaked through the refusal: %s", raw)
	}
}

// TestReadEngineRootsExplicitSetSkipsDiscovery pins the opt-out: RegisterReadEngineRoots
// performs NO discovery and NO cwd defaulting, so a caller that pins roots keeps exactly the
// confinement it asked for even when a discoverable companion sits right next to it. This is
// the escape hatch for a caller that does not want the widening at all.
func TestReadEngineRootsExplicitSetSkipsDiscovery(t *testing.T) {
	parent := t.TempDir()
	priv := filepath.Join(parent, "priv")
	pub := filepath.Join(parent, "pub")
	workspace := filepath.Join(priv, "workspace")
	for _, dir := range []string{workspace, pub} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	fakRootDir(t, pub)
	if err := os.WriteFile(filepath.Join(priv, "go.work"), []byte("go 1.26.0\n\nuse (\n\t.\n\t../pub\n)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pub, "hello.txt"), []byte("companion payload"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("FAK_ROOT", "")
	t.Cleanup(func() { RegisterReadEngine("") })
	t.Chdir(priv) // the companion IS discoverable from here

	RegisterReadEngineRoots(workspace)

	eng := abi.Engine(FakReadEngineID)
	if eng == nil {
		t.Fatalf("no driver registered under %q after RegisterReadEngineRoots", FakReadEngineID)
	}
	re, ok := eng.(readEngine)
	if !ok {
		t.Fatalf("registered driver is %T, want agent.readEngine", eng)
	}
	if len(re.roots) != 1 || !sameDir(t, re.roots[0], workspace) {
		t.Fatalf("roots = %v, want exactly the explicit set [%s]: RegisterReadEngineRoots must run no discovery", re.roots, workspace)
	}
	status, body, raw := readViaRegisteredEngine(t, filepath.Join(pub, "hello.txt"))
	if status != abi.StatusError {
		t.Fatalf("opt-out read of a discoverable companion: status %v, want StatusError", status)
	}
	assertRefused(t, body, []byte(raw), "path_escape", "confinement", wantEscapeMessage)

	// Registration order is the contract: roots[0] resolves relative paths, and a repeated
	// root must not be admitted twice (a duplicate root would widen nothing but would make
	// the confinement story harder to reason about).
	RegisterReadEngineRoots(workspace, pub, workspace)
	re = abi.Engine(FakReadEngineID).(readEngine)
	if len(re.roots) != 2 || !sameDir(t, re.roots[0], workspace) || !sameDir(t, re.roots[1], pub) {
		t.Fatalf("roots = %v, want [workspace pub] — the first argument stays primary and duplicates are dropped", re.roots)
	}
	status, body, raw = readViaRegisteredEngine(t, filepath.Join(pub, "hello.txt"))
	if status != abi.StatusOK || body["content"] != "companion payload" {
		t.Fatalf("explicitly declared companion root not readable: status %v body %v", status, body)
	}
}
