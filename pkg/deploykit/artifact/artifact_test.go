package artifact

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testgit"
	"github.com/anthony-chaudhary/fak/pkg/deploykit"
)

// TestMain drops inherited GIT_* variables (a hook or guarded session can export GIT_DIR or
// GIT_INDEX_FILE, which would point the temp-repo git commands at the live checkout) and
// serves the helper-process mode TestLocalRunEnvOverridesAmbient re-executes.
func TestMain(m *testing.M) {
	if os.Getenv("FAK_ARTIFACT_TEST_HELPER") == "printenv" {
		fmt.Print(os.Getenv("FAK_ARTIFACT_TEST_PROBE"))
		os.Exit(0)
	}
	for _, kv := range os.Environ() {
		if k, _, _ := strings.Cut(kv, "="); strings.HasPrefix(strings.ToUpper(k), "GIT_") {
			_ = os.Unsetenv(k)
		}
	}
	// Hermetic git beyond the scrub above: a fixture `git config --global` write
	// lands in a private copy, and a changed checkout identity fails the run.
	os.Exit(testgit.RunGitIsolated(m))
}

const (
	commitA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	commitB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

// call is one command an executor was asked to run.
type call struct {
	dir, name  string
	args, env  []string
	withEnvSet bool
}

func (c call) isGo() bool { return c.name == "go" }

func (c call) isSmoke() bool {
	return len(c.args) == 2 && c.args[0] == "version" && c.args[1] == "--json"
}

func (c call) String() string { return filepath.Base(c.name) + " " + strings.Join(c.args, " ") }

// scriptExec records every command and answers it through handle; with no handler every
// command fails, so a test can prove nothing ran.
type scriptExec struct {
	mu     sync.Mutex
	calls  []call
	handle func(ctx context.Context, c call) (string, bool)
}

func (e *scriptExec) Run(ctx context.Context, dir, name string, args ...string) (string, bool) {
	return e.do(ctx, call{dir: dir, name: name, args: args})
}

func (e *scriptExec) RunEnv(ctx context.Context, dir string, env []string, name string, args ...string) (string, bool) {
	return e.do(ctx, call{dir: dir, name: name, args: args, env: env, withEnvSet: true})
}

func (e *scriptExec) do(ctx context.Context, c call) (string, bool) {
	e.mu.Lock()
	e.calls = append(e.calls, c)
	handle := e.handle
	e.mu.Unlock()
	if handle == nil {
		return "unexpected command " + c.String(), false
	}
	return handle(ctx, c)
}

func (e *scriptExec) recorded() []call {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]call(nil), e.calls...)
}

func (e *scriptExec) count(match func(call) bool) int {
	n := 0
	for _, c := range e.recorded() {
		if match(c) {
			n++
		}
	}
	return n
}

// realRun runs c on this host.
func realRun(ctx context.Context, c call) (string, bool) {
	if c.withEnvSet {
		return Local().RunEnv(ctx, c.dir, c.env, c.name, c.args...)
	}
	return Local().Run(ctx, c.dir, c.name, c.args...)
}

// runOnly hides RunEnv, leaving a plain deploykit.Executor.
type runOnly struct{ deploykit.Executor }

func versionJSON(commit string, dirty, stamped bool, app string) string {
	b, _ := json.Marshal(map[string]any{"app_version": app, "commit": commit, "dirty": dirty, "stamped": stamped})
	return string(b)
}

// smokeOnly answers only the candidate's `version --json`.
func smokeOnly(out string) func(context.Context, call) (string, bool) {
	return func(_ context.Context, c call) (string, bool) {
		if c.isSmoke() {
			return out, true
		}
		return "unexpected command " + c.String(), false
	}
}

// isolateTemp points os.TempDir at a fresh directory so a test can prove Acquire leaves no
// staging or workspace directory behind.
func isolateTemp(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, k := range []string{"TMPDIR", "TMP", "TEMP"} {
		t.Setenv(k, dir)
	}
	return dir
}

func assertNoLeftovers(t *testing.T, tmp string) {
	t.Helper()
	entries, err := os.ReadDir(tmp)
	if err != nil {
		t.Fatalf("read temp dir: %v", err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "fak-deploykit-") {
			t.Errorf("left behind %s", e.Name())
		}
	}
}

func sha(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// artifactServer serves payload and counts requests. chunked drops the Content-Length header.
func artifactServer(t *testing.T, payload []byte, chunked bool) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if r.URL.Path != "/fak.bin" {
			http.NotFound(w, r)
			return
		}
		if chunked {
			_, _ = w.Write(payload[:1])
			w.(http.Flusher).Flush()
			_, _ = w.Write(payload[1:])
			return
		}
		w.Header().Set("Content-Length", fmt.Sprint(len(payload)))
		_, _ = w.Write(payload)
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func TestFetchShaMismatchNotPromoted(t *testing.T) {
	payload := []byte("fetched fak artifact bytes")
	wrong := sha([]byte("some other artifact"))
	cases := []struct {
		name    string
		chunked bool
		src     func(url string) Fetch
		want    error
	}{
		{"wrong digest", false, func(u string) Fetch {
			return Fetch{URL: u, SHA256: wrong, Size: int64(len(payload)), Commit: commitA, Generation: 1}
		}, ErrDigestMismatch},
		{"announced size too large", false, func(u string) Fetch {
			return Fetch{URL: u, SHA256: sha(payload), Size: int64(len(payload)) + 1, Commit: commitA, Generation: 1}
		}, ErrSizeMismatch},
		{"streamed body longer than pinned", true, func(u string) Fetch {
			return Fetch{URL: u, SHA256: sha(payload[:4]), Size: 4, Commit: commitA, Generation: 1}
		}, ErrSizeMismatch},
		{"streamed body shorter than pinned", true, func(u string) Fetch {
			return Fetch{URL: u, SHA256: sha(payload), Size: int64(len(payload)) + 9, Commit: commitA, Generation: 1}
		}, ErrSizeMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := isolateTemp(t)
			srv, hits := artifactServer(t, payload, tc.chunked)
			x := &scriptExec{handle: smokeOnly(versionJSON(commitA, false, true, "1.0.0"))}
			target := filepath.Join(t.TempDir(), "fak.exe")

			// The driver's shape: only an acquired candidate is ever stored.
			c, err := Acquire(context.Background(), x, tc.src(srv.URL+"/fak.bin"))
			if err == nil {
				_, err = StoreSlot(target, c)
			}
			if !errors.Is(err, tc.want) {
				t.Fatalf("Acquire err = %v, want %v", err, tc.want)
			}
			if c != (Candidate{}) {
				t.Fatalf("refused Acquire returned a candidate: %+v", c)
			}
			if hits.Load() != 1 {
				t.Fatalf("server hits = %d, want 1", hits.Load())
			}
			if calls := x.recorded(); len(calls) != 0 {
				t.Fatalf("mismatched bytes were executed: %v", calls)
			}
			if _, err := os.Stat(slotRoot(target)); !os.IsNotExist(err) {
				t.Fatalf("slot root exists after a refused fetch (stat err %v)", err)
			}
			assertNoLeftovers(t, tmp)
		})
	}

	t.Run("bytes changed after acquire are never promoted", func(t *testing.T) {
		tmp := isolateTemp(t)
		srv, _ := artifactServer(t, payload, false)
		x := &scriptExec{handle: smokeOnly(versionJSON(commitA, false, true, "1.0.0"))}
		c, err := Acquire(context.Background(), x, Fetch{URL: srv.URL + "/fak.bin", SHA256: sha(payload), Size: int64(len(payload)), Commit: commitA, Generation: 1})
		if err != nil {
			t.Fatalf("Acquire: %v", err)
		}
		tampered := bytes.ToUpper(payload)
		if err := os.WriteFile(c.Path, tampered, 0o700); err != nil {
			t.Fatal(err)
		}
		target := filepath.Join(t.TempDir(), "fak.exe")
		if _, err := StoreSlot(target, c); !errors.Is(err, ErrDigestMismatch) {
			t.Fatalf("StoreSlot(tampered) err = %v, want ErrDigestMismatch", err)
		}
		if _, err := os.Stat(slotRoot(target)); !os.IsNotExist(err) {
			t.Fatalf("slot root exists after a refused store (stat err %v)", err)
		}
		if err := c.Remove(); err != nil {
			t.Fatal(err)
		}
		assertNoLeftovers(t, tmp)
	})
}

func TestFetchManifestFailsClosed(t *testing.T) {
	tmp := isolateTemp(t)
	payload := []byte("fetched fak artifact bytes")
	srv, hits := artifactServer(t, payload, false)
	x := &scriptExec{handle: smokeOnly(versionJSON(commitA, false, true, "1.0.0"))}
	// Everything else matches, so a SHA-only pass would succeed: the manifest must still refuse.
	src := Fetch{
		URL: srv.URL + "/fak.bin", SHA256: sha(payload), Size: int64(len(payload)), Commit: commitA, Generation: 1,
		Manifest: &Manifest{Payload: json.RawMessage(`{"schema":"x"}`), Signature: "c2ln"},
	}
	c, err := Acquire(context.Background(), x, src)
	if !errors.Is(err, ErrManifestNotWired) {
		t.Fatalf("Acquire err = %v, want ErrManifestNotWired", err)
	}
	if c != (Candidate{}) {
		t.Fatalf("manifest fetch produced a candidate: %+v", c)
	}
	if hits.Load() != 0 {
		t.Fatalf("manifest fetch downloaded before failing closed (%d hits)", hits.Load())
	}
	if calls := x.recorded(); len(calls) != 0 {
		t.Fatalf("manifest fetch executed commands: %v", calls)
	}
	assertNoLeftovers(t, tmp)

	// The same source without the manifest passes, proving the refusal is the manifest's.
	src.Manifest = nil
	c, err = Acquire(context.Background(), x, src)
	if err != nil {
		t.Fatalf("Acquire without manifest: %v", err)
	}
	_ = c.Remove()
}

func TestFetchAcquireVerifiesAndSmokes(t *testing.T) {
	tmp := isolateTemp(t)
	payload := []byte("fetched fak artifact bytes")
	srv, _ := artifactServer(t, payload, false)
	x := &scriptExec{handle: smokeOnly(versionJSON(strings.ToUpper(commitA), false, true, "2.4.0"))}
	src := Fetch{URL: srv.URL + "/fak.bin", SHA256: strings.ToUpper(sha(payload)), Size: int64(len(payload)), Commit: commitA, Generation: 3}
	c, err := Acquire(context.Background(), x, src)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	want := Candidate{Kind: KindFetch, Path: c.Path, Generation: 3, Commit: commitA, Digest: sha(payload), Size: int64(len(payload)), AppVersion: "2.4.0", dir: c.dir}
	if c != want {
		t.Fatalf("candidate = %+v, want %+v", c, want)
	}
	if got, _ := os.ReadFile(c.Path); !bytes.Equal(got, payload) {
		t.Fatalf("candidate bytes = %q", got)
	}
	// Two smokes: the gate's own and Verify's, both after the digest check.
	if n := x.count(call.isSmoke); n != 2 || len(x.recorded()) != 2 {
		t.Fatalf("calls = %v, want exactly two smokes", x.recorded())
	}
	if err := c.Remove(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(c.Path); !os.IsNotExist(err) {
		t.Fatalf("Remove left the candidate (stat err %v)", err)
	}
	assertNoLeftovers(t, tmp)
}

func TestFetchRefusals(t *testing.T) {
	payload := []byte("fetched fak artifact bytes")
	srv, _ := artifactServer(t, payload, false)
	good := Fetch{URL: srv.URL + "/fak.bin", SHA256: sha(payload), Size: int64(len(payload)), Commit: commitA, Generation: 1}
	cases := []struct {
		name  string
		edit  func(*Fetch)
		smoke string
		want  error
	}{
		{"remote plain http", func(f *Fetch) { f.URL = "http://example.com/fak.bin" }, "", ErrInvalidSource},
		{"ftp", func(f *Fetch) { f.URL = "ftp://127.0.0.1/fak.bin" }, "", ErrInvalidSource},
		{"relative url", func(f *Fetch) { f.URL = "/fak.bin" }, "", ErrInvalidSource},
		{"short digest", func(f *Fetch) { f.SHA256 = "abc" }, "", ErrInvalidSource},
		{"zero size", func(f *Fetch) { f.Size = 0 }, "", ErrInvalidSource},
		{"short commit", func(f *Fetch) { f.Commit = "aaaaaaa" }, "", ErrInvalidSource},
		{"zero generation", func(f *Fetch) { f.Generation = 0 }, "", ErrInvalidSource},
		{"http 404", func(f *Fetch) { f.URL = srv.URL + "/missing" }, "", ErrFetchFailed},
		{"smoked commit differs", nil, versionJSON(commitB, false, true, "1.0.0"), ErrSmokeCommitMismatch},
		{"smoked dirty", nil, versionJSON(commitA, true, true, "1.0.0"), ErrProvenance},
		{"smoked unstamped", nil, versionJSON("", false, false, "1.0.0"), ErrProvenance},
		{"smoke not json", nil, "fak 1.0.0", ErrProvenance},
		{"smoke without app version", nil, versionJSON(commitA, false, true, ""), ErrProvenance},
		{"pinned app version differs", func(f *Fetch) { f.AppVersion = "9.9.9" }, versionJSON(commitA, false, true, "1.0.0"), ErrProvenance},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := isolateTemp(t)
			src := good
			if tc.edit != nil {
				tc.edit(&src)
			}
			x := &scriptExec{}
			if tc.smoke != "" {
				x.handle = smokeOnly(tc.smoke)
			}
			c, err := Acquire(context.Background(), x, src)
			if !errors.Is(err, tc.want) {
				t.Fatalf("Acquire err = %v, want %v", err, tc.want)
			}
			if c != (Candidate{}) {
				t.Fatalf("refused Acquire returned a candidate: %+v", c)
			}
			if tc.smoke == "" && len(x.recorded()) != 0 {
				t.Fatalf("refused before download but executed %v", x.recorded())
			}
			assertNoLeftovers(t, tmp)
		})
	}
}

func TestAcquireRejectsNilArguments(t *testing.T) {
	if _, err := Acquire(context.Background(), nil, Fetch{}); !errors.Is(err, ErrInvalidSource) {
		t.Fatalf("nil executor: err = %v", err)
	}
	if _, err := Acquire(context.Background(), &scriptExec{}, nil); !errors.Is(err, ErrInvalidSource) {
		t.Fatalf("nil source: err = %v", err)
	}
	for _, src := range []Source{(*Fetch)(nil), (*BuildPinned)(nil)} {
		if _, err := Acquire(context.Background(), &scriptExec{}, src); !errors.Is(err, ErrInvalidSource) {
			t.Fatalf("typed nil source %T: err = %v", src, err)
		}
	}
}

// handCandidate writes payload to a file and returns a candidate recording its identity.
func handCandidate(t *testing.T, payload []byte) Candidate {
	t.Helper()
	path := filepath.Join(t.TempDir(), candidateName("windows"))
	if err := os.WriteFile(path, payload, 0o700); err != nil {
		t.Fatal(err)
	}
	return Candidate{Kind: KindBuild, Path: path, Generation: 4, Commit: commitA, Digest: sha(payload), Size: int64(len(payload)), AppVersion: "1.0.0"}
}

func TestStoreSlotReusesAndRefusesCorrupt(t *testing.T) {
	c := handCandidate(t, []byte("verified candidate bytes"))
	target := filepath.Join(t.TempDir(), "bin", "fak.exe")

	first, err := StoreSlot(target, c)
	if err != nil {
		t.Fatalf("first StoreSlot: %v", err)
	}
	if got, _ := os.ReadFile(first); string(got) != "verified candidate bytes" {
		t.Fatalf("slot bytes = %q", got)
	}
	slotDir := filepath.Dir(first)
	if filepath.Dir(slotDir) != slotRoot(target) {
		t.Fatalf("slot %s is not under %s: slotRoot drifted from selfinstall", first, slotRoot(target))
	}
	if !strings.HasSuffix(filepath.Base(slotDir), "-"+c.Digest) || filepath.Base(first) != "fak.exe" {
		t.Fatalf("slot %s is not generation+digest addressed", first)
	}
	meta, err := os.ReadFile(filepath.Join(slotDir, "slot.json"))
	if err != nil {
		t.Fatalf("slot metadata: %v", err)
	}

	// Re-storing the identical candidate re-verifies and reuses the slot.
	second, err := StoreSlot(target, c)
	if err != nil || second != first {
		t.Fatalf("re-store = %q, %v; want reuse of %q", second, err, first)
	}
	if again, _ := os.ReadFile(filepath.Join(slotDir, "slot.json")); !bytes.Equal(again, meta) {
		t.Fatalf("re-store rewrote the slot metadata")
	}

	// A corrupt existing slot is refused and left as it is, never overwritten.
	if err := os.WriteFile(first, []byte("corrupted slot bytes!!!!"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := StoreSlot(target, c); !errors.Is(err, ErrSlot) {
		t.Fatalf("StoreSlot over a corrupt slot err = %v, want ErrSlot", err)
	}
	if got, _ := os.ReadFile(first); string(got) != "corrupted slot bytes!!!!" {
		t.Fatalf("refused store touched the existing slot: %q", got)
	}

	// A frozen slot generation with a different identity is refused too.
	other := c
	other.AppVersion = "2.0.0"
	if err := os.WriteFile(first, []byte("verified candidate bytes"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := StoreSlot(target, other); !errors.Is(err, ErrSlot) {
		t.Fatalf("StoreSlot with a changed identity err = %v, want ErrSlot", err)
	}
}

func TestStoreSlotRefusesInvalidCandidate(t *testing.T) {
	good := handCandidate(t, []byte("verified candidate bytes"))
	cases := []struct {
		name string
		edit func(*Candidate)
		want error
	}{
		{"bytes differ from digest", func(c *Candidate) { c.Digest = sha([]byte("other")) }, ErrDigestMismatch},
		{"size differs", func(c *Candidate) { c.Size++ }, ErrSizeMismatch},
		{"missing file", func(c *Candidate) { c.Path += ".missing" }, ErrInvalidCandidate},
		{"upper-case digest", func(c *Candidate) { c.Digest = strings.ToUpper(c.Digest) }, ErrInvalidCandidate},
		{"zero generation", func(c *Candidate) { c.Generation = 0 }, ErrInvalidCandidate},
		{"short commit", func(c *Candidate) { c.Commit = "abc1234" }, ErrInvalidCandidate},
		{"no app version", func(c *Candidate) { c.AppVersion = " " }, ErrInvalidCandidate},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := good
			tc.edit(&c)
			target := filepath.Join(t.TempDir(), "fak.exe")
			if _, err := StoreSlot(target, c); !errors.Is(err, tc.want) {
				t.Fatalf("StoreSlot err = %v, want %v", err, tc.want)
			}
			if _, err := os.Stat(slotRoot(target)); !os.IsNotExist(err) {
				t.Fatalf("slot root created for a refused candidate (stat err %v)", err)
			}
		})
	}
	if _, err := StoreSlot(" ", good); !errors.Is(err, ErrInvalidCandidate) {
		t.Fatalf("empty target: err = %v", err)
	}
}

func TestDiscardNewSlotsKeepsPreexisting(t *testing.T) {
	digest := sha([]byte("x"))
	root := filepath.Join(t.TempDir(), "fak.exe.self-update-slots")
	old := filepath.Join(root, "g00000000000000000001-"+digest)
	if err := os.MkdirAll(old, 0o700); err != nil {
		t.Fatal(err)
	}
	before, existed := slotNames(root)
	fresh := filepath.Join(root, "g00000000000000000002-"+digest)
	unrelated := filepath.Join(root, "g00000000000000000003-"+sha([]byte("y")))
	for _, d := range []string{fresh, unrelated} {
		if err := os.MkdirAll(d, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	discardNewSlots(root, before, existed, digest)
	for path, want := range map[string]bool{old: true, fresh: false, unrelated: true} {
		if _, err := os.Stat(path); (err == nil) != want {
			t.Errorf("%s exists=%v, want %v", filepath.Base(path), err == nil, want)
		}
	}

	// A root the failed store created is removed once its new slot is gone.
	root2 := filepath.Join(t.TempDir(), "fak.exe.self-update-slots")
	before2, existed2 := slotNames(root2)
	if err := os.MkdirAll(filepath.Join(root2, "g00000000000000000001-"+digest), 0o700); err != nil {
		t.Fatal(err)
	}
	discardNewSlots(root2, before2, existed2, digest)
	if _, err := os.Stat(root2); !os.IsNotExist(err) {
		t.Fatalf("created slot root survived (stat err %v)", err)
	}
}

func TestVerifyBytesFirstThenProvenance(t *testing.T) {
	payload := []byte("verified candidate bytes")
	c := handCandidate(t, payload)

	good := &scriptExec{handle: smokeOnly(versionJSON(commitA, false, true, "1.0.0"))}
	if err := Verify(context.Background(), good, c); err != nil {
		t.Fatalf("Verify: %v", err)
	}

	// Tampered bytes are refused before anything runs.
	tampered := c
	tampered.Digest = sha([]byte("other"))
	x := &scriptExec{handle: smokeOnly(versionJSON(commitA, false, true, "1.0.0"))}
	if err := Verify(context.Background(), x, tampered); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("Verify(tampered) err = %v, want ErrDigestMismatch", err)
	}
	if len(x.recorded()) != 0 {
		t.Fatalf("tampered candidate was executed: %v", x.recorded())
	}

	for name, out := range map[string]string{
		"other commit":      versionJSON(commitB, false, true, "1.0.0"),
		"dirty":             versionJSON(commitA, true, true, "1.0.0"),
		"other app version": versionJSON(commitA, false, true, "1.0.1"),
	} {
		x := &scriptExec{handle: smokeOnly(out)}
		if err := Verify(context.Background(), x, c); !errors.Is(err, ErrProvenance) {
			t.Errorf("Verify(%s) err = %v, want ErrProvenance", name, err)
		}
	}
	if err := Verify(context.Background(), nil, c); !errors.Is(err, ErrInvalidCandidate) {
		t.Fatalf("nil executor: err = %v", err)
	}
}

func TestCandidateRemoveOnlyOwnsItsStagingDir(t *testing.T) {
	c := handCandidate(t, []byte("keep me"))
	if err := c.Remove(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(c.Path); err != nil {
		t.Fatalf("Remove deleted a candidate Acquire did not stage: %v", err)
	}
}

func TestLocalRunEnvOverridesAmbient(t *testing.T) {
	t.Setenv("FAK_ARTIFACT_TEST_PROBE", "ambient")
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	out, ok := Local().RunEnv(context.Background(), t.TempDir(),
		[]string{"FAK_ARTIFACT_TEST_HELPER=printenv", "FAK_ARTIFACT_TEST_PROBE=pinned"}, exe)
	if !ok || out != "pinned" {
		t.Fatalf("RunEnv = %q, %v; want the pinned value to win over the ambient one", out, ok)
	}
}
