package skilldiscovery

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type fakeHTTP func(*http.Request) (*http.Response, error)

func (f fakeHTTP) Do(r *http.Request) (*http.Response, error) { return f(r) }

func skillFixture(version, body string) (Index, map[string][]byte) {
	entry := []byte("---\nname: example\ndescription: Harmless fixture\n---\n" + body + "\n")
	ref := []byte("reference data\n")
	files := map[string][]byte{"example/SKILL.md": entry, "example/references/guide.md": ref}
	s := CatalogSkill{Name: "example", Version: version, Files: []string{"SKILL.md", "references/guide.md"}, SHA256: map[string]string{}, Sizes: map[string]int64{}}
	for _, f := range s.Files {
		b := files["example/"+f]
		s.SHA256[f] = digest(b)
		s.Sizes[f] = int64(len(b))
	}
	return Index{Skills: []CatalogSkill{s}}, files
}
func validateFixture(name string, b []byte) error {
	s, err := agent.ParseSkill(b, "")
	if err != nil || s.Name != name || strings.TrimSpace(s.Metadata["canonical"]) != "" {
		return fail("ENTRYPOINT_INVALID")
	}
	return nil
}
func fixtureOptions(t *testing.T, index Index) (Options, []byte) {
	t.Helper()
	b, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return Options{Source: "https://skills.example.test/catalog/", IndexSHA256: digest(b), CacheRoot: root, ValidateSkill: validateFixture}, b
}
func fixtureHTTP(index []byte, files map[string][]byte, calls *[]string) httpDoer {
	return fakeHTTP(func(r *http.Request) (*http.Response, error) {
		if calls != nil {
			*calls = append(*calls, r.URL.String())
		}
		if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" {
			return nil, errors.New("unexpected credentials")
		}
		if err := r.Context().Err(); err != nil {
			return nil, err
		}
		p := strings.TrimPrefix(r.URL.Path, "/catalog/")
		var b []byte
		var ok bool
		if p == "index.json" {
			b = index
			ok = true
		} else {
			b, ok = files[p]
		}
		status := http.StatusOK
		if !ok {
			status = http.StatusNotFound
		}
		return &http.Response{StatusCode: status, Body: io.NopCloser(bytes.NewReader(b)), ContentLength: int64(len(b)), Header: make(http.Header)}, nil
	})
}
func mustPull(t *testing.T, o Options, b []byte, files map[string][]byte) Receipt {
	t.Helper()
	r, err := pull(context.Background(), o, fixtureHTTP(b, files, nil))
	if err != nil {
		t.Fatal(err)
	}
	return r
}
func assertReason(t *testing.T, err error, want string) {
	t.Helper()
	var e *Error
	if !errors.As(err, &e) || e.Code != want {
		t.Fatalf("reason=%v want=%s", err, want)
	}
}

// fak-test:runtime fast est=2s lane=default
// Declared estimate, not measured; HTTP is injected except one bounded local TLS fixture.
func TestRemoteSkillCatalog(t *testing.T) {
	t.Run("immutable refresh cache and real native consumer", func(t *testing.T) {
		i1, f1 := skillFixture("v1", "harmless first version")
		o, b1 := fixtureOptions(t, i1)
		r1 := mustPull(t, o, b1, f1)
		if r1.Activated || r1.CacheHit || r1.SkillsRoot == "" {
			t.Fatalf("receipt=%+v", r1)
		}
		reg, err := agent.DiscoverSkills(t.TempDir(), r1.SkillsRoot)
		if err != nil {
			t.Fatal(err)
		}
		text, err := reg.Execute("example")
		if err != nil || !strings.Contains(text, "harmless first version") {
			t.Fatalf("real consumer text=%q err=%v", text, err)
		}
		i2, f2 := skillFixture("v2", "harmless second version")
		b2, _ := json.Marshal(i2)
		o.IndexSHA256 = digest(b2)
		r2 := mustPull(t, o, b2, f2)
		if r1.SkillsRoot == r2.SkillsRoot {
			t.Fatal("replaced in-flight root")
		}
		old, _ := os.ReadFile(filepath.Join(r1.SkillsRoot, "example", "SKILL.md"))
		if !bytes.Equal(old, f1["example/SKILL.md"]) {
			t.Fatal("old version changed")
		}
		if got, _ := reg.Execute("example"); got != text {
			t.Fatal("existing registry changed")
		}
		noNetwork := fakeHTTP(func(*http.Request) (*http.Response, error) {
			t.Fatal("cache hit made network request")
			return nil, nil
		})
		hit, err := pull(context.Background(), o, noNetwork)
		if err != nil || !hit.CacheHit || hit.SkillsRoot != r2.SkillsRoot {
			t.Fatalf("hit=%+v err=%v", hit, err)
		}
	})
	t.Run("failed new version preserves old data and has no record", func(t *testing.T) {
		i1, f1 := skillFixture("v1", "old")
		o, b1 := fixtureOptions(t, i1)
		old := mustPull(t, o, b1, f1)
		i2, f2 := skillFixture("v2", "new")
		b2, _ := json.Marshal(i2)
		o.IndexSHA256 = digest(b2)
		delete(f2, "example/references/guide.md")
		_, err := pull(context.Background(), o, fixtureHTTP(b2, f2, nil))
		assertReason(t, err, "HTTP_STATUS_REFUSED")
		b, _ := os.ReadFile(filepath.Join(old.SkillsRoot, "example", "SKILL.md"))
		if !bytes.Equal(b, f1["example/SKILL.md"]) {
			t.Fatal("lost old entrypoint")
		}
		records, _ := os.ReadDir(filepath.Join(o.CacheRoot, "records"))
		objects, _ := os.ReadDir(filepath.Join(o.CacheRoot, "objects"))
		if len(records) != 1 || len(objects) != 1 {
			t.Fatalf("published incomplete data: records=%d objects=%d", len(records), len(objects))
		}
	})
	t.Run("untrusted pin and tampered files refuse", func(t *testing.T) {
		i, f := skillFixture("v1", "content")
		o, b := fixtureOptions(t, i)
		o.IndexSHA256 = strings.Repeat("0", 64)
		_, err := pull(context.Background(), o, fixtureHTTP(b, f, nil))
		assertReason(t, err, "INDEX_DIGEST_MISMATCH")
		o.IndexSHA256 = digest(b)
		f["example/SKILL.md"] = []byte("not pinned")
		_, err = pull(context.Background(), o, fixtureHTTP(b, f, nil))
		assertReason(t, err, "FILE_DIGEST_MISMATCH")
	})
	t.Run("cache tamper never refetches or erases", func(t *testing.T) {
		i, f := skillFixture("v1", "content")
		o, b := fixtureOptions(t, i)
		r := mustPull(t, o, b, f)
		name := filepath.Join(r.SkillsRoot, "example", "SKILL.md")
		if err := os.WriteFile(name, []byte("changed"), 0600); err != nil {
			t.Fatal(err)
		}
		calls := []string{}
		_, err := pull(context.Background(), o, fixtureHTTP(b, f, &calls))
		assertReason(t, err, "CACHE_CONTENT_MISMATCH")
		got, _ := os.ReadFile(name)
		if len(calls) != 0 || string(got) != "changed" {
			t.Fatal("tampered cache was retried or rewritten")
		}
	})
	t.Run("extra skill directory is not silently activated", func(t *testing.T) {
		i, f := skillFixture("v1", "content")
		o, b := fixtureOptions(t, i)
		r := mustPull(t, o, b, f)
		if err := os.Mkdir(filepath.Join(r.SkillsRoot, "extra"), 0700); err != nil {
			t.Fatal(err)
		}
		_, err := pull(context.Background(), o, fixtureHTTP(b, f, nil))
		assertReason(t, err, "CACHE_CONTENT_MISMATCH")
	})
	t.Run("path version size and collision checks precede file requests", func(t *testing.T) {
		for _, bad := range []string{"../outside", "/absolute", `C:\\outside`, "a/../../x", "a\\b", "https://evil.test/x", "a%2fb", "a//b", "a/./b", "CON", "a.", "a?secret", "a#fragment"} {
			t.Run(bad, func(t *testing.T) {
				i, f := skillFixture("v1", "content")
				i.Skills[0].Files = append(i.Skills[0].Files, bad)
				i.Skills[0].SHA256[bad] = digest(nil)
				i.Skills[0].Sizes[bad] = 0
				o, b := fixtureOptions(t, i)
				calls := []string{}
				_, err := pull(context.Background(), o, fixtureHTTP(b, f, &calls))
				if err == nil || len(calls) != 1 {
					t.Fatalf("unsafe path accepted or fetched: err=%v calls=%d", err, len(calls))
				}
			})
		}
		for _, mutate := range []func(*Index){
			func(i *Index) { i.Skills[0].Version = "" },
			func(i *Index) { i.Skills[0].Name = "../outside" },
			func(i *Index) { i.Skills[0].Sizes["SKILL.md"] = -1 },
			func(i *Index) { i.Skills[0].Sizes["SKILL.md"] = MaxFileBytes + 1 },
			func(i *Index) { i.Skills[0].SHA256["SKILL.md"] = "untrusted" },
			func(i *Index) {
				i.Skills[0].Files = append(i.Skills[0].Files, "skill.md")
				i.Skills[0].SHA256["skill.md"] = digest(nil)
				i.Skills[0].Sizes["skill.md"] = 0
			},
			func(i *Index) {
				i.Skills[0].Files = append(i.Skills[0].Files, "references")
				i.Skills[0].SHA256["references"] = digest(nil)
				i.Skills[0].Sizes["references"] = 0
			},
			func(i *Index) { i.Skills = append(i.Skills, i.Skills[0]) },
		} {
			i, f := skillFixture("v1", "content")
			mutate(&i)
			o, b := fixtureOptions(t, i)
			calls := []string{}
			_, err := pull(context.Background(), o, fixtureHTTP(b, f, &calls))
			if err == nil || len(calls) != 1 {
				t.Fatalf("invalid index admitted: %v", err)
			}
		}
	})
	t.Run("actual byte limit despite absent length", func(t *testing.T) {
		i, _ := skillFixture("v1", "content")
		o, b := fixtureOptions(t, i)
		client := fakeHTTP(func(r *http.Request) (*http.Response, error) {
			data := b
			if !strings.HasSuffix(r.URL.Path, "index.json") {
				data = bytes.Repeat([]byte("x"), int(i.Skills[0].Sizes["SKILL.md"])+1)
			}
			return &http.Response{StatusCode: 200, ContentLength: -1, Body: io.NopCloser(bytes.NewReader(data))}, nil
		})
		_, err := pull(context.Background(), o, client)
		assertReason(t, err, "BYTE_LIMIT_EXCEEDED")
	})
	t.Run("symlink cache parent refuses", func(t *testing.T) {
		i, f := skillFixture("v1", "content")
		o, b := fixtureOptions(t, i)
		outside := t.TempDir()
		link := filepath.Join(o.CacheRoot, "records")
		if err := os.Symlink(outside, link); err != nil {
			t.Skip("symlink creation unavailable on this host")
		}
		calls := []string{}
		_, err := pull(context.Background(), o, fixtureHTTP(b, f, &calls))
		assertReason(t, err, "CACHE_PATH_REFUSED")
		if len(calls) != 0 {
			t.Fatal("network preceded symlink refusal")
		}
		files, _ := os.ReadDir(outside)
		if len(files) != 0 {
			t.Fatal("outside root modified")
		}
	})
	t.Run("entrypoint validator has no canonical filesystem read", func(t *testing.T) {
		i, f := skillFixture("v1", "content")
		entry := []byte("---\nname: example\nmetadata:\n  canonical: ../../secret\n---\nignored")
		f["example/SKILL.md"] = entry
		i.Skills[0].SHA256["SKILL.md"] = digest(entry)
		i.Skills[0].Sizes["SKILL.md"] = int64(len(entry))
		o, b := fixtureOptions(t, i)
		_, err := pull(context.Background(), o, fixtureHTTP(b, f, nil))
		assertReason(t, err, "ENTRYPOINT_INVALID")
	})
	t.Run("concurrent immutable publication is create-only", func(t *testing.T) {
		i, f := skillFixture("v1", "content")
		o, b := fixtureOptions(t, i)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		var arrived atomic.Int32
		ready := make(chan struct{})
		base := fixtureHTTP(b, f, nil)
		client := fakeHTTP(func(r *http.Request) (*http.Response, error) {
			if r.URL.Path == "/catalog/index.json" {
				if arrived.Add(1) == 2 {
					close(ready)
				}
				// Both cache lookups must miss before either index is returned.
				// Each pull then reaches create-only publication, including its loser.
				select {
				case <-ready:
				case <-r.Context().Done():
					return nil, r.Context().Err()
				}
			}
			return base.Do(r)
		})
		var wg sync.WaitGroup
		receipts := make([]Receipt, 2)
		errs := make([]error, 2)
		for n := 0; n < 2; n++ {
			wg.Add(1)
			go func(n int) {
				defer wg.Done()
				receipts[n], errs[n] = pull(ctx, o, client)
			}(n)
		}
		wg.Wait()
		for _, err := range errs {
			if err != nil {
				t.Fatal(err)
			}
		}
		if arrived.Load() != 2 || receipts[0].CacheHit == receipts[1].CacheHit {
			t.Fatal("did not witness both initial misses and one create-only winner")
		}
		if receipts[0].SkillsRoot != receipts[1].SkillsRoot {
			t.Fatal("multiple roots published for same pin")
		}
		objects, _ := os.ReadDir(filepath.Join(o.CacheRoot, "objects"))
		if len(objects) != 1 {
			t.Fatalf("unowned or incomplete generation remains: %d", len(objects))
		}
	})
	t.Run("URL errors are redacted and redirects refused", func(t *testing.T) {
		i, f := skillFixture("v1", "content")
		o, b := fixtureOptions(t, i)
		for _, source := range []string{"http://skills.test/", "https://user:secret@skills.test/", "https://skills.test/?token=secret", "https://skills.test/#secret", "https://skills.test/a%2Fb/", "https://skills.test/?"} {
			o.Source = source
			calls := []string{}
			_, err := pull(context.Background(), o, fixtureHTTP(b, f, &calls))
			if err == nil || strings.Contains(err.Error(), "secret") || len(calls) != 0 {
				t.Fatal("untrusted URL/secret escaped validation")
			}
		}
		var hits atomic.Int64
		server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits.Add(1)
			http.Redirect(w, r, "/redirected", http.StatusFound)
		}))
		defer server.Close()
		o.Source = server.URL + "/catalog/"
		client := newHTTPClient()
		client.Transport = server.Client().Transport
		defer client.CloseIdleConnections()
		_, err := pull(context.Background(), o, client)
		assertReason(t, err, "FETCH_FAILED")
		if got := hits.Load(); got != 1 {
			t.Fatalf("redirect followed: %d requests", got)
		}
	})
	t.Run("cancellation leaves no published generation", func(t *testing.T) {
		i, f := skillFixture("v1", "content")
		o, b := fixtureOptions(t, i)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := pull(ctx, o, fixtureHTTP(b, f, nil))
		if err == nil {
			t.Fatal("cancelled acquisition succeeded")
		}
		records, _ := os.ReadDir(filepath.Join(o.CacheRoot, "records"))
		if len(records) != 0 {
			t.Fatal("cancelled data published")
		}
	})
}
