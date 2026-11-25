package studymonitor

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func shardTestRow(repo, why string) string {
	return fmt.Sprintf(`{"repository":%q,"url":"https://github.com/%s","status":"candidate","priority":1,"why":%q,"last_checked":"2026-08-01","checked_revision":"abc"}`, repo, repo, why)
}

func writeShardTestRegistry(t *testing.T, rows ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "repos.json")
	doc := `{"schema":"fak-monitored-repositories/1","updated_at":"2026-08-14","methodology":"ranked","repositories":[` + strings.Join(rows, ",") + `]}`
	if err := os.WriteFile(path, []byte(doc), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeShardFile(t *testing.T, registryPath, name, data string) {
	t.Helper()
	file := filepath.Join(ShardDir(registryPath), name)
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
}

func shardTestWhys(r Registry) string {
	var out []string
	for _, repo := range r.Repositories {
		out = append(out, repo.Repository+"="+repo.Why)
	}
	return strings.Join(out, ";")
}

func TestShardFileName(t *testing.T) {
	got, err := ShardFileName(" Owner/Repo.Name ")
	if err != nil || got != "owner__repo.name.json" {
		t.Fatalf("ShardFileName = %q, %v", got, err)
	}
	if _, err := ShardFileName("no-slash"); err == nil {
		t.Fatal("ShardFileName(no-slash) succeeded")
	}
	if got := ShardDir(filepath.Join("a", "repos.json")); got != filepath.Join("a", "repos") {
		t.Fatalf("ShardDir = %q", got)
	}
}

func TestReadWithoutShardDirIsUnchanged(t *testing.T) {
	path := writeShardTestRegistry(t, shardTestRow("b/two", "header"), shardTestRow("a/one", "header"))
	r, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := shardTestWhys(r); got != "b/two=header;a/one=header" {
		t.Fatalf("rows = %s", got)
	}
}

func TestReadShardOverridesHeaderAndAppendsSorted(t *testing.T) {
	path := writeShardTestRegistry(t, shardTestRow("Keep/Me", "header"), shardTestRow("Over/Ride", "header"))
	writeShardFile(t, path, "over__ride.json", shardTestRow("over/ride", "shard"))
	writeShardFile(t, path, "zz__last.json", shardTestRow("zz/last", "shard-only"))
	writeShardFile(t, path, "aa__first.json", shardTestRow("AA/First", "shard-only"))
	r, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	want := "Keep/Me=header;over/ride=shard;AA/First=shard-only;zz/last=shard-only"
	if got := shardTestWhys(r); got != want {
		t.Fatalf("rows = %s, want %s", got, want)
	}
}

func TestReadShardOnlyRegistryWithEmptyHeader(t *testing.T) {
	path := writeShardTestRegistry(t)
	writeShardFile(t, path, "a__one.json", shardTestRow("a/one", "shard"))
	r, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := shardTestWhys(r); got != "a/one=shard" {
		t.Fatalf("rows = %s", got)
	}
}

func TestReadShardNameMismatch(t *testing.T) {
	path := writeShardTestRegistry(t)
	writeShardFile(t, path, "a__one.json", shardTestRow("a/other", "shard"))
	if _, err := Read(path); !errors.Is(err, ErrShardNameMismatch) {
		t.Fatalf("Read error = %v, want %v", err, ErrShardNameMismatch)
	}
}

func TestReadShardSkipsNonRows(t *testing.T) {
	path := writeShardTestRegistry(t, shardTestRow("a/one", "header"))
	bad := "not json"
	writeShardFile(t, path, ".hidden.json", bad)
	writeShardFile(t, path, "a__one.json.lock", bad)
	writeShardFile(t, path, "a__one.json.tmp-123", bad)
	writeShardFile(t, path, "notes.txt", bad)
	writeShardFile(t, path, filepath.Join("nested", "b__two.json"), shardTestRow("b/two", "nested"))
	r, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := shardTestWhys(r); got != "a/one=header" {
		t.Fatalf("rows = %s", got)
	}
}

func TestReadShardMalformedFails(t *testing.T) {
	path := writeShardTestRegistry(t)
	writeShardFile(t, path, "a__one.json", "[1]")
	if _, err := Read(path); err == nil || !strings.Contains(err.Error(), "decode registry shard") {
		t.Fatalf("Read error = %v, want decode failure", err)
	}
}

func TestReadShardPerFileCap(t *testing.T) {
	path := writeShardTestRegistry(t)
	padded := strings.Replace(shardTestRow("a/one", "x"), `"why":"x"`, `"why":"`+strings.Repeat("x", MaxRegistryShardBytes)+`"`, 1)
	writeShardFile(t, path, "a__one.json", padded)
	if _, err := Read(path); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("Read error = %v, want per-shard cap", err)
	}
}

func TestReadShardTotalCap(t *testing.T) {
	path := writeShardTestRegistry(t)
	why := strings.Repeat("x", MaxRegistryShardBytes-512)
	for i := 0; i < MaxRegistryBytes/MaxRegistryShardBytes+1; i++ {
		repo := fmt.Sprintf("o/r%02d", i)
		writeShardFile(t, path, fmt.Sprintf("o__r%02d.json", i), shardTestRow(repo, why))
	}
	if _, err := Read(path); err == nil || !strings.Contains(err.Error(), "assembled") {
		t.Fatalf("Read error = %v, want total cap", err)
	}
}

func TestReadShardNullFailsAsNameMismatch(t *testing.T) {
	path := writeShardTestRegistry(t)
	writeShardFile(t, path, "a__one.json", "null")
	if _, err := Read(path); !errors.Is(err, ErrShardNameMismatch) {
		t.Fatalf("Read error = %v, want %v", err, ErrShardNameMismatch)
	}
}

func TestReadNonJSONRegistryPathIgnoresShardDir(t *testing.T) {
	json := writeShardTestRegistry(t, shardTestRow("a/one", "header"))
	path := strings.TrimSuffix(json, ".json") + ".registry"
	if err := os.Rename(json, path); err != nil {
		t.Fatal(err)
	}
	writeShardFile(t, json, "a__one.json", "not json")
	r, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := shardTestWhys(r); got != "a/one=header" {
		t.Fatalf("rows = %s", got)
	}
}

func TestReadOversizeHeaderFailsOnceShardDirExists(t *testing.T) {
	why := strings.Repeat("x", MaxRegistryBytes)
	path := writeShardTestRegistry(t, shardTestRow("a/one", why))
	if _, err := Read(path); err != nil {
		t.Fatalf("Read without shard dir = %v, want unchanged behaviour", err)
	}
	if err := os.MkdirAll(ShardDir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(path); err == nil || !strings.Contains(err.Error(), "assembled") {
		t.Fatalf("Read error = %v, want total cap", err)
	}
}
