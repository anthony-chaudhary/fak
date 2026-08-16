package harnessrelease

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestVerifyChecksum(t *testing.T) {
	path := filepath.Join(t.TempDir(), "asset.zip")
	if err := os.WriteFile(path, []byte("release"), 0o644); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256([]byte("release"))
	sidecar := path + ".sha256"
	if err := os.WriteFile(sidecar, []byte(fmt.Sprintf("%x  asset.zip\n", sum)), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := VerifyChecksum(path, sidecar); err != nil || got != fmt.Sprintf("%x", sum) {
		t.Fatalf("got=%q err=%v", got, err)
	}
	if err := os.WriteFile(path, []byte("changed"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyChecksum(path, sidecar); err == nil {
		t.Fatal("checksum mismatch accepted")
	}
}

func TestExtractReleaseFormatsAndRejectTraversal(t *testing.T) {
	for _, format := range []string{"zip", "tar.gz"} {
		t.Run(format, func(t *testing.T) {
			root := t.TempDir()
			archive := filepath.Join(root, "asset."+format)
			writeArchive(t, archive, format, "nested/fak", "binary")
			out := filepath.Join(root, "out")
			if err := Extract(archive, out); err != nil {
				t.Fatal(err)
			}
			body, err := os.ReadFile(filepath.Join(out, "nested", "fak"))
			if err != nil || string(body) != "binary" {
				t.Fatalf("body=%q err=%v", body, err)
			}
			bad := filepath.Join(root, "bad."+format)
			writeArchive(t, bad, format, "../escape", "bad")
			if err := Extract(bad, filepath.Join(root, "badout")); err == nil || !strings.Contains(err.Error(), "unsafe") {
				t.Fatalf("traversal err=%v", err)
			}
		})
	}
}

func writeArchive(t *testing.T, path, format, name, body string) {
	writeArchiveMode(t, path, format, name, body, 0o644)
}

func writeArchiveMode(t *testing.T, path, format, name, body string, mode os.FileMode) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if format == "zip" {
		z := zip.NewWriter(f)
		header := &zip.FileHeader{Name: name, Method: zip.Deflate}
		header.SetMode(mode)
		w, err := z.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
		if err := z.Close(); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		return
	}
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	if err := tw.WriteHeader(&tar.Header{Name: name, Mode: int64(mode.Perm()), Size: int64(len(body)), Typeflag: tar.TypeReg}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write([]byte(body)); err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRunEndToEndWithFixtureRelease(t *testing.T) {
	root := t.TempDir()
	fakeDir := filepath.Join(root, "fake")
	if err := os.MkdirAll(fakeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(fakeDir, "go.mod"), []byte("module example.test/fakefak\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(fakeDir, "main.go"), []byte(fakeFAKSource), 0o644); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(fakeDir, "fak")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	cmd := exec.Command("go", "build", "-o", binary, ".")
	cmd.Dir = fakeDir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build fixture: %v\n%s", err, out)
	}
	archive := filepath.Join(root, "fak_fixture.zip")
	body, err := os.ReadFile(binary)
	if err != nil {
		t.Fatal(err)
	}
	writeArchiveMode(t, archive, "zip", filepath.Base(binary), string(body), 0o755)
	sum, err := fileSHA256(archive)
	if err != nil {
		t.Fatal(err)
	}
	sidecar := archive + ".sha256"
	if err := os.WriteFile(sidecar, []byte(sum+"  "+filepath.Base(archive)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	product := filepath.Join(root, "product")
	receiptPath := filepath.Join(root, "receipt.json")
	r, err := Run(context.Background(), Options{Archive: archive, Checksum: sidecar, Target: runtime.GOOS + "_" + runtime.GOARCH, ProductDir: product, Module: "example.test/product", Receipt: receiptPath, RollbackCommand: "install previous release"})
	if err != nil {
		t.Fatal(err)
	}
	if r.Outcome != "success" || !r.UserConfigPreserved || r.UpgradeCommand == "" || r.RollbackCommand == "" || len(r.Commands) != 4 {
		t.Fatalf("receipt=%+v", r)
	}
	var stored Receipt
	storedBody, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(storedBody, &stored); err != nil {
		t.Fatal(err)
	}
	if stored.ArchiveSHA256 != sum || stored.BinarySHA256 == "" || stored.FAKVersion != "v9.9.9" {
		t.Fatalf("stored=%+v", stored)
	}
}

const fakeFAKSource = `package main
import (
 "encoding/json"
 "fmt"
 "os"
 "path/filepath"
)
func main() {
 if len(os.Args) < 2 || os.Args[1] != "harness" { os.Exit(2) }
 var dir, module, version string
 for i := 3; i+1 < len(os.Args); i += 2 { switch os.Args[i] { case "--dir": dir=os.Args[i+1]; case "--module": module=os.Args[i+1]; case "--fak-version": version=os.Args[i+1] } }
 if version == "" { version="v9.9.9" }
 must(os.MkdirAll(filepath.Join(dir,"product"),0755)); must(os.MkdirAll(filepath.Join(dir,"cmd","product"),0755))
 writeIfMissing(filepath.Join(dir,"product","config.go"), "package product\n\ntype Config struct { ID, Version, Profile, SystemPrompt, Task string }\nfunc DefaultConfig() Config { return Config{} }\nfunc OfflineReply(prompt string) string { return prompt }\n")
 must(os.WriteFile(filepath.Join(dir,"go.mod"), []byte("module "+module+"\n\ngo 1.26\n"),0644))
 must(os.WriteFile(filepath.Join(dir,"cmd","product","main.go"), []byte("package main\nimport (\"flag\";\"fmt\")\nfunc main(){ self:=flag.Bool(\"selfcheck\",false,\"\"); flag.Parse(); if !*self { panic(\"selfcheck required\") }; fmt.Println(\"ok\") }\n"),0644))
 lock:=map[string]string{"generator":"fixture/v1","contract_version":"v1alpha1","fak_module":"example.test/fak","fak_version":version,"upgrade":"fak harness init --dir . --module "+module+" --fak-version "+version}
 body,_:=json.Marshal(lock); must(os.WriteFile(filepath.Join(dir,"harness.lock.json"),body,0644)); fmt.Println("created")
}
func writeIfMissing(path, body string){ if _,err:=os.Stat(path); os.IsNotExist(err){ must(os.WriteFile(path,[]byte(body),0644)) } }
func must(err error){ if err!=nil { panic(err) } }
`
