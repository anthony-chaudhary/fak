package maturity

import (
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func TestPackageAnatomyDeterminism(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "internal", "sample")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"a.go": "package sample\n// Package sample is deterministic.\nfunc A() error { return nil }\n",
		"b.go": "package sample\nfunc B(ok bool) error { if !ok { return errSample{} }; return nil }\ntype errSample struct{}\nfunc (errSample) Error() string { return \"sample\" }\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	baseline, err := AnalyzeAnatomy(root, "internal/sample")
	if err != nil {
		t.Fatal(err)
	}

	const runs = 100
	errCh := make(chan error, runs)
	var wg sync.WaitGroup
	for i := 0; i < runs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, gotErr := AnalyzeAnatomy(root, "internal/sample")
			if gotErr != nil {
				errCh <- gotErr
				return
			}
			if !reflect.DeepEqual(got, baseline) {
				errCh <- errSample{}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for gotErr := range errCh {
		t.Error(gotErr)
	}
}
