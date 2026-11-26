package ggufload

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"unsafe"
)

// The draft checkpoint cache hands every opener of an unchanged checkpoint the
// same mapping, keeps it while idle, remaps on a changed config, and stays
// consistent under concurrent open/close.
func TestInkling2DFlash2CacheSharesMapping(t *testing.T) {
	src := inkling2DFlash2TestDir(t)
	if src == "" {
		t.Skip("no Inkling-2-Mini-DFlash2 directory (set FAK_INKLING2_DFLASH2_DIR or FAK_INKLING2_GGUF)")
	}
	dir := t.TempDir()
	if err := os.Link(filepath.Join(src, inkling2DFlash2Weights), filepath.Join(dir, inkling2DFlash2Weights)); err != nil {
		t.Skipf("cannot link the checkpoint into a temp dir: %v", err)
	}
	cfg, err := os.ReadFile(filepath.Join(src, inkling2DFlash2Config))
	if err != nil {
		t.Fatal(err)
	}
	writeCfg := func(b []byte) {
		if err := os.WriteFile(filepath.Join(dir, inkling2DFlash2Config), b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeCfg(cfg)
	t.Cleanup(func() { inkling2DFlash2CacheDrop(t, dir) })
	base := func(m *Inkling2DFlash2Model) uintptr { return uintptr(unsafe.Pointer(unsafe.SliceData(m.mapped))) }
	open := func() *Inkling2DFlash2Model {
		m, err := OpenInkling2DFlash2(dir)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}

	a, b := open(), open()
	if base(a) != base(b) || a.shared == nil || a.shared != b.shared {
		t.Fatalf("two opens map %#x and %#x", base(a), base(b))
	}
	a.Close()
	a.Close() // a second close of one handle must not release b's reference
	if refs := inkling2DFlash2CacheRefs(dir); refs != 1 {
		t.Fatalf("after closing one of two handles twice, %d refs, want 1", refs)
	}
	b.Close()
	c := open()
	if base(c) != base(b) {
		t.Fatalf("reopening an idle checkpoint mapped %#x, cached %#x", base(c), base(b))
	}
	c.Close()

	writeCfg(append(cfg, '\n'))
	d := open()
	if d.shared == c.shared || !c.shared.stale {
		t.Fatalf("a changed config reused the old mapping %#x", base(d))
	}
	d.Close()

	inkling2DFlash2Uncached = true
	e := open()
	inkling2DFlash2Uncached = false
	if e.shared != nil || base(e) == base(d) {
		t.Fatal("an uncached open shared the cached mapping")
	}
	e.Close()

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				m, err := OpenInkling2DFlash2(dir)
				if err != nil {
					t.Error(err)
					return
				}
				if base(m) != base(d) {
					t.Errorf("concurrent open mapped %#x, cached %#x", base(m), base(d))
				}
				m.Close()
			}
		}()
	}
	wg.Wait()
	if refs := inkling2DFlash2CacheRefs(dir); refs != 0 {
		t.Fatalf("after every handle closed, %d refs", refs)
	}
}

func inkling2DFlash2CacheRefs(dir string) int {
	abs, _ := filepath.Abs(dir)
	c := &inkling2DFlash2Cache
	c.Lock()
	defer c.Unlock()
	if e := c.entries[abs]; e != nil {
		return e.refs
	}
	return -1
}

// inkling2DFlash2CacheDrop unmaps a test's idle checkpoint.
func inkling2DFlash2CacheDrop(t *testing.T, dir string) {
	abs, _ := filepath.Abs(dir)
	c := &inkling2DFlash2Cache
	c.Lock()
	defer c.Unlock()
	if e := c.entries[abs]; e != nil {
		if e.refs != 0 {
			t.Errorf("%d refs left on %s", e.refs, dir)
		}
		delete(c.entries, abs)
		e.free()
	}
}
