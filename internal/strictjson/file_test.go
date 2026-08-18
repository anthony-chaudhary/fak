package strictjson

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "value.json")
	if err := os.WriteFile(path, []byte(`{"n":3}`), 0600); err != nil {
		t.Fatal(err)
	}
	value, err := LoadFile[struct {
		N int `json:"n"`
	}](path)
	if err != nil || value.N != 3 {
		t.Fatalf("LoadFile=%+v, %v", value, err)
	}
}
