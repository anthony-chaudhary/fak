package exclusivefile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCreatePIDTime(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lock")
	if err := CreatePIDTime(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	if err := CreatePIDTime(path); !os.IsExist(err) {
		t.Fatalf("second create = %v", err)
	}
}
