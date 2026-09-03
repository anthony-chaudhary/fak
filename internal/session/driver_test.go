package session

import (
	"errors"
	"testing"
)

func TestStoreDriverRegistry(t *testing.T) {
	// 1. Get default "file" driver
	fileDrv, err := GetStoreDriver("file")
	if err != nil {
		t.Fatalf("expected 'file' driver registered, got error: %v", err)
	}
	if fileDrv.DriverName() != "file" {
		t.Errorf("expected driver name 'file', got %q", fileDrv.DriverName())
	}

	// 2. Get "memory" driver
	memDrv, err := GetStoreDriver("memory")
	if err != nil {
		t.Fatalf("expected 'memory' driver registered, got error: %v", err)
	}
	if memDrv.DriverName() != "memory" {
		t.Errorf("expected driver name 'memory', got %q", memDrv.DriverName())
	}

	// 3. Unknown driver returns ErrUnknownDriver
	_, err = GetStoreDriver("nonexistent_store")
	if !errors.Is(err, ErrUnknownDriver) {
		t.Errorf("expected ErrUnknownDriver, got %v", err)
	}

	// 4. StoreDrivers lists registered drivers sorted
	drivers := StoreDrivers()
	foundFile := false
	foundMem := false
	for _, d := range drivers {
		if d == "file" {
			foundFile = true
		}
		if d == "memory" {
			foundMem = true
		}
	}
	if !foundFile || !foundMem {
		t.Errorf("StoreDrivers() missing expected default drivers: %v", drivers)
	}

	// 5. Duplicate registration panics
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic on duplicate driver registration")
		}
	}()
	RegisterStoreDriver("file", &FileStore{})
}

func TestStoreDriverPanicsOnInvalidInput(t *testing.T) {
	// Empty name
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("expected panic on empty driver name")
			}
		}()
		RegisterStoreDriver("", &FileStore{})
	}()

	// Nil driver
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("expected panic on nil driver")
			}
		}()
		RegisterStoreDriver("nil-test", nil)
	}()
}
