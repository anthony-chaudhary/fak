package session

import (
	"errors"
	"fmt"
	"sort"
	"sync"
)

var (
	ErrUnknownDriver           = errors.New("session: unknown store driver")
	ErrDriverAlreadyRegistered = errors.New("session: store driver already registered")
)

// StoreDriver defines the pluggable storage driver seam for durable session state and descriptors.
type StoreDriver interface {
	DescriptorStore
	DriverName() string
}

var (
	storeDriversMu sync.RWMutex
	storeDrivers   = make(map[string]StoreDriver)
)

// RegisterStoreDriver registers a session store driver by name.
// Duplicate registration panics at init, matching the fail-loud plugin contract.
func RegisterStoreDriver(name string, d StoreDriver) {
	storeDriversMu.Lock()
	defer storeDriversMu.Unlock()
	if name == "" {
		panic("session: store driver name must be non-empty")
	}
	if d == nil {
		panic("session: store driver must not be nil")
	}
	if _, exists := storeDrivers[name]; exists {
		panic(fmt.Sprintf("session: store driver %q already registered", name))
	}
	storeDrivers[name] = d
}

// GetStoreDriver returns the registered driver by name, or an error if unknown.
func GetStoreDriver(name string) (StoreDriver, error) {
	storeDriversMu.RLock()
	defer storeDriversMu.RUnlock()
	d, ok := storeDrivers[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownDriver, name)
	}
	return d, nil
}

// StoreDrivers returns the sorted list of registered driver names.
func StoreDrivers() []string {
	storeDriversMu.RLock()
	defer storeDriversMu.RUnlock()
	names := make([]string, 0, len(storeDrivers))
	for n := range storeDrivers {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}
