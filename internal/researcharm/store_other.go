//go:build !linux && !darwin

package researcharm

import "fmt"

type leaseStore struct{}

// NewDurableCoordinator fails closed where private durable stores are unsupported.
func NewDurableCoordinator(max int, path string) (*Coordinator, error) {
	return nil, fmt.Errorf("researcharm: durable lease stores require Linux or Darwin")
}
func (*leaseStore) save(map[string]*LeaseInfo, bool) (bool, error) {
	return false, fmt.Errorf("researcharm: unsupported durable lease store")
}
func (*leaseStore) close() error { return nil }
