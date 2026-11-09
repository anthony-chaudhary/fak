//go:build !windows

package service

import "errors"

// defaultSCMBoundary refuses off-Windows hosts: the live service control
// manager only exists on Windows. Tests on any host inject an SCMBoundary via
// WithSCM instead.
func defaultSCMBoundary() (SCMBoundary, error) {
	return nil, errors.New("deploykit/service: no Windows service control manager on this platform; inject an SCMBoundary with WithSCM")
}
