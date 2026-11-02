//go:build !windows

package main

func dosLeaseAuthorityEnvironment(base []string) ([]string, error) {
	return base, nil
}
