// Package suiteverify provides shared validation primitives for JSON-backed test suites.
package suiteverify

import "fmt"

// OptionalSchema accepts an omitted schema or requires it to equal want.
func OptionalSchema(kind, got, want string) error {
	if got != "" && got != want {
		return fmt.Errorf("%s suite: schema %q != %q", kind, got, want)
	}
	return nil
}
