package corelocks

import _ "embed"

// fixtureTOML is the shipped declarative core-lock taxonomy. It is embedded so
// the package carries a valid, self-contained example of the declaration shape
// and so tests (and any future audit) can load the canonical fixture without a
// filesystem path.
//
//go:embed testdata/core_locks.toml
var fixtureTOML []byte

// Fixture returns the shipped, embedded core-lock declaration bytes. It is the
// canonical example of the declaration shape this package parses.
func Fixture() []byte {
	out := make([]byte, len(fixtureTOML))
	copy(out, fixtureTOML)
	return out
}

// LoadFixture parses the shipped embedded taxonomy. A non-nil error means the
// shipped fixture itself is malformed (which the package's own tests catch).
func LoadFixture() (*Taxonomy, error) {
	return Parse(fixtureTOML)
}
