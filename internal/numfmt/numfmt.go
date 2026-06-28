// Package numfmt holds the small allocation-light numeric and environment
// formatting helpers that were copy-pasted across the kernel. Each function here
// is a behavior-preserving hoist of byte-identical clones that the per-package
// de-duplication waves could not reach because they lived in different packages.
//
// Scope is deliberately narrow: only helpers that were genuinely identical across
// two or more packages land here. Variants that differ in behavior (e.g. a
// positive-only itoa, or a byte-size formatter with a different decimal
// precision) are intentionally left in place rather than papered over with a
// parameter that would change a call site's contract.
package numfmt

import (
	"os"
	"strconv"
)

// Itoa renders an unsigned integer in base 10 without going through the fmt
// reflection path. It is the hoist of the identical hand-rolled uint64->string
// helpers in internal/abi and internal/gateway.
func Itoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// EnvPositiveInt reads key from the environment and parses it as a base-10
// integer, returning def unless the value parses cleanly AND is strictly
// positive. It is the hoist of the identical envPositiveInt helpers in
// internal/ctxmmu and internal/normgate.
//
// Note: internal/secretgate has a sibling that additionally trims surrounding
// whitespace and parses with fmt.Sscan; that variant has different behavior on
// padded or trailing-garbage values and is therefore left in place.
func EnvPositiveInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}
