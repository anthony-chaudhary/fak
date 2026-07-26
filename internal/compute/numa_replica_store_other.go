//go:build !(linux && amd64)

package compute

// allocNodeRegion has no node-bound storage off linux/amd64: there is no mbind to bind with.
// It still returns correct, writable storage so the replica set's BYTES are exact everywhere
// (which is what the bit-identity contract tests), but reports bound=false so no caller can
// mistake this for node locality. Placement — and only placement — is the linux-only half.
func allocNodeRegion(n int, node int) ([]byte, func() error, bool, error) {
	if n <= 0 {
		return nil, nil, false, errReplicaUnsupported
	}
	return make([]byte, n), func() error { return nil }, false, nil
}
