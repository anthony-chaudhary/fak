package skillenv

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// This file implements STATIC ABI-hash admission: a plugin (skill page or tool
// plugin) is admitted only when its declared ABI digest matches the digest the
// kernel derives from its own frozen ABI surface. Match = admit; mismatch or
// undecodable digest = refuse (fail-closed). The digest is computed here, at
// admission time, from the frozen constants of internal/abi — no runtime
// negotiation, no fallback.

// ABIAdmissionEnv is the identity string the kernel derives: the frozen ABI
// version plus the negotiated capability namespace the plugin will be served
// under. It is stable for a given kernel build.
func ABIAdmissionEnv() string {
	return fmt.Sprintf("fak-abi v%d.%d", abi.ABIMajor, abi.ABIMinor)
}

// ComputePluginABIDigest derives the canonical static ABI hash a plugin must
// declare: sha256 over the kernel's frozen ABI identity string.
func ComputePluginABIDigest() string {
	sum := sha256.Sum256([]byte(ABIAdmissionEnv()))
	return hex.EncodeToString(sum[:])
}

// VerifyPluginABI gates plugin admission on the static ABI hash. fail-closed:
// only an exact hex match against the kernel-derived digest admits.
func VerifyPluginABI(declared string) error {
	want := ComputePluginABIDigest()
	if declared == "" {
		return fmt.Errorf("skillenv: plugin admission refused: empty ABI digest (fail-closed)")
	}
	if declared != want {
		return fmt.Errorf("skillenv: plugin admission refused: ABI digest mismatch (declared %s, kernel %s)", truncateDigest(declared), truncateDigest(want))
	}
	return nil
}

func truncateDigest(s string) string {
	if len(s) > 16 {
		return s[:16] + "…"
	}
	return s
}
