//go:build !windows

package codexresume

// nativeOwnershipProbe deliberately does not guess from lock age, contents, or
// PID scans. Platforms without a native resource-owner witness fail closed.
type nativeOwnershipProbe struct{}

func (nativeOwnershipProbe) inspect(string) (ownershipWitness, error) {
	return ownershipWitness{source: "platform_unsupported", conclusive: false}, nil
}
