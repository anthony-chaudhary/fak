//go:build !windows && !linux

package procguard

import "errors"

type unsupportedFaultDomain struct{}

func newNativeFaultDomain(_ string, e ResourceEnvelope) (nativeFaultDomain, FaultDomainReceipt, error) {
	limits := requestedSupport(e, nil)
	return &unsupportedFaultDomain{}, FaultDomainReceipt{Mode: EnforcementObserveOnly, Primitive: "process-observation", Limits: limits}, nil
}
func (*unsupportedFaultDomain) bindCurrent() error {
	return errors.New("no hard fault-domain primitive on this OS")
}
func (*unsupportedFaultDomain) usage() (ResourceUsage, error) {
	return ResourceUsage{}, errors.New("fault-domain usage unavailable")
}
func (*unsupportedFaultDomain) close() error { return nil }
