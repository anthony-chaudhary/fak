//go:build !linux

package localbench

import (
	"context"
	"errors"
)

type v41EPOwnerLiveHandoff struct {
	Request V41EPOwnerRequest
	Rank    int
	Context context.Context
}

func v41EPHUnavailable() error {
	return errors.New("v41 EP owner: Linux supervisor handoff is unavailable")
}

// No portable fallback can establish the required Linux launch attributes.
func openV41EPOwnerLiveHandoff(context.Context) (*v41EPOwnerLiveHandoff, error) {
	return nil, v41EPHUnavailable()
}
func (*v41EPOwnerLiveHandoff) ObserveProcess(context.Context) (v41EPOwnerProcessIdentity, error) {
	return v41EPOwnerProcessIdentity{}, v41EPHUnavailable()
}
func (*v41EPOwnerLiveHandoff) Verify(context.Context, v41EPOwnerSupervisorChallenge) (v41EPOwnerSupervisorHandoff, error) {
	return v41EPOwnerSupervisorHandoff{}, v41EPHUnavailable()
}
func (*v41EPOwnerLiveHandoff) Finish(V41EPOwnerReceipt) error { return v41EPHUnavailable() }
func (*v41EPOwnerLiveHandoff) Close() error                   { return nil }
