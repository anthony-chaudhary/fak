//go:build !darwin

package parentwatch

import "context"

func watch(ctx context.Context, cancel context.CancelFunc, parentPID int) func() {
	return pollWatch(ctx, cancel, parentPID)
}
