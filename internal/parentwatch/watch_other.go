//go:build !darwin

package parentwatch

import "context"

func watch(ctx context.Context, cancel context.CancelFunc, id identity) func() {
	return pollWatch(ctx, cancel, id)
}
