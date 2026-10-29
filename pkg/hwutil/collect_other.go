//go:build !linux && !windows

package hwutil

import (
	"context"
	"runtime"
)

// Unsupported platforms get an honest all-nil host plane: identity fields are
// set, every family is nil, and the snapshot carries a closed error code
// instead of invented zeros. GPU probing still runs below.
func collectFamilies(ctx context.Context, o Options) familyResults {
	return familyResults{
		errs: []string{"host: collector not implemented for " + runtime.GOOS},
		code: CodeCollectorUnreadable,
	}
}
