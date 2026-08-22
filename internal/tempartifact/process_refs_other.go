//go:build !windows

package tempartifact

import "context"

func inspectLiveProcessPaths(_ context.Context, _ []string) Inspection {
	return Inspection{Reason: ReasonInspectionUnavailable, References: map[string]bool{}}
}
