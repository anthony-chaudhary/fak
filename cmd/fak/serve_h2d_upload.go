package main

import "fmt"

// formatServeH2DUpload renders the post-load upload split: h2d_direct bytes were written through a
// mapped weight buffer with no staging copy or transfer submit, and h2d_total includes them.
func formatServeH2DUpload(direct, total uint64) string {
	staged := "unknown"
	if direct <= total {
		staged = fmt.Sprint(total - direct)
	}
	return fmt.Sprintf("h2d_direct=%d h2d_total=%d h2d_staged=%s scope=process-cumulative", direct, total, staged)
}
