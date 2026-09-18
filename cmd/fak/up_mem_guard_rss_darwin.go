//go:build darwin && cgo

package main

/*
#include <mach/mach.h>
#include <stdint.h>
static int fak_up_current_rss(uint64_t *rss) {
  mach_task_basic_info_data_t info; mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
  kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &n);
  if (kr != KERN_SUCCESS) return (int)kr;
  *rss = (uint64_t)info.resident_size; return 0;
}
*/
import "C"

// platformCurrentRSS reports this process's current resident set size via the
// Mach task basic-info resident_size, which tracks the LIVE resident footprint
// (not the monotonic peak) so the mem guard can detect a sustained breach and
// clear it when the working set shrinks. A mach failure returns 0, which makes
// the guard fall back to the Go runtime's own accounting.
func platformCurrentRSS() uint64 {
	var rss C.uint64_t
	if code := C.fak_up_current_rss(&rss); code != 0 {
		return 0
	}
	return uint64(rss)
}
