//go:build darwin && cgo

package harnessres

/*
#include <mach/mach.h>
#include <stdint.h>

static int fak_harnessres_current_rss(uint64_t *rss) {
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  kern_return_t result = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                  (task_info_t)&info, &count);
  if (result != KERN_SUCCESS) return (int)result;
  *rss = (uint64_t)info.resident_size;
  return 0;
}
*/
import "C"

// DarwinSelfRSSReader returns a read-only Mach reader for this process's current
// resident set in bytes. The value is resident_size, not getrusage's peak RSS or
// macOS phys_footprint; compressed and swapped pages are not resident bytes.
// A failed task_info call returns (0, false) so callers keep read failure distinct
// from an unavailable reader. Builds without Darwin cgo support return nil.
func DarwinSelfRSSReader() func() (uint64, bool) {
	return readDarwinSelfRSSBytes
}

func readDarwinSelfRSSBytes() (uint64, bool) {
	var rss C.uint64_t
	if result := C.fak_harnessres_current_rss(&rss); result != 0 {
		return 0, false
	}
	return uint64(rss), true
}
