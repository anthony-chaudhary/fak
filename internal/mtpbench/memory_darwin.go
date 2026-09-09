//go:build darwin && cgo

package mtpbench

/*
#include <mach/mach.h>
#include <stdint.h>
static int fak_current_rss(uint64_t *rss) {
  mach_task_basic_info_data_t info; mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
  kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &n);
  if (kr != KERN_SUCCESS) return (int)kr;
  *rss = (uint64_t)info.resident_size; return 0;
}
*/
import "C"

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"syscall"
)

func observeMemory() (MemorySnapshot, error) {
	var rss C.uint64_t
	if code := C.fak_current_rss(&rss); code != 0 {
		return MemorySnapshot{}, fmt.Errorf("mtpbench: current RSS unavailable: mach error %d", int(code))
	}
	raw, err := syscall.Sysctl("vm.swapusage")
	if err != nil {
		return MemorySnapshot{}, fmt.Errorf("mtpbench: swap unavailable: %w", err)
	}
	used, ok := parseSwapUsed(raw)
	if !ok {
		return MemorySnapshot{}, fmt.Errorf("mtpbench: swap unavailable: malformed vm.swapusage %q", raw)
	}
	return MemorySnapshot{CurrentRSSBytes: uint64(rss), SwapUsedBytes: used, RSSAvailable: true, SwapAvailable: true}, nil
}

func parseSwapUsed(raw string) (uint64, bool) {
	fields := strings.Fields(raw)
	for i := 0; i+2 < len(fields); i++ {
		if fields[i] != "used" || fields[i+1] != "=" {
			continue
		}
		token := fields[i+2]
		if !strings.HasSuffix(token, "M") || strings.Count(token, "M") != 1 {
			return 0, false
		}
		mb := strings.TrimSuffix(token, "M")
		value, err := strconv.ParseFloat(mb, 64)
		// 2^44 MiB converts to 2^64 bytes, one past uint64. Reject the
		// boundary too because float64 cannot represent MaxUint64 precisely.
		if err != nil || value < 0 || math.IsNaN(value) || math.IsInf(value, 0) || value >= math.Ldexp(1, 44) {
			return 0, false
		}
		return uint64(value * (1 << 20)), true
	}
	return 0, false
}
