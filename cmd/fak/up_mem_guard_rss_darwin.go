//go:build darwin && cgo

package main

// Adapted from carloslfu/slotstream, Sources/Slotstream/ProcessMemory.swift:52-69
// at a855c49090330df88b2464814bcc003c8a7c5bfe (MIT).
// Copyright (c) 2026 Carlos Galarza
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
#include <mach/mach.h>
#include <stdint.h>
static int fak_up_current_rss(uint64_t *rss) {
  task_vm_info_data_t info = {0}; mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &n);
  if (kr != KERN_SUCCESS) return (int)kr;
  if (n < TASK_VM_INFO_REV1_COUNT) return (int)KERN_INVALID_ARGUMENT;
  *rss = (uint64_t)info.phys_footprint; return 0;
}
*/
import "C"

// platformCurrentRSS reports the current physical footprint charged to this
// process, including memory the kernel accounts beyond its resident_size.
// Use the current value rather than the lifetime peak so a sustained breach
// clears when the working set shrinks. Older truncated Mach revisions and Mach
// failures return 0, preserving the Go-runtime accounting fallback.
func platformCurrentRSS() uint64 {
	var rss C.uint64_t
	if code := C.fak_up_current_rss(&rss); code != 0 {
		return 0
	}
	return uint64(rss)
}
