//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdint.h>
int mg_graph_encoder_sample_count(void *graph);
int mg_graph_segments(void *graph);
int mg_graph_encoder_sample(void *graph, int index, int *segment, int *ordinal, uint64_t *start, uint64_t *end, const char **operation);
*/
import "C"

import "unsafe"

func graphEncoderTiming(ptr unsafe.Pointer) *ExecutionEncoderTiming {
	count := int(C.mg_graph_encoder_sample_count(ptr))
	if count <= 0 || count > executionEncoderSampleLimit {
		reason := "native encoder samples unavailable"
		switch count {
		case -1:
			reason = "Apple GPU stage-boundary timestamp counters unsupported"
		case -2:
			reason = "native encoder counter allocation or encoder creation failed"
		case -3:
			reason = "native encoder capture exceeded bounded capacity"
		case -4:
			reason = "native command-buffer completion unavailable"
		case -5:
			reason = "native timestamp resolution failed or returned invalid samples"
		}
		return &ExecutionEncoderTiming{Status: "unavailable", Reason: reason}
	}
	samples := make([]rawEncoderSample, 0, count)
	for i := 0; i < count; i++ {
		var segment, ordinal C.int
		var start, end C.uint64_t
		var operation *C.char
		if C.mg_graph_encoder_sample(ptr, C.int(i), &segment, &ordinal, &start, &end, &operation) == 0 {
			return &ExecutionEncoderTiming{Status: "unavailable", Reason: "native encoder sample readback failed"}
		}
		samples = append(samples, rawEncoderSample{int(segment), int(ordinal), uint64(start), uint64(end), C.GoString(operation)})
	}
	return encoderTimingFromSamples(samples, int(C.mg_graph_segments(ptr)))
}

func executionEventsAvailable() bool { return true }
