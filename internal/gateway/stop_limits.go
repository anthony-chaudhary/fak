package gateway

import (
	"fmt"
	"net/http"
)

// Adapted from SGLang SamplingParams.normalize at
// https://github.com/sgl-project/sglang/blob/ab9b2a173c18dc32614325319f631b016f543aad/python/sglang/srt/sampling/sampling_params.py
// Copyright 2023-2024 SGLang Team. Licensed under the Apache License, Version 2.0
// (https://www.apache.org/licenses/LICENSE-2.0). The Go cardinality adaptation
// does not port stop-regex limits; the separate literal byte policy below is local.
const maxStopSequences = 32

// maxStopSequenceBytes is Fak's explicit local policy from #12406. SGLang's
// referenced PR bounds stop-regex bytes, not literal-stop bytes; this is not a
// claim of upstream literal-stop behavior.
const maxStopSequenceBytes = 256

// rejectStopLimits runs before dispatch and streaming headers on each generation
// wire accepting stop strings. Count entries without filtering or deduplicating,
// then measure decoded string bytes (not runes or JSON-escaped wire bytes).
// Accepted requests retain exact stop bytes, ordering and semantics.
// #12406 remains partial: guided regexes are not bounded here.
func rejectStopLimits(w http.ResponseWriter, stops []string) bool {
	if len(stops) > maxStopSequences {
		writeErrCode(w, http.StatusBadRequest, "parameter_out_of_range",
			fmt.Sprintf("at most %d stop strings are allowed, got %d", maxStopSequences, len(stops)))
		return true
	}
	for i, stop := range stops {
		if len(stop) > maxStopSequenceBytes {
			writeErrCode(w, http.StatusBadRequest, "parameter_out_of_range",
				fmt.Sprintf("stop string %d is %d bytes, over the %d-byte limit", i, len(stop), maxStopSequenceBytes))
			return true
		}
	}
	return false
}
