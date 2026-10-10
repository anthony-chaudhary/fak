package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// Local request policy from #12406 for the existing provider-native guided
// regex carriers. This does not add stop_regex or compile a regex locally.
const maxGuidedRegexBytes = 4096

func rejectGuidedRegexLimits(w http.ResponseWriter, req ChatRequest) bool {
	for _, field := range []struct {
		name string
		raw  json.RawMessage
	}{
		{"guided_regex", req.GuidedRegex}, {"regex", req.Regex},
	} {
		var pattern string
		// Preserve legacy provider handling of absent/null/non-string values.
		if json.Unmarshal(field.raw, &pattern) != nil {
			continue
		}
		if len(pattern) > maxGuidedRegexBytes {
			writeErrCode(w, http.StatusBadRequest, "parameter_out_of_range",
				fmt.Sprintf("%s is %d bytes, over the %d-byte limit", field.name, len(pattern), maxGuidedRegexBytes))
			return true
		}
	}
	return false
}
