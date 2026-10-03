package model

import (
	"encoding/json"
	"testing"
)

// These compatibility cases were authored after reviewing the issue 12604
// candidate. They supplement, rather than replace, the blind regression suite.
// fak-test:runtime fast est=1ms lane=default
func TestAuditMissingEOSPreservesDecodeHistory(t *testing.T) {
	for _, tc := range []struct {
		name       string
		initialID  int
		initialIDs []int
		firstJSON  string
		wantError  bool
		wantID     int
	}{
		{
			name:       "programmatic_nonzero_default",
			initialID:  2,
			initialIDs: []int{2, 3},
			wantID:     2,
		},
		{
			name:      "decoded_null",
			firstJSON: `{"eos_token_id":null}`,
			wantID:    0,
		},
		{
			name:      "decoded_scalar_zero",
			firstJSON: `{"eos_token_id":0}`,
			wantID:    0,
		},
		{
			name:      "failed_semantic_decode_is_not_successful_history",
			firstJSON: `{"eos_token_id":null,"block_topology":"inside_out"}`,
			wantError: true,
			wantID:    -1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{EOSTokenID: tc.initialID, EOSTokenIDs: tc.initialIDs}
			if tc.firstJSON != "" {
				err := json.Unmarshal([]byte(tc.firstJSON), &cfg)
				if (err != nil) != tc.wantError {
					t.Fatalf("first decode error = %v, want error = %v", err, tc.wantError)
				}
			}
			// Value copies must retain decode history. Do not require rollback
			// after the failed decode: existing decoding can mutate other fields.
			copied := cfg
			for overlay := 1; overlay <= 2; overlay++ {
				if err := json.Unmarshal([]byte(`{}`), &copied); err != nil {
					t.Fatalf("overlay %d: %v", overlay, err)
				}
				if copied.EOSTokenID != tc.wantID || len(copied.EOSTokenIDs) != 0 {
					t.Fatalf("overlay %d: EOS = %d/%v, want %d with no list", overlay, copied.EOSTokenID, copied.EOSTokenIDs, tc.wantID)
				}
				for _, token := range []int{0, 2, 3} {
					if got, want := copied.IsEOS(token), token == tc.wantID; got != want {
						t.Fatalf("overlay %d: IsEOS(%d) = %v, want %v", overlay, token, got, want)
					}
				}
			}
		})
	}
}
