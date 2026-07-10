package resume

import "testing"

func TestClassifyAttemptError(t *testing.T) {
	cases := []struct {
		text  string
		want  AttemptErrorClass
		fatal bool
	}{
		{"CHILD_CRASH upstream 400 malformed request", AttemptErrorMalformed400, true},
		{"401 unauthorized invalid api key", AttemptErrorAuth, true},
		{"429 usage limit reached", AttemptErrorUsage, false},
		{"transport connection reset by peer", AttemptErrorWireCrash, false},
		{"mystery exit", AttemptErrorUnknown, false},
	}
	for _, tc := range cases {
		if got := ClassifyAttemptError(tc.text); got != tc.want || got.Unrecoverable() != tc.fatal {
			t.Errorf("ClassifyAttemptError(%q)=%s fatal=%v, want %s fatal=%v", tc.text, got, got.Unrecoverable(), tc.want, tc.fatal)
		}
	}
}
