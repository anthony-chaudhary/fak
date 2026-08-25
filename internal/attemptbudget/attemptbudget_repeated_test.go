package attemptbudget

import "testing"

func TestRepeatedFailureTracker(t *testing.T) {
	tests := []struct {
		name    string
		key     string
		success bool
		want    bool
	}{
		{name: "failure A call 1", key: "A", want: false},
		{name: "failure A call 2", key: "A", want: false},
		{name: "failure A call 3", key: "A", want: true},
		{name: "success resets", key: "A", success: true, want: false},
		{name: "failure A after success", key: "A", want: false},
		{name: "changed key B resets", key: "B", want: false},
	}

	var tracker RepeatedFailureTracker
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tracker.Record(tt.key, tt.success); got != tt.want {
				t.Fatalf("Record(%q, %t) = %t, want %t", tt.key, tt.success, got, tt.want)
			}
		})
	}
}
