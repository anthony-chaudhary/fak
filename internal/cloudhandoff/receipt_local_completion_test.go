package cloudhandoff

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestReceiptLocalCompletionSurvivesRemoteOutcome(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		local         Attempt
		wantCompleted bool
	}{
		{Attempt{Engine: "fak-native", Location: "local", Outcome: "completed"}, true},
		{Attempt{Engine: "fak-native", Location: "local", Outcome: "failed"}, false},
		{Attempt{Engine: "app-provider", Location: "remote", Outcome: "completed"}, false},
	} {
		local := tc.local
		for _, terminal := range []string{"denied", "failed", "completed"} {
			t.Run(local.Location+"/"+local.Outcome+"/"+terminal, func(t *testing.T) {
				p, r := base()
				if terminal == "denied" {
					p.Consent = ConsentNever
				}
				calls := 0
				got, err := New().Handoff(p, r, nil, func(Package) error {
					calls++
					if terminal == "failed" {
						return errors.New("offline")
					}
					return nil
				}, local)
				want := Receipt{
					Schema:          "fak.cloud-handoff-receipt/1",
					OperationID:     r.OperationID,
					Terminal:        terminal,
					Attempts:        []Attempt{local},
					LocalCompleted:  tc.wantCompleted,
					RemoteCompleted: terminal == "completed",
				}
				var wantErr error
				wantCalls := 0
				if terminal == "denied" {
					wantErr = ErrDenied
				} else {
					wantCalls = 1
					want.Attempts = append(want.Attempts, Attempt{Engine: "app_provider", Location: "remote", Outcome: terminal})
					if terminal == "failed" {
						wantErr = ErrRemote
					}
				}
				if !errors.Is(err, wantErr) || calls != wantCalls || !reflect.DeepEqual(got, want) {
					t.Fatalf("receipt=%+v err=%v sends=%d; want receipt=%+v err=%v sends=%d", got, err, calls, want, wantErr, wantCalls)
				}
			})
		}
	}
}
