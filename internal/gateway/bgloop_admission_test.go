package gateway

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/bgloop"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestAdmitBackgroundWakeQueuesStormInsteadOfRefusing(t *testing.T) {
	controller := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, TokenBudget: 100, MaxWaiting: 100, AgingRounds: 1})
	admit := AdmitBackgroundWake(controller, 1)
	first, err := admit(context.Background(), bgloop.WakeRequest{Job: gatewayWakeJob("loop-000")})
	if err != nil {
		t.Fatal(err)
	}
	defer first()
	var wg sync.WaitGroup
	errs := make(chan error, 99)
	for i := 1; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			release, err := admit(context.Background(), bgloop.WakeRequest{Job: gatewayWakeJob(fmt.Sprintf("loop-%03d", i))})
			if err != nil {
				errs <- err
				return
			}
			release()
		}(i)
	}
	first()
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("storm wake refused: %v", err)
	}
	stats := controller.Stats()
	if stats.Admitted != 100 || stats.Shed != 0 || stats.Denied != 0 {
		t.Fatalf("stats=%+v, want admitted=100 shed=0 denied=0", stats)
	}
}

func gatewayWakeJob(id string) loopmgr.Job {
	return loopmgr.Job{Schedule: loopmgr.Schedule{JobID: id, IntervalSeconds: 3600, MissedRun: loopmgr.MissedSkip}, State: loopmgr.JobArmed}
}
