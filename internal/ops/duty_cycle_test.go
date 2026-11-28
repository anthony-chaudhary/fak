package ops

import (
	"testing"
	"time"
)

func dutyCycle(cpuMS int64) LoadCycle {
	return LoadCycle{WallMS: 60000, SelfCPUMillis: cpuMS, HaveSelfCPU: true}
}

func testDutyPolicy() DutyCyclePolicy {
	return DutyCyclePolicy{MaxDutyPercent: 2, MinInterval: time.Second, MaxInterval: 30 * time.Minute, Window: 4}
}

func TestDutyCycleCapStretchesToUnderCap(t *testing.T) {
	c := NewDutyCycleController(testDutyPolicy())
	d := c.Observe(dutyCycle(2000), time.Minute, false)
	if !d.Stretched || d.ReasonClass != PauseReasonDutyCycleCap {
		t.Fatalf("expected stretch with DUTY_CYCLE_CAP, got %+v", d)
	}
	if d.NextInterval <= time.Minute || d.PredictedDuty > 2 {
		t.Fatalf("predicted duty %.2f%% over cap, interval %v", d.PredictedDuty, d.NextInterval)
	}
}

func TestDutyCycleCapStretchIsBounded(t *testing.T) {
	p := testDutyPolicy()
	p.MaxInterval = 5 * time.Minute
	c := NewDutyCycleController(p)
	d := c.Observe(dutyCycle(60000), time.Minute, false)
	if d.NextInterval != p.MaxInterval || !d.Overloaded {
		t.Fatalf("expected clamp to max and overloaded, got %+v", d)
	}
}

func TestDutyCycleCapBelowCapDoesNotStretch(t *testing.T) {
	c := NewDutyCycleController(testDutyPolicy())
	d := c.Observe(dutyCycle(500), time.Minute, false)
	if d.Stretched || d.NextInterval != time.Minute || d.ReasonClass != "" {
		t.Fatalf("below-cap tick stretched: %+v", d)
	}
}

func TestDutyCycleCapDiskRefuseOverrides(t *testing.T) {
	c := NewDutyCycleController(testDutyPolicy())
	d := c.Observe(dutyCycle(20000), time.Minute, true)
	if d.Stretched || !d.Yielded || d.NextInterval != time.Minute {
		t.Fatalf("cap must yield under disk refuse: %+v", d)
	}
}

func TestDutyCycleCapUnmeasuredTickNotObserved(t *testing.T) {
	c := NewDutyCycleController(testDutyPolicy())
	d := c.Observe(LoadCycle{WallMS: 60000}, time.Minute, false)
	if d.Stretched {
		t.Fatalf("unmeasured tick stretched: %+v", d)
	}
}

func TestDutyCycleCapMonotonicInCost(t *testing.T) {
	var prev time.Duration
	for cpu := int64(0); cpu <= 40000; cpu += 500 {
		d := NewDutyCycleController(testDutyPolicy()).Observe(dutyCycle(cpu), time.Minute, false)
		if d.NextInterval < prev {
			t.Fatalf("interval fell from %v to %v at cpu=%dms", prev, d.NextInterval, cpu)
		}
		prev = d.NextInterval
	}
}
