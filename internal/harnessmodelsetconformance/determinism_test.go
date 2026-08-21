package harnessmodelsetconformance_test

import (
	"bytes"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelsetreceipt"
)

func TestTwoRoleModelSetConformanceDeterminism(t *testing.T) {
	intent := readIntent(t)
	inventory, baselineInventory := normalize(t, successObservations())
	baselineResolution := resolve(t, intent, inventory)
	baselineExpectation, err := modelsetreceipt.Bind(intent, baselineResolution, inventory)
	if err != nil {
		t.Fatal(err)
	}
	baselineReceipt, err := modelsetreceipt.Evaluate(baselineExpectation, intent, baselineResolution, inventory, conformanceTime)
	if err != nil {
		t.Fatal(err)
	}

	const runs = 100
	errCh := make(chan string, runs)
	var wg sync.WaitGroup
	for i := 0; i < runs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			gotInventory, gotRaw := normalize(t, successObservations())
			gotResolution := resolve(t, intent, gotInventory)
			gotExpectation, bindErr := modelsetreceipt.Bind(intent, gotResolution, gotInventory)
			if bindErr != nil {
				errCh <- bindErr.Error()
				return
			}
			gotReceipt, evaluateErr := modelsetreceipt.Evaluate(gotExpectation, intent, gotResolution, gotInventory, conformanceTime)
			if evaluateErr != nil {
				errCh <- evaluateErr.Error()
				return
			}
			if !bytes.Equal(gotRaw, baselineInventory) || !reflect.DeepEqual(gotResolution, baselineResolution) || !reflect.DeepEqual(gotExpectation, baselineExpectation) || !reflect.DeepEqual(gotReceipt, baselineReceipt) {
				errCh <- "conformance pipeline changed across identical runs"
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for failure := range errCh {
		t.Error(failure)
	}
}
