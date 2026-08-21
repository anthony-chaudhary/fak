package main

import (
	"context"
	"reflect"
	"sync"
	"testing"
)

func TestMicroharnessDeterminism(t *testing.T) {
	baseline, err := run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := check(baseline); err != nil {
		t.Fatal(err)
	}
	const runs = 100
	errCh := make(chan error, runs)
	var wg sync.WaitGroup
	for i := 0; i < runs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, gotErr := run(context.Background())
			if gotErr != nil {
				errCh <- gotErr
				return
			}
			if checkErr := check(got); checkErr != nil {
				errCh <- checkErr
				return
			}
			if !reflect.DeepEqual(got, baseline) {
				errCh <- nondeterministicReport{}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for gotErr := range errCh {
		t.Error(gotErr)
	}
}

type nondeterministicReport struct{}

func (nondeterministicReport) Error() string {
	return "microharness report changed across identical runs"
}
