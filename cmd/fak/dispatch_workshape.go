package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/workshape"
)

type dispatchWorkShapeEnvelope struct {
	WorkShape *workshape.Contract `json:"work_shape"`
}

func readDispatchWorkShape(path string) (*workshape.Result, error) {
	if path == "" {
		return nil, nil
	}
	var body []byte
	var err error
	if path == "-" {
		body, err = io.ReadAll(os.Stdin)
	} else {
		body, err = os.ReadFile(path)
	}
	if err != nil {
		return nil, err
	}
	var envelope dispatchWorkShapeEnvelope
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, err
	}
	if envelope.WorkShape == nil {
		return nil, fmt.Errorf("input has no work_shape contract")
	}
	result := workshape.Evaluate(*envelope.WorkShape)
	return &result, nil
}
