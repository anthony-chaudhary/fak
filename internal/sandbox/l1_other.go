//go:build !linux && !darwin && !windows

package sandbox

import (
	"os/exec"
)

type otherConfinement struct {
	spec Spec
}

func newOSConfinement(spec Spec) (osConfinement, error) {
	return &otherConfinement{spec: spec}, nil
}

func (o *otherConfinement) PrepareCommand(cmd *exec.Cmd, req ExecutionRequest) error {
	return nil
}

func (o *otherConfinement) OnProcessStart(pid int) error {
	return nil
}

func (o *otherConfinement) PostProcess(res *ExecutionResult) error {
	return nil
}

func (o *otherConfinement) Close() error {
	return nil
}
