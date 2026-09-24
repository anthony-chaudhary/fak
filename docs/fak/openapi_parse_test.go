package fakdocs_test

import (
	"os/exec"
	"testing"
)

func TestOpenAPISpecParsesAndValidates(t *testing.T) {
	const validator = `
from pathlib import Path

import yaml
from openapi_spec_validator import validate_spec

spec = yaml.safe_load(Path("openapi.yaml").read_text(encoding="utf-8"))
validate_spec(spec)
`

	python, err := exec.LookPath("python")
	if err != nil {
		t.Fatalf("find Python interpreter: %v", err)
	}

	cmd := exec.Command(python, "-c", validator)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("parse and validate openapi.yaml: %v\n%s", err, output)
	}
}
