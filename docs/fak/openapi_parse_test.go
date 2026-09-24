package fakdocs_test

import (
	"os"
	"os/exec"
	"testing"
)

func TestOpenAPISpecParsesAndValidates(t *testing.T) {
	requireValidator := os.Getenv("FAK_REQUIRE_OPENAPI_VALIDATOR") == "1"

	python := ""
	for _, candidate := range []string{"python", "python3"} {
		if path, err := exec.LookPath(candidate); err == nil {
			python = path
			break
		}
	}
	if python == "" {
		if requireValidator {
			t.Fatal("FAK_REQUIRE_OPENAPI_VALIDATOR=1 but no Python interpreter was found")
		}
		t.Skip("Python interpreter unavailable; skipping optional OpenAPI validation")
	}

	dependencyProbe := exec.Command(python, "-c", "import yaml; import openapi_spec_validator")
	if output, err := dependencyProbe.CombinedOutput(); err != nil {
		if requireValidator {
			t.Fatalf("FAK_REQUIRE_OPENAPI_VALIDATOR=1 but validator dependencies are unavailable: %v\n%s", err, output)
		}
		t.Skipf("OpenAPI validator dependencies unavailable; skipping optional validation: %v", err)
	}

	const validator = `
from pathlib import Path

import yaml
from openapi_spec_validator import validate_spec

spec = yaml.safe_load(Path("openapi.yaml").read_text(encoding="utf-8"))
validate_spec(spec)
`

	cmd := exec.Command(python, "-c", validator)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("parse and validate openapi.yaml: %v\n%s", err, output)
	}
}
