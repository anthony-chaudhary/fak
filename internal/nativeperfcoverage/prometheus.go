package nativeperfcoverage

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// validatePrometheusJob intentionally understands only the bounded
// scrape_configs/job_name/static_configs/targets shape committed by fak. It is
// not a general YAML parser and fails closed if the job has no active target.
func validatePrometheusJob(path, required string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("load Prometheus config: %w", err)
	}
	defer file.Close()

	found, hasTarget := false, false
	insideRequired := false
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(strings.SplitN(scanner.Text(), "#", 2)[0])
		if strings.HasPrefix(line, "- job_name:") {
			if insideRequired {
				break
			}
			name := strings.Trim(strings.TrimSpace(strings.TrimPrefix(line, "- job_name:")), `"'`)
			insideRequired = name == required
			found = found || insideRequired
			continue
		}
		if insideRequired && (strings.HasPrefix(line, "- targets:") || strings.HasPrefix(line, "targets:")) {
			value := strings.TrimSpace(strings.SplitN(line, ":", 2)[1])
			hasTarget = value != "" && value != "[]"
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("required supervised Prometheus job %q is absent", required)
	}
	if !hasTarget {
		return fmt.Errorf("required supervised Prometheus job %q has no enabled target", required)
	}
	return nil
}
