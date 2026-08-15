package studymonitor

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"
)

const Schema = "fak-monitored-repositories/1"

type Registry struct {
	Schema       string       `json:"schema"`
	UpdatedAt    string       `json:"updated_at"`
	Methodology  string       `json:"methodology"`
	Repositories []Repository `json:"repositories"`
}

type Repository struct {
	Repository      string `json:"repository"`
	URL             string `json:"url"`
	Status          string `json:"status"`
	Priority        int    `json:"priority"`
	Why             string `json:"why"`
	LastChecked     string `json:"last_checked"`
	CheckedRevision string `json:"checked_revision"`
	StarsAtCheck    int    `json:"stars_at_check"`
	LastPushAtCheck string `json:"last_push_at_check"`
	StudyNote       string `json:"study_note,omitempty"`
}

type Report struct {
	Schema       string       `json:"schema"`
	RegistryPath string       `json:"registry_path"`
	AsOf         string       `json:"as_of"`
	DueDays      int          `json:"due_days"`
	Repositories []Repository `json:"repositories"`
}

func Read(path string) (Registry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Registry{}, err
	}
	var registry Registry
	if err := json.Unmarshal(data, &registry); err != nil {
		return Registry{}, fmt.Errorf("decode registry: %w", err)
	}
	if err := registry.Validate(); err != nil {
		return Registry{}, err
	}
	return registry, nil
}

func (r Registry) Validate() error {
	if r.Schema != Schema {
		return fmt.Errorf("schema must be %q", Schema)
	}
	if strings.TrimSpace(r.Methodology) == "" {
		return fmt.Errorf("methodology is required")
	}
	seen := make(map[string]bool, len(r.Repositories))
	for i, repo := range r.Repositories {
		prefix := fmt.Sprintf("repositories[%d]", i)
		if repo.Repository == "" || repo.URL == "" || repo.Why == "" {
			return fmt.Errorf("%s: repository, url, and why are required", prefix)
		}
		if seen[strings.ToLower(repo.Repository)] {
			return fmt.Errorf("%s: duplicate repository %q", prefix, repo.Repository)
		}
		seen[strings.ToLower(repo.Repository)] = true
		if repo.Priority < 1 {
			return fmt.Errorf("%s: priority must be positive", prefix)
		}
		if repo.Status != "candidate" && repo.Status != "studied" && repo.Status != "watch" && repo.Status != "dismissed" {
			return fmt.Errorf("%s: unsupported status %q", prefix, repo.Status)
		}
		if _, err := time.Parse("2006-01-02", repo.LastChecked); err != nil {
			return fmt.Errorf("%s: last_checked must be YYYY-MM-DD", prefix)
		}
		if repo.CheckedRevision == "" {
			return fmt.Errorf("%s: checked_revision is required", prefix)
		}
	}
	return nil
}

func BuildReport(path string, registry Registry, now time.Time, dueDays int) Report {
	repositories := append([]Repository(nil), registry.Repositories...)
	sort.Slice(repositories, func(i, j int) bool {
		if repositories[i].Priority != repositories[j].Priority {
			return repositories[i].Priority < repositories[j].Priority
		}
		return repositories[i].Repository < repositories[j].Repository
	})
	return Report{Schema: Schema, RegistryPath: path, AsOf: now.Format("2006-01-02"), DueDays: dueDays, Repositories: repositories}
}

func RenderHuman(w io.Writer, report Report) {
	fmt.Fprintf(w, "MONITORED_REPOSITORIES as_of=%s due_days=%d count=%d\n", report.AsOf, report.DueDays, len(report.Repositories))
	now, _ := time.Parse("2006-01-02", report.AsOf)
	for _, repo := range report.Repositories {
		checked, _ := time.Parse("2006-01-02", repo.LastChecked)
		age := int(now.Sub(checked).Hours() / 24)
		due := age >= report.DueDays
		fmt.Fprintf(w, "%d. %s status=%s checked=%s age_days=%d due=%t stars=%d rev=%s\n   %s\n", repo.Priority, repo.Repository, repo.Status, repo.LastChecked, age, due, repo.StarsAtCheck, shortRevision(repo.CheckedRevision), repo.Why)
	}
}

func WriteJSON(w io.Writer, report Report) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func shortRevision(revision string) string {
	if len(revision) > 12 {
		return revision[:12]
	}
	return revision
}
