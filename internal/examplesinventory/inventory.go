// Package examplesinventory verifies the categorized corpus summary in examples/README.md.
package examplesinventory

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
)

type Counts struct{ Directories, WithREADME, Policies int }

var summaryRE = regexp.MustCompile(`(?m)^\| \*\*Runnable demos\*\* \|.*\| (\d+) directories, (\d+) with their own README \|$`)

func Discover(root string) (Counts, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return Counts{}, err
	}
	var c Counts
	for _, e := range entries {
		if e.IsDir() {
			c.Directories++
			if _, err := os.Stat(filepath.Join(root, e.Name(), "README.md")); err == nil {
				c.WithREADME++
			}
			continue
		}
		if filepath.Ext(e.Name()) != ".json" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(root, e.Name()))
		if err != nil {
			return Counts{}, err
		}
		// Policy manifests use the fak-policy schema; unrelated model/config/report JSON does not.
		if regexp.MustCompile(`"schema"\s*:\s*"fak-policy`).Match(b) {
			c.Policies++
		}
	}
	return c, nil
}
func CheckREADME(root string) error {
	c, err := Discover(root)
	if err != nil {
		return err
	}
	b, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		return err
	}
	m := summaryRE.FindSubmatch(b)
	if m == nil {
		return fmt.Errorf("runnable demo summary not found")
	}
	want := fmt.Sprintf("%d directories, %d with their own README", c.Directories, c.WithREADME)
	got := string(m[1]) + " directories, " + string(m[2]) + " with their own README"
	if got != want {
		return fmt.Errorf("examples README summary = %s, want %s", got, want)
	}
	return nil
}
