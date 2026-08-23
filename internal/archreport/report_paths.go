package archreport

import (
	"sort"
	"strings"
)

func dependencyPaths(name string, leaves []Leaf, byName map[string]int) []DependencyPath {
	paths := map[string][]string{name: {name}}
	pending := []string{name}
	for len(pending) > 0 {
		current := pending[0]
		pending = pending[1:]
		dependencies := append([]string(nil), leaves[byName[current]].Dependencies...)
		sort.Strings(dependencies)
		for _, dependency := range dependencies {
			candidate := append(append([]string(nil), paths[current]...), dependency)
			existing, seen := paths[dependency]
			if seen && (len(existing) < len(candidate) || len(existing) == len(candidate) && strings.Join(existing, "\x00") <= strings.Join(candidate, "\x00")) {
				continue
			}
			paths[dependency] = candidate
			pending = append(pending, dependency)
		}
	}
	delete(paths, name)
	dependencies := make([]string, 0, len(paths))
	for dependency := range paths {
		dependencies = append(dependencies, dependency)
	}
	sort.Strings(dependencies)
	out := make([]DependencyPath, 0, len(dependencies))
	for _, dependency := range dependencies {
		out = append(out, DependencyPath{Dependency: dependency, Path: paths[dependency]})
	}
	return out
}

func blastPaths(name string, leaves []Leaf, byName map[string]int) []BlastPath {
	paths := map[string][]string{name: {name}}
	pending := []string{name}
	for len(pending) > 0 {
		current := pending[0]
		pending = pending[1:]
		dependents := append([]string(nil), leaves[byName[current]].Dependents...)
		sort.Strings(dependents)
		for _, dependent := range dependents {
			if _, ok := paths[dependent]; ok {
				continue
			}
			paths[dependent] = append(append([]string(nil), paths[current]...), dependent)
			pending = append(pending, dependent)
		}
	}
	delete(paths, name)
	dependents := make([]string, 0, len(paths))
	for dependent := range paths {
		dependents = append(dependents, dependent)
	}
	sort.Strings(dependents)
	out := make([]BlastPath, 0, len(dependents))
	for _, dependent := range dependents {
		out = append(out, BlastPath{Dependent: dependent, Path: paths[dependent]})
	}
	return out
}
