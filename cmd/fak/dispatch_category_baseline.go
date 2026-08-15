package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/categorybaseline"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

const reasonCategoryBaselineComplete = "CATEGORY_BASELINE_COMPLETE"

func holdCompletedCategoryBaselines(root string, payload dispatchtick.RouterPayload) dispatchtick.RouterPayload {
	registry := categorybaseline.Load(root)
	if len(registry.Categories) == 0 {
		return payload
	}
	held := map[int]categorybaseline.Decision{}
	var routes []dispatchtick.IssueRoute
	steps := map[int]int{}
	for _, route := range payload.Issues {
		steps[route.Number] = route.ExpectedSteps
		if steps[route.Number] <= 0 {
			steps[route.Number] = 1
		}
		decision := categorybaseline.Evaluate(registry, route.Category, route.Layer, categoryBaselineRegression(route))
		if decision.Hold {
			held[route.Number] = decision
			routes = append(routes, route)
		}
	}
	if len(held) == 0 {
		return payload
	}
	return applyDispatchHold(payload, held, routes, steps, reasonCategoryBaselineComplete, func(issue dispatchtick.IssueRoute) string {
		d := held[issue.Number]
		return fmt.Sprintf("held: %s/%s baseline is complete (%s); move capacity to %s/%s; regression fixes remain dispatchable", d.Category, d.CompletedLayer, d.Witness, d.Category, d.NextLayer)
	})
}

func categoryBaselineRegression(route dispatchtick.IssueRoute) bool {
	for _, label := range route.Labels {
		label = strings.ToLower(strings.TrimSpace(label))
		if label == "bug" || label == "regression" || label == "type/bug" {
			return true
		}
	}
	title := strings.ToLower(strings.TrimSpace(route.Title))
	return strings.HasPrefix(title, "fix(") || strings.HasPrefix(title, "fix:") || strings.Contains(title, "regression")
}
