package fabricmap

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// EndpointSelector matches endpoint metadata without assigning semantics to a
// label name or value. ID is optional; Kind and Labels are conjunctive.
type EndpointSelector struct {
	ID     string            `json:"id,omitempty"`
	Kind   string            `json:"kind,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`
}
type SelectionRequest struct {
	From  EndpointSelector `json:"from"`
	To    EndpointSelector `json:"to"`
	Route Request          `json:"route"`
}
type Mapping struct {
	Source      Endpoint `json:"source"`
	Destination Endpoint `json:"destination"`
	Route       Route    `json:"route"`
}

var ErrNoEndpointMatch = errors.New("no endpoint matches selector")

// SelectRoute evaluates every concrete directed pair selected by context and
// chooses by route score, then source ID and destination ID. The concrete pair
// is returned so a broad selector can never hide what was actually chosen.
func (g Graph) SelectRoute(request SelectionRequest) (Mapping, error) {
	if err := g.Validate(); err != nil {
		return Mapping{}, err
	}
	sources := matchingEndpoints(g.Endpoints, request.From)
	if len(sources) == 0 {
		return Mapping{}, fmt.Errorf("%w: source", ErrNoEndpointMatch)
	}
	destinations := matchingEndpoints(g.Endpoints, request.To)
	if len(destinations) == 0 {
		return Mapping{}, fmt.Errorf("%w: destination", ErrNoEndpointMatch)
	}
	type candidate struct {
		mapping Mapping
		score   score
	}
	candidates := make([]candidate, 0)
	for _, source := range sources {
		for _, destination := range destinations {
			if source.ID == destination.ID {
				continue
			}
			routeRequest := request.Route
			routeRequest.From = source.ID
			routeRequest.To = destination.ID
			route, err := g.Plan(routeRequest)
			if err != nil {
				if errors.Is(err, ErrNoRoute) {
					continue
				}
				return Mapping{}, err
			}
			key := ""
			for _, link := range route.Links {
				key += "\x00" + link.ID
			}
			candidates = append(candidates, candidate{mapping: Mapping{Source: cloneEndpoint(source), Destination: cloneEndpoint(destination), Route: route}, score: score{objective: route.Objective, cost: route.TotalCost, latency: route.TotalLatencyNanos, readyTime: route.EstimatedReadyTimeNanos, hops: len(route.Links), key: key + "\x00" + source.ID + "\x00" + destination.ID}})
		}
	}
	if len(candidates) == 0 {
		return Mapping{}, fmt.Errorf("%w: selected endpoint sets", ErrNoRoute)
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].score.less(candidates[j].score) })
	return candidates[0].mapping, nil
}

func matchingEndpoints(endpoints []Endpoint, selector EndpointSelector) []Endpoint {
	out := make([]Endpoint, 0)
	for _, endpoint := range endpoints {
		if endpointMatches(endpoint, selector) {
			out = append(out, cloneEndpoint(endpoint))
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}
func endpointMatches(endpoint Endpoint, selector EndpointSelector) bool {
	if id := strings.TrimSpace(selector.ID); id != "" && endpoint.ID != id {
		return false
	}
	if kind := strings.TrimSpace(selector.Kind); kind != "" && endpoint.Kind != kind {
		return false
	}
	for key, value := range selector.Labels {
		if endpoint.Labels[key] != value {
			return false
		}
	}
	return true
}
