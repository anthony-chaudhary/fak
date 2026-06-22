// Tests for the pure conversion helpers in webbench-convert: difficultyFromWeb,
// categoryFromWeb, and extractDomain. These are deterministic, resource-free
// mappings/parsers, so the expected values below are computed by hand from the
// source switch arms and the URL-splitting logic.
package main

import "testing"

func TestDifficultyFromWeb(t *testing.T) {
	tests := []struct {
		name    string
		webName string
		want    string
	}{
		{"allrecipes medium", "Allrecipes", "medium"},
		{"amazon hard", "Amazon", "hard"},
		{"apple hard", "Apple", "hard"},
		{"bbc medium", "BBC", "medium"},
		{"coursera hard", "Coursera", "hard"},
		{"espn easy", "ESPN", "easy"},
		{"globotours hard", "Globotours", "hard"},
		{"google medium", "Google", "medium"},
		{"google flights medium", "Google Flights", "medium"},
		{"google maps medium", "Google Maps", "medium"},
		{"google search easy", "Google Search", "easy"},
		{"instagram hard", "Instagram", "hard"},
		{"nike medium", "NIKE", "medium"},
		{"reddit medium", "Reddit", "medium"},
		{"wikipedia easy", "Wikipedia", "easy"},
		{"youtube easy", "Youtube", "easy"},
		{"unknown defaults medium", "SomeUnknownSite", "medium"},
		{"empty defaults medium", "", "medium"},
		{"case sensitive miss defaults medium", "amazon", "medium"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := difficultyFromWeb(tt.webName); got != tt.want {
				t.Errorf("difficultyFromWeb(%q) = %q, want %q", tt.webName, got, tt.want)
			}
		})
	}
}

func TestCategoryFromWeb(t *testing.T) {
	tests := []struct {
		name    string
		webName string
		want    string
	}{
		{"allrecipes shopping", "Allrecipes", "shopping"},
		{"amazon shopping", "Amazon", "shopping"},
		{"nike shopping", "NIKE", "shopping"},
		{"apple information", "Apple", "information"},
		{"bbc information", "BBC", "information"},
		{"coursera information", "Coursera", "information"},
		{"reddit information", "Reddit", "information"},
		{"wikipedia information", "Wikipedia", "information"},
		{"espn media", "ESPN", "media"},
		{"youtube media", "Youtube", "media"},
		{"google flights travel", "Google Flights", "travel"},
		{"globotours travel", "Globotours", "travel"},
		{"google maps navigation", "Google Maps", "navigation"},
		{"google search search", "Google Search", "search"},
		{"instagram social", "Instagram", "social"},
		{"unknown defaults general", "Google", "general"},
		{"empty defaults general", "", "general"},
		{"case sensitive miss defaults general", "amazon", "general"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := categoryFromWeb(tt.webName); got != tt.want {
				t.Errorf("categoryFromWeb(%q) = %q, want %q", tt.webName, got, tt.want)
			}
		})
	}
}

func TestExtractDomain(t *testing.T) {
	tests := []struct {
		name string
		url  string
		want string
	}{
		{"https host only", "https://example.com", "example.com"},
		{"https host with path", "https://example.com/path/to/page", "example.com"},
		{"http with port and path", "http://example.com:8080/x", "example.com:8080"},
		{"https with query string", "https://www.amazon.com/?node=1", "www.amazon.com"},
		{"trailing slash", "https://www.bbc.com/", "www.bbc.com"},
		{"no scheme returns input", "example.com/path", "example.com/path"},
		{"empty returns empty", "", ""},
		{"scheme only no host", "https://", ""},
		{"subdomain preserved", "https://maps.google.com/maps", "maps.google.com"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := extractDomain(tt.url); got != tt.want {
				t.Errorf("extractDomain(%q) = %q, want %q", tt.url, got, tt.want)
			}
		})
	}
}
