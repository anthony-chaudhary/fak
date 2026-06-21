// browser - web automation layer for webbench
// Uses Playwright CLI for browser control (cross-platform)
package browser

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// Browser represents a browser session.
type Browser struct {
	cmd     *exec.Cmd
	script  string
	context string
}

// PageState represents the captured DOM and page state.
type PageState struct {
	URL       string    `json:"url"`
	Title     string    `json:"title"`
	Content   string    `json:"content"`
	HTML      string    `json:"html,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// Action represents a browser action.
type Action struct {
	Type     string `json:"type"` // click, fill, navigate, wait
	Selector string `json:"selector,omitempty"`
	Value    string `json:"value,omitempty"`
	URL      string `json:"url,omitempty"`
}

// Config holds browser configuration.
type Config struct {
	Headless bool
	Timeout  time.Duration
}

// DefaultConfig returns sensible browser defaults.
func DefaultConfig() Config {
	return Config{
		Headless: true,
		Timeout:  30 * time.Second,
	}
}

// New launches a new browser session.
func New(cfg Config) (*Browser, error) {
	// For now, we'll use Playwright's CLI via node
	// In production, this would use a long-running browser process
	return &Browser{
		script:  "",
		context: "",
	}, nil
}

// Navigate to a URL and return the page state.
func (b *Browser) Navigate(url string) (*PageState, error) {
	// Use playwright CLI to navigate and capture page state
	script := fmt.Sprintf(`
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto('%s', { waitUntil: 'domcontentloaded' });
  const state = {
    url: page.url(),
    title: await page.title(),
    content: await page.textContent('body'),
  };
  console.log(JSON.stringify(state));
  await browser.close();
})();
`, url)

	cmd := exec.Command("node", "-e", script)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("navigate: %w: %s", err, string(output))
	}

	var state PageState
	if err := json.Unmarshal(output, &state); err != nil {
		return nil, fmt.Errorf("parse state: %w", err)
	}
	state.Timestamp = time.Now()

	return &state, nil
}

// Click an element on the page.
func (b *Browser) Click(selector string) error {
	script := fmt.Sprintf(`
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  // TODO: restore page state from context
  await page.click('%s');
  await browser.close();
})();
`, selector)

	cmd := exec.Command("node", "-e", script)
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("click: %w: %s", err, string(output))
	}
	return nil
}

// Fill a form field with text.
func (b *Browser) Fill(selector, value string) error {
	script := fmt.Sprintf(`
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.fill('%s', '%s');
  await browser.close();
})();
`, selector, value)

	cmd := exec.Command("node", "-e", script)
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("fill: %w: %s", err, string(output))
	}
	return nil
}

// DetectPlaywright checks if Playwright is available.
func DetectPlaywright() bool {
	// Check for Node.js
	if _, err := exec.LookPath("node"); err != nil {
		return false
	}

	// Check for Playwright package
	cmd := exec.Command("node", "-e", "try { require('playwright'); console.log('OK'); } catch(e) { console.error('MISSING'); }")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return false
	}
	return strings.Contains(string(output), "OK")
}

// InstallPlaywright installs Playwright and browser binaries.
func InstallPlaywright() error {
	// Install via npm
	cmd := exec.Command("npm", "install", "-g", "playwright")
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("npm install: %w: %s", err, string(output))
	}

	// Install chromium
	cmd = exec.Command("npx", "playwright", "install", "chromium")
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("install chromium: %w: %s", err, string(output))
	}

	return nil
}
