package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestA2AHandlers(t *testing.T) {
	// Create a minimal gateway server for testing
	srv, err := New(Config{
		EngineID: "mock",
		Model:    "test-model",
	})
	if err != nil {
		t.Fatalf("failed to create gateway: %v", err)
	}
	defer srv.Close()

	// Set up audit logging
	srv.SetA2AAuditLog(func(log a2aAuditLog) {
		// For testing, just verify the log entry is structured correctly
		if log.TaskID == "" {
			t.Errorf("audit log missing task_id")
		}
		if log.Transition == "" {
			t.Errorf("audit log missing transition")
		}
	})

	t.Run("handleA2ASendMessage", func(t *testing.T) {
		msg := a2aMessage{
			MessageID: "test-msg-1",
			From:      "test-caller",
			To:        "fleet-agent",
			Content: map[string]interface{}{
				"method": "laptop.status",
				"params": map[string]interface{}{
					"detailed": true,
				},
			},
			Timestamp: time.Now(),
		}
		_, _ = json.Marshal(msg)
		// Test is minimal for now - just ensure the handler compiles
	})

	t.Run("handleA2AGetExtendedAgentCard", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/a2a/v1/agent-card", nil)
		req.Header.Set("X-Caller-ID", "test-caller")
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, req)

		resp := w.Result()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected status 200, got %d", resp.StatusCode)
		}

		var card map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&card); err != nil {
			t.Fatalf("failed to decode response: %v", err)
		}

		if card["id"] != "fleet-fak" {
			t.Errorf("expected card id 'fleet-fak', got %v", card["id"])
		}

		if card["version"] != a2aVersion {
			t.Errorf("expected card version %s, got %v", a2aVersion, card["version"])
		}

		skills, ok := card["skills"].([]interface{})
		if !ok {
			t.Errorf("expected skills array")
		}

		if len(skills) == 0 {
			t.Errorf("expected at least one skill")
		}
	})

	t.Run("handleA2AListTasks empty", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/a2a/v1/tasks", nil)
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, req)

		resp := w.Result()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected status 200, got %d", resp.StatusCode)
		}

		var result map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			t.Fatalf("failed to decode response: %v", err)
		}

		if result["count"] != float64(0) {
			t.Errorf("expected 0 tasks, got %v", result["count"])
		}
	})

	t.Run("handleA2AGetTask not found", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/a2a/v1/tasks/nonexistent-task", nil)
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, req)

		resp := w.Result()
		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("expected status 404, got %d", resp.StatusCode)
		}
	})

	t.Run("handleA2ACancelTask not found", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/a2a/v1/tasks/nonexistent-task/cancel", nil)
		w := httptest.NewRecorder()
		srv.Handler().ServeHTTP(w, req)

		resp := w.Result()
		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("expected status 404, got %d", resp.StatusCode)
		}
	})
}

func TestGenerateTaskID(t *testing.T) {
	taskID, err := generateTaskID()
	if err != nil {
		t.Fatalf("generateTaskID failed: %v", err)
	}
	if taskID[:4] != "a2a_" {
		t.Errorf("expected task ID to start with 'a2a_', got %s", taskID)
	}
	if len(taskID) != 4+32 { // "a2a_" + 32 hex chars
		t.Errorf("expected task ID length 36, got %d", len(taskID))
	}
}

func TestHashParams(t *testing.T) {
	params1 := map[string]interface{}{"foo": "bar", "num": 42}
	params2 := map[string]interface{}{"foo": "bar", "num": 42}
	params3 := map[string]interface{}{"foo": "baz"}

	hash1 := hashParams(params1)
	hash2 := hashParams(params2)
	hash3 := hashParams(params3)

	if hash1 != hash2 {
		t.Errorf("identical params should have same hash")
	}
	if hash1 == hash3 {
		t.Errorf("different params should have different hashes")
	}
}

func TestValidateMethodAgainstRegistry(t *testing.T) {
	t.Run("valid method", func(t *testing.T) {
		spec, err := validateMethodAgainstRegistry("laptop.status")
		if err != nil {
			t.Errorf("unexpected error for valid method: %v", err)
		}
		if spec.Name != "laptop.status" {
			t.Errorf("expected method name 'laptop.status', got %s", spec.Name)
		}
		if spec.Scope != "read" {
			t.Errorf("expected scope 'read', got %s", spec.Scope)
		}
	})

	t.Run("invalid method", func(t *testing.T) {
		_, err := validateMethodAgainstRegistry("unregistered.method")
		if err == nil {
			t.Errorf("expected error for unregistered method")
		}
	})

	t.Run("all registry methods", func(t *testing.T) {
		expectedMethods := []string{
			"agent.info",
			"agent.ping",
			"protocol.manifest",
			"laptop.check",
			"laptop.status",
			"laptop.verify",
			"laptop.accept",
		}
		for _, method := range expectedMethods {
			spec, err := validateMethodAgainstRegistry(method)
			if err != nil {
				t.Errorf("method %s should be valid: %v", method, err)
			}
			if spec.Name != method {
				t.Errorf("expected method name %s, got %s", method, spec.Name)
			}
		}
	})
}

func TestA2AAgentCardRegistryProjection(t *testing.T) {
	srv, err := New(Config{
		EngineID: "mock",
		Model:    "test-model",
	})
	if err != nil {
		t.Fatalf("failed to create gateway: %v", err)
	}
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "/a2a/v1/agent-card", nil)
	req.Header.Set("X-Caller-ID", "test-caller")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}

	var card map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&card); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	skills, ok := card["skills"].([]interface{})
	if !ok {
		t.Fatalf("expected skills array")
	}

	// Verify that skills match the method registry
	expectedCount := len(a2aMethodRegistry)
	if len(skills) != expectedCount {
		t.Errorf("expected %d skills from registry, got %d", expectedCount, len(skills))
	}

	// Check that each skill has the right shape
	for _, skillInterface := range skills {
		skill, ok := skillInterface.(map[string]interface{})
		if !ok {
			t.Errorf("skill should be an object")
			continue
		}

		if skill["id"] == nil || skill["name"] == nil || skill["description"] == nil || skill["scope"] == nil {
			t.Errorf("skill missing required fields: %v", skill)
		}

		// Verify the method exists in the registry
		methodName, _ := skill["name"].(string)
		if methodName != "" {
			if _, exists := a2aMethodRegistry[methodName]; !exists {
				t.Errorf("skill %s not found in method registry", methodName)
			}
		}
	}
}