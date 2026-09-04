package cloudroute

import (
	"strings"
	"testing"
)

var (
	benchSinkRoute  Route
	benchSinkBool   bool
	benchSinkNames  []string
	benchSinkString string
)

var (
	benchBaseEnv = []string{
		"PATH=/usr/local/bin:/usr/bin:/bin",
		"HOME=/home/agent",
		"USER=agent",
		"SHELL=/bin/bash",
		"TERM=xterm-256color",
		"LANG=en_US.UTF-8",
		"LC_ALL=en_US.UTF-8",
		"PWD=/work/fak",
		"GIT_AUTHOR_NAME=Agent",
		"GIT_AUTHOR_EMAIL=agent@fak.internal",
		"GIT_COMMITTER_NAME=Agent",
		"GIT_COMMITTER_EMAIL=agent@fak.internal",
		"EDITOR=nano",
		"TMPDIR=/tmp",
		"SSH_AUTH_SOCK=/run/user/1000/keyring/ssh",
		"LOGNAME=agent",
		"HOSTNAME=devbox-01",
		"COLORTERM=truecolor",
		"NODE_ENV=production",
		"GOPATH=/home/agent/go",
		"GOROOT=/usr/local/go",
		"SHLVL=1",
		"XDG_RUNTIME_DIR=/run/user/1000",
		"DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
	}

	benchBedrockEnv = append(append([]string(nil), benchBaseEnv...),
		"CLAUDE_CODE_USE_BEDROCK=1",
		"ANTHROPIC_BEDROCK_BASE_URL=https://bedrock-runtime.us-east-1.amazonaws.com",
		"AWS_PROFILE=corp-sso-production",
		"AWS_REGION=us-east-1",
		"AWS_DEFAULT_REGION=us-east-1",
		"AWS_ROLE_ARN=arn:aws:iam::123456789012:role/FakAgentWorkerRole",
		"AWS_ROLE_SESSION_NAME=fak-worker-session-01",
		"AWS_ACCESS_KEY_ID=ASIAIOSFODNN7EXAMPLE",
		"AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
		"AWS_SESSION_TOKEN=AQoDYXdzEJr1EXAMPLE1234567890abcdefghijklmnopqrstuvwxyz",
		"AWS_CONFIG_FILE=/home/agent/.aws/config",
		"AWS_SHARED_CREDENTIALS_FILE=/home/agent/.aws/credentials",
	)

	benchVertexEnv = append(append([]string(nil), benchBaseEnv...),
		"CLAUDE_CODE_USE_VERTEX=1",
		"ANTHROPIC_VERTEX_PROJECT_ID=corp-fak-production",
		"GOOGLE_CLOUD_PROJECT=corp-fak-production",
		"GOOGLE_CLOUD_QUOTA_PROJECT=corp-fak-quota",
		"CLOUD_ML_REGION=us-central1",
		"GOOGLE_APPLICATION_CREDENTIALS=/etc/credentials/vertex-sa.json",
		"GCLOUD_PROJECT=corp-fak-production",
	)

	benchIncidentalCloudEnv = append(append([]string(nil), benchBaseEnv...),
		"AWS_PROFILE=personal",
		"AWS_REGION=us-west-2",
		"GOOGLE_CLOUD_PROJECT=side-project",
		"AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
		"AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
	)

	benchDualRouteEnv = append(append([]string(nil), benchBaseEnv...),
		"CLAUDE_CODE_USE_BEDROCK=1",
		"CLAUDE_CODE_USE_VERTEX=1",
		"AWS_PROFILE=corp-sso-production",
		"GOOGLE_CLOUD_PROJECT=corp-fak-production",
	)

	benchWaivedBedrockEnv = append(append([]string(nil), benchBedrockEnv...),
		WaiverKey+"=1",
	)
)

// BenchmarkDetect measures route discrimination across realistic environment snapshots.
func BenchmarkDetect(b *testing.B) {
	b.Run("Baseline_NoCloudRoute", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkRoute, benchSinkBool = Detect(benchBaseEnv)
		}
	})

	b.Run("Bedrock_Selected", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkRoute, benchSinkBool = Detect(benchBedrockEnv)
		}
	})

	b.Run("Vertex_Selected", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkRoute, benchSinkBool = Detect(benchVertexEnv)
		}
	})

	b.Run("Incidental_Credentials", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkRoute, benchSinkBool = Detect(benchIncidentalCloudEnv)
		}
	})

	b.Run("Dual_Route_Conflict", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkRoute, benchSinkBool = Detect(benchDualRouteEnv)
		}
	})

	b.Run("Waived_Bedrock", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkRoute, benchSinkBool = Detect(benchWaivedBedrockEnv)
		}
	})
}

// BenchmarkCredentialNames measures keep-set extraction for secret stripping.
func BenchmarkCredentialNames(b *testing.B) {
	routeBedrock, _ := Detect(benchBedrockEnv)
	routeVertex, _ := Detect(benchVertexEnv)

	b.Run("Bedrock_WithCredentials", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkNames = routeBedrock.CredentialNames(benchBedrockEnv)
		}
	})

	b.Run("Bedrock_WithoutCredentials", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkNames = routeBedrock.CredentialNames(benchBaseEnv)
		}
	})

	b.Run("Vertex_WithCredentials", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkNames = routeVertex.CredentialNames(benchVertexEnv)
		}
	})
}

// BenchmarkExplain measures operator-facing refusal rendering.
func BenchmarkExplain(b *testing.B) {
	routeBedrock, _ := Detect(benchBedrockEnv)
	routeVertex, _ := Detect(benchVertexEnv)
	routeMinimal := Route{
		Kind:     KindBedrock,
		Selector: SelectorFor[KindBedrock],
	}

	b.Run("Bedrock_Observed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = routeBedrock.Explain()
		}
	})

	b.Run("Vertex_Observed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = routeVertex.Explain()
		}
	})

	b.Run("Minimal_NoObserved", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = routeMinimal.Explain()
		}
	})
}

// BenchmarkEnvNames measures defensive vocabulary slice retrieval per cloud route kind.
func BenchmarkEnvNames(b *testing.B) {
	b.Run("Bedrock", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkNames = EnvNames(KindBedrock)
		}
	})

	b.Run("Vertex", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkNames = EnvNames(KindVertex)
		}
	})
}

// BenchmarkAllEnvNames measures full union and sorting of all cloud route environment names.
func BenchmarkAllEnvNames(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkNames = AllEnvNames()
	}
}

// BenchmarkNonSecretEnvNames measures static sandbox allow-list filtering.
func BenchmarkNonSecretEnvNames(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkNames = NonSecretEnvNames()
	}
}

// BenchmarkDetectParallel measures concurrent detection throughput across goroutines.
func BenchmarkDetectParallel(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			r, ok := Detect(benchBedrockEnv)
			if !ok || r.Kind != KindBedrock {
				b.Fatalf("unexpected detect result: ok=%v, kind=%v", ok, r.Kind)
			}
		}
	})
}

// TestBenchmarkOperationsSanity verifies that all benchmark fixtures and operations execute
// correctly and produce valid results during standard test passes.
func TestBenchmarkOperationsSanity(t *testing.T) {
	rBedrock, ok := Detect(benchBedrockEnv)
	if !ok || rBedrock.Kind != KindBedrock {
		t.Fatalf("Detect(benchBedrockEnv) = %+v, ok=%v; want KindBedrock", rBedrock, ok)
	}
	if len(rBedrock.Observed) == 0 {
		t.Fatal("expected non-empty Observed for Bedrock env")
	}
	creds := rBedrock.CredentialNames(benchBedrockEnv)
	if len(creds) == 0 {
		t.Fatal("expected credentials in benchBedrockEnv")
	}
	exp := rBedrock.Explain()
	if !strings.Contains(exp, "AWS Bedrock") {
		t.Fatalf("Explain missing AWS Bedrock: %q", exp)
	}

	rVertex, ok := Detect(benchVertexEnv)
	if !ok || rVertex.Kind != KindVertex {
		t.Fatalf("Detect(benchVertexEnv) = %+v, ok=%v; want KindVertex", rVertex, ok)
	}

	rNone, ok := Detect(benchBaseEnv)
	if ok {
		t.Fatalf("Detect(benchBaseEnv) = %+v, want false", rNone)
	}

	rIncidental, ok := Detect(benchIncidentalCloudEnv)
	if ok {
		t.Fatalf("Detect(benchIncidentalCloudEnv) = %+v, want false", rIncidental)
	}

	rWaived, ok := Detect(benchWaivedBedrockEnv)
	if !ok || !rWaived.Waived {
		t.Fatalf("Detect(benchWaivedBedrockEnv) = %+v, ok=%v; want Waived=true", rWaived, ok)
	}

	all := AllEnvNames()
	if len(all) == 0 {
		t.Fatal("AllEnvNames returned empty")
	}
	nonSec := NonSecretEnvNames()
	if len(nonSec) == 0 || len(nonSec) >= len(all) {
		t.Fatalf("NonSecretEnvNames len=%d, all len=%d", len(nonSec), len(all))
	}
}
