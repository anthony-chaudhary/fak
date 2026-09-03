package gateway

import (
	"os"
	"reflect"
	"testing"
)

func TestUCXTopologyDefaultsWitness(t *testing.T) {
	// First witness requirements (#9918):
	// Table-driven preflight covering:
	// 1. IB devices present/usable.
	// 2. Container sysfs-only environment.
	// 3. Explicit operator override precedence.
	// 4. TCP fallback when no IB devices present.
	// 5. Emits scrubbed plan WITHOUT mutating os.Environ().

	// Capture initial environment snapshot
	initialEnv := os.Environ()

	tests := []struct {
		name                   string
		env                    UCXTopologyEnvironment
		wantTransport          UCXTransportType
		wantOperatorPrecedence bool
		wantTLSContains        string
		wantNetDevices         string
	}{
		{
			name: "native_ib_present",
			env: UCXTopologyEnvironment{
				AvailableIBDevices: []string{"mlx5_0", "mlx5_1"},
				ContainerSysfsOnly: false,
			},
			wantTransport:          UCXTransportInfiniBand,
			wantOperatorPrecedence: false,
			wantTLSContains:        "rc_x",
			wantNetDevices:         "mlx5_0,mlx5_1",
		},
		{
			name: "container_sysfs_only",
			env: UCXTopologyEnvironment{
				AvailableIBDevices: []string{"mlx5_0"},
				ContainerSysfsOnly: true,
			},
			wantTransport:          UCXTransportInfiniBand,
			wantOperatorPrecedence: false,
			wantTLSContains:        "rc_x,cuda_copy",
			wantNetDevices:         "mlx5_0",
		},
		{
			name: "explicit_operator_override_wins",
			env: UCXTopologyEnvironment{
				AvailableIBDevices: []string{"mlx5_0"},
				OperatorOverrides: map[string]string{
					"UCX_TLS":         "tcp",
					"UCX_NET_DEVICES": "eth0",
				},
			},
			wantTransport:          UCXTransportTCP,
			wantOperatorPrecedence: true,
			wantTLSContains:        "tcp",
			wantNetDevices:         "eth0",
		},
		{
			name: "tcp_fallback_no_ib",
			env: UCXTopologyEnvironment{
				AvailableIBDevices: nil,
			},
			wantTransport:          UCXTransportTCP,
			wantOperatorPrecedence: false,
			wantTLSContains:        "tcp,cuda_copy",
			wantNetDevices:         "all",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := ResolveUCXTopologyDefaults(tt.env)

			if res.Transport != tt.wantTransport {
				t.Fatalf("transport = %s, want %s", res.Transport, tt.wantTransport)
			}
			if res.OperatorPrecedence != tt.wantOperatorPrecedence {
				t.Fatalf("operatorPrecedence = %t, want %t", res.OperatorPrecedence, tt.wantOperatorPrecedence)
			}
			if res.NetDevices != tt.wantNetDevices {
				t.Fatalf("netDevices = %s, want %s", res.NetDevices, tt.wantNetDevices)
			}
			if res.ResolutionRationale == "" {
				t.Fatal("expected non-empty resolution rationale")
			}
			if len(res.ScrubbedSettings) == 0 {
				t.Fatal("expected non-empty scrubbed settings")
			}

			// 5. Verify os.Environ() was NOT mutated
			currentEnv := os.Environ()
			if !reflect.DeepEqual(initialEnv, currentEnv) {
				t.Fatal("os.Environ() was mutated during topology resolution")
			}
		})
	}
}
