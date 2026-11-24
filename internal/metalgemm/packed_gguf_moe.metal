struct PackedMoE {
    uint slots, hidden, width, activationStride;
    float beta, linearBeta;
};

kernel void packed_situ(device const float *gateUp [[buffer(0)]],
                        device const float *scales [[buffer(1)]],
                        device float *activation [[buffer(2)]],
                        constant PackedMoE &p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    uint slot = i / p.width, col = i % p.width;
    uint base = slot * 2 * p.width + col;
    float scale = scales[3*slot];
    float g = gateUp[base] * scale, u = gateUp[base + p.width] * scale;
    float a = p.beta * tanh(g / p.beta) / (1.0f + exp(-g));
    if (p.linearBeta > 0) u = p.linearBeta * tanh(u / p.linearBeta);
    activation[slot * p.activationStride + col] = a * u;
}

kernel void packed_expert_sum(device const float *down [[buffer(0)]],
                              device const float *scales [[buffer(1)]],
                              device float *out [[buffer(2)]],
                              constant PackedMoE &p [[buffer(3)]],
                              uint col [[thread_position_in_grid]]) {
    float sum = 0;
    for (uint slot = 0; slot < p.slots; ++slot) {
        float d = down[slot*p.hidden + col] * scales[3*slot + 1];
        sum += d * scales[3*slot + 2];
    }
    out[col] = sum;
}

// The router's selection bias ranks experts. The mixture weights use unbiased
// sigmoid probabilities and normalize over the selected slots.
struct PackedRoute { uint experts, slots; float weightScale; };

kernel void packed_route(device float *probs [[buffer(0)]],
                         device const float *table [[buffer(1)]],
                         device uint *ids [[buffer(2)]],
                         device float *mixture [[buffer(3)]],
                         constant PackedRoute &p [[buffer(4)]]) {
    for (uint e = 0; e < p.experts; ++e) probs[e] = 1.0f / (1.0f + exp(-probs[e]));
    float sum = 0;
    for (uint s = 0; s < p.slots; ++s) {
        float highest = -INFINITY;
        uint best = 0;
        for (uint e = 0; e < p.experts; ++e) {
            float score = probs[e] + table[3*e];
            if (score > highest) { highest = score; best = e; }
        }
        ids[s] = best;
        mixture[3*s] = table[3*best+1];
        mixture[3*s+1] = table[3*best+2];
        mixture[3*s+2] = probs[best];
        sum += probs[best];
        probs[best] = -INFINITY;
    }
    for (uint s = 0; s < p.slots; ++s) mixture[3*s+2] = (mixture[3*s+2]/(sum+1e-20f)) * p.weightScale;
}
