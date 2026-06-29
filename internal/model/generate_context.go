package model

import "context"

func contextOrBackground(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	return ctx
}

// GenerateContext greedily decodes n tokens after prompt, checking ctx before
// prefill and between every decode step. On cancellation it returns the tokens
// produced so far plus ctx.Err().
func (s *Session) GenerateContext(ctx context.Context, prompt []int, n int) ([]int, error) {
	ctx = contextOrBackground(ctx)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	logits := s.Prefill(prompt)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		next := argmaxF32(logits)
		out = append(out, next)
		if s.M.Cfg.IsEOS(next) {
			return out, nil
		}
		if err := ctx.Err(); err != nil {
			return out, err
		}
		logits = s.Step(next)
	}
	return out, nil
}

// GenerateBatchContext is GenerateBatch with a cancellation point before prefill
// and between every ragged-batch decode step. On cancellation it returns the
// per-lane tokens produced so far plus ctx.Err().
func (bs *BatchSession) GenerateBatchContext(ctx context.Context, prompts [][]int, n int) ([][]int, error) {
	ctx = contextOrBackground(ctx)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	B := len(bs.Seqs)
	out := make([][]int, B)
	logits := bs.PrefillEach(prompts)
	if err := ctx.Err(); err != nil {
		return out, err
	}
	done := make([]bool, B)
	next := make([]int, B)
	active := make([]bool, B)
	for i := 0; i < n; i++ {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		anyLive := false
		for b := 0; b < B; b++ {
			active[b] = false
			if done[b] {
				continue
			}
			t := argmaxF32(logits[b])
			out[b] = append(out[b], t)
			next[b] = t
			if bs.M.Cfg.IsEOS(t) {
				done[b] = true
				continue
			}
			active[b] = true
			anyLive = true
		}
		if !anyLive {
			return out, nil
		}
		if err := ctx.Err(); err != nil {
			return out, err
		}
		logits = bs.StepBatchActive(next, active)
	}
	return out, nil
}
