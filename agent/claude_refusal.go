package agent

import (
	"context"
	"fmt"
	"iter"

	"github.com/Autumn-27/artex/llmpool"
	"github.com/Autumn-27/norma/llm"
)

// Refusal handling for Claude models.
//
// A declined request (a safety-classifier block, or the model declining) comes back
// as HTTP 200 with stop_reason "refusal" and no usable content. Left alone, the turn
// loop treats it as an empty answer and carries on. refusalGuard turns it into
// llmpool.ErrModelRefused, so the run stops with an explicit error. Failover and
// retries do not act on that error: switching to another profile to get past a
// refusal is not a fix.
//
// Server-side fallback (claude_fallback.go) runs before this. If it already re-ran
// the request on another model, a refusal still seen here means the whole chain
// declined.
type refusalGuard struct {
	inner llm.Provider
}

func refusedError() error {
	return fmt.Errorf("%w: the model declined this request (stop_reason=refusal); server-side fallback, if any, also declined", llmpool.ErrModelRefused)
}

func (g refusalGuard) Complete(ctx context.Context, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	msg, stop, usage, err := g.inner.Complete(ctx, req)
	if err == nil && stop == "refusal" {
		err = refusedError()
	}
	return msg, stop, usage, err
}

func (g refusalGuard) Stream(ctx context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		refused := false
		for ev, err := range g.inner.Stream(ctx, req) {
			if err == nil && ev.Type == llm.SEMessageDelta && ev.StopReason == "refusal" {
				refused = true
			}
			if !yield(ev, err) {
				return
			}
		}
		if refused {
			yield(llm.StreamEvent{}, refusedError())
		}
	}
}

// withRefusalGuard wraps Anthropic-format providers. OpenAI-format providers use
// their own finish reasons and are left alone.
func withRefusalGuard(prov llm.Provider, format llm.Format) llm.Provider {
	if format == llm.FormatOpenAI || format == llm.FormatOpenAIResponses {
		return prov
	}
	return refusalGuard{inner: prov}
}
