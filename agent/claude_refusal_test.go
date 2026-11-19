package agent

import (
	"context"
	"errors"
	"iter"
	"testing"

	"github.com/Autumn-27/artex/llmpool"
	"github.com/Autumn-27/norma/llm"
)

// fakeProvider replays scripted stream events and a fixed Complete result.
type fakeProvider struct {
	events []llm.StreamEvent
	stop   string
}

func (f fakeProvider) Stream(context.Context, llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		for _, ev := range f.events {
			if !yield(ev, nil) {
				return
			}
		}
	}
}

func (f fakeProvider) Complete(context.Context, llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	return llm.Message{}, f.stop, llm.Usage{}, nil
}

func refusalEvents() []llm.StreamEvent {
	return []llm.StreamEvent{
		{Type: llm.SETextDelta, Text: "partial"},
		{Type: llm.SEMessageDelta, StopReason: "refusal"},
		{Type: llm.SEMessageStop},
	}
}

func TestRefusalGuardCompleteReturnsSentinel(t *testing.T) {
	g := refusalGuard{inner: fakeProvider{stop: "refusal"}}
	_, _, _, err := g.Complete(context.Background(), llm.CompletionRequest{})
	if !errors.Is(err, llmpool.ErrModelRefused) {
		t.Fatalf("err = %v, want ErrModelRefused", err)
	}
	ok := refusalGuard{inner: fakeProvider{stop: "end_turn"}}
	if _, stop, _, err := ok.Complete(context.Background(), llm.CompletionRequest{}); err != nil || stop != "end_turn" {
		t.Fatalf("end_turn: stop=%q err=%v", stop, err)
	}
}

func TestRefusalGuardStreamAppendsSentinelAfterEvents(t *testing.T) {
	g := refusalGuard{inner: fakeProvider{events: refusalEvents()}}
	var got []llm.StreamEvent
	var last error
	for ev, err := range g.Stream(context.Background(), llm.CompletionRequest{}) {
		if err != nil {
			last = err
			continue
		}
		got = append(got, ev)
	}
	if len(got) != 3 {
		t.Fatalf("passed %d events, want all 3 (stream content must still reach the consumer)", len(got))
	}
	if !errors.Is(last, llmpool.ErrModelRefused) {
		t.Fatalf("final error = %v, want ErrModelRefused", last)
	}
}

func TestRefusalGuardStreamPassesNormalEnds(t *testing.T) {
	ev := []llm.StreamEvent{
		{Type: llm.SETextDelta, Text: "ok"},
		{Type: llm.SEMessageDelta, StopReason: "end_turn"},
	}
	g := refusalGuard{inner: fakeProvider{events: ev}}
	for _, err := range g.Stream(context.Background(), llm.CompletionRequest{}) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
}

func TestRefusalGuardStopsWhenConsumerStops(t *testing.T) {
	g := refusalGuard{inner: fakeProvider{events: refusalEvents()}}
	n := 0
	for range g.Stream(context.Background(), llm.CompletionRequest{}) {
		n++
		break
	}
	if n != 1 {
		t.Fatalf("consumer saw %d events after break, want 1", n)
	}
}

func TestWithRefusalGuardSkipsOpenAIFormats(t *testing.T) {
	p := fakeProvider{}
	if _, ok := withRefusalGuard(p, llm.FormatOpenAI).(refusalGuard); ok {
		t.Error("OpenAI format must not be wrapped")
	}
	if _, ok := withRefusalGuard(p, llm.FormatAnthropic).(refusalGuard); !ok {
		t.Error("Anthropic format must be wrapped")
	}
}
