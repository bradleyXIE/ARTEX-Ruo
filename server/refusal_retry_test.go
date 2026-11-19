package server

import (
	"fmt"
	"testing"

	"github.com/Autumn-27/artex/llmpool"
)

func TestRefusalIsNotRetried(t *testing.T) {
	err := fmt.Errorf("%w: declined", llmpool.ErrModelRefused)
	if isRetryableStreamError(err) {
		t.Fatal("a refusal must not be retried on the same provider")
	}
}
