package llmpool

import (
	"context"
	"fmt"
	"testing"
)

// A refusal carries no HTTP status, which shouldFailover reads as a transport
// failure. It must still never trigger a profile switch.
func TestRefusalNeverFailsOver(t *testing.T) {
	err := fmt.Errorf("%w: declined", ErrModelRefused)
	if shouldFailover(context.Background(), err) {
		t.Fatal("refusal must not fail over to another profile")
	}
}
