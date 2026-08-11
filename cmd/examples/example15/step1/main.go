// This example is reconnaissance: the attacker asks the assistant to describe
// its own tools, their parameters and its system prompt. Nothing leaves and no
// rule is broken, but every later step needs to know tool_browse exists, and
// this is where that is handed over.
//
// Tool definitions go to the model as structured data on every request, so the
// name was never a secret to begin with.
//
// # Running the example:
//
//	$ make example15-step1
//
// # This requires running the following commands:
//
//	$ make kronk-up
//
// No database is needed for this step.

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/ardanlabs/ai-training/cmd/examples/example15/attacks"
	"github.com/ardanlabs/ai-training/cmd/examples/example15/harness"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	h := harness.New()
	defer h.Close()

	h.Config()

	harness.Banner(
		"STEP 1: Reconnaissance - tool discovery",
		"The attacker asks the assistant what it can do. No defenses, no",
		"retrieval, no attack - just a single, polite, entirely legal question.",
	)

	// No defenses: what the system gives away when nobody is attacking it.
	out, err := attacks.Recon(ctx, h, attacks.Defenses{})
	if err != nil {
		return err
	}

	harness.Banner(fmt.Sprintf("OUTCOME: %s", out))

	// If the outcome is LEAKED, the attacker has a tool name and every later
	// step is possible. Cost: one question, no defense tripped, nothing in the
	// audit log that looks like an attack.
	//
	// The tools were never hidden. They go to the model as structured
	// definitions on every request. A design that depends on the attacker not
	// knowing a tool name has no defense at all.

	return nil
}
