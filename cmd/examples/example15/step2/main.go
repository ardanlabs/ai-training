// This example exfiltrates the same confidential record to the same attacker
// endpoint twice, with nothing in the way. Only the delivery differs:
//
//   - Direct: the attacker types the instructions into the chat box.
//   - Indirect: the attacker writes one comment into the document store, and
//     retrieval carries it into a real user's conversation later.
//
// The second is why RAG changes the threat model. The attacker is not in the
// conversation, was never authenticated to it, and may have poisoned the store
// months earlier.
//
// # Running the example:
//
//	$ make example15-step2
//
// # Watch the data arrive - run this in a second terminal FIRST:
//
//	$ go run ./cmd/examples/example15/attacker
//
// # This requires running the following commands:
//
//	$ make compose-up
//	$ make kronk-up

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
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	h := harness.New()
	defer h.Close()

	h.Config()

	// The baseline a team ships before anyone asks the question.
	var none attacks.Defenses

	harness.Banner(
		"STEP 2a: Direct injection - no defenses",
		"The attacker types the instructions straight into the chat box,",
		"wrapped in a benign question so retrieval pulls the confidential",
		"record into context first.",
	)

	direct, err := attacks.Direct(ctx, h, none)
	if err != nil {
		return err
	}

	harness.Banner(
		"STEP 2b: Indirect injection - no defenses",
		"Same goal, but the attacker never talks to the model. One poisoned",
		"comment in the store, and a legitimate user's legitimate question",
		"is what triggers the exfiltration.",
	)

	indirect, err := attacks.Indirect(ctx, h, none)
	if err != nil {
		return err
	}

	harness.Banner(
		"OUTCOMES",
		fmt.Sprintf("Direct:   %s", direct),
		fmt.Sprintf("Indirect: %s", indirect),
	)

	// If the attacker terminal printed the confidential profile: SSN, card,
	// password hash and address left the system over plain HTTP because a
	// document asked nicely. Nothing was simulated - a real tool call opened
	// a real socket. In step 5 the same call gets made and goes nowhere.
	//
	// The vulnerability is one line of the system prompt - step 2 of the
	// workflow, which tells the model to act on directives found in its
	// retrieved context. That line reads like good instruction-following.
	// It is the entire indirect-injection hole.

	return nil
}
