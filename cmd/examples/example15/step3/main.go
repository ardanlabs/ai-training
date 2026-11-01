// This example runs step 2's direct injection against each content-side defense
// alone - hardened prompt, regex blocklist, LLM classifier - and then against
// all three. Corpus, wrapper, injection text, model and sampling are unchanged;
// the defense is the only variable.
//
// Expect this step to look like a success. That is the setup for step 6.
//
// Read the hardened-prompt row carefully. The blocklist and the classifier
// reject the request before the model is called, and score BLOCKED. The
// hardened prompt cannot reject anything: it is more text in the model's
// context, and if the model declines, the outcome is ANSWERED - the same label
// an undefended model gets for declining on its own, because within one run the
// two are indistinguishable.
//
// # Running the example:
//
//	$ make example15-step3
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
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	h := harness.New()
	defer h.Close()

	h.Config()

	// Each alone before the stack: a stack that holds tells you nothing about
	// which layer did the work.
	configs := []struct {
		name        string
		description string
		defenses    attacks.Defenses
	}{
		{
			"Hardened system prompt",
			"Append a rule telling the model to treat context as data, not instructions.",
			attacks.Defenses{HardenedPrompt: true},
		},
		{
			"Input sanitization",
			"Reject untrusted text that matches a regex blocklist of known injection phrases.",
			attacks.Defenses{InputSanitizer: true},
		},
		{
			"Detection classifier",
			"Ask a second model whether the text is an injection, and block it if so.",
			attacks.Defenses{Classifier: true},
		},
		{
			"All three together",
			"The full content-side stack, in the order a real pipeline would apply it.",
			attacks.Defenses{HardenedPrompt: true, InputSanitizer: true, Classifier: true},
		},
	}

	results := make(map[string]attacks.Outcome, len(configs))

	for _, c := range configs {
		harness.Banner("STEP 3: "+c.name, c.description, "Payload: identical to step 2's direct injection.")

		out, err := attacks.Direct(ctx, h, c.defenses)
		if err != nil {
			return fmt.Errorf("%s: %w", c.name, err)
		}

		results[c.name] = out
	}

	harness.Banner(
		"OUTCOMES - direct injection vs each content-side defense",
		"BLOCKED  = a filter rejected the request before the model saw it.",
		"ANSWERED = the request reached the model and the model declined on its own.",
	)

	for _, c := range configs {
		fmt.Printf("  %-28s %s\n", c.name, results[c.name])
	}

	// The blocklist and the classifier both had something to work with here:
	// the payload says "ignore all previous instructions" and names tool_browse
	// in plain English. A regex can match that. A classifier can read it.
	//
	// The hardened prompt works differently. It does not stop a request; it asks
	// the model to stop itself, so the most it can ever produce is ANSWERED.
	// Whether the rule is why the model declined cannot be read from that row -
	// only from comparing it against step 2's undefended run - and either way it
	// rests on the model's judgment, which is what step 6 goes after. The "All
	// three together" row is BLOCKED by the blocklist, the first layer to fire.
	//
	// The question the session turns on: what is each of these keyed on?
	//
	// Step 4 changes the channel the payload arrives through. Step 5 changes
	// where the defense sits. Step 6 changes only the alphabet.

	return nil
}
