// This example reruns the step-3 attack against the full content-side stack
// with the payload written in Morse code. Same intent, same request, same
// defenses; only the alphabet changed, so none of the words the blocklist
// matches are present as text.
//
// Every content-side defense is on and nothing structural is. One claim,
// isolated: watch what each filter does with a message that has no keyword
// in it.
//
// # Running the example:
//
//	$ make example15-step6
//
// # Watch the data arrive - run this in a second terminal FIRST:
//
//	$ make example15-attacker
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

	// The same stack that held against the plain-English payload in step 3.
	content := attacks.Defenses{
		HardenedPrompt:      true,
		InputSanitizer:      true,
		ToolOutputSanitizer: true,
		Classifier:          true,
	}

	harness.Banner(
		"STEP 6: The same attack, in Morse code",
		"Every content-side defense is active: hardened prompt, regex blocklist,",
		"LLM classifier, tool-output sanitizer. The payload asks for exactly what",
		"step 2's payload asked for. Only the alphabet changed.",
		"Defenses: "+content.String(),
	)

	out, err := attacks.Encoded(ctx, h, content)
	if err != nil {
		return err
	}

	harness.Banner(fmt.Sprintf("OUTCOME: %s", out))

	// The input sanitizer reports matching zero patterns. It did not fail. It had
	// no opportunity to fail - which is why ApplyInputSanitizer reports if it
	// redacted anything rather than just returning the text. A defense that
	// abstained should never print as a defense that worked.
	//
	// The classifier did get a chance and took it: it read a plain question with
	// a Morse attachment and ruled it benign. It was not wrong about the surface.
	// It was reading the wrong thing.
	//
	// This is not a Morse trick. Base64, ROT13, leetspeak, emoji, pig latin, or
	// asking in Dutch all land in the same place. The attacker chooses the
	// encoding, and they choose it last - after reading whatever list of
	// encodings the defense anticipated.

	return nil
}
