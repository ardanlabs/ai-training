// This example changes the door the payload comes through. Steps 2 and 3 had it
// typed by the attacker or carried by a document; here it arrives as the output
// of a tool the model chose to call on its own.
//
// Three parts:
//
//  1. The shell tool called directly with hardcoded arguments and no model in
//     the picture, so the unsafe contract is visible before anyone can blame the
//     model for being fooled.
//
//  2. A compromised weather API appends instructions to its response. The user
//     asked about the weather in Miami, there is no document store, and tool
//     output re-enters the conversation as prompt.
//
//  3. The obvious fix: blocklist the tool output. It works here, and it is the
//     same class of defense step 6 walks through.
//
// # Running the example:
//
//	$ make example15-step4
//
// # This requires running the following commands:
//
//	$ make kronk-up
//
// No database is needed: there is no retrieval anywhere in this step, which is
// itself part of the lesson.

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

	// -------------------------------------------------------------------------
	// 4a. The unsafe interface, no model.

	harness.Banner(
		"STEP 4a: The unsafe interface - no model involved",
		"tool_shell_command takes a free-form command string. Here it is called",
		"directly with hardcoded arguments. There is no LLM in this section.",
	)

	attacks.DirectCalls(ctx)

	// Read the tool document, not the output. One string parameter, described as
	// a shell command line, handed to `sh -c`. That contract is unsafe for every
	// caller it could ever have - a model, a retry loop, a queue consumer, a
	// test. Nothing above required anyone to be tricked.

	// -------------------------------------------------------------------------
	// 4b. The injection arrives as tool output.

	harness.Banner(
		"STEP 4b: Indirect injection through tool output - no defenses",
		"The user asks for the weather. The weather API's response carries",
		"instructions. Tool output goes back into the conversation, so that",
		"text is now prompt, and the model may call the shell tool on its own.",
	)

	undefended, err := attacks.ToolOutput(ctx, h, attacks.Defenses{})
	if err != nil {
		return fmt.Errorf("undefended: %w", err)
	}

	// -------------------------------------------------------------------------
	// 4c. The obvious fix.

	harness.Banner(
		"STEP 4c: Sanitize the tool output",
		"The natural response to 4b: run the same kind of regex blocklist over",
		"tool output before it re-enters the conversation, and wrap what is left",
		"in markers the hardened prompt tells the model to distrust.",
	)

	sanitized, err := attacks.ToolOutput(ctx, h, attacks.Defenses{
		HardenedPrompt:      true,
		ToolOutputSanitizer: true,
	})
	if err != nil {
		return fmt.Errorf("sanitized: %w", err)
	}

	// -------------------------------------------------------------------------

	harness.Banner(
		"OUTCOMES - injection via tool output",
		fmt.Sprintf("No defenses:                       %s", undefended),
		fmt.Sprintf("Tool output sanitizer + hardened:  %s", sanitized),
	)

	// The attacker was never in the conversation and there was no document store
	// to poison. Every input filter in step 3 was inspecting the user's message,
	// and the user's message was "what is the weather in Miami".
	//
	// 4c works - and notice what it is. It is the step-3 blocklist, moved to a
	// new position. Moving a content filter somewhere new does not change what
	// kind of defense it is, and step 6 will treat it accordingly.

	return nil
}
