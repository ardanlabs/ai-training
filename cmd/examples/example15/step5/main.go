// This example moves the defense instead of improving it. Every mitigation so
// far read the untrusted text and judged it. The four here never look at it:
// they act on where a socket points, what a tool's schema can express, who the
// caller is, and what retrieval was allowed to return.
//
// The payloads are step 2's plain-English injection and step 4's poisoned tool
// output, both already seen to land. Step 3 also stopped the plain-English one,
// so the claim here is not that these succeed where content filters failed - it
// is about why they hold, which step 6 tests.
//
// REFUSED in the outcomes means the model asked for the tool and the system
// declined: the injection worked perfectly and did not matter.
//
// # Running the example:
//
//	$ make example15-step5
//
// # Watch the attacker terminal stay empty - run this in a second terminal:
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
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	h := harness.New()
	defer h.Close()

	h.Config()

	// Caller identity rides on the context, set at the request boundary. Nothing
	// the model emits can reach it.
	adminCtx := attacks.WithUser(ctx, attacks.Admin)
	guestCtx := attacks.WithUser(ctx, attacks.Guest)

	results := make(map[string]attacks.Outcome, 5)

	// -------------------------------------------------------------------------
	// 5a. Egress policy: the socket.

	harness.Banner(
		"STEP 5a: Egress policy - acts on the socket",
		"The model may call tool_browse. The tool checks the destination against",
		"an allowlist of the application's real outbound dependencies before it",
		"opens a connection. The attacker's collector is not on that list.",
	)

	out, err := attacks.Direct(adminCtx, h, attacks.Defenses{Egress: true})
	if err != nil {
		return fmt.Errorf("egress: %w", err)
	}

	results["5a Egress policy"] = out

	// -------------------------------------------------------------------------
	// 5b. Tool allowlist: the schema.

	harness.Banner(
		"STEP 5b: Schema redesign + verb allowlist - acts on the call",
		"tool_shell_command no longer takes a command string. It takes a verb",
		"from a fixed set and a list of arguments passed to the program directly,",
		"with no shell in between. The injection asks for `cat /etc/passwd`.",
		"There is nowhere in this schema to put that.",
	)

	out, err = attacks.ToolOutput(adminCtx, h, attacks.Defenses{ToolAllowlist: true})
	if err != nil {
		return fmt.Errorf("allowlist: %w", err)
	}

	results["5b Tool allowlist"] = out

	// -------------------------------------------------------------------------
	// 5c. Authorization: the caller.

	harness.Banner(
		"STEP 5c: Authorization - acts on the caller",
		"Same poisoned weather response, but the request is running as a guest.",
		"The tool reads the caller off the context and requires the admin role,",
		"so even a legitimate call from this user is refused.",
	)

	out, err = attacks.ToolOutput(guestCtx, h, attacks.Defenses{Authz: true})
	if err != nil {
		return fmt.Errorf("authz: %w", err)
	}

	results["5c Authorization"] = out

	// -------------------------------------------------------------------------
	// 5d. Access filter: the context.

	harness.Banner(
		"STEP 5d: Document access level - acts on the retrieval",
		"One argument to the retrieval query: return public documents only. The",
		"injection is untouched and the model is free to obey it. Watch what it",
		"finds to send.",
	)

	out, err = attacks.Direct(adminCtx, h, attacks.Defenses{AccessFilter: true})
	if err != nil {
		return fmt.Errorf("access filter: %w", err)
	}

	results["5d Access filter"] = out

	// -------------------------------------------------------------------------
	// 5e. All four.

	harness.Banner(
		"STEP 5e: All four structural defenses",
		"Both vectors, everything on. No content filtering anywhere in this run -",
		"no blocklist, no classifier, not even the hardened prompt.",
	)

	all := attacks.Defenses{
		AccessFilter:  true,
		ToolAllowlist: true,
		Authz:         true,
		Egress:        true,
	}

	out, err = attacks.Direct(adminCtx, h, all)
	if err != nil {
		return fmt.Errorf("all four, direct: %w", err)
	}

	results["5e All four (direct)"] = out

	out, err = attacks.ToolOutput(adminCtx, h, all)
	if err != nil {
		return fmt.Errorf("all four, tool output: %w", err)
	}

	results["5e All four (tool output)"] = out

	// -------------------------------------------------------------------------

	harness.Banner(
		"OUTCOMES - structural defenses vs the plain-English payload",
		"REFUSED     = the model asked and the system said no.",
		"NEUTRALIZED = the call went through and carried nothing.",
	)

	for _, name := range []string{
		"5a Egress policy",
		"5b Tool allowlist",
		"5c Authorization",
		"5d Access filter",
		"5e All four (direct)",
		"5e All four (tool output)",
	} {
		fmt.Printf("  %-28s %s\n", name, results[name])
	}

	// Be careful about what this step proves. Step 3 also stopped the
	// plain-English payload. On this table alone, the blocklist and the
	// allowlist look equally good.
	//
	// The difference is what each one is keyed on. No wording gets the tool to
	// accept `cat` when `cat` is not in the map, because the check does not read
	// words.

	return nil
}
