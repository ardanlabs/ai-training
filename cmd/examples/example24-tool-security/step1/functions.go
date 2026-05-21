package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
	fntools "github.com/ardanlabs/ai-training/foundation/tools"
	"github.com/jmoiron/sqlx"
	"github.com/lib/pq"
)

// User is the identity record loaded from the users table. The authorization
// gate matches the user's roles against the authorization map.
type User struct {
	ID    string         `db:"user_id"`
	Name  string         `db:"name"`
	Email string         `db:"email"`
	Roles pq.StringArray `db:"roles"`
}

// currentUser simulates the active user for authorization checks. It is
// loaded from the database before each tool call.
var currentUser User

// loadUser fetches a user by name from the users table.
func loadUser(ctx context.Context, db *sqlx.DB, name string) (User, error) {
	const q = `SELECT user_id, name, email, roles FROM users WHERE name = $1`

	var u User
	if err := db.GetContext(ctx, &u, q, name); err != nil {
		return User{}, fmt.Errorf("load user %q: %w", name, err)
	}

	return u, nil
}

// =============================================================================
// VulnerableShellCommand — UNSAFE shell tool (prints, never executes)

type VulnerableShellCommand struct {
	name string
}

func (t *VulnerableShellCommand) Call(ctx context.Context, toolCall client.ToolCall) client.D {
	command, ok := toolCall.Function.Arguments["command"].(string)
	if !ok || command == "" {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("missing or invalid 'command' argument"))
	}

	fmt.Printf("  ⚠️  VULNERABLE: Would execute: %s\n", command)

	return fntools.SuccessResponse(toolCall.ID,
		"output", fmt.Sprintf("[SIMULATED] Executed: %s", command),
	)
}

// =============================================================================
// SecuredShellCommand — Hardened shell tool

var commandAllowlist = map[string]bool{
	"ls":     true,
	"echo":   true,
	"date":   true,
	"whoami": true,
	"pwd":    true,
}

var shellMetachars = []string{";", "|", "&", "`", "$", "(", ")", "{", "}", "<", ">", "\\", "!"}

// authorization defines which roles can use which tools. Roles come from the
// users table; a tool is allowed when any of the user's roles grants it.
var authorization = map[string]map[string]bool{
	"admin": {
		"tool_shell_command": true,
		"tool_get_weather":   true,
	},
	"user": {
		"tool_get_weather": true,
	},
}

type SecuredShellCommand struct {
	name string
}

func RegisterSecuredShellCommand(tools map[string]fntools.Tool) client.D {
	t := &SecuredShellCommand{name: "tool_shell_command"}
	tools[t.name] = t

	return t.toolDocument()
}

func (t *SecuredShellCommand) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        t.name,
			"description": "Run a limited set of safe system commands (ls, echo, date, whoami, pwd). Returns simulated output.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"command": client.D{
						"type":        "string",
						"description": "The command to run. Only allowed: ls, echo, date, whoami, pwd.",
					},
				},
				"required": []string{"command"},
			},
		},
	}
}

func (t *SecuredShellCommand) Call(ctx context.Context, toolCall client.ToolCall) client.D {
	command, ok := toolCall.Function.Arguments["command"].(string)
	if !ok || command == "" {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("missing or invalid 'command' argument"))
	}

	// Authorization gate.
	if !isAllowed(t.name, currentUser) {
		fmt.Printf("  🔒 BLOCKED: User %q (roles=%v) not authorized for %s\n", currentUser.Name, []string(currentUser.Roles), t.name)
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("user %q is not authorized to use %s", currentUser.Name, t.name))
	}

	// Argument validation: reject shell metacharacters.
	for _, meta := range shellMetachars {
		if strings.Contains(command, meta) {
			fmt.Printf("  🔒 BLOCKED: Shell metacharacter %q in command\n", meta)
			return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("rejected: shell metacharacter %q detected in command", meta))
		}
	}

	// Command allowlist: extract base command.
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("empty command"))
	}

	baseCommand := parts[0]
	if !commandAllowlist[baseCommand] {
		fmt.Printf("  🔒 BLOCKED: Command %q not in allowlist\n", baseCommand)
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("rejected: command %q is not in the allowlist [ls, echo, date, whoami, pwd]", baseCommand))
	}

	fmt.Printf("  ✅ ALLOWED: Would execute: %s\n", command)

	return fntools.SuccessResponse(toolCall.ID,
		"output", fmt.Sprintf("[SIMULATED] Executed: %s", command),
	)
}

func isAllowed(toolName string, user User) bool {
	for _, role := range user.Roles {
		if authorization[role][toolName] {
			return true
		}
	}

	return false
}

// =============================================================================
// GetWeather Tool — a safe demonstration tool

type GetWeather struct {
	name string
}

func RegisterGetWeather(tools map[string]fntools.Tool) client.D {
	t := &GetWeather{name: "tool_get_weather"}
	tools[t.name] = t

	return t.toolDocument()
}

func (t *GetWeather) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        t.name,
			"description": "Get the current weather for a location.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"location": client.D{
						"type":        "string",
						"description": "The location to get the weather for, e.g. Miami, FL.",
					},
				},
				"required": []string{"location"},
			},
		},
	}
}

func (t *GetWeather) Call(ctx context.Context, toolCall client.ToolCall) client.D {
	location, ok := toolCall.Function.Arguments["location"].(string)
	if !ok || location == "" {
		return fntools.ErrorResponse(toolCall.ID, fmt.Errorf("missing or invalid 'location' argument"))
	}

	fmt.Printf("  🌤  Weather lookup: %s\n", location)

	return fntools.SuccessResponse(toolCall.ID,
		"location", location,
		"temperature", "82°F",
		"condition", "Sunny",
		"humidity", "65%",
	)
}
