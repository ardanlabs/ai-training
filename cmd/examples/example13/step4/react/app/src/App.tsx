import { useState, useRef, useEffect } from "react";
import "./App.css";

interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

// Tool calls stream in pieces using the OpenAI shape. The first piece carries
// the id and name, later pieces with the same index add to the arguments.
interface ToolCallDelta {
  index: number;
  id?: string;
  type?: string;
  function?: {
    name?: string;
    arguments?: string;
  };
}

interface Message {
  role: string;
  content: string;
  reasoning?: string;
  toolCall?: ToolCall;
  usage?: Usage;
}

interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  completion_tokens_details?: {
    reasoning_tokens: number;
  };
  tokens_per_second: number;
}

interface ChatRequest {
  stream: boolean;
  messages: Message[];
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
}

interface Choice {
  index: number;
  delta?: {
    role?: string;
    content?: string;
    reasoning_content?: string;
    tool_calls?: ToolCallDelta[];
  };
  finish_reason?: string | null;
}

interface Response {
  id?: string;
  created?: number;
  model?: string;
  choices?: Choice[];
  error?: {
    message: string;
  };
  usage?: Usage;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [temperature, setTemperature] = useState(1.0);
  const [topP, setTopP] = useState(0.95);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(1000);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: "user",
      content: input.trim(),
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setIsLoading(true);

    const assistantMessage: Message = {
      role: "assistant",
      content: "",
    };
    setMessages([...updatedMessages, assistantMessage]);

    try {
      const requestBody: ChatRequest = {
        stream: true,
        messages: updatedMessages,
        temperature,
        top_p: topP,
        top_k: topK,
        max_tokens: maxTokens,
      };

      const response = await fetch("http://localhost:8080/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No response body");
      }

      let accumulatedContent = "";
      let accumulatedReasoning = "";
      // The cast stops TypeScript narrowing toolCall to undefined, which it
      // otherwise does because the only assignment is deeper in the loop.
      let toolCall = undefined as ToolCall | undefined;
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // A read can end in the middle of a line, so keep the last partial
        // line in the buffer until the rest of it arrives.
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.substring(6);

            if (data === "[DONE]") {
              break;
            }

            try {
              const parsed: Response = JSON.parse(data);

              if (parsed.error) {
                console.error("Error from server:", parsed.error);
                accumulatedContent += `\n[Error: ${parsed.error.message}]`;
              } else if (parsed.choices && parsed.choices.length > 0) {
                const delta = parsed.choices[0].delta;

                if (delta?.content) {
                  accumulatedContent += delta.content;
                }
                if (delta?.reasoning_content) {
                  accumulatedReasoning += delta.reasoning_content;
                }

                // The tool call arrives before the chunk with the
                // "tool_calls" finish reason, whose delta is empty. Only the
                // first tool call is shown.
                const tc = delta?.tool_calls?.find((t) => t.index === 0);
                if (tc) {
                  toolCall = {
                    id: tc.id || toolCall?.id || "",
                    name: tc.function?.name || toolCall?.name || "",
                    arguments:
                      (toolCall?.arguments || "") +
                      (tc.function?.arguments || ""),
                  };
                }
              }

              setMessages((prev) => {
                const newMessages = [...prev];
                const currentMsg = newMessages[newMessages.length - 1];
                newMessages[newMessages.length - 1] = {
                  role: "assistant",
                  content: accumulatedContent,
                  reasoning: accumulatedReasoning,
                  toolCall: toolCall || currentMsg.toolCall,
                  usage: parsed.usage || currentMsg.usage,
                };
                return newMessages;
              });
            } catch (e) {
              console.error("Failed to parse JSON:", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          role: "assistant",
          content: "Error: Failed to get response from server",
        };
        return newMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearHistory = () => {
    setMessages([]);
  };

  return (
    <div className="app">
      <div className="chat-container">
        <div className="parameters">
          <div className="param">
            <label>Temperature: {temperature}</label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              disabled={isLoading}
            />
          </div>
          <div className="param">
            <label>Top P: {topP}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={topP}
              onChange={(e) => setTopP(parseFloat(e.target.value))}
              disabled={isLoading}
            />
          </div>
          <div className="param">
            <label>Top K: {topK}</label>
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              disabled={isLoading}
            />
          </div>
          <div className="param">
            <label>Max Tokens: {maxTokens}</label>
            <input
              type="range"
              min="10"
              max="4000"
              step="10"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              disabled={isLoading}
            />
          </div>
          <button
            onClick={clearHistory}
            className="clear-button"
            disabled={isLoading}
          >
            Clear History
          </button>
        </div>
        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="role">
                {msg.role === "user" ? "You" : "Assistant"}
              </div>
              <div className="content">
                {msg.reasoning && (
                  <div
                    style={{
                      color: "red",
                      whiteSpace: "pre-wrap",
                      marginBottom: "10px",
                    }}
                  >
                    {msg.reasoning}
                  </div>
                )}
                {msg.content}
                {msg.toolCall && (
                  <div
                    style={{
                      color: "#006400",
                      whiteSpace: "pre-wrap",
                      marginTop: "10px",
                      fontFamily: "monospace",
                    }}
                  >
                    Model Asking For Tool Call:
                    <br />
                    ToolID[{msg.toolCall.id}]: {msg.toolCall.name}(
                    {msg.toolCall.arguments})
                  </div>
                )}
                {msg.usage && (
                  <div
                    className="usage-info"
                    style={{
                      fontSize: "0.8em",
                      color: "#888",
                      marginTop: "8px",
                      paddingTop: "8px",
                      borderTop: "1px solid #eee",
                    }}
                  >
                    Input: {msg.usage.prompt_tokens} | Output:{" "}
                    {msg.usage.completion_tokens} | Speed:{" "}
                    {msg.usage.tokens_per_second.toFixed(2)} t/s
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div className="input-container">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={isLoading}
          />
          <button onClick={sendMessage} disabled={isLoading || !input.trim()}>
            {isLoading ? "Sending..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
