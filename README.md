# Ultimate AI

Copyright 2024, 2025 Ardan Labs  
hello@ardanlabs.com

## My Information

```
Name:    Bill Kennedy
Company: Ardan Labs
Title:   Managing Partner
Email:   bill@ardanlabs.com
Twitter: goinggodotnet

Name:    Florin Pățan
Company: Ardan Labs
Title:   Senior Engineer
Email:   florin.patan@ardanlabs.com
Twitter: dlsniper
```

## Description

This class provides you with a strong foundation for understanding all the semantics and mechanisms behind adding AI
technologies to your Go applications.

## Licensing

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Examples

| Example                           | Purpose                                       | How to run                  |
|-----------------------------------|-----------------------------------------------|-----------------------------|
| example01-vectors                 | Hand-crafted vectors and cosine similarity    | `make example01`            |
| example02-embeddings              | LLM-generated embeddings                      | `make example02`            |
| example03-context-injection       | Context injection into a prompt               | `make example03`            |
| example04-chat-streaming          | Streaming chat completions via SSE            | `make example04-step{1,2}`  |
| example05-rag-motivation          | Same question with and without RAG            | `make example05-step{1,2}`  |
| example06-vector-db               | pgvector nearest-neighbor search              | `make example06`            |
| example07-ingestion               | Ingest the Go notebook into pgvector          | `make example07-step{1..4}` |
| example08-rag-pipeline            | End-to-end RAG over the Go notebook           | `make example08-step{1..3}` |
| example09-retrieval-debug         | Debug retrieval in isolation (K + threshold)  | `make example09-step{1,2}`  |
| example10-rag-end-to-end          | Interactive RAG REPL                          | `make example10-step{1,3}`  |
| example11-rag-perf                | Parallel + batched embeddings, response cache | `make example11-step{1,3}`  |
| example12-tool-calling            | Two-phase tool-calling REPL                   | `make example12-step{1,4}`  |
| example13-agent-loop              | Minimal multi-tool agent loop                 | `make example13`            |
| example14-streaming-agent         | Streaming agent with reasoning + usage panel  | `make example14-step{1,5}`  |
| example15-sql-tool                | Read-only NL→SQL tool                         | `make example15-step{1,3}`  |
| example16-tool-hardening          | Panic recovery + per-tool timeouts            | `make example16-step{1,3}`  |
| example17-mcp                     | Agent over an MCP server                      | `make example17-step{1,3}`  |
| example18-prefix-cache            | Incremental message caching (IMC)             | `make example18-step{1,2}`  |
| example19-speculative             | Speculative decoding with a draft model       | `make example19-step{1,2}`  |
| example20-semantic-cache          | Embedding-based semantic cache                | `make example20`            |
| example21-adaptive-retrieval      | Classifier gate for retrieval                 | `make example21`            |
| example22-cascade                 | Cascading model router                        | `make example22`            |
| example23-prompt-injection        | Direct/indirect prompt injection + defenses   | `make example23-step{1,8}`  |
| example24-tool-security           | Hardened shell tool with RBAC                 | `make example24-step{1,3}`  |
| example25-rag-poisoning           | Poisoned-document attacks + defenses          | `make example25-step{1,3}`  |
| example26-output-sanitization     | HTML sanitization + exfiltration defenses     | `make example26-step{1,4}`  |
| example27-chain-escalation        | Tool-chain escalation budgets and audit       | `make example27-step{1,5}`  |
| example28-image-vision-rag        | Image search via vision model + pgvector      | `make example28-step{1,5}`  |
| example29-video-transcription-rag | Chat over transcribed video chunks            | `make example29-step{1,2}`  |
| example30-pdf-docling             | PDF extraction via Docling + LLM              | `make example30`            |
| example31-coding-agent            | Coding agent with file tools                  | `make example31`            |
| example32-chat-web-service        | Embedded React chat over RAG                  | `make example32`            |
| example33-jupyter-tutorial        | Go-powered Jupyter notebook (GoMLX/GoNB)      |                             |

## Kronk

<a href="https://github.com/ardanlabs/kronk" target="_blank">
<img src="https://github.com/ardanlabs/kronk/blob/main/images/project/kronk_logo1.png?raw=true" width="150" alt="Kronk logo" align="left" style="margin-right: 10px">
</a>

[Kronk](https://github.com/ardanlabs/kronk) lets you use Go for hardware-accelerated local inference with llama.cpp
directly integrated into your applications via the [yzma](https://github.com/ardanlabs/yzma) module.
Kronk provides a high-level API that feels similar to using an OpenAI compatible API.

Kronk is used as the LLM and embeddings client throughout the examples in this repo.

<br />

## Installing Software

To run the examples in this repo, start by installing the required CLI tooling (postgres/mongo clients, ffmpeg, kronk, etc.)
using these make commands.

```
make install
make docker
make install-python
```

With the software installed, you will want to start your Kronk service. Open a terminal and run the following command.
This will show logs so start a terminal window you can see but won't need.

```
make kronk-up
```

Now start the Postgres, MongoDB, and Docling containers in Docker Compose. Open a new terminal window for this.

```
make compose-up
```

Now you need to pull down the models you will be using. Open a new terminal window for this. 
This might take several minutes depending on your bandwidth.

```
make install-models
```

## Workshop Helpers

A small set of helpers tracks which example you ran last and what comes next, so
you don't have to scroll back through `lessons.md` while teaching.

State is persisted in `.workshop-state` (gitignored). Both flavors share the
same state file, so you can mix them freely.

### Make targets (shell-agnostic)

| Command                                | What it does                                          |
|----------------------------------------|-------------------------------------------------------|
| `make ws-list`                         | Print all 82 units in order (full slug form)          |
| `make ws-run UNIT=example09-step1`     | Run the example and mark it as current                |
| `make ws-current`                      | Print the unit last marked/run                        |
| `make ws-next`                         | Print the unit that follows the current one           |
| `make ws-set UNIT=example13`           | Mark a unit as current without running it             |
| `make ws-reset`                        | Forget the current state                              |

Unit names use the short form `exampleNN` or `exampleNN-stepM`.

### Zsh + Powerlevel10k integration (optional)

If you're on zsh with [direnv](https://direnv.net) and Powerlevel10k, you get:

- Tab-completed `ws-run`, `ws-set`, `ws-current`, `ws-next`, `ws-list`, `ws-reset` commands.
- A yellow prompt segment after `git` showing
  `📚 ex09-step1 → ex09-step2` (current → next), styled to match the vcs band.
- Auto-loads only when you `cd` into this repo; unloads when you leave.

Setup:

1. Create `.envrc` at the repo root with `export WORKSHOP_DIR="$PWD"`, then run
   `direnv allow`.
2. Add to `~/.zshrc` (before `~/.p10k.zsh` is sourced):
   ```zsh
   prompt_workshop() { return; }   # stub, replaced when entering the repo

   _workshop_dir_hook() {
     if [[ -n "${WORKSHOP_DIR:-}" && -z "${WORKSHOP_LOADED:-}" ]]; then
       source "${WORKSHOP_DIR}/zarf/shell/workshop.zsh"
     elif [[ -z "${WORKSHOP_DIR:-}" && -n "${WORKSHOP_LOADED:-}" ]]; then
       unfunction ws-list ws-current ws-next ws-set ws-run ws-reset \
                  _ws_units _ws_resolve _ws_display _ws_compact _ws_complete 2>/dev/null
       prompt_workshop() { return; }
       unset WORKSHOP_LOADED WORKSHOP_STATE_FILE
     fi
   }
   autoload -Uz add-zsh-hook
   add-zsh-hook precmd _workshop_dir_hook
   ```
   Then after `~/.p10k.zsh` sources, splice the hook to run right after
   `_direnv_hook` and before `_p9k_precmd`:
   ```zsh
   () {
     local -a reordered=()
     local fn
     for fn in $precmd_functions; do
       [[ $fn == _workshop_dir_hook ]] && continue
       reordered+=( $fn )
       [[ $fn == _direnv_hook ]] && reordered+=( _workshop_dir_hook )
     done
     precmd_functions=( $reordered )
   }
   ```
3. In `~/.p10k.zsh`, add `workshop` to `POWERLEVEL9K_LEFT_PROMPT_ELEMENTS`
   right after `vcs`, and (optionally) match the vcs styling:
   ```zsh
   typeset -g POWERLEVEL9K_WORKSHOP_BACKGROUND=3
   typeset -g POWERLEVEL9K_WORKSHOP_FOREGROUND=0
   ```

The full zsh implementation lives in `zarf/shell/workshop.zsh`.

## Learn More

**Reach out about corporate training events, open enrollment live training sessions, and on-demand learning options.**

Ardan Labs (www.ardanlabs.com)  
hello@ardanlabs.com

## Purchase Video

The entire training class has been recorded to be made available to those who can't have the class taught at their 
company or who can't attend a conference. This is the entire class material.

[ardanlabs.com/education](https://www.ardanlabs.com/education/)

## Our Experience

We have taught Go to thousands of developers all around the world since 2014. There is no other company that has been 
doing it longer, and our material has proven to help jump-start developers 6 to 12 months ahead of their knowledge of Go.
We know what knowledge developers need to be productive and efficient when writing software in Go.

Our classes are perfect for intermediate-level developers who have at least a few months to years of experience writing
code in Go. Our classes provide a very deep knowledge of the programming language with a big push on language mechanics,
design philosophies, and guidelines. We focus on teaching how to write code with a priority on consistency, integrity,
readability, and simplicity.

We cover a lot about “if performance matters” with a focus on mechanical sympathy, data-oriented design, decoupling and
writing/debugging production software.

## Our Teacher

### William Kennedy ([@goinggodotnet](https://twitter.com/goinggodotnet))

_William Kennedy is a managing partner at Ardan Labs in Miami, Florida. Ardan Labs is a high-performance development and
training firm working with startups and Fortune 500 companies. He is also a co-author of the book Go in Action, the author
of the blog GoingGo.Net, and a founding member of GoBridge, which is working to increase Go adoption through diversity.

## More About Go

Go is an open source programming language that makes it easy to build simple, reliable, and efficient software. Although
it borrows ideas from existing languages, it has a unique and simple nature that makes Go programs different in character
from programs written in other languages. It balances the capabilities of a low-level systems language with some
high-level features you see in modern languages today. This creates a programming environment that allows you to be
incredibly productive, performant, and fully in control; in Go, you can write less code and do so much more.

Go is the fusion of performance and productivity wrapped in a language that software developers can learn, use, and
understand. Go is not C, yet we have many of the benefits of C with the benefits of higher level programming languages.

[The Ecosystem of the Go Programming Language](https://henvic.dev/posts/go/) - Henrique Vicente  
[The Why of Go](https://www.infoq.com/presentations/go-concurrency-gc) - Carmen Andoh  
[Go Ten Years and Climbing](https://commandcenter.blogspot.com/2017/09/go-ten-years-and-climbing.html) - Rob Pike  
[The eigenvector of "Why we moved from language X to language Y"](https://erikbern.com/2017/03/15/the-eigenvector-of-why-we-moved-from-language-x-to-language-y.html) - Erik Bernhardsson  
[Learn More](https://talks.golang.org/2012/splash.article) - Go Team  
[Simplicity is Complicated](https://www.youtube.com/watch?v=rFejpH_tAHM) - Rob Pike  
[Getting Started In Go](http://aarti.github.io/2016/08/13/getting-started-in-go) - Aarti Parikh

## Minimal Qualified Student

The material has been designed to be taught in a classroom environment. The code is well commented but missing some
contextual concepts and ideas that will be covered in class. Students with the following minimal background will get the
most out of the class.

- Studied CS in school or has a minimum of two years of experience programming full time professionally.
- Familiar with structural and object-oriented programming styles.
- Has worked with arrays, lists, queues, and stacks.
- Understands processes, threads, and synchronization at a high level.
- Operating Systems
  - Has worked with a command shell.
  - Knows how to maneuver around the file system.
  - Understands what environment variables are.

## Joining the Go Slack Community

We use a Slack channel to share links, code, and examples during the training. This is free. This is also the same Slack
community you will use after training to ask for help and interact with many Go experts around the world in the community.

1. Using the following link, fill out your name and email address: https://invite.slack.gobridge.org
2. Check your email and follow the link to the slack application.
3. Join the training channel by clicking on this link: https://gophers.slack.com/messages/training/
4. Click the “Join Channel” button at the bottom of the screen.

---

All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
