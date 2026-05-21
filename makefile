# Check to see if we can use ash, in Alpine images, or default to BASH.
SHELL_PATH = /bin/ash
SHELL = $(if $(wildcard $(SHELL_PATH)),/bin/ash,/bin/bash)

# ==============================================================================
# Go Installation
#
#	You need to have Go version 1.26 to run this code.
#
#	https://go.dev/dl/
#
#	If you are not allowed to update your Go frontend, you can install
#	and use a 1.26 frontend.
#
#	$ go install golang.org/dl/go1.26@latest
#	$ go1.26 download
#
#	This means you need to use `go1.26` instead of `go` for any command
#	using the Go frontend tooling from the makefile.

# ==============================================================================
# Brew Installation
#
#	Having brew installed will simplify the process of installing all the tooling.
#
#	Run this command to install brew on your machine. This works for Linux, Mac and Windows.
#	The script explains what it will do and then pauses before it does it.
#	$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#
#	WINDOWS MACHINES
#	These are extra things you will most likely need to do after installing brew
#
# 	Run these three commands in your terminal to add Homebrew to your PATH:
# 	Replace <name> with your username.
#	$ echo '# Set PATH, MANPATH, etc., for Homebrew.' >> /home/<name>/.profile
#	$ echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/<name>/.profile
#	$ eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
#
# 	Install Homebrew's dependencies:
#	$ sudo apt-get install build-essential
#
# 	Install GCC:
#	$ brew install gcc

# ==============================================================================
# Install Tooling and Dependencies
#
#	This project uses Docker and it is expected to be installed. Please provide
#	Docker at least 4 CPUs. To use Podman instead please alias Docker CLI to
#	Podman CLI or symlink the Docker socket to the Podman socket. More
#	information on migrating from Docker to Podman can be found at
#	https://podman-desktop.io/docs/migrating-from-docker.
#
#	Run these commands to install everything needed.
#	$ make install
#	$ make docker
#	$ make install-python

# ==============================================================================
# Pulling Model Images
#
# Start Kronk and pull down all the images we need for this project.
#
#	Run these commands to download the models we need.
#	$ make install-models

# ==============================================================================
# CLASS NOTES
#
# 	Mongo support
# 		db.book.find({id: 300})
#
# 		db.book.aggregate([
# 		{
# 			"$vectorSearch": {
# 				"index": "vector_index",
# 				"exact": true,
# 				"path": "embedding",
# 				"queryVector": [1.2, 2.2, 3.2, 4.2],
# 				"limit": 10
# 			}
# 		},
# 		{
# 			"$project": {
# 				"text": 1,
# 				"embedding": 1,
# 				"score": {
# 					"$meta": "vectorSearchScore"
# 				}
# 			}
# 		}
# 	}])

# ==============================================================================
# Install dependencies

install:
	brew install mongosh
	brew install mplayer
	brew install pgcli
	brew install uv
	brew install pkgconf
	brew install whisper-cpp
	brew tap homebrew-ffmpeg/ffmpeg/ffmpeg
	brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-whisper-cpp
	go install github.com/janpfeifer/gonb@latest
	go install github.com/ardanlabs/kronk/cmd/kronk@latest

docker:
	docker pull pgvector/pgvector:pg18
	docker pull mongodb/mongodb-atlas-local:8.2
	docker pull quay.io/docling-project/docling-serve:v1.9.0

install-python:
	rm -rf .venv
	uv venv --python 3.12
	uv lock
	uv sync
	uv pip install vllm
	uv pip install jupyterlab
	uv pip list > pydeps.txt

# Use this to install models. Needed to run examples. You can install a model
# as the example calls for it. Just copy/paste in terminal.
install-models:
	@echo ========== INSTALL MODELS ==========
	kronk model pull --local "unsloth/Qwen3-0.6B-Q8_0"
	@echo
	kronk model pull --local "unsloth/Qwen3-8B-Q8_0"
	@echo
	kronk model pull --local "ggml-org/embeddinggemma-300m-qat-Q8_0"
	@echo
	kronk model pull --local "ggml-org/Qwen2.5-VL-3B-Instruct-Q8_0"
	@echo
	kronk model pull --local "mradermacher/Qwen2-Audio-7B.Q8_0"
	@echo
	kronk model pull --local "unsloth/gpt-oss-20b-Q8_0"
	@echo
	kronk model pull --local "bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-Q8_0"
	@echo

# ==============================================================================
# Examples

app-attacker:
	go run ./cmd/app/attacker

# Part 0 — RAG Concepts (branch demos)

example01:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example01-vectors

example02:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example02-embeddings

example03:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example03-context-injection

example04-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example04-chat-streaming/step1

example04-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example04-chat-streaming/step2

example05-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example05-rag-motivation/step1

example05-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example05-rag-motivation/step2

example06:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example06-vector-db

example07-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example07-ingestion/step1

example07-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example07-ingestion/step2

example07-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example07-ingestion/step3

example07-step4:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example07-ingestion/step4

example08-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example08-rag-pipeline/step1

example08-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example08-rag-pipeline/step2

example08-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example08-rag-pipeline/step3

example09-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example09-retrieval-debug/step1

example09-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example09-retrieval-debug/step2

# Part 1 — RAG Application (additive)

example10-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example10-rag-end-to-end/step1

example10-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example10-rag-end-to-end/step2

example10-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example10-rag-end-to-end/step3

example11-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example11-rag-perf/step1

example11-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example11-rag-perf/step2

example11-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example11-rag-perf/step3

# Part 2 — Tools & MCP (additive)

example12-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example12-tool-calling/step1

example12-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example12-tool-calling/step2

example12-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example12-tool-calling/step3

example12: example12-step3

example13-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example13-agent-loop/step1

example13-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example13-agent-loop/step2

example13-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example13-agent-loop/step3

example13: example13-step3

example14-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example14-streaming-agent/step1

example14-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example14-streaming-agent/step2

example14-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example14-streaming-agent/step3

example14-step4:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example14-streaming-agent/step4

example14-step5:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example14-streaming-agent/step5

example15-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example15-sql-tool/step1

example15-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example15-sql-tool/step2

example15: example15-step2

example16-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example16-tool-hardening/step1

example16-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example16-tool-hardening/step2

example16-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example16-tool-hardening/step3

example17-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example17-mcp/step1

example17-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example17-mcp/step2

example17-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example17-mcp/step3

# Part 3 — Optimizations (branch demos)

example18-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example18-prefix-cache/step1

example18-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example18-prefix-cache/step2

example19-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example19-speculative/step1

example19-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example19-speculative/step2

example20-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example20-semantic-cache/step1

example20-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example20-semantic-cache/step2

example21-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example21-adaptive-retrieval/step1

example21-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example21-adaptive-retrieval/step2

example22:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example22-cascade

# Part 4 — Security (branch demos)

example23-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example23-prompt-injection/step1

example23-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example23-prompt-injection/step2

example23-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example23-prompt-injection/step3

example23-step4:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example23-prompt-injection/step4

example23-step5:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example23-prompt-injection/step5

example23-step6:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example23-prompt-injection/step6

example23-step7:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example23-prompt-injection/step7

example23-step8:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example23-prompt-injection/step8

example23-step9:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example23-prompt-injection/step9

example24-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example24-tool-security/step1

example24-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example24-tool-security/step2

example24-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example24-tool-security/step3

example25-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example25-rag-poisoning/step1

example25-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example25-rag-poisoning/step2

example26-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example26-output-sanitization/step1

example26-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example26-output-sanitization/step2

example26-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example26-output-sanitization/step3

example26-step4:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example26-output-sanitization/step4

example27-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example27-chain-escalation/step1

example27-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example27-chain-escalation/step2

example27-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example27-chain-escalation/step3

example27-step4:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example27-chain-escalation/step4

example27-step5:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example27-chain-escalation/step5

example28-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example28-image-vision-rag/step1

example28-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example28-image-vision-rag/step2

example28-step3:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example28-image-vision-rag/step3

example28-step4:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example28-image-vision-rag/step4

example28-step5:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example28-image-vision-rag/step5

example29-step1:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example29-video-transcription-rag/step1

example29-step2:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example29-video-transcription-rag/step2

example30:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example30-pdf-docling

example31:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example31-coding-agent

example32:
	@echo $@ > $(WORKSHOP_STATE)
	go run ./cmd/examples/example32-chat-web-service

# ==============================================================================
# Run Postgres, MongoDB, and Docling

compose-up:
	docker compose -f zarf/docker/compose.yaml up

compose-down:
	docker compose -f zarf/docker/compose.yaml down

compose-clean-sql:
	rm -rf zarf/docker/sql-data

compose-clean-mongo:
	rm -rf zarf/docker/mongodb && \
	mkdir -p zarf/docker/mongodb/db zarf/docker/mongodb/configdb zarf/docker/mongodb/mongot && \
	chmod -R 777 zarf/docker/mongodb

compose-clean: compose-clean-sql compose-clean-mongo

compose-logs:
	docker compose logs -n 100

# ==============================================================================
# Running Docling only

docling-compose-up:
	docker compose -f zarf/docker/compose.yaml up docling

docling-compose-down:
	docker compose -f zarf/docker/compose.yaml down docling

docling-browse:
	open -a "Google Chrome" http://localhost:5001/ui/

# ==============================================================================
# Running Mongo only

mongo-compose-up:
	docker compose -f zarf/docker/compose.yaml up mongodb

mongo-compose-down:
	docker compose -f zarf/docker/compose.yaml down mongodb

# ==============================================================================
# Kronk tooling

kronk-up:
	export KRONK_CACHE_MODEL_CONFIG_FILE=zarf/kms/model_config.yaml && \
	kronk server start

kronk-logs:
	kronk server logs

kronk-list-models:
	kronk model list --local

# ==============================================================================
# Run Tooling

download-data:
	curl -o zarf/data/example3.gz -X GET https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz \
	&& gunzip -k -d zarf/data/example3.gz \
	&& mv zarf/data/example3 zarf/data/example3.json

clean-data:
	go run cmd/cleaner/main.go

mongo:
	mongosh -u ardan -p ardan mongodb://localhost:27017

pgcli:
	pgcli postgresql://postgres:postgres@localhost

# ==============================================================================
# VLLM
# You need to add this to your .env file
# 	export VLLM_CPU_KVCACHE_SPACE=26

vllm-run:
	source .env && uv run vllm serve --host 0.0.0.0 --port 8000 --max_num_batched_tokens 131072 "NousResearch/Hermes-3-Llama-3.1-8B"

vllm-test:
	curl -X POST "http://localhost:8000/v1/chat/completions" \
		-H "Content-Type: application/json" \
		--data '{ \
			"model": "NousResearch/Hermes-3-Llama-3.1-8B", \
			"messages": [ \
				{"role": "system", "content": [{"type": "text", "text": "You are an expert developer and you are helping the user with their question."}]}, \
				{"role": "user", "content": [{"type": "text", "text": "How do you declare a variable in Python?"}]} \
			] \
		}'

# ==============================================================================
# Jupyter Notebook using Go

jupyter-run:
	uv run jupyter lab

# ==============================================================================
# Llamacpp support

llama-bench:
	zarf/libraries/llama-bench --list-devices

# ==============================================================================
# Go Modules support

tidy:
	go mod tidy

deps-upgrade:
	go get -u -v ./...
	go mod tidy

# ==============================================================================
# Python Dependencies

deps-python-sync:
	uv sync

deps-python-upgrade:
	uv lock --upgrade && uv sync
	uv pip install vllm
	uv pip install jupyterlab
	uv pip list > pydeps.txt

deps-python-outdated:
	uv pip list --outdated

# ==============================================================================
# FFMpeg test commands

ffmpeg-extract-chunks:
	rm -rf zarf/samples/videos/chunks/*
	ffmpeg -i zarf/samples/videos/test_rag_video.mp4 \
		-c copy -map 0 -f segment -segment_time 15 -reset_timestamps 1 \
		-loglevel error \
		zarf/samples/videos/chunks/output_%05d.mp4

ffmpeg-extract-frames:
	rm -rf zarf/samples/videos/frames/*
	ffmpeg -skip_frame nokey -i zarf/samples/videos/chunks/output_00000.mp4 \
		-frame_pts true -fps_mode vfr \
		-loglevel error \
		zarf/samples/videos/frames/frame-%05d.jpg

ffmpeg-extract-different-frames:
	rm -rf zarf/samples/videos/frames/*
	ffmpeg -i zarf/samples/videos/test_rag_video.mp4 \
		-vf "select='gt(scene,0.05)',setpts=N/FRAME_RATE/TB" \
		-fps_mode vfr \
		-loglevel error \
		zarf/samples/videos/frames/frame-%05d.jpg

ffmpeg-check-chunk-duration:
	ffprobe -v quiet -print_format json -show_entries format=duration zarf/samples/videos/chunks/output_00000.mp4
	ffprobe -v quiet -print_format json -show_entries format=duration zarf/samples/videos/chunks/output_00002.mp4
	ffprobe -v quiet -print_format json -show_entries format=duration zarf/samples/videos/chunks/output_00003.mp4

# ==============================================================================
# curl test commands

curl-tooling:
	curl http://localhost:11434/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{ \
	"model": "gpt-oss:latest", \
	"messages": [ \
		{ \
			"role": "user", \
			"content": "What is the weather like in New York, NY?" \
		} \
	], \
	"stream": false, \
	"tools": [ \
		{ \
			"type": "function", \
			"function": { \
				"name": "get_current_weather", \
				"description": "Get the current weather for a location", \
				"parameters": { \
					"type": "object", \
					"properties": { \
						"location": { \
							"type": "string", \
							"description": "The location to get the weather for, e.g. San Francisco, CA" \
						} \
					}, \
					"required": ["location"] \
				} \
			} \
		} \
  	], \
	"tool_selection": "auto", \
	"options": { "num_ctx": 32000 } \
	}'

# ==============================================================================

# This will establish a SSE session and this is where we will get the sessionID
# and the results of the call.
curl-mcp-get-session:
	curl -N -H "Accept: text/event-stream" http://localhost:11435/tool_list_files

# Once we have the sessionID, we can initialize the session.
# Replace the sessionID with the one you get from the SSE session.
curl-mcp-init:
	curl -X POST http://localhost:11435/tool_list_files?sessionid=$(SESSIONID) \
	-H "Content-Type: application/json" \
	-d '{ \
		"jsonrpc": "2.0", \
		"id": 1, \
		"method": "initialize", \
		"params": { \
			"protocolVersion": "2024-11-05", \
			"capabilities": {}, \
			"clientInfo": {"name": "curl-client", "version": "1.0.0"} \
		} \
	}'

# Then we can make the actual tool call. The response will be streamed in the
# session call. Replace the sessionID with the one you get from the SSE session.
curl-mcp-tool-call:
	curl -X POST http://localhost:11435/tool_list_files?sessionid=$(SESSIONID) \
	-H "Content-Type: application/json" \
	-d '{ \
		"jsonrpc": "2.0", \
		"id": 2, \
		"method": "tools/call", \
		"params": { \
			"name": "tool_list_files", \
			"arguments": {"filter": "list any files that have the name example"} \
		} \
	}'

curl-embed-triton:
	curl -i -X POST https://api.predictionguard.com/embeddings \
     -H "Authorization: Bearer $(PG_API_PREDICTIONGUARD_API_KEY)" \
     -H "Content-Type: application/json" \
     -d '{ \
		"model": "bridgetower-large-itm-mlm-itc", \
		"input": [ \
			{ \
				"text": "This is Bill Kennedy, a decent Go developer.", \
				"image": "$(IMAGE)" \
			} \
		] \
	}'

# =============================================================================
# Docling

basic-doc:
	curl -i -X POST "http://0.0.0.0:5001/v1/convert/file" \
		-H "Content-Type: multipart/form-data" \
		-F 'files=@zarf/samples/docs/dinner_menu.pdf;type=application/pdf' \
		-F 'to_formats=md' \
		-F 'include_images=false' \
		-F 'table_mode=accurate' \
		-F 'md_page_break_placeholder=---' \
		-F 'pdf_backend=dlparse_v4' \
		-F 'image_export_mode=placeholder'

# =============================================================================
# Workshop helpers (mirrors the zsh ws-* functions in zarf/shell/workshop.zsh)
#
# Tracks the "current" unit in .workshop-state so the p10k prompt segment can
# show progress. Run these from anywhere (zsh, bash, sh, CI).
#
# Usage:
#   make ws-list                        # full-slug listing
#   make ws-run UNIT=example09-step1    # runs and marks as current
#   make ws-run example09-step1         # positional form (same as above)
#   make ws-current
#   make ws-next
#   make ws-set UNIT=example13
#   make ws-set example13               # positional form (same as above)
#   make ws-reset

WORKSHOP_STATE := .workshop-state

# Allow a positional UNIT for ws-run / ws-set, e.g.
#   make ws-run example01   ==  make ws-run UNIT=example01
# When positional args are present, ws-run lets make build them as goals
# itself (so we don't redefine and warn about existing example targets).
ifneq (,$(filter ws-run ws-set,$(MAKECMDGOALS)))
  WS_POS_ARGS := $(filter-out ws-run ws-set,$(MAKECMDGOALS))
  ifneq ($(WS_POS_ARGS),)
    UNIT ?= $(firstword $(WS_POS_ARGS))
  endif
endif

# Internal: short-form listing (exampleNN[-stepM]). One unit per line.
.ws-list-short:
	@for d in cmd/examples/example*/; do \
		base=$$(basename "$$d"); \
		num=$${base#example}; num=$${num%%-*}; \
		has_steps=0; \
		for s in $$d/step*/; do \
			[ -d "$$s" ] || continue; \
			has_steps=1; \
			echo "example$$num-$$(basename $$s)"; \
		done; \
		if [ $$has_steps -eq 0 ] && [ -f "$$d/main.go" ]; then \
			echo "example$$num"; \
		fi; \
	done

# Internal: resolve a short-form UNIT to its full-slug display form.
.ws-display:
	@u="$(UNIT)"; \
	num=$${u#example}; num=$${num%%-*}; \
	step=""; case "$$u" in *-step[0-9]*) step=$${u##*-};; esac; \
	for d in cmd/examples/example$$num-*/; do \
		[ -d "$$d" ] || continue; \
		full=$$(basename "$$d"); \
		if [ -n "$$step" ]; then echo "$$full-$$step"; else echo "$$full"; fi; \
		exit 0; \
	done; \
	echo "$$u"

ws-list:
	@$(MAKE) -s .ws-list-short | while read u; do \
		$(MAKE) -s .ws-display UNIT=$$u; \
	done

ws-current:
	@[ -f $(WORKSHOP_STATE) ] && cat $(WORKSHOP_STATE) || true

ws-next:
	@cur=""; [ -f $(WORKSHOP_STATE) ] && cur=$$(cat $(WORKSHOP_STATE)); \
	$(MAKE) -s .ws-list-short | awk -v cur="$$cur" ' \
		cur=="" && !done { print; done=1; next } \
		prev=="x" && !done { print; done=1 } \
		$$0==cur { prev="x" }'

ws-set:
	@if [ -z "$(UNIT)" ]; then echo "usage: make ws-set UNIT=example09-step1"; exit 2; fi
	@echo "$(UNIT)" > $(WORKSHOP_STATE)
	@echo "current: $$($(MAKE) -s .ws-display UNIT=$(UNIT))"

ws-run:
	@if [ -z "$(UNIT)" ]; then echo "usage: make ws-run UNIT=example09-step1"; exit 2; fi
	@echo "$(UNIT)" > $(WORKSHOP_STATE)
	@echo "▶ $$($(MAKE) -s .ws-display UNIT=$(UNIT))"
ifeq ($(strip $(WS_POS_ARGS)),)
	@$(MAKE) $(UNIT)
endif

ws-reset:
	@rm -f $(WORKSHOP_STATE)
