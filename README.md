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

This class provides you a strong foundation for understanding all the semantics and mechanincs behind adding AI technologies to your Go applications.

## Licensing

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Examples

**- Example 01-06**: Vectors, Embeddings, RAG

- **Example 01**: This example shows you what a vector and embedding is by hand crafting a relationship of data. It also shows you how cosine similarity works between different vectors.

- **Example 02**: This example shows you how to use an LLM to create vector embeddings and get the same results from the hand crafted solution.

- **Example 03**: This example shows you how to use MongoDB as a vector database to perform a nearest neighbor vector search. The example will create a vector search index, store 2 documents, and perform a vector search.

- **Example 04**: This example shows you how to use Kronk to create a reasonable response to a question with provided content.

- **Example 05**: This example shows you how to use MongoDB and Kronk to create a proper vector embedding database for the Ultimate Go Notebook. With this vector database, you will be able to query for content that has a strong similarity to your question.

- **Example 06**: This example shows you how to use MongoDB and Kronk to perform a vector search for a user question. The search will return the top 5 chunks from the database. Then these chunks are sent to the Llama model to create a coherent response. You must run example05 first.

**- Example 07**: SQL Code Generation

- This example shows you how to get a model to generate SQL queries.

**- Example 08**: Vision Models

- **Step 1**: This example shows you how to use a vision model to generate an image description.

- **Step 2**: This examples takes step1 and shows you how to generate a vector embedding from the image description and for the image itself.

- **Step 3**: This example takes step2 and shows you how to store the image details into a vector database for similarity searching.

- **Step 4**: This example takes step3 and shows you how to search for an image based on its description.

- **Step 5**: This example takes step4 and shows you how to process a set of images from a location on disk and provide search capabilities by text or similar image.

**- Example 09**: AI Agents

- **Step 1**: This example shows you how to create a terminal based chat agent using the Kronk service.

- **Step 2**: This example shows you the workflow and mechanics for tool calling.

- **Step 3**: This example shows you how to add tool calling to the chat agent from steps 1 and 2.

- **Step 4**: This example shows you how introduce "real" tooling into the coding agent from step3. We will add support for reading, listing, creating, and editing files. We also enhance the agent's UI.

**- Example 10**: AI Agents with MCP

- **Step 1**: This example shows you how to create a basic MCP interaction where the Server runs as a service and extends the set of tools as endpoints. The Client makes a call to the Server via the MCP SSE protocol. The makefile shows you the raw CURL calls that are used to make the client/server interaction.

- **Step 2**: This example shows you how to use the program from cmd/examples/example10/step4/main.go and move the tooling to a MCP service that is called by the tooling.

**- Example 11**: Video Transcription and Text Extraction

- **Step 1**: This example provides a proof of concept for extracting transcriptions and, code examples from videos using the Kronk and a vision model. It then stores the extracted data in a MongoDB database for vector search and RAG functionality.

- **Step 2**: This example provides a chat interface for the video that was processed in step1.

**- Example 12**: PDF Processing with Docling

- This example shows you how to query the Docling API to extract data from a PDF and have it processed by an LLM.

**- Example 13**: Kronk (API Based Model Server)

- **Step 1**: This example shows you how to use Kronk to create a simple chat application against an inference model using llama.cpp directly via yzma and a native Go application.

- **Step 2**: This example shows you how to use Kronk to execute a simple prompt against a vision model using llama.cpp directly via yzma and a native Go application.

- **Step 3**: This example shows you a complete RAG application using DuckDB as an embedding DB and an embedding model to generate embeddings, and a chat model for answering a question using Kronk directly via yzma and a native Go application.

- **Step 4**: This example shows you a web service that provides a chat endpoint for asking questions about the Go notebook. It uses the code from step3 for the RAG aspects of the application. The code also provides an embedded react app that can be used to interact with the chat endpoint. The react app is built using vite and the code is in the app directory.

**- Example 14**: Jupyter Notebook using Go

- This example shows you how to use GoMLX and GoNB projects so we can run a Jupyter notebook that can execute Go code.

## Kronk

<a href="https://github.com/ardanlabs/kronk" target="_blank">
<img src="https://github.com/ardanlabs/kronk/blob/main/images/project/kronk_logo1.png?raw=true" width="150" alt="Kronk logo" align="left" style="margin-right: 10px">
</a>

[Kronk](https://github.com/ardanlabs/kronk) lets you use Go for hardware accelerated local inference with llama.cpp directly integrated into your applications via the [yzma](https://github.com/ardanlabs/yzma) module. Kronk provides a high-level API that feels similar to using an OpenAI compatible API.

Examples for Kronk can be found under Example13.

<br />

## Installing Software

To run the examples in this repo, start by installing `mongosh` and `kronk` using these make commands.

```
make install
make docker
make install-python
```

With the software installed, you will want to start your Kronk service. Open a terminal and run the following command. This will show logs so start a terminal window you can see but won't need.

```
make kronk-up
```

Now start the Mongo and Open Web containers in Docker Compose. Open a new terminal window for this.

```
make compose-up
```

Now you need to pull down the models you will be using. Open a new terminal window for this. This might take several minutes depending on your bandwidth.

```
make install-models
```

## Learn More

**Reach out about corporate training events, open enrollment live training sessions, and on-demand learning options.**

Ardan Labs (www.ardanlabs.com)  
hello@ardanlabs.com

## Purchase Video

The entire training class has been recorded to be made available to those who can't have the class taught at their company or who can't attend a conference. This is the entire class material.

[ardanlabs.com/education](https://www.ardanlabs.com/education/)

## Our Experience

We have taught Go to thousands of developers all around the world since 2014. There is no other company that has been doing it longer and our material has proven to help jump-start developers 6 to 12 months ahead of their knowledge of Go. We know what knowledge developers need in order to be productive and efficient when writing software in Go.

Our classes are perfect for intermediate-level developers who have at least a few months to years of experience writing code in Go. Our classes provide a very deep knowledge of the programming langauge with a big push on language mechanics, design philosophies and guidelines. We focus on teaching how to write code with a priority on consistency, integrity, readability and simplicity. We cover a lot about “if performance matters” with a focus on mechanical sympathy, data oriented design, decoupling and writing/debugging production software.

## Our Teacher

### William Kennedy ([@goinggodotnet](https://twitter.com/goinggodotnet))

_William Kennedy is a managing partner at Ardan Labs in Miami, Florida. Ardan Labs is a high-performance development and training firm working with startups and fortune 500 companies. He is also a co-author of the book Go in Action, the author of the blog GoingGo.Net, and a founding member of GoBridge which is working to increase Go adoption through diversity._

## More About Go

Go is an open source programming language that makes it easy to build simple, reliable, and efficient software. Although it borrows ideas from existing languages, it has a unique and simple nature that make Go programs different in character from programs written in other languages. It balances the capabilities of a low-level systems language with some high-level features you see in modern languages today. This creates a programming environment that allows you to be incredibly productive, performant and fully in control; in Go, you can write less code and do so much more.

Go is the fusion of performance and productivity wrapped in a language that software developers can learn, use and understand. Go is not C, yet we have many of the benefits of C with the benefits of higher level programming languages.

[The Ecosystem of the Go Programming Language](https://henvic.dev/posts/go/) - Henrique Vicente  
[The Why of Go](https://www.infoq.com/presentations/go-concurrency-gc) - Carmen Andoh  
[Go Ten Years and Climbing](https://commandcenter.blogspot.com/2017/09/go-ten-years-and-climbing.html) - Rob Pike  
[The eigenvector of "Why we moved from language X to language Y"](https://erikbern.com/2017/03/15/the-eigenvector-of-why-we-moved-from-language-x-to-language-y.html) - Erik Bernhardsson  
[Learn More](https://talks.golang.org/2012/splash.article) - Go Team  
[Simplicity is Complicated](https://www.youtube.com/watch?v=rFejpH_tAHM) - Rob Pike  
[Getting Started In Go](http://aarti.github.io/2016/08/13/getting-started-in-go) - Aarti Parikh

## Minimal Qualified Student

The material has been designed to be taught in a classroom environment. The code is well commented but missing some contextual concepts and ideas that will be covered in class. Students with the following minimal background will get the most out of the class.

- Studied CS in school or has a minimum of two years of experience programming full time professionally.
- Familiar with structural and object oriented programming styles.
- Has worked with arrays, lists, queues and stacks.
- Understands processes, threads and synchronization at a high level.
- Operating Systems
  - Has worked with a command shell.
  - Knows how to maneuver around the file system.
  - Understands what environment variables are.

## Joining the Go Slack Community

We use a Slack channel to share links, code, and examples during the training. This is free. This is also the same Slack community you will use after training to ask for help and interact with may Go experts around the world in the community.

1. Using the following link, fill out your name and email address: https://invite.slack.gobridge.org
1. Check your email, and follow the link to the slack application.
1. Join the training channel by clicking on this link: https://gophers.slack.com/messages/training/
1. Click the “Join Channel” button at the bottom of the screen.

---

All material is licensed under the [Apache License Version 2.0, January 2004](http://www.apache.org/licenses/LICENSE-2.0).
