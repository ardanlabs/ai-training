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

**THIS IS A WORK IN PROGRESS**

These examples provide a foundation for understanding the semantics behind vector embeddings and the basics behind writing LLM based applications to allow people to interact with your own content. Each example builds on the next, providing you a strong understanding of how LLM based AI applications work.

You will find in the source code file for each example notes to help you understand the example and how to run it.

The makefile has everything you need to get up and running quickly.

### Installing Software

To run the examples in this repo, start by installing `mongosh` and `ollama` using Brew. If you don't have Brew installed, I highly recommend it.

```
make install
```

Next you want to pull down these images:

- `mongodb/mongodb-atlas-local:8.0`
- `ghcr.io/open-webui/open-webui:v0.6.18`
- `postgres:18.0`
- `quay.io/docling-project/docling-serve`

Run the following command to do so:

```
make docker
```

With the software installed, you will want to start your Ollama service. Open a terminal and run the following command. This will show logs so start a terminal window you can see but won't need.

```
make ollama-up
```

Now start the Mongo and Open Web containers in Docker Compose. Open a new terminal window for this.

```
make compose-up
```

Now you need to pull down the models you will be using. Open a new terminal window for this. This might take several minutes depending on your bandwidth.

```
make ollama-pull
```

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
