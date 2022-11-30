# Progressive Prompts

This repo includes an original implementation of Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis and Amjad Almahairi. ["Progressive Prompts: Continual Learning for Language Models without Forgetting"](https://openreview.net/pdf?id=UJTgQBc91_), 2022.

### Table of contents
* [Introduction](#star2-introduction)
* [What's in this repository](#question-whats-in-this-repository)
* [Installation](#wrench-installation)
* [How to run](#zap-how-to-run) 


## :star2: Introduction
We introduce **Progressive Prompts** – a novel Continual Learning (CL) approach for language models. Our
method is inspired by progressive networks, but is significantly more memory-efficient - it
only learns a fixed number of tokens, or prompt, for each new task. In Progressive Prompts, we learn a separate prompt for each incoming task and sequentially concatenate it with previously learned prompts. 

Our method can: 

1) **alleviate catastrophic forgetting** by preserving the knowledge acquired in previous prompts, and 
2) **transfer knowledge to future tasks** by sequentially learning new prompts given
previous ones. 


![Progressive Prompts schematics](/images/illustration.png)
Figure: *Illustrating our proposed method **Progressive Prompts** and contrasting it with a simple
adaptation of progressive networks using prompt tuning. In the simple adaptation of progressive
networks we learn a separate prompt and repeat the frozen input embeddings for each new task.
This setup requires repeating input tokens for each task. In Progressive Prompts we use the same
input and progressively append new prompt for each new task. Prior task prompts are not modified
by the addition of new prompts.*

## :question: What's in this repository

## :wrench: Installation

The configuration keys are as follows:
| Argument |   Default     |  Description |
|----------|:-------------:|------:   |
| col 1 is |  left-aligned | $1600    |
| col 2 is |    centered   |   $12.   |
| col 3 is | right-aligned |    $1    |
## :zap: How to run 


**Note**: if you have any questions about the paper or code, please contact Anastasia Razdaibiedina (anastasia.razdaibiedina[at]mail.utoronto.ca) or open an issue.
