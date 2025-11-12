# Conversation Summary: Debatable-Intelligence vs MADE Frameworks (Nov 12, 2025)

## Key Points

- **Debatable-Intelligence Framework** is an *evaluation/benchmarking* system designed to judge debate speeches—especially for benchmarking LLM (Large Language Model) judges. It is not a debate-running agent like MADE, but assesses argument strength, relevance, coherence, style, and more.

- **MADE Frameworks** focus on *multi-agent debate*, where models/agents argue and respond across multiple turns to collaboratively or adversarially reach an answer or verdict.

---

## Paper Context

- The referenced paper ([arXiv:2506.05062](https://arxiv.org/abs/2506.05062)) introduces Debate Speech Evaluation as a benchmark for LLM judges. The core: evaluating debate speeches on multiple dimensions using annotated data, comparing how LLMs and humans rate debate performance. The framework judges, it doesn't simulate debates itself.

- The *debating* element in the paper is in the *input data* (debate speeches), not in orchestrating debate sessions. LLMs are benchmarked on their judgment, not generation in a debate setting.

---

## Customization/Extension Potential

- It is feasible to extend Debatable-Intelligence to simulate multi-agent debating (MADE-style):
  - Add orchestration scripts for turn-based, multi-agent rounds.
  - Use prompt engineering for adversarial/affirmative/negative stances.
  - Judge each agent's arguments turn-by-turn, then aggregate results.

- Requires scripting/model orchestration and prompt templates to create "live" debate interaction.

---

## Features Comparison Table

| Feature                  | Debatable-Intelligence | MADE-style (desired)       |
|--------------------------|------------------------|---------------------------|
| Speech Generation        | Single speech          | Multi-agent debate        |
| Judging/Scoring          | Per speech             | Per agent, per turn        |
| Turn-based Debate Sim    | No                     | Yes (extension needed)     |
| Winner/Metrics           | Single speech/score    | Aggregate of debate round  |

---

This summary covers the conceptual difference, referenced paper, and steps for customization to enable debating agents using the evaluation logic of Debatable-Intelligence.

*(Generated Nov 12, 2025 as per user request)*
