# Thinking in LLM - Outline

> A first-principles guide to how LLMs "think," starting from next-token prediction and ending with practical system design.

**Target reader**: Engineers who can program and are using, or want to use, LLMs. No ML background required.

**Tone**: Like DDIA (Designing Data-Intensive Applications): principled and deep, but always in service of practice.

**Language**: Chinese and English bilingual (Chinese first, then English version).

---

## Part I: What an LLM Is (The Machine)

Build the correct mental model from the bottom up. After reading this part, you will no longer treat LLMs as "smart search engines."

### Chapter 1: Everything Is Continuation
- An LLM does only one thing: predict the next token.
- Token != text: how the tokenizer shapes the model's "cognitive boundary."
- Temperature, top-p, top-k: not parameter tuning, but choosing a "thinking mode."
- From continuation to conversation: the essence of a chat template is the conditional probability P(response | system + history + user).
- **Thought experiment**: If you can only predict the next character, can you "understand" language?

### Chapter 2: Attention Is Information Routing
- The intuition of self-attention: every token asks, "Where should I look?"
- QKV is not three matrices; it is query-match-read.
- Multi-head: attending to relationships across different dimensions at the same time (syntax, semantics, position...).
- Induction heads: the first "algorithm" the model learns - copy and paste.
- KV Cache: why inference does not need to recompute history.
- **Visualization**: use BertViz/attention patterns to see what the model is "looking" at.

### Chapter 3: Emergence From Scale
- Scaling Laws: loss is a power-law function of parameter count and data size.
- Emergent abilities: why a task a 10B model cannot handle may suddenly work with a 100B model.
- Chinchilla law: the optimal ratio between model and data.
- Over-training: why real training uses more data than the Chinchilla optimum.
- Grokking: why a model can suddenly generalize after training for a long time.
- **Philosophical question**: Intelligence = compression? A larger compressor = more intelligent?

### Chapter 4: From Pretraining to Alignment
- Base model capabilities and limitations: it can do many things, but it does not follow instructions reliably.
- SFT: teaches format, not knowledge.
- RLHF/DPO: teaches preferences and lets the model "choose" better answers.
- Constitutional AI: replace human annotation with principles.
- Safety training: teach the art of refusal.
- **Key insight**: Alignment does not change the model's capabilities; it only changes how those capabilities are expressed.

---

## Part II: The Capability Boundaries of LLMs (The Boundaries)

Knowing what LLMs can do is important. Knowing what they **cannot** do is even more important.

### Chapter 5: What LLMs Are Truly Good At
- Pattern recognition and analogy: after seeing enough code, a model can "write" code.
- Translation and transformation: mapping between formats is the sweet spot for LLMs.
- Summarization and extraction: compressing information is a direct product of the training objective.
- Few-shot learning: why a few examples are enough to learn a new task.
- The nature of in-context learning: implicit gradient descent, or Bayesian inference?
- **Experiment**: Compare 0-shot vs 1-shot vs 5-shot on the same task.

### Chapter 6: The Hard Limitations of LLMs
- Counting goes wrong: the tokenizer breaks character boundaries.
- Arithmetic is unreliable: it is not computation, but token-level pattern matching that "looks like" an answer.
- Long-range reasoning breaks: autoregressive generation has no global planning.
- Time cutoff: knowledge is frozen in the training data.
- Faithfulness hallucination: the model always produces the "most likely continuation," even when that continuation is fabricated.
- **Key framework**: A checklist of reliable vs unreliable tasks.

### Chapter 7: The Nature of Hallucination
- Hallucination is not a bug; it follows from the mechanism: the continuation engine must continue.
- Knowledge hallucination vs reasoning hallucination vs instruction hallucination.
- Calibration: does the model know what it does not know? (Partly.)
- Detecting hallucination: self-consistency, multiple sampling, logprob analysis.
- Reducing hallucination: RAG, citations, structured output, and making the model say "I don't know."
- **Experiment**: Deliberately trigger hallucination and observe the model's confidence.

### Chapter 8: Reasoning or Imitation?
- Chain-of-Thought: give the model "scratch paper."
- The essence of CoT: more tokens = more computation steps.
- Reasoning models (o1/R1/Claude): internalized CoT.
- Test-time compute scaling: trade inference time for accuracy.
- Is the LLM "really reasoning," or is it "imitating the appearance of reasoning"?
- **Open question**: System 1 vs System 2 thinking in LLMs.

---

## Part III: Building With LLMs (The Practice)

The first two parts give us the mental model; this part derives the right way to build from it.

### Chapter 9: Prompt Is Programming
- A prompt is not just a natural-language instruction; it constructs a conditional probability scenario.
- System prompt = class definition, few-shot = unit tests, CoT = forced intermediate variables.
- Structured output = type system: JSON mode, function calling, constrained decoding.
- Prompt composability: templates, variables, conditional branches.
- Why small changes can have very different effects: the butterfly effect in token space.
- **Hands-on**: The full process of iterating from a bad prompt to a good one.

### Chapter 10: Three Paths for Knowledge Injection
- RAG = open-book exam: runtime retrieval, real-time updates, auditable.
- Fine-tuning = etched into the model: changes behavior, format, and style.
- Long context = working memory: simple but expensive.
- Decision framework: when to use each method, and when to combine them.
- The intuition of embeddings: semantic similarity = close vector distance.
- Vector retrieval engineering: choosing an index, choosing a database, chunk strategy.
- **Decision tree**: Given a scenario, choose the best knowledge-injection method.

### Chapter 11: First Principles of Agents
- Tool use: not "letting AI use tools," but extending token space into the real world.
- The fundamental difficulty of planning: autoregressive models do not have lookahead ability.
- ReAct: thought -> action -> observation loop.
- Reflection: letting the model inspect its own output.
- Multi-agent: the benefit of division of labor and the cost of communication.
- When to use an agent, and when a single prompt is enough.
- **Counterintuitive**: The best agent designs are often the simplest.

### Chapter 12: Evaluation: The Most Underestimated Step
- Vibe checks are not enough, and benchmarks are not enough either.
- LLM-as-judge: the principles and traps of using models to evaluate models.
- Human evaluation: Chatbot Arena's ELO system.
- Single-call evaluation vs system-level evaluation.
- Regression testing: after changing a prompt, how do you know you did not break something else?
- Eval-driven development: write evals first, then tune the system.
- **Hands-on**: Build an evaluation pipeline for a RAG system.

---

## Part IV: Frontier and Future (The Frontier)

### Chapter 13: Interpretability: Opening the Black Box
- Superposition: one neuron encodes multiple concepts.
- Sparse autoencoders: decompose the model's internal representations.
- Circuits: find the "algorithms" inside the model.
- Feature steering: control model behavior by modifying internal representations.
- Why interpretability is key to safety.
- **Experiment**: Explore the model's internals with TransformerLens.

### Chapter 14: Multimodal: Beyond Text
- Vision-Language Models: images become token sequences.
- CLIP's insight: image-text alignment is the foundation of everything.
- Image generation: from Diffusion to DiT.
- Audio: Whisper (listen) -> TTS (speak).
- Video: the most expensive modality and the biggest opportunity.
- Omni Models: one model understands everything.

### Chapter 15: The Future of LLMs
- Will scaling hit a wall? Data wall, energy wall, economic wall.
- Synthetic data: let models generate their own training data.
- Longer context -> less RAG?
- Stronger reasoning -> less prompt engineering?
- Agents -> from tools to colleagues.
- Open source vs closed source: who will win?
- **Reflection**: How will the role of the LLM engineer evolve?

---

## Appendices

- A: Mathematical basics quick reference (softmax, cross-entropy, cosine similarity).
- B: Key paper list (5 must-read papers per chapter).
- C: Hands-on experiment guide (notebooks/code for each chapter).
- D: Glossary.

---

## Difference From the LLM Training Guide

| | The Complete Guide for LLM Training Engineers | Thinking in LLM |
|---|---|---|
| **Perspective** | How to **build** LLMs | How to **understand and use** LLMs |
| **Reader** | Training engineers | All LLM developers |
| **Depth** | Engineering implementation details | Concepts and mental models |
| **Goal** | Train models | Design LLM systems |
| **Prerequisite** | Requires ML background | Only requires programming background |
