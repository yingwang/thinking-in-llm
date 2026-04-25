# Thinking in LLM — 大纲

> 从 next-token prediction 的本质出发，理解 LLM 的思维机制，掌握构建 LLM 系统的第一性原理。

**目标读者**: 有编程基础、在用或想用 LLM 的工程师。不要求 ML 背景。

**调性**: 像 DDIA（Designing Data-Intensive Applications）——有原理深度，但始终服务于实践。

**语言**: 中英双语（先写中文，再出英文版）

---

## Part I: LLM 是什么（The Machine）

从底层建立正确的心智模型。读完这部分，你不会再把 LLM 当"聪明的搜索引擎"。

### 第1章：一切都是续写
- LLM 只做一件事：预测下一个 token
- Token ≠ 文字：tokenizer 如何塑造模型的"认知边界"
- Temperature、top-p、top-k：不是调参，是选择"思维模式"
- 从续写到对话：chat template 的本质是条件概率 P(response | system + history + user)
- **思维实验**: 如果你只能预测下一个字，你能"理解"语言吗？

### 第2章：Attention 是信息路由
- Self-attention 的直觉：每个 token 在问"我该看哪里？"
- QKV 不是三个矩阵，是查询-匹配-读取
- Multi-head：同时关注不同维度的关系（语法、语义、位置...）
- Induction heads：模型学会的第一个"算法"——复制粘贴
- KV Cache：为什么推理时不需要重算历史
- **可视化**: 用 BertViz/attention pattern 看模型在"看"什么

### 第3章：规模涌现
- Scaling Laws：loss 是参数量和数据量的幂律函数
- 涌现能力：为什么 10B 做不到的事，100B 突然能做到
- Chinchilla 定律：模型和数据的最优比例
- 过度训练（over-training）：为什么实际训练比 Chinchilla 最优用更多数据
- Grokking：为什么训练久了模型会突然"顿悟"
- **哲学问题**: 智能 = 压缩？更大的压缩器 = 更智能？

### 第4章：从预训练到对齐
- Base model 的能力与局限：什么都会，但不听话
- SFT：教格式，不教知识
- RLHF/DPO：教偏好，让模型"选择"更好的回答
- Constitutional AI：用原则替代人工标注
- Safety training：教拒绝的艺术
- **关键洞察**: 对齐不改变模型的能力，只改变能力的表达方式

---

## Part II: LLM 的能力边界（The Boundaries）

知道 LLM 能做什么很重要，知道它**不能**做什么更重要。

### 第5章：LLM 真正擅长什么
- 模式识别与类比：看过足够多的代码，就能"写"代码
- 翻译与转换：格式之间的映射是 LLM 的甜区
- 摘要与提取：压缩信息是训练目标的直接产物
- Few-shot learning：为什么几个例子就能学会新任务
- In-context learning 的本质：隐式的梯度下降？还是贝叶斯推断？
- **实验**: 相同任务，0-shot vs 1-shot vs 5-shot 的效果对比

### 第6章：LLM 的硬伤
- 数数数不对：tokenizer 打碎了字符边界
- 算术不可靠：不是计算，是模式匹配"看起来像答案"的 token
- 长程推理断裂：自回归生成没有全局规划
- 时间截止：知识冻结在训练数据
- 忠实性幻觉：模型永远会给出"最可能的续写"，即使是编造的
- **关键框架**: 可靠 vs 不可靠的任务清单

### 第7章：幻觉的本质
- 幻觉不是 bug，是 feature：续写器必须续写
- 知识幻觉 vs 推理幻觉 vs 指令幻觉
- Calibration：模型知道自己不知道吗？（部分知道）
- 检测幻觉：self-consistency、多次采样、logprob 分析
- 减少幻觉：RAG、引用、结构化输出、让模型说"我不知道"
- **实验**: 故意触发幻觉，观察模型的 confidence

### 第8章：推理还是模仿？
- Chain-of-Thought：给模型"草稿纸"
- CoT 的本质：更多 token = 更多计算步骤
- Reasoning models (o1/R1/Claude)：内化的 CoT
- Test-time compute scaling：用推理时间换准确率
- LLM 是在"真的推理"还是在"模仿推理的样子"？
- **开放问题**: System 1 vs System 2 thinking in LLM

---

## Part III: 用 LLM 构建（The Practice）

基于前两部分的理解，推导出正确的构建方式。

### 第9章：Prompt 是编程
- Prompt 不是自然语言指令，是在构造条件概率场景
- System prompt = 类定义，few-shot = 单元测试，CoT = 强制中间变量
- 结构化输出 = 类型系统：JSON mode、function calling、constrained decoding
- Prompt 的可组合性：模板、变量、条件分支
- 为什么小改动效果差很多：token 空间里的蝴蝶效应
- **实战**: 从一个烂 prompt 迭代到好 prompt 的完整过程

### 第10章：知识注入的三条路
- RAG = 开卷考试：运行时检索，实时更新，可审计
- Fine-tuning = 刻进大脑：改变行为/格式/风格
- Long context = 工作记忆：简单但贵
- 决策框架：什么时候用哪个，什么时候组合
- Embedding 的直觉：语义相似 = 向量距离近
- 向量检索的工程：选 index、选数据库、chunk 策略
- **决策树**: 给定场景，选择最佳知识注入方式

### 第11章：Agent 的第一性原理
- Tool use：不是"让 AI 用工具"，是扩展 token 空间到真实世界
- Planning 的根本困难：自回归模型没有前瞻能力
- ReAct：思考→行动→观察 循环
- Reflection：让模型审视自己的输出
- Multi-agent：分工的好处与通信的代价
- 什么时候该用 agent，什么时候一个 prompt 就够
- **反直觉**: 最好的 agent 设计往往是最简单的

### 第12章：评估——最被低估的环节
- Vibe check 不够，benchmark 也不够
- LLM-as-judge：用模型评模型的原理和陷阱
- 人类评估：Chatbot Arena 的 ELO 系统
- 单次调用评估 vs 系统级评估
- Regression testing：改了 prompt 怎么知道没搞坏别的
- 评估驱动开发：先写 eval，再调系统
- **实战**: 为一个 RAG 系统搭建评估 pipeline

---

## Part IV: 前沿与未来（The Frontier）

### 第13章：Interpretability——打开黑箱
- Superposition：一个神经元编码多个概念
- Sparse autoencoders：拆解模型的内部表示
- Circuits：找到模型内部的"算法"
- Feature steering：通过修改内部表示控制模型行为
- 为什么 interpretability 是安全的关键
- **实验**: 用 TransformerLens 探索模型内部

### 第14章：多模态——超越文本
- Vision-Language Models：图像变成 token 序列
- CLIP 的洞察：图文对齐是一切的基础
- 图像生成：从 Diffusion 到 DiT
- 音频：Whisper (听) → TTS (说)
- Video：最贵的模态，最大的机会
- Omni Models：一个模型理解一切

### 第15章：LLM 的未来
- Scaling 会撞墙吗？数据墙、能源墙、经济墙
- 合成数据：让模型自己生成训练数据
- 更长的上下文 → 更少的 RAG？
- 更强的推理 → 更少的 prompt engineering？
- Agent → 从工具到同事
- 开源 vs 闭源：谁会赢？
- **思考**: LLM 工程师这个角色会如何演变

---

## 附录

- A: 数学基础速查（softmax, cross-entropy, cosine similarity）
- B: 关键论文清单（每章 5 篇 must-read）
- C: 动手实验指南（每章配套的 notebook/代码）
- D: 术语表

---

## 与 LLM 训练指南的区别

| | LLM 训练工程师完全指南 | Thinking in LLM |
|---|---|---|
| **视角** | 怎么**造** LLM | 怎么**理解和用** LLM |
| **读者** | 训练工程师 | 所有 LLM 开发者 |
| **深度** | 工程实现细节 | 概念与心智模型 |
| **目标** | 能训练模型 | 能设计 LLM 系统 |
| **前置** | 需要 ML 基础 | 只需编程基础 |
