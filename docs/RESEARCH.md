# Sources and implementation decisions

Reviewed 11 October 2026. Package versions were checked against PyPI and the installed APIs.

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
  - Keep the system small; observe actual tool feedback and define stopping conditions.
  - Applied here: one model, three tools, deterministic validation, call budgets, isolated working copies.
- [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents)
  - Use the current `create_agent` API for a model/tools feedback loop.
  - Applied here: `agent.build_agent`; structured output and middleware are configured in one readable file.
- [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)
  - Describe tools through Python signatures/docstrings and inject private runtime context.
  - Applied here: `@tool`, `ToolRuntime[RunContext]`, typed operation choices, no model-supplied file paths.
- [LangChain event streaming](https://docs.langchain.com/oss/python/langchain/event-streaming)
  - The current documentation recommends typed event projections for frontend applications.
  - Applied here: `stream_events(version="v3")`, `messages` and `tool_calls` projections, completed observation payloads.
  - Installed API marks v3 experimental; lock tested versions and exercise the stream adapter in tests.
- [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
  - Use a schema and an explicit output strategy.
  - Applied here: `ToolStrategy(CleaningSummary)`; host code verifies business correctness separately.
- [LangChain middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
  - Apply execution limits around the agent loop.
  - Applied here: model/tool call limits plus recoverable tool-error observations.
- [OpenAI GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
  - Verified tool calling and structured-output support for the configurable default model.
  - Live evaluations use the user's authorized API credential outside the repository.

# Autonomy boundary

- Python defines valid transformations and quality rules.
- The model chooses actions, arguments, ordering, and additional inspections.
- The runner records the execution and verifies the final outcome; it does not script the cleaning sequence.
- Evaluation checks output invariants and unnecessary edits rather than requiring one exact tool sequence.
