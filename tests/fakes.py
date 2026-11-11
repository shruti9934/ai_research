"""A scripted chat model so the whole graph and API can be tested without API keys."""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool


class ScriptedChatModel(BaseChatModel):
    """Responds according to which tool/schema is bound:

    * ``Plan``  -> planner decision (from ``plan``)
    * ``Grade`` -> next item of ``grades``
    * otherwise -> writer: ``writer(messages)`` returns an AIMessage (text or a calculator tool call)
    """

    plan: dict = {"route": "documents", "doc_query": "", "web_query": ""}
    grades: list = []
    writer: Callable[[list[BaseMessage]], AIMessage] = lambda msgs: AIMessage("Answer [1].")
    bound: list[str] = []
    calls: dict = {}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs: Any):
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"bound": names})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        kind = "Plan" if "Plan" in self.bound else "Grade" if "Grade" in self.bound else "writer"
        self.calls.setdefault(kind, []).append(messages)
        if kind == "Plan":
            msg = AIMessage("", tool_calls=[{"name": "Plan", "args": self.plan, "id": "p1"}])
        elif kind == "Grade":
            grade = self.grades.pop(0) if self.grades else {"grounded": True, "reason": "ok"}
            msg = AIMessage("", tool_calls=[{"name": "Grade", "args": grade, "id": "g1"}])
        else:
            msg = self.writer(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])
