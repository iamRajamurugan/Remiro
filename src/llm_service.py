from __future__ import annotations

from typing import Any

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI


AGENT_PROMPT_TEMPLATE = """
You are REMIRO AI, a dedicated Expert Career Development Coach and Skills Mentor.

Your mission is to help users navigate their career journey, unlock their full potential, and build a professional life that brings them both success and deep fulfillment.

When a user asks who you are or what you can do, introduce yourself warmly as REMIRO AI and explain exactly what you can do.

1. Career Discovery & Alignment  
2. Hyper-Personalized Skill Roadmaps  
3. Actionable Strategy  
4. Unwavering Support  

When a user interacts with you:

1. Deeply understand the user  
2. Provide personalized career suggestions  
3. Create actionable skill roadmaps  
4. Engage naturally and empathetically  
5. Tailor response length  
6. Maintain context  

**TOOLS**
------
{tools}

**INSTRUCTIONS**
---------------
Use this format when using tools:

Thought: Do I need to use a tool? Yes  
Action: one of [{tool_names}]  
Action Input: input  
Observation: result  

If no tool:

Thought: Do I need to use a tool? No  
Final Answer: your response  

**BEGIN!**

Previous conversation history:
{chat_history}

New input:
{input}

{agent_scratchpad}
"""


def normalize_model_name(model_name: str) -> str:
    """Normalize Gemini model name."""
    name = (model_name or "").strip()

    if not name:
        return "gemini-1.5-flash"

    if name.lower().startswith("model="):
        name = name.split("=", 1)[1].strip()

    if name.lower().startswith("models/"):
        name = name.split("/", 1)[1].strip()

    return name


class CareerGuideLLM:
    def __init__(
        self,
        google_api_key: str,
        model_name: str = "gemini-1.5-flash",
        serper_api_key: str | None = None,
    ) -> None:

        self.model = ChatGoogleGenerativeAI(
            model=normalize_model_name(model_name),
            google_api_key=google_api_key,
            temperature=0.4,
            convert_system_message_to_human=True,  # ✅ IMPORTANT FIX
        )

        self.agent_executor = self._create_agent_executor(serper_api_key)

    def _create_agent_executor(
        self, serper_api_key: str | None
    ) -> AgentExecutor | None:

        if not serper_api_key:
            print("⚠️ WARNING: No SERPER_API_KEY found. Web search DISABLED.")
            return None

        print("✅ SERPER enabled")

        search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)

        tools = [
            Tool(
                name="web_search",
                func=search.run,
                description="Use this to get latest data from internet.",
            )
        ]

        prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)

        agent = create_react_agent(self.model, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

    def generate_reply(self, history: list[dict[str, Any]]) -> str:

        if not self.agent_executor:
            return self._generate_reply_no_agent(history)

        chat_history = []

        for msg in history[:-1]:
            role = (msg.get("role") or "").lower()
            content = (msg.get("content") or "").strip()

            if not content:
                continue

            if role == "assistant":
                chat_history.append(AIMessage(content=content))
            elif role == "user":
                chat_history.append(HumanMessage(content=content))

        user_input = (history[-1].get("content") or "").strip()

        try:
            response = self.agent_executor.invoke(
                {
                    "input": user_input,
                    "chat_history": chat_history,
                }
            )

            return (response.get("output") or "").strip()

        except Exception as e:
            print(f"Agent failed: {e}")
            return self._generate_reply_no_agent(history)

    def _generate_reply_no_agent(self, history: list[dict[str, Any]]) -> str:

        messages = [
            HumanMessage(content=AGENT_PROMPT_TEMPLATE.split("{tools}")[0])
        ]

        for msg in history[-20:]:
            role = (msg.get("role") or "").lower()
            content = (msg.get("content") or "").strip()

            if not content:
                continue

            if role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))

        response = self.model.invoke(messages)
        content = response.content

        if isinstance(content, list):
            return "".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            ).strip()

        if isinstance(content, dict):
            return (content.get("text") or "").strip()

        return str(content).strip()
