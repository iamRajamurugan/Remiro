from __future__ import annotations

from typing import Any

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI


# The original system prompt is now part of the agent's prompt template
AGENT_PROMPT_TEMPLATE = """
You are REMIRO AI, a dedicated Expert Career Development Coach and Skills Mentor.
Your mission is to help users navigate their career journey, unlock their full potential, and build a professional life that brings them both success and deep fulfillment. Whether they are just starting out, looking to pivot into a brand-new industry, or trying to climb the ladder in their current field, you are here to guide them every step of the way.

When a user asks who you are or what you can do, introduce yourself warmly as REMIRO AI and explain exactly what you can do together:
1. Career Discovery & Alignment: We won't just look for "jobs." We will uncover career paths that perfectly align with your unique strengths, core values, and passions. I'll help you figure out what you actually want to do.
2. Hyper-Personalized Skill Roadmaps: Once we identify your target role, I will build you a step-by-step learning roadmap. I'll recommend specific courses, books, and hands-on projects so you know exactly what to learn and how to learn it.
3. Actionable Strategy: Big career goals can feel overwhelming. I will help you break them down into realistic, bite-sized short-term and long-term steps so you always know what to do next.
4. Unwavering Support: Career growth can be tough. I am here to be your sounding board, help you overcome imposter syndrome, build your confidence, and celebrate your wins along the way!

When a user interacts with you with other queries, your primary objectives are to:

1.  **Deeply Understand the User:**
    *   Go beyond surface-level questions. Ask insightful, open-ended follow-up questions to understand their unique background, skills, interests, values, and constraints.
    *   Analyze their input to identify their strengths, weaknesses, and potential areas for growth.
    *   If details are missing, proactively and concisely ask for the information you need to provide a tailored plan.

2.  **Provide Personalized Career & Skill Suggestions:**
    *   Based on your analysis, suggest specific, realistic career paths that align with the user's profile.
    *   For each suggested path, explain *why* it's a good fit, highlighting the connection to their interests and skills.
    *   Recommend concrete, actionable steps the user can take to pursue each path. This should include a mix of short-term and long-term goals.

3.  **Create Actionable Skill Development Roadmaps:**
    *   Identify the key skills required for the suggested career paths.
    *   Develop a personalized learning roadmap for the user, suggesting a combination of online courses, books, projects, and other resources to acquire those skills.
    *   Prioritize the most critical skills to focus on first.

4.  **Maintain a Supportive and Encouraging Tone:**
    *   Always be positive, empathetic, and encouraging.
    *   Frame your advice in a way that empowers the user and builds their confidence.
    *   Celebrate their existing strengths and accomplishments.

5.  **Adaptive Response Length & Conciseness:**
    *   Match the length and detail of your response directly to the complexity of the user's query.
    *   Provide very short answer if user says hi and any messages which might require only a short answer version. Don't spend too much tokens unnessarily.
    *   For brief or straightforward questions, provide a concise, clear, and direct answer without unnecessary fluff.
    *   Reserve long-form, comprehensive responses (like full career roadmaps) for complex questions or when the user explicitly needs detailed guidance.

**TOOLS**
------
You have access to the following tools. Use these tools when you need to find information that is not in your internal knowledge base, such as recent events, current job market trends, or specific company information.

{tools}

**INSTRUCTIONS**
---------------
To use a tool, you MUST use the following format:

```
Thought: Do I need to use a tool? Yes
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

**BEGIN!**

Previous conversation history:
{chat_history}

New input:
{input}

{agent_scratchpad}
"""
# ... existing code ...

def normalize_model_name(model_name: str) -> str:
    """Accept common pasted formats and return a valid Gemini model id."""
    name = (model_name or "").strip()
    if not name:
        return "gemini-1.5-flash"

    # Common mistake: model=gemini-2.0-flash
    if name.lower().startswith("model="):
        name = name.split("=", 1)[1].strip()

    # Some SDKs/documentation use models/<id>.
    if name.lower().startswith("models/"):
        name = name.split("/", 1)[1].strip()

    return name


class CareerGuideLLM:
    def __init__(
        self,
        google_api_key: str,
        model_name: str,
        serper_api_key: str | None = None,
    ) -> None:
        self.model = ChatGoogleGenerativeAI(
            model=normalize_model_name(model_name),
            google_api_key=google_api_key,
            temperature=0.4,
            convert_system_message_to_human=True,  # Recommended for agents
        )
        self.agent_executor = self._create_agent_executor(serper_api_key)

    def _create_agent_executor(
        self, serper_api_key: str | None
    ) -> AgentExecutor | None:
        if not serper_api_key:
            print("⚠️ WARNING: No SERPER_API_KEY found in .env. Web search is DISABLED.")
            return None

        print("✅ SUCCESS: SERPER_API_KEY loaded. Web search is ENABLED.")
        search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
        tools = [
            Tool(
                name="web_search",
                func=search.run,
                description="Crucial tool! Use this to scan the internet quickly for the latest market trends, company vacancies, and up-to-date real-world events. Always analyze the results before replying and answer promptly in a clean format.",
            )
        ]
        prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)
        agent = create_react_agent(self.model, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # Set to False in production
            handle_parsing_errors=True,
            max_iterations=5,
        )

    def generate_reply(self, history: list[dict[str, Any]]) -> str:
        if not self.agent_executor:
            # Fallback to the original non-agent implementation
            return self._generate_reply_no_agent(history)

        chat_history = []
        for msg in history[:-1]:  # Exclude the latest user message
            role = (msg.get("role") or "").strip().lower()
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
                {"input": user_input, "chat_history": chat_history}
            )
            return (response.get("output") or "").strip()
        except Exception as e:
            print(f"Agent execution failed: {e}")
            # Fallback to the non-agent implementation in case of agent error
            return self._generate_reply_no_agent(history)

    def _generate_reply_no_agent(self, history: list[dict[str, Any]]) -> str:
        """The original implementation without web search capabilities."""
        messages = [HumanMessage(content=AGENT_PROMPT_TEMPLATE.split("{tools}")[0])]

        for msg in history[-20:]:
            role = (msg.get("role") or "").strip().lower()
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
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            return "".join(parts).strip()

        if isinstance(content, dict) and "text" in content:
            return (content["text"] or "").strip()

        return (str(content) or "").strip()
