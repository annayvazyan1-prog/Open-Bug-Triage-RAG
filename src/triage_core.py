# -*- coding: utf-8 -*-
"""
Open-Bug Triage RAG — core logic
================================

Shared, side-effect-free module used by both the CLI (open_bug_triage_rag.py)
and the Streamlit app (app.py).

It provides:
  - OpenTicketTriageTools : loads tickets, builds the vector index, and exposes
    both string tools (for the LLM agent) and structured helpers (for the UI)
  - build_llm_with_tools  : wires the tools to an OpenAI chat model
  - run_agent             : the tool-calling loop

Nothing here runs at import time except load_dotenv(); building the index (which
needs an API key) only happens when you create OpenTicketTriageTools().
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# Data file, resolved relative to this file so the working directory doesn't matter.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "house_search_tickets_open.json"

# Default similarity cutoff for the duplicate detector (0..1, higher = stricter).
DUPLICATE_THRESHOLD = 0.78


class OpenTicketTriageTools:
    """Loads open tickets, indexes them, and exposes search/triage operations."""

    def __init__(self, tickets_path=DATA_PATH):
        with open(tickets_path, "r", encoding="utf-8") as f:
            self.tickets = json.load(f)

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._setup_vectorstore()

    def _setup_vectorstore(self):
        """Embed each ticket once and store it for similarity/duplicate search."""
        documents = []
        for t in self.tickets:
            content = f"{t['title']}. {t['description']}"
            documents.append(Document(
                page_content=content,
                metadata={
                    "ticket_id": t["ticket_id"],
                    "title": t["title"],
                    "category": t["category"],
                    "priority": t["priority"],
                },
            ))

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="open_bug_triage",
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Small lookups used by both the UI and the string tools
    # ------------------------------------------------------------------
    def categories(self):
        return sorted({t["category"] for t in self.tickets})

    def priorities(self):
        order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        return sorted({t["priority"] for t in self.tickets}, key=lambda p: order.get(p, 99))

    def get(self, ticket_id):
        ticket_id = ticket_id.upper().strip()
        return next((t for t in self.tickets if t["ticket_id"] == ticket_id), None)

    def stats(self):
        categories, priorities = {}, {}
        for t in self.tickets:
            categories[t["category"]] = categories.get(t["category"], 0) + 1
            priorities[t["priority"]] = priorities.get(t["priority"], 0) + 1
        return {"total": len(self.tickets), "by_category": categories, "by_priority": priorities}

    # ------------------------------------------------------------------
    # Structured search helpers (return data, used by the Streamlit UI)
    # ------------------------------------------------------------------
    def similar_rows(self, query, k=5, priority=None):
        """Return similar tickets as a list of dict rows (with similarity score)."""
        if not query or not query.strip():
            return []
        if priority:
            raw = self.vectorstore.similarity_search_with_score(
                query, k=k, filter={"priority": priority}
            )
        else:
            raw = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            {
                "Ticket": doc.metadata["ticket_id"],
                "Title": doc.metadata["title"],
                "Category": doc.metadata["category"],
                "Priority": doc.metadata["priority"],
                "Similarity": round(1.0 - float(dist), 3),
            }
            for doc, dist in raw
        ]

    def duplicate_rows(self, query, threshold=DUPLICATE_THRESHOLD):
        """Return likely duplicates as dict rows. Accepts a description or ticket ID."""
        result = {"error": None, "exclude_id": None, "rows": []}
        if not query or not query.strip():
            result["error"] = "Provide a bug description or a ticket ID (e.g. TICK-003)."
            return result

        query = query.strip()
        search_text = query
        if query.upper().startswith("TICK-"):
            result["exclude_id"] = query.upper()
            source = self.get(query)
            if source is None:
                result["error"] = f"Ticket {query.upper()} not found."
                return result
            search_text = f"{source['title']}. {source['description']}"

        raw = self.vectorstore.similarity_search_with_score(search_text, k=6)
        for doc, dist in raw:
            score = 1.0 - float(dist)
            if doc.metadata.get("ticket_id") == result["exclude_id"]:
                continue
            if score >= threshold:
                result["rows"].append({
                    "Ticket": doc.metadata["ticket_id"],
                    "Title": doc.metadata["title"],
                    "Priority": doc.metadata["priority"],
                    "Similarity": round(score, 3),
                })
        return result

    def by_priority(self, priority):
        return [t for t in self.tickets if t["priority"].lower() == priority.lower()]

    def by_category(self, category):
        return [t for t in self.tickets if t["category"].lower() == category.lower()]

    # ------------------------------------------------------------------
    # String tools (consumed by the LLM agent)
    # ------------------------------------------------------------------
    def search_similar_tickets(self, query: str) -> str:
        if not query or not query.strip():
            return "Error: Please describe the bug or paste a ticket title to search."
        rows = self.similar_rows(query, k=5)
        if not rows:
            return "No similar open tickets found."
        out = f"Found {len(rows)} similar OPEN tickets:\n\n"
        for r in rows:
            out += f"• [{r['Ticket']}] {r['Title']} (Priority: {r['Priority']}, similarity: {r['Similarity']})\n"
        return out

    def find_duplicate_tickets(self, query: str) -> str:
        res = self.duplicate_rows(query)
        if res["error"]:
            return res["error"]
        target = res["exclude_id"] or "your description"
        if not res["rows"]:
            return f"No likely duplicates of {target} found (threshold {DUPLICATE_THRESHOLD:.2f})."
        out = f"Likely DUPLICATES of {target} (similarity >= {DUPLICATE_THRESHOLD:.2f}):\n\n"
        for r in res["rows"]:
            out += f"• [{r['Ticket']}] {r['Title']} (Priority: {r['Priority']}, similarity: {r['Similarity']})\n"
        return out

    def search_by_priority(self, priority: str) -> str:
        if not priority or not priority.strip():
            return f"Error: Please provide a priority. Available: {', '.join(self.priorities())}"
        rows = self.by_priority(priority.strip())
        if not rows:
            return f"No open tickets with priority '{priority}'. Available: {', '.join(self.priorities())}"
        out = f"Found {len(rows)} OPEN '{priority}' priority tickets:\n\n"
        for t in rows:
            out += f"• [{t['ticket_id']}] {t['title']} ({t['category']})\n"
        return out

    def search_by_category(self, category: str) -> str:
        if not category or not category.strip():
            return f"Error: Please provide a category. Available: {', '.join(self.categories())}"
        rows = self.by_category(category.strip())
        if not rows:
            return f"No open tickets in category '{category}'. Available: {', '.join(self.categories())}"
        out = f"Found {len(rows)} OPEN tickets in '{category}':\n\n"
        for t in rows:
            out += f"• [{t['ticket_id']}] {t['title']} (Priority: {t['priority']})\n"
        return out

    def get_ticket_by_id(self, ticket_id: str) -> str:
        if not ticket_id or not ticket_id.strip():
            return "Error: Please provide a ticket ID (e.g., TICK-001)"
        ticket_id = ticket_id.upper().strip()
        if not ticket_id.startswith("TICK-"):
            return f"Error: Invalid format. Ticket IDs look like TICK-001. You provided: '{ticket_id}'"
        t = self.get(ticket_id)
        if t is None:
            return f"Ticket '{ticket_id}' not found among open tickets."
        return (
            f"Ticket ID: {t['ticket_id']}\nTitle: {t['title']}\n"
            f"Description: {t['description']}\nCategory: {t['category']}\n"
            f"Priority: {t['priority']}\nCreated: {t['created_date']}\nStatus: OPEN"
        )

    def get_ticket_statistics(self, input: str = "") -> str:
        s = self.stats()
        out = f"Open Ticket Backlog Statistics:\nTotal Open Tickets: {s['total']}\n\nBy Category:\n"
        for cat, c in sorted(s["by_category"].items(), key=lambda x: x[1], reverse=True):
            out += f"  • {cat}: {c}\n"
        out += "\nBy Priority:\n"
        for pri, c in sorted(s["by_priority"].items(), key=lambda x: x[1], reverse=True):
            out += f"  • {pri}: {c}\n"
        return out

    # ------------------------------------------------------------------
    def get_tools(self):
        return [
            Tool(name="SearchSimilarTickets", func=self.search_similar_tickets,
                 description="""Find OPEN tickets similar to a described problem.
Use when an employee describes a bug and wants related open issues.
To filter by priority, use SearchByPriority instead.
Input: a bug description or ticket title."""),
            Tool(name="FindDuplicateTickets", func=self.find_duplicate_tickets,
                 description="""Detect likely DUPLICATE open tickets.
Use for "is this a duplicate?" or "duplicates of TICK-003?".
Input: a bug description OR an existing ticket ID (e.g. 'TICK-003')."""),
            Tool(name="SearchByPriority", func=self.search_by_priority,
                 description="""List ALL open tickets at a given priority level.
Input: one of Critical, High, Medium, Low."""),
            Tool(name="SearchByCategory", func=self.search_by_category,
                 description="""List ALL open tickets in a specific category.
Input: a category name (e.g. 'Search', 'Listings')."""),
            Tool(name="GetTicketByID", func=self.get_ticket_by_id,
                 description="""Get full details of one open ticket by exact ID.
Input: a ticket ID. Do NOT use this for searching."""),
            Tool(name="GetTicketStatistics", func=self.get_ticket_statistics,
                 description="""Overview of the open backlog: counts by category and
priority. No meaningful input needed."""),
        ]


TRIAGE_SYSTEM_PROMPT = """You are an open-bug triage assistant for a house-search /
home-buying platform. You help engineers and support staff SEARCH and GROUP the
open ticket backlog: surfacing similar tickets, flagging duplicates, and listing
high-priority bugs.

Follow these rules:
1. State which tool you're about to use before calling it.
2. If the request is ambiguous, ask one clarifying question first.
3. When showing tickets, always reference their ticket IDs and priority.
4. For "is this a duplicate?" questions, use FindDuplicateTickets and clearly say
   whether likely duplicates were found.
5. After answering, suggest one useful next triage step.
"""


def build_llm_with_tools(tools, model=None, temperature=0):
    """Create a chat model and bind the tool schemas to it."""
    model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=temperature)
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "The input to the tool"}
                    },
                    "required": ["input"],
                },
            },
        }
        for tool in tools
    ]
    return llm.bind(tools=tool_definitions)


def run_agent(query, tools, llm_with_tools, system_prompt=TRIAGE_SYSTEM_PROMPT,
              max_iterations=5):
    """Run the tool-calling loop. Returns {response, tools_used, iterations}."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
    tools_used = []
    tools_by_name = {t.name: t for t in tools}

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return {"response": response.content, "tools_used": tools_used,
                    "iterations": i + 1}

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            tool_input = tool_call["args"].get("input", "")
            tools_used.append(name)
            tool = tools_by_name.get(name)
            output = tool.func(tool_input) if tool else f"Error: Tool {name} not found"
            messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))

    return {"response": "Maximum iterations reached.", "tools_used": tools_used,
            "iterations": max_iterations}
