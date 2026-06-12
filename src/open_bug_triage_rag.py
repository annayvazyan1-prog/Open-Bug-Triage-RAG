# -*- coding: utf-8 -*-
"""
Open-Bug Triage RAG
===================

A retrieval system for support/engineering employees to search the **open**
ticket backlog:

  - Find SIMILAR open tickets to a problem description
  - Detect likely DUPLICATE open tickets
  - List all HIGH / CRITICAL priority open bugs
  - Browse open bugs by category
  - Look up a specific ticket and get backlog statistics

Dataset notes:
  - Themed around a house-search / home-buying platform
  - The tool helps employees locate and group related open work.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# Default data path resolved relative to THIS file (src/), so the script runs
# correctly no matter what the current working directory is:
#   <repo>/src/open_bug_triage_rag.py  ->  <repo>/data/house_search_tickets_open.json
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "house_search_tickets_open.json"

# Distance/relevance threshold above which two tickets are flagged as likely
# duplicates. Relevance scores from Chroma are normalized to [0, 1] (higher =
# more similar). Tune this for your data — 0.78 is a sensible starting point.
DUPLICATE_THRESHOLD = 0.78

print("Setting up open-bug triage system...")


class OpenTicketTriageTools:
    """Tools for searching and grouping open tickets."""

    def __init__(self, tickets_path=DATA_PATH):
        # Load the canonical ticket dataset once at startup for all tools.
        with open(tickets_path, 'r', encoding='utf-8') as f:
            self.tickets = json.load(f)

        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        self._setup_vectorstore()

    def _setup_vectorstore(self):
        """Index every ticket for semantic similarity / duplicate search."""
        documents = []
        for ticket in self.tickets:
            # Content is title + description + category + priority.
            # Triage matches on the *symptom*, which is exactly what we want
            # for similarity and duplicate detection.
            content = f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Category: {ticket['category']}
Priority: {ticket['priority']}"""

            documents.append(Document(
                page_content=content,
                metadata={
                    'ticket_id': ticket['ticket_id'],
                    'title': ticket['title'],
                    'category': ticket['category'],
                    'priority': ticket['priority'],
                }
            ))

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="open_bug_triage"
        )

    # ------------------------------------------------------------------
    # Similar open tickets
    # ------------------------------------------------------------------
    def search_similar_tickets(self, query: str) -> str:
        """Find open tickets semantically similar to a problem description."""
        if not query or not query.strip():
            return "Error: Please describe the bug or paste a ticket title to search."

        # If the employee is filtering to a priority, honor it in the search.
        priority_filter = None
        for level in ("Critical", "High", "Medium", "Low"):
            if level.lower() in query.lower():
                priority_filter = level
                break

        if priority_filter:
            results = self.vectorstore.similarity_search(
                query, k=5, filter={"priority": priority_filter}
            )
        else:
            results = self.vectorstore.similarity_search(query, k=5)

        if not results:
            return "No similar open tickets found."

        output = f"Found {len(results)} similar OPEN tickets:\n\n"
        for i, doc in enumerate(results, 1):
            output += f"--- Match {i} ---\n{doc.page_content}\n\n"
        return output

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------
    def find_duplicate_tickets(self, query: str) -> str:
        """Flag open tickets that are likely DUPLICATES of the given input.

        Input may be either a free-text bug description OR an existing ticket ID
        (e.g. 'TICK-003'). When an ID is given, we search using that ticket's own
        title + description and exclude it from its own results.
        """
        if not query or not query.strip():
            return "Error: Provide a bug description or a ticket ID (e.g. TICK-003)."

        query = query.strip()
        exclude_id = None

        # If the employee passed a ticket ID, search by that ticket's content.
        if query.upper().startswith("TICK-"):
            exclude_id = query.upper()
            source = next(
                (t for t in self.tickets if t['ticket_id'] == exclude_id), None
            )
            if source is None:
                return f"Ticket '{exclude_id}' not found among open tickets."
            search_text = f"{source['title']}. {source['description']}"
        else:
            search_text = query

        # Relevance scores are in [0, 1]; higher = more similar.
        scored = self.vectorstore.similarity_search_with_relevance_scores(
            search_text, k=6
        )

        duplicates = []
        for doc, score in scored:
            tid = doc.metadata.get('ticket_id')
            if tid == exclude_id:
                continue  # never report a ticket as its own duplicate
            if score >= DUPLICATE_THRESHOLD:
                duplicates.append((doc, score))

        if not duplicates:
            header = f"No likely duplicates found (threshold {DUPLICATE_THRESHOLD:.2f})."
            if exclude_id:
                header = f"No likely duplicates of {exclude_id} found " \
                         f"(threshold {DUPLICATE_THRESHOLD:.2f})."
            return header

        target = exclude_id or "your description"
        output = f"Likely DUPLICATES of {target} " \
                 f"(similarity ≥ {DUPLICATE_THRESHOLD:.2f}):\n\n"
        for doc, score in duplicates:
            output += (
                f"• [{doc.metadata['ticket_id']}] {doc.metadata['title']} "
                f"(Priority: {doc.metadata['priority']}, "
                f"similarity: {score:.2f})\n"
            )
        return output

    # ------------------------------------------------------------------
    # Priority search (first-class triage tool)
    # ------------------------------------------------------------------
    def search_by_priority(self, priority: str) -> str:
        """List all open tickets at a given priority (e.g. High, Critical)."""
        if not priority or not priority.strip():
            available = sorted(set(t['priority'] for t in self.tickets))
            return f"Error: Please provide a priority. Available: {', '.join(available)}"

        priority = priority.strip()
        matching = [t for t in self.tickets if t['priority'].lower() == priority.lower()]

        if not matching:
            available = sorted(set(t['priority'] for t in self.tickets))
            return f"No open tickets with priority '{priority}'. Available: {', '.join(available)}"

        output = f"Found {len(matching)} OPEN '{priority}' priority tickets:\n\n"
        for t in matching:
            output += f"• [{t['ticket_id']}] {t['title']} ({t['category']})\n"
        return output

    # ------------------------------------------------------------------
    # Category browse
    # ------------------------------------------------------------------
    def search_by_category(self, category: str) -> str:
        """List all open tickets in a specific category."""
        if not category or not category.strip():
            available = sorted(set(t['category'] for t in self.tickets))
            return f"Error: Please provide a category. Available: {', '.join(available)}"

        category = category.strip()
        matching = [t for t in self.tickets if t['category'].lower() == category.lower()]

        if not matching:
            available = sorted(set(t['category'] for t in self.tickets))
            return f"No open tickets in category '{category}'. Available: {', '.join(available)}"

        output = f"Found {len(matching)} OPEN tickets in '{category}':\n\n"
        for t in matching:
            output += f"• [{t['ticket_id']}] {t['title']} (Priority: {t['priority']})\n"
        return output

    # ------------------------------------------------------------------
    # Single ticket lookup
    # ------------------------------------------------------------------
    def get_ticket_by_id(self, ticket_id: str) -> str:
        """Retrieve one open ticket by its exact ID."""
        if not ticket_id or not ticket_id.strip():
            return "Error: Please provide a ticket ID (e.g., TICK-001)"

        ticket_id = ticket_id.upper().strip()
        if not ticket_id.startswith("TICK-"):
            return (f"Error: Invalid format. Ticket IDs look like TICK-001, "
                    f"TICK-002, etc. You provided: '{ticket_id}'")

        for t in self.tickets:
            if t['ticket_id'] == ticket_id:
                return f"""Ticket ID: {t['ticket_id']}
Title: {t['title']}
Description: {t['description']}
Category: {t['category']}
Priority: {t['priority']}
Created: {t['created_date']}
Status: OPEN"""
        return f"Ticket '{ticket_id}' not found among open tickets."

    # ------------------------------------------------------------------
    # Backlog statistics
    # ------------------------------------------------------------------
    def get_ticket_statistics(self, input: str = "") -> str:
        """Overview of the OPEN ticket backlog: totals by category and priority."""
        total = len(self.tickets)
        categories, priorities = {}, {}
        for t in self.tickets:
            categories[t['category']] = categories.get(t['category'], 0) + 1
            priorities[t['priority']] = priorities.get(t['priority'], 0) + 1

        output = "Open Ticket Backlog Statistics:\n"
        output += f"Total Open Tickets: {total}\n\n"
        output += "By Category:\n"
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            output += f"  • {cat}: {count}\n"
        output += "\nBy Priority:\n"
        for pri, count in sorted(priorities.items(), key=lambda x: x[1], reverse=True):
            output += f"  • {pri}: {count}\n"
        return output

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------
    def get_tools(self):
        return [
            Tool(
                name="SearchSimilarTickets",
                func=self.search_similar_tickets,
                description="""Find OPEN tickets similar to a described problem.
Use when an employee describes a bug and wants to see related open issues, or
asks "are there other tickets like this?". You can include a priority word
(e.g. 'critical login bugs') to bias the search toward that priority.
Input: a bug description or ticket title."""
            ),
            Tool(
                name="FindDuplicateTickets",
                func=self.find_duplicate_tickets,
                description="""Detect likely DUPLICATE open tickets.
Use when an employee asks "is this a duplicate?", "are there duplicates of
TICK-003?", or before filing a new ticket. Returns matches above a similarity
threshold with their scores.
Input: a bug description OR an existing ticket ID (e.g. 'TICK-003')."""
            ),
            Tool(
                name="SearchByPriority",
                func=self.search_by_priority,
                description="""List ALL open tickets at a given priority level.
Use when an employee asks for high/critical/urgent open bugs.
Input: one of Critical, High, Medium, Low."""
            ),
            Tool(
                name="SearchByCategory",
                func=self.search_by_category,
                description="""List ALL open tickets in a specific category.
Use for "show me all <X> open tickets".
Categories include: Search, Listings, Mortgage Calculator, Notifications,
Payment, Media, Performance, API, Real-time, Integration, File Upload,
User Management.
Input: a category name (e.g. 'Search', 'Listings')."""
            ),
            Tool(
                name="GetTicketByID",
                func=self.get_ticket_by_id,
                description="""Get the full details of one open ticket by exact ID.
Use when the employee names a specific ticket (TICK-001, etc.).
Input: a ticket ID. Do NOT use this for searching."""
            ),
            Tool(
                name="GetTicketStatistics",
                func=self.get_ticket_statistics,
                description="""Overview of the open backlog: counts by category and
priority. Use for "how many open tickets", "give me an overview".
No meaningful input needed."""
            ),
        ]


# ---------------------------------------------------------------------------
# Build tools + LLM agent
# ---------------------------------------------------------------------------
print("Creating triage tools...")
tool_manager = OpenTicketTriageTools()
tools = tool_manager.get_tools()
print(f"✓ Created {len(tools)} tools over {len(tool_manager.tickets)} open tickets")

llm = ChatOpenAI(model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'), temperature=0)

tool_definitions = []
for tool in tools:
    tool_definitions.append({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "The input to the tool"}
                },
                "required": ["input"]
            }
        }
    })

llm_with_tools = llm.bind(tools=tool_definitions)


TRIAGE_SYSTEM_PROMPT = """You are an open-bug triage assistant for a house-search /
home-buying platform. You help engineers and support staff SEARCH and GROUP the
open ticket backlog: surfacing similar tickets, flagging duplicates, and listing
high-priority bugs.

Follow these rules:
1. State which tool you're about to use before calling it.
2. If the request is ambiguous, ask one clarifying question first.
3. When showing tickets, always reference their ticket IDs and priority.
4. For "is this a duplicate?" style questions, use FindDuplicateTickets and clearly
   state whether likely duplicates were found.
5. After answering, suggest one useful next triage step (e.g. "want me to check
   for duplicates?" or "want all High priority ones in this category?").

Your tools:
- SearchSimilarTickets  : related open tickets for a described problem
- FindDuplicateTickets  : likely duplicates of a description or ticket ID
- SearchByPriority      : all open tickets at a priority level
- SearchByCategory      : all open tickets in a category
- GetTicketByID         : full details of one ticket
- GetTicketStatistics   : backlog overview
"""


def run_agent(query: str, max_iterations: int = 5, custom_prompt: str = None) -> dict:
    """Run the triage agent, tracking which tools were used."""
    system_prompt = custom_prompt or TRIAGE_SYSTEM_PROMPT
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
    tools_used = []

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return {'response': response.content, 'tools_used': tools_used,
                    'iterations': i + 1}

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"].get("input", "")
            print(f"🔧 Tool: {tool_name} | Input: {str(tool_input)[:60]}...")
            tools_used.append(tool_name)

            tool_output = None
            for tool in tools:
                if tool.name == tool_name:
                    tool_output = tool.func(tool_input)
                    break
            if tool_output is None:
                tool_output = f"Error: Tool {tool_name} not found"

            messages.append(ToolMessage(content=tool_output,
                                        tool_call_id=tool_call["id"]))

    return {'response': "Maximum iterations reached.", 'tools_used': tools_used,
            'iterations': max_iterations}


# ---------------------------------------------------------------------------
# Demo queries
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEMO: Triage searches over the open backlog")
    print("=" * 80)

    demo_queries = [
        # Find similar open tickets
        "Are there open tickets about search returning the wrong listings?",
        # Duplicate detection by description
        "Is there already a ticket about the map freezing with too many pins?",
        # Duplicate detection by ID
        "Find duplicates of TICK-001",
        # High priority sweep
        "Show me all high priority open bugs",
        # Critical sweep
        "What critical open tickets do we have?",
        # Category browse
        "List all open Listings tickets",
        # Specific lookup
        "Show me TICK-009",
        # Backlog overview
        "Give me an overview of the open backlog",
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n--- Query {i}: '{query}' ---")
        result = run_agent(query)
        print(f"Tools used: {result['tools_used']}")
        print(f"Response: {result['response'][:220]}...")
