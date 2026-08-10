import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)

# 3. Define the Edges & Routing Logic
def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"

workflow.set_entry_point("planner")


# Conditional Edge: Planner -> Router -> (Retriever OR Responder)
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder"
    }
)


workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)


# --- PRODUCTION MEMORY: PostgreSQL Checkpointer ---
# Persists conversation state across server restarts and multiple workers.
# Requires POSTGRES_URI in .env  e.g.:
#   postgresql://user:password@localhost:5432/floatchat
POSTGRES_URI = os.getenv("POSTGRES_URI")

if not POSTGRES_URI:
    raise RuntimeError(
        "POSTGRES_URI is not set. "
        "Add it to your .env file: postgresql://user:pass@host:5432/dbname"
    )

# psycopg3 connection pool — min 1 conn, max 10 conns
_pool = ConnectionPool(
    conninfo=POSTGRES_URI,
    max_size=10,
    kwargs={"autocommit": True},   # required by LangGraph checkpointer
)

checkpointer = PostgresSaver(_pool)

# One-time DDL: create the LangGraph checkpoint tables if they don't exist
checkpointer.setup()


# 4. Compile the Graph with PostgreSQL-backed Memory
rag_agent = workflow.compile(checkpointer=checkpointer)


