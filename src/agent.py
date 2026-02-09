from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    PROBLEM: str
    retrieved_actions: List[str]
    best_actions: str
    final_action: str


def build_agent(retriever, llm):

    def retrieve_node(state: AgentState):
        docs = retriever.invoke(state["PROBLEM"])
        actions = [d.metadata["ACTION"] for d in docs]
        return {"retrieved_actions": actions}

    def rerank_node(state: AgentState):
        prompt = f"""
You are an expert aircraft maintenance supervisor.

PROBLEM:
{state['PROBLEM']}

Candidate Actions:
{state['retrieved_actions']}

Select the 4 best actions most relevant to the PROBLEM.
Return them as a list.
"""
        response = llm.invoke(prompt).content
        return {"best_actions": response}

    def generate_node(state: AgentState):
        prompt = f"""
You are an aircraft maintenance engineer.

Fault:
{state['PROBLEM']}

Relevant Past Actions:
{state['best_actions']}

Generate a precise and technically grounded maintenance action.
Do NOT hallucinate. Only use the references if possible.
Generate a short, concise maintenance action.
Limit to one sentence. No extra explanation.
"""
        response = llm.invoke(prompt).content
        return {"final_action": response}

    workflow = StateGraph(AgentState)
    workflow.add_node("Retrieve", retrieve_node)
    workflow.add_node("Rerank", rerank_node)
    workflow.add_node("Generate", generate_node)

    workflow.set_entry_point("Retrieve")
    workflow.add_edge("Retrieve", "Rerank")
    workflow.add_edge("Rerank", "Generate")
    workflow.add_edge("Generate", END)

    return workflow.compile()
