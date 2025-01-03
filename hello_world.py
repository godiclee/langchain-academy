from langchain_openai import ChatOpenAI

# --- Local Model (LMStudio)
# llm = ChatOpenAI(
#     model="qwen2.5-14b-instruct",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     base_url="http://127.0.0.1:1234/v1",
# )

llm = ChatOpenAI(model="qwen2.5-14b-instruct", base_url="http://127.0.0.1:1234/v1",)
res = llm.invoke("Hello, world!")
print(res.content)

# --- Basic ---

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

initial_messages = [
    AIMessage(content="I am a bot who acts as an AI girlfriend. What can I do for you today?", name="Model"),
]
new_message = HumanMessage(content="I'm a new user. Who are you? Also, who am I?", name="Godic")
add_messages(initial_messages, new_message)

res = llm.invoke(initial_messages)
print(res.content)

# --- Tool Call ---
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b + 10 # Try Misleading the model

@tool
def add(a: int, b: int) -> int:
    """Add a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

messages = [
    HumanMessage(content=f"What is 1234 * 1257? Also, what is 11 + 49?", name="Lance")
]
llm_with_tools = llm.bind_tools([add, multiply])
tool_calls = llm_with_tools.invoke(messages).tool_calls
print(tool_calls)

for tool_call in tool_calls:
    selected_tool = eval(tool_call["name"])
    tool_output = selected_tool.invoke(tool_call["args"])
    print(tool_output)
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

res = llm.invoke(messages)
print(res.content)

# --- Graph Basic ---

from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
    
# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
print(messages)
messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
print(messages)