from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

from agent.executor import ToolExecutor
from agent.query_engine import QueryEngine
from agent.tool_registry import ToolRegistry

app = FastAPI(
    title="SatQuery AI",
    description="Agentic Vision-Language Assistant for Remote-Sensing Image Analysis",
    version="0.2.0",
)

query_engine = QueryEngine()
tool_registry = ToolRegistry()
tool_executor = ToolExecutor(tool_registry)

class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "name": "SatQuery AI",
        "problem_statement": "SIH26167",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "satquery-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/query/plan")
def plan_query(request: QueryRequest):
    plan = query_engine.plan(request.query)

    return {
        "query": request.query,
        "plan": {
            "task": plan.task,
            "description": plan.description,
            "required_inputs": plan.required_inputs,
            "selected_tool": plan.selected_tool,
            "confidence": plan.confidence,
        },
    }

@app.get("/api/tools")
def list_tools():
    tools = tool_registry.list_tools()

    return {
        "count": len(tools),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "supported_tasks": tool.supported_tasks,
                "required_inputs": tool.required_inputs,
                "model_name": tool.model_name,
                "parameters": tool.parameters,
            }
            for tool in tools
        ],
    }


@app.post("/api/tools/{tool_name}/execute")
def execute_tool(tool_name: str, inputs: dict):
    result = tool_executor.execute(tool_name, inputs)

    return {
        "tool_name": result.tool_name,
        "status": result.status,
        "message": result.message,
        "outputs": result.outputs,
        "evidence": result.evidence,
    }