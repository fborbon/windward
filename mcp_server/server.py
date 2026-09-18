"""MCP server exposing Windward's forecast/recommendation/RAG tools."""
from mcp.server.mcpserver import MCPServer

from mcp_server import tools

mcp = MCPServer("windward")


@mcp.tool()
def get_forecast(farm_id: str, horizon_hours: int) -> dict:
    """Get the production and price forecast for a wind farm."""
    from schemas.models import ForecastRequest

    result = tools.get_forecast(ForecastRequest(farm_id=farm_id, horizon_hours=horizon_hours))
    return result.model_dump()


@mcp.tool()
def get_recommendation(farm_id: str) -> dict:
    """Get the current operational recommendation for a wind farm."""
    return tools.get_recommendation(farm_id)


@mcp.tool()
def query_maintenance_docs(farm_id: str, question: str) -> str:
    """Ask a question against a farm's real incident log + technical reference corpus."""
    return tools.query_maintenance_docs(farm_id, question)


@mcp.tool()
def query_edp_incidents(question: str) -> str:
    """Ask a question against EDP Wind Farm A's real labeled fault case studies (22
    anonymized turbine windows, diagnosis/RAG only, no forecasting for this farm)."""
    return tools.query_edp_incidents(question)


@mcp.tool()
def query_dswe_reference(question: str) -> str:
    """Ask a question against the DSWE Inland-Offshore dataset's real turbine/met-mast
    facts and the Measure-Correlate-Predict methodology used to work around its undisclosed
    location."""
    return tools.query_dswe_reference(question)


if __name__ == "__main__":
    mcp.run()
