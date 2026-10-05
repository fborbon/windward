"""MCP server exposing Windward's forecast/recommendation/RAG, OpenOA operational assessment
and per-turbine tools."""
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


@mcp.tool()
def get_operational_assessment(farm_id: str) -> dict:
    """Long-term operational assessment from NREL/NLR's OpenOA, run on the full multi-year
    data: electrical losses (turbines vs substation meter), long-term AEP P50/P90 with a
    per-component uncertainty breakdown, wake losses per turbine, and static yaw
    misalignment estimates with their status (flag / ok / inconclusive)."""
    return tools.get_operational_assessment(farm_id)


@mcp.tool()
def get_turbine_details(farm_id: str, turbine: str) -> dict | str:
    """Everything known about one turbine (name or number, e.g. 'Kelmarsh 3' or '3'):
    capacity factor and fleet rank, peak Cp, QC-removed hours, its anomalies, OpenOA wake loss
    and yaw status, and its real status log (last maintenance stop, forced-outage hours with
    overlapping alarms counted once, top causes)."""
    return tools.get_turbine_details(farm_id, turbine)


@mcp.tool()
def rank_turbines(farm_id: str, metric: str) -> dict | str:
    """Rank a farm's turbines by one metric, highest first: capacity_factor, peak_cp,
    wake_loss_lt_pct, abs_yaw_misalignment_deg, qc_removed_share, forced_outage_hours or
    maintenance_events."""
    return tools.rank_turbines(farm_id, metric)


if __name__ == "__main__":
    mcp.run()
