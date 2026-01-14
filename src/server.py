"""
Model Context Protocol (MCP) for ev_onehot

This MCP server provides tools for protein fitness prediction combining evolutionary data and one-hot encoding approaches.

This MCP Server contains the following tools:
1. ev_onehot_train_fitness_predictor: Train and evaluate combined EV+Onehot predictor for protein fitness prediction
2. ev_onehot_predict_fitness: Predict fitness for protein sequences using a pretrained EV+Onehot model
"""

from fastmcp import FastMCP

# Import statements
from tools.train import train_mcp
from tools.pred import pred_mcp

# Server definition and mounting
mcp = FastMCP(name="ev_onehot")
mcp.mount(train_mcp)
mcp.mount(pred_mcp)

if __name__ == "__main__":
    mcp.run()