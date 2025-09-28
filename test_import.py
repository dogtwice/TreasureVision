import os

# VS Code 会把 ${workspaceFolder} 替换成实际路径再传给 Python
workspace_folder = os.environ.get("workspaceFolder")

print(f"workspaceFolder from env: {workspace_folder}")
