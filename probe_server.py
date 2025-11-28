#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 18:01:34 2025

@author: localadmin
"""

# probe_server.py
import subprocess
import sys
import os

# Path to your server script
server_script = os.path.join("server", "mcp_server.py")

print(f"🕵️  Probing Server at: {server_script}")

# Launch the server process manually
process = subprocess.Popen(
    [sys.executable, server_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=os.environ.copy() # Pass current env
)

print("✅ Server process launched. Listening for 'First Words'...")

try:
    # We try to read the first few lines of output
    # If Gurobi is printing, it will show up here.
    output = process.stdout.readline()
    error = process.stderr.readline()

    if output:
        print(f"\n[STDOUT] The server said:\n---> {output.strip()}")
        print("(This should be blank or a JSON string. If it is text, that is the bug!)")
    else:
        print("\n[STDOUT] (No output received yet - implies Server is waiting silently, which is GOOD)")

    if error:
        print(f"\n[STDERR] The server logged error:\n---> {error.strip()}")

    # Send a handshake to see if it responds validly
    print("\nAttempting JSON Handshake...")
    # This is a standard MCP initialization message
    msg = '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-04-01", "capabilities": {}, "clientInfo": {"name": "probe", "version": "1.0"}}}\n'
    
    process.stdin.write(msg)
    process.stdin.flush()
    
    # Read response
    response = process.stdout.readline()
    print(f"\n[RESPONSE] Server replied:\n---> {response}")

except Exception as e:
    print(f"Probe crashed: {e}")

finally:
    process.terminate()