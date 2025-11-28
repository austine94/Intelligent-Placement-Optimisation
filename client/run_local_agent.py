#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 17:46:46 2025

@author: localadmin
"""

# client/run_local_agent.py
import asyncio
import os
import sys
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

#set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
server_script = os.path.join(project_root, 'server', 'mcp_server.py')

#CONFIG - check model has been pulled from Ollama
MODEL = "llama3.2" 

async def run_agent():
    print(f"🚀 Launching Server Process: {server_script}")
    
    #runs the server and scripts 
    server_params = StdioServerParameters(
        command=sys.executable, 
        args=[server_script, "run"],
        env=os.environ.copy()
    )

    #server connection
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            #identify available tools
            tools_response = await session.list_tools()
            tools = tools_response.tools
            print(f"Available Tools: {[t.name for t in tools]}")
            
            # --- CHAT LOOP ---
            messages = []
            print("\n--- AGENT READY (Type 'quit' to exit) ---")
            print(f"Using Model: {MODEL}")
            
            while True:
                #First we get the input from the user
                user_input = input("\nUser: ")
                if user_input.lower() in ['quit', 'exit']: break
                
                messages.append({"role": "user", "content": user_input})
                
                #this loop converts MCP tools to Ollama tools
                ollama_tools = []
                for tool in tools:
                    ollama_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })

                #this then connects up the model with the available tools
                #and the prompt that has been passed by the user.
                #using this information the model will decide if we need a tool
                print("Thinking...")
                response = ollama.chat(
                    model=MODEL,
                    messages=messages,
                    tools=ollama_tools
                )
                
                msg = response['message']
                messages.append(msg) #this stores messages in a log 
                
                #check if agent wants to coll a tool
                if msg.get('tool_calls'):
                    print(f"Plan: Agent is calling {len(msg['tool_calls'])} tool(s)...")
                    
                    for tool_call in msg['tool_calls']:
                        fn_name = tool_call['function']['name']
                        fn_args = tool_call['function']['arguments']
                        
                        print(f"Agent Executing: {fn_name}({fn_args})")
                        
                        #run tool on MCP server
                        try:
                            result = await session.call_tool(fn_name, fn_args)
                            
                            #first entry will be the text output from tool call
                            output_text = result.content[0].text
                            print(f"   -> Result: {output_text[:150]}...") #only store truncated logs
                            
                            #pass tool output back to LLM
                            messages.append({
                                "role": "tool",
                                "content": output_text,
                            })
                        except Exception as e:
                            print(f"   -> Tool Error: {e}")
                            messages.append({
                                "role": "tool",
                                "content": f"Error executing tool: {e}",
                            })
                    
                    #We now have the user prompt and the tool call outputs.
                    #These are used to produce the final LLM response.
                    final_response = ollama.chat(model=MODEL, messages=messages)
                    print(f"\nAgent: {final_response['message']['content']}")
                    messages.append(final_response['message'])
                
                else:
                    # AI didn't use a tool, just talked
                    print(f"\nAgent: {msg['content']}")

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nGoodbye!")