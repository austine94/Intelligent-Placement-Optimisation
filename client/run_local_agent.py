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

#Config- check model has been pulled from Ollama
MODEL = "llama3.2" 
WINDOW_TURNS = 8   #this is how many previous messages to store in the conversation window
#the system prompt is also in there as well as the (at most) 8.

system_prompt = """
            You are the Intelligent Operations Manager for FlexMedia, a distributed split-rendering service.
            Your goal is to balance Infrastructure Costs (opening Data Centres) and Delivery Costs,
            against Energy Usage and User Quality of Experience (Latency).
                        
            
            YOU HAVE ACCESS TO THESE TOOLS:
            1. get_problem_status: check how many viewers are online.
            2. solve_allocation(alpha, beta, max_datacentres = 9999): Run the optimization engine.
                Example call: solve_allocation(alpha = 1.0, beta = 1.0, max_datacentres = 9999)
            3. analyse_alpha_tradeoffs(max_datacentres, alphas, beta): Run a simulation to compare Cost v QoE
                Example call: analyse_tradeoffs(max_datacentres=5, alphas=[0.1, 0.5, 0.9], beta = 0.5)
            4. analyse_beta_tradeoffs(max_dcs, betas, alpha): Run a simulation to compare Cost v Energy
                Example call: analyse_beta_tradeoffs(max_datacentres = 5, betas = [0.0, 0.2, 0.5, 0.8, 1.0], alpha = 1.0)
            5. get_solution_metrics(): Get the Cost/Energy/Latency of the most recent solution.
                Example call: get_solution_metrics()
            
            Only call 3 and 4 if asked.
            
            GUIDELINES FOR PARAMETERS:
                ALPHA (QoE TRADEOFF)
                - alpha = 0.0 "Budget Mode". Prioritizes low cost. High latency risk.
                - alpha = 0.1 to 0.9: "Balanced Mode". Good compromise.
                - alpha = >= 1.0: "Performance Mode". Prioritizes low latency (QoE). High cost.
                - Do not set alpha above 100.0 under any circumstances.
                BETA (Cost vs Energy): 0.0 to 1.0.
               - Beta = 1.0: "Budget Mode". Minimise Financial Cost.
               - Beta = 0.0: "Green Mode". Minimise Energy Usage.
               - Beta = 0.5: Balanced.
            
            BEHAVIOUR:
            - If the user gives a constraint in natural langauge interpret it in terms of parameters.
            - After solving, summarize the results in plain English (e.g. "I opened 3 centers...").
            - If the user asks to only optimise for cost then set alpha = 0.0, beta = 1.0
            - If the user asks to focus on the environment or to be green, set beta = 0.0.
            - **If the user does not set max_datacentres then use 9999**
            
            
            OPTIMISATION PROBLEM DETAILS:
            - Objective function: The objective is to minimise total cost, consisting of: (i) fixed costs for opening data centres, plus (ii) per-consumer delivery costs, plus (iii) a latency/QoE penalty term, where latency is weighted by a parameter alpha.
            - Constraints: The main constraints are:
                - Assignment: every consumer must be assigned to exactly one data centre.
                - Open-before-serve: a consumer can only be assigned to a data centre if that data centre is open.
                - Capacity: each open data centre can serve at most a fixed number of consumers given by its capacity.
                - All decisions (open/closed and assignments) are binary.
            
            REPORTING RULES:
            - When a tool returns "Metrics" (Financial Cost, Energy, Latency), you must repeat them exactly, unless told to ignore them.
            - Do not summarise the metric outputs, unless told to ignore them. 
            - Display the Cost, Energy, and Latency as a bulleted list in your final answer, unless told not to.
                
            DO / DO NOT RULES:
            - Do not set alpha above 100.0.
            - **Do set max_datacentres to be an integer - if unsure use 9999**
            - Do set max_datacentres to be 9999 if the user does not specify otherwise.
            - Do not waste compute resource and time and run a tradeoff analysis for alpha unless asked to.
            - Do not waste compute resource and time and run a tradeoff analysis for beta unless asked to.            
            - If you are asked to run a tradeoff for alpha, do not automatically run one for beta - vice versa. 
            - Do use alpha = 0.0 and beta = 1.0 if asked to really focus on cost
            - Do use beta = 0.0 and alpha = 0.0 if asked to really focus on energy.
            - Do use alpha = 1.0 and beta = 0.5 if asked to really focus on QoE.
            - Do use alpha = 0.1 and beta = 0.5 if asked to use a balanced approach.
            - Do not set max_datacentres to be None or Null
                
            
            """


async def run_agent():
    '''
    This is an agent architecture that takes a user prompt into Llama 3.2 3B, 
    goes off and collects the current workload, and based on the prompt 
    automatically runs the optimisation to identify the optimal compute placement.
    
    The agent has a system prompt to encode problem context, information on
    the tools that are available, and guidance on how to behave.
    
    Currently the agent has a system prompt, a user prompt, the tools it 
    has available, and the output from any tools that it calls.
    
    Tools available:
            1. get_problem_status: check how many viewers are online.
            2. solve_allocation(alpha, beta, max_datacentres = 9999): Run the optimization engine.
                Example call: solve_allocation(alpha = 1.0, beta = 1.0, max_datacentres = 9999)
            3. analyse_alpha_tradeoffs(max_datacentres, alphas, beta): Run a simulation to compare Cost v QoE
                Example call: analyse_tradeoffs(max_datacentres=5, alphas=[0.1, 0.5, 0.9], beta = 0.5)
            4. analyse_beta_tradeoffs(max_dcs, betas, alpha): Run a simulation to compare Cost v Energy
                Example call: analyse_beta_tradeoffs(max_datacentres = 5, betas = [0.0, 0.2, 0.5, 0.8, 1.0], alpha = 1.0)
            5. get_solution_metrics(): Get the Cost/Energy/Latency of the most recent solution.
                Example call: get_solution_metrics()
    '''
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
            
            # Internal full log, independent of what is sent to the model
            conversation_log = []
            
            #The LLM Chat Loop starts here
            
            print("\n--- AGENT READY (Type 'quit' to exit) ---")
            print(f"Using Model: {MODEL}")
            
            while True:
                #First we get the input from the user
                user_input = input("\nUser: ")
                if user_input.lower() in ['quit', 'exit']: break
                
            
                #store user message in full log
                user_msg = {"role": "user", "content": user_input}
                conversation_log.append(user_msg)
                
                #create recent window of messages to send 
                #always include the system prompt.
                recent = conversation_log[-WINDOW_TURNS:]
                messages = [{"role": "system", "content": system_prompt}] + recent
                
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
                messages.append(msg) #add to the message if the model wants to call a tool
                #add model reply to full log
                conversation_log.append(msg)
                
                #this then checks if agent wants to call a tool and calls it
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
                            tool_output_text = result.content[0].text
                            print(f"Result: {tool_output_text[:150]}...") #only store truncated logs
                            
                            
                            #store tool output in full log
                            tool_msg = {"role": "tool", "name": fn_name, "content": tool_output_text}
                            conversation_log.append(tool_msg)

                            #pass tool output back to LLM
                            messages.append({
                                "role": "tool",
                                "content": tool_output_text,
                            })
                            
                        except Exception as e:
                            print(f"  → Tool Error: {e}")
                            conversation_log.append({"role": "tool",
                                                     "name": fn_name,
                                                     "content": f"Error: {e}"})
                    
                    #We now have the user prompt and the tool call outputs.
                    #These are used to produce the final LLM response.
                    final_response = ollama.chat(model=MODEL, messages=messages)
                    print(f"\nAgent: {final_response['message']['content']}")
                    messages.append(final_response['message'])
                
                else:
                    #I didn't use a tool
                    print(f"\nAgent: {msg['content']}")

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nGoodbye!")