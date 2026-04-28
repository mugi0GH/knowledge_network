import sys
import os
import asyncio
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_server.py")
MODEL = "qwen2.5:7b"


def to_ollama_tools(mcp_tools) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


async def run_once(user_input: str, session: ClientSession, ollama_tools: list) -> str:
    messages = [{"role": "user", "content": user_input}]

    while True:
        response = ollama.chat(model=MODEL, messages=messages, tools=ollama_tools)
        msg = response.message

        if not msg.tool_calls:
            return msg.content

        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in msg.tool_calls
        ]})

        for call in msg.tool_calls:
            print(f"[调用工具: {call.function.name}]")
            result = await session.call_tool(
                call.function.name,
                arguments=call.function.arguments if isinstance(call.function.arguments, dict)
                          else json.loads(call.function.arguments),
            )
            content = result.content[0].text if result.content else ""
            messages.append({"role": "tool", "content": content})


async def main():
    server_params = StdioServerParameters(
        command=sys.executable, args=[MCP_SERVER]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            ollama_tools = to_ollama_tools(mcp_tools)

            print(f"Agent 已启动，加载了 {len(mcp_tools)} 个工具（输入 exit 退出）")
            for t in mcp_tools:
                print(f"  - {t.name}: {t.description}")

            while True:
                sys.stdout.write("\n问: ")
                sys.stdout.flush()
                raw = sys.stdin.buffer.readline()
                if not raw:
                    break
                user_input = raw.decode("utf-8", errors="replace").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break

                answer = await run_once(user_input, session, ollama_tools)
                print(f"答: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
