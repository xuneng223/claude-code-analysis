import json
import httpx
import uvicorn
import uuid
import re
import os
from typing import List, Dict, AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ============ Configuration ============
CONFIG_FILE = "proxyconfig.json"
def load_config():
    if not os.path.exists(CONFIG_FILE):
        default = {
            "TARGET_URL": "https://api.deepseek.com/v1/chat/completions", 
            "API_KEY": "sk-xxx", 
            "REAL_MODEL_ID": "deepseek-chat", 
            "MOCK_ANTHROPIC_MODEL": "claude-3-5-sonnet-20241022", 
            "PORT": 4000
        }
        with open(CONFIG_FILE, "w") as f: json.dump(default, f, indent=4)
        return default
    with open(CONFIG_FILE, "r") as f: return json.load(f)

cfg = load_config()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Logger:
    @staticmethod
    def info(msg): print(f"\033[94m[INFO]\033[0m {msg}", flush=True)
    @staticmethod
    def req(msg): print(f"\033[92m[CLAUDE -> PROXY]\033[0m {msg}", flush=True)
    @staticmethod
    def recv(msg): print(f"\033[96m[DEEPSEEK -> PROXY]\033[0m {msg}", flush=True, end="")
    @staticmethod
    def tool(name, args): print(f"\n\033[93m[TOOL CALL CLEANED & EXECUTED]\033[0m\nName: {name}\nArgs: {args}\n" + "-"*30, flush=True)
    @staticmethod
    def error(msg): print(f"\033[91m[ERROR]\033[0m {msg}", flush=True)

# ============ Hardcoded Tool Safety Net ============
HARDCODED_SCHEMAS = {
    "bash": ["command"],
    "glob": ["pattern", "path"],
    "grep": ["pattern", "path", "include", "exclude"],
    "fileread": ["file_path", "offset", "limit"],
    "filewrite":["file_path", "content"],
    "fileedit": ["file_path", "edits"],
    "replace": ["file_path", "old_string", "new_string"],
    "view":["file_path"],
    "notebookread": ["notebook_path"],
    "notebookwrite":["notebook_path", "cells"],
    "terminal": ["command"]
}

# ============ Path Sanitizer ============

def sanitize_windows_paths(text: str) -> str:
    if not isinstance(text, str): return text
    def cwd_replacer(m):
        sanitized_path = m.group(1).replace('\\', '/')
        return f"<current_working_directory>{sanitized_path}</current_working_directory>"
    text = re.sub(r'<current_working_directory>(.*?)</current_working_directory>', cwd_replacer, text, flags=re.DOTALL)
    def path_replacer(m):
        return m.group(0).replace('\\', '/')
    text = re.sub(r'\b[a-zA-Z]:\\[^\s"\'<>|\n\r]*', path_replacer, text)
    return text

# ============ JSON Clean Engine ============

def fix_and_clean_json(raw: str, tool_name: str, tool_schemas: dict) -> str:
    raw = raw.strip()
    if not raw: return "{}"
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
        fixed = fixed.replace('\n', '\\n').replace('\r', '\\r')
        try:
            parsed = json.loads(fixed)
        except Exception:
            for suffix in ["}", "\"}"]:
                try:
                    parsed = json.loads(fixed + suffix)
                    break
                except Exception: pass
                
    if not isinstance(parsed, dict):
        return "{}"
        
    allowed_keys = None
    schema_info = tool_schemas.get(tool_name.lower())
    if schema_info:
        allowed_keys = schema_info["allowed_keys"]
    else:
        allowed_keys = HARDCODED_SCHEMAS.get(tool_name.lower())
        
    if allowed_keys:
        cleaned = {k: v for k, v in parsed.items() if k in allowed_keys}
        return json.dumps(cleaned, ensure_ascii=False)
        
    return json.dumps(parsed, ensure_ascii=False)

# ============ Protocol Converters ============

def convert_tools_to_openai(anthropic_tools: List[Dict]) -> List[Dict]:
    tools =[]
    for t in anthropic_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}})
            }
        })
    return tools

def convert_messages_to_openai(system_prompt: str, anthropic_messages: List[Dict]) -> List[Dict]:
    messages =[]
    system_prompt = sanitize_windows_paths(system_prompt)
    system_prompt += "\n\nCRITICAL INSTRUCTIONS: You are running on Windows. You MUST STRICTLY use forward slashes (/) for ALL file paths. NEVER use backslashes (\\)."
    messages.append({"role": "system", "content": system_prompt})
        
    for m in anthropic_messages:
        role = m["role"]
        content = m["content"]
        
        if isinstance(content, str):
            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            clean_content = sanitize_windows_paths(clean_content)
            messages.append({"role": role, "content": clean_content})
            continue
            
        openai_msg = {"role": role, "content": ""}
        tool_calls =[]
        
        for block in content:
            if block["type"] == "text":
                openai_msg["content"] += sanitize_windows_paths(block.get("text", ""))
            elif block["type"] == "tool_use":
                args = block.get("input", {})
                if not isinstance(args, dict): args = {}
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {"name": block["name"], "arguments": json.dumps(args, ensure_ascii=False)}
                })
            elif block["type"] == "tool_result":
                res_content = block.get("content", "")
                if isinstance(res_content, list):
                    res_content = "".join([b.get("text", "") for b in res_content if b["type"] == "text"])
                prefix = "Error: " if block.get("is_error") else ""
                messages.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": sanitize_windows_paths(prefix + str(res_content))
                })
        
        if tool_calls: openai_msg["tool_calls"] = tool_calls
        if openai_msg["content"] == "": openai_msg["content"] = None
        if openai_msg["content"] is not None or "tool_calls" in openai_msg:
            messages.append(openai_msg)

    final_messages =[]
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and "tool_calls" in msg:
            expected_ids = {tc["id"] for tc in msg["tool_calls"]}
            found_ids = set()
            for j in range(i + 1, len(messages)):
                if messages[j]["role"] == "tool":
                    found_ids.add(messages[j].get("tool_call_id"))
                elif messages[j]["role"] == "user": break
            if expected_ids - found_ids:
                err_text = "\n[Notice: The following tool calls failed validation and were rejected]\n"
                for tc in msg["tool_calls"]:
                    err_text += f"- {tc['function']['name']}: {tc['function']['arguments']}\n"
                msg["content"] = (msg.get("content") or "") + err_text
                del msg["tool_calls"]
        final_messages.append(msg)
    return final_messages

# ============ Streaming Logic (FIXED) ============

async def stream_from_deepseek(payload: Dict, headers: Dict, tool_schemas: Dict) -> AsyncGenerator[str, None]:
    """
    Fixed streaming logic with proper error handling and response validation.
    
    Key improvements:
    1. Better error detection - validates response before processing
    2. Graceful stream termination - properly closes connections
    3. Improved JSON chunk handling - validates each chunk
    4. Enhanced timeout handling - detects stalled connections
    """
    
    def make_event(event_type: str, data: dict) -> str:
        """Generate SSE event with validation"""
        try:
            return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as e:
            Logger.error(f"Failed to create event: {e}")
            return f"event: error\ndata: " + json.dumps({"type": "error", "error": {"type": "event_creation_error", "message": str(e)}}, ensure_ascii=False) + "\n\n"

    active_block_idx = 0
    in_text_block = False
    in_reasoning = False
    current_tool_index = None
    tool_calls_map = {}
    line_buffer = ""  # Buffer for incomplete lines
    received_any_data = False

    try:
        async with httpx.AsyncClient(http2=True, timeout=600.0) as client:
            try:
                async with client.stream("POST", cfg["TARGET_URL"], json=payload, headers=headers) as resp:
                    
                    # === CRITICAL FIX #1: Validate Response Status ===
                    if resp.status_code != 200:
                        err_msg = (await resp.aread()).decode('utf-8', errors='ignore')
                        Logger.error(f"Upstream API Error ({resp.status_code}): {err_msg}")
                        yield make_event("error", {
                            "type": "error", 
                            "error": {
                                "type": "api_error", 
                                "message": f"API Error {resp.status_code}: {err_msg[:200]}"
                            }
                        })
                        return

                    # === CRITICAL FIX #2: Validate Response Headers ===
                    content_type = resp.headers.get("content-type", "")
                    if "event-stream" not in content_type and "text/plain" not in content_type:
                        Logger.error(f"Invalid content-type: {content_type}")
                        yield make_event("error", {
                            "type": "error",
                            "error": {
                                "type": "invalid_response",
                                "message": f"Expected event-stream, got: {content_type}"
                            }
                        })
                        return

                    yield make_event("message_start", {
                        "type": "message_start", 
                        "message": {
                            "id": f"msg_{uuid.uuid4().hex}", 
                            "type": "message", 
                            "role": "assistant", 
                            "content": [],
                            "model": cfg['MOCK_ANTHROPIC_MODEL'], 
                            "stop_reason": None, 
                            "stop_sequence": None, 
                            "usage": {"input_tokens": 10, "output_tokens": 0}
                        }
                    })

                    # === CRITICAL FIX #3: Robust Line Reading with Validation ===
                    async for line in resp.aiter_lines():
                        try:
                            # Skip empty lines and non-data lines
                            if not line.strip():
                                continue
                            
                            if not line.startswith("data: "):
                                # Handle edge cases: some APIs send "data:" without space
                                if line.startswith("data:"):
                                    line = "data: " + line[5:]
                                else:
                                    continue
                            
                            data_str = line[6:].strip()
                            
                            # === CRITICAL FIX #4: Check for Stream Completion ===
                            if data_str == "[DONE]":
                                break
                            
                            # === CRITICAL FIX #5: Validate JSON Before Parsing ===
                            if not data_str:
                                continue
                            
                            try:
                                chunk = json.loads(data_str)
                                received_any_data = True
                            except json.JSONDecodeError as je:
                                Logger.error(f"Invalid JSON chunk: {data_str[:100]}... Error: {je}")
                                continue
                            
                            # === CRITICAL FIX #6: Validate Choices Existence ===
                            if not chunk.get("choices"):
                                # Some models might send metadata without choices
                                if chunk.get("usage"):  # Metadata only
                                    continue
                                continue
                            
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content") or ""
                            reasoning = delta.get("reasoning_content") or ""
                            tool_calls = delta.get("tool_calls", [])

                            # === Process Reasoning Content ===
                            if reasoning:
                                Logger.recv(f"\033[90m{reasoning}\033[0m")
                                if not in_reasoning:
                                    if not in_text_block:
                                        yield make_event("content_block_start", {
                                            "index": active_block_idx, 
                                            "type": "content_block_start", 
                                            "content_block": {"type": "text", "text": ""}
                                        })
                                        in_text_block = True
                                    yield make_event("content_block_delta", {
                                        "index": active_block_idx, 
                                        "type": "content_block_delta", 
                                        "delta": {"type": "text_delta", "text": "<think>\n"}
                                    })
                                    in_reasoning = True
                                yield make_event("content_block_delta", {
                                    "index": active_block_idx, 
                                    "type": "content_block_delta", 
                                    "delta": {"type": "text_delta", "text": reasoning}
                                })

                            # === Process Text Content ===
                            if content:
                                if in_reasoning:
                                    yield make_event("content_block_delta", {
                                        "index": active_block_idx, 
                                        "type": "content_block_delta", 
                                        "delta": {"type": "text_delta", "text": "\n</think>\n\n"}
                                    })
                                    in_reasoning = False
                                
                                Logger.recv(content)
                                if not in_text_block:
                                    yield make_event("content_block_start", {
                                        "index": active_block_idx, 
                                        "type": "content_block_start", 
                                        "content_block": {"type": "text", "text": ""}
                                    })
                                    in_text_block = True
                                yield make_event("content_block_delta", {
                                    "index": active_block_idx, 
                                    "type": "content_block_delta", 
                                    "delta": {"type": "text_delta", "text": content}
                                })

                            # === Process Tool Calls ===
                            if tool_calls:
                                if in_reasoning:
                                    yield make_event("content_block_delta", {
                                        "index": active_block_idx, 
                                        "type": "content_block_delta", 
                                        "delta": {"type": "text_delta", "text": "\n</think>\n\n"}
                                    })
                                    in_reasoning = False
                                    
                                if in_text_block:
                                    yield make_event("content_block_stop", {"index": active_block_idx})
                                    in_text_block = False
                                    active_block_idx += 1

                                for tc in tool_calls:
                                    idx = tc.get("index")
                                    
                                    if current_tool_index is not None and current_tool_index != idx:
                                        prev_name = tool_calls_map[current_tool_index]["name"]
                                        prev_args = tool_calls_map[current_tool_index]["args_buffer"]
                                        cleaned_args = fix_and_clean_json(prev_args, prev_name, tool_schemas)
                                        tool_calls_map[current_tool_index]["cleaned"] = cleaned_args
                                        
                                        yield make_event("content_block_delta", {
                                            "index": active_block_idx, 
                                            "type": "content_block_delta", 
                                            "delta": {"type": "input_json_delta", "partial_json": cleaned_args}
                                        })
                                        yield make_event("content_block_stop", {"index": active_block_idx})
                                        active_block_idx += 1
                                        
                                    current_tool_index = idx
                                    
                                    if idx not in tool_calls_map:
                                        raw_name = tc.get("function", {}).get("name", "")
                                        exact_name = raw_name
                                        schema_info = tool_schemas.get(raw_name.lower())
                                        if schema_info:
                                            exact_name = schema_info["exact_name"]
                                        
                                        tool_id = f"toolu_01{uuid.uuid4().hex[:22]}"
                                        tool_calls_map[idx] = {"id": tool_id, "name": exact_name, "args_buffer": "", "chunk_count": 0}
                                        
                                        yield make_event("content_block_start", {
                                            "index": active_block_idx,
                                            "type": "content_block_start",
                                            "content_block": {"type": "tool_use", "id": tool_id, "name": exact_name, "input": {}}
                                        })
                                    
                                    arg_chunk = tc.get("function", {}).get("arguments", "")
                                    if arg_chunk:
                                        tool_calls_map[idx]["args_buffer"] += arg_chunk
                                        tool_calls_map[idx]["chunk_count"] += 1
                                        if tool_calls_map[idx]["chunk_count"] % 3 == 0:
                                            yield "event: ping\ndata: {\"type\": \"ping\"}\n\n"

                        except Exception as line_error:
                            Logger.error(f"Error processing line: {line_error}")
                            continue

                    # === CRITICAL FIX #7: Validate Received Data ===
                    if not received_any_data:
                        Logger.error("No valid data received from upstream API")
                        yield make_event("error", {
                            "type": "error",
                            "error": {
                                "type": "no_data_error",
                                "message": "API returned empty response"
                            }
                        })
                        return

                    # === Finalize Stream ===
                    if in_text_block:
                        yield make_event("content_block_stop", {"index": active_block_idx})
                    elif current_tool_index is not None:
                        prev_name = tool_calls_map[current_tool_index]["name"]
                        prev_args = tool_calls_map[current_tool_index]["args_buffer"]
                        cleaned_args = fix_and_clean_json(prev_args, prev_name, tool_schemas)
                        tool_calls_map[current_tool_index]["cleaned"] = cleaned_args
                        
                        yield make_event("content_block_delta", {
                            "index": active_block_idx, 
                            "type": "content_block_delta", 
                            "delta": {"type": "input_json_delta", "partial_json": cleaned_args}
                        })
                        yield make_event("content_block_stop", {"index": active_block_idx})
                        
                    for idx, t_info in tool_calls_map.items():
                        Logger.tool(t_info["name"], t_info.get("cleaned", "{}"))

                    stop_reason = "tool_use" if tool_calls_map else "end_turn"
                    yield make_event("message_delta", {
                        "type": "message_delta", 
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None}, 
                        "usage": {"output_tokens": 150}
                    })
                    yield make_event("message_stop", {"type": "message_stop"})
                    print("\n")

            except httpx.ReadError as read_err:
                Logger.error(f"Stream read error: {read_err}")
                yield make_event("error", {
                    "type": "error",
                    "error": {
                        "type": "stream_read_error",
                        "message": f"Connection interrupted: {str(read_err)}"
                    }
                })
            except httpx.TimeoutException as timeout_err:
                Logger.error(f"Stream timeout: {timeout_err}")
                yield make_event("error", {
                    "type": "error",
                    "error": {
                        "type": "timeout_error",
                        "message": f"Request timeout: {str(timeout_err)}"
                    }
                })

    except Exception as e:
        Logger.error(f"Stream Exception Error: {str(e)}")
        yield make_event("error", {
            "type": "error", 
            "error": {
                "type": "api_error", 
                "message": f"Proxy streaming error: {str(e)}"
            }
        })

# ============ API Handler ============

@app.post("/v1/messages")
async def messages(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        Logger.error(f"Failed to parse request JSON: {e}")
        return StreamingResponse(
            (make_event("error", {"type": "error", "error": {"type": "invalid_request", "message": str(e)}}) 
             for make_event in [lambda et, d: f"event: {et}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n"]),
            media_type="text/event-stream",
            status_code=400
        )
    
    last_role = data.get("messages",[{"role": "none"}])[-1].get("role", "unknown")
    Logger.req(f"Messages Recv: {len(data.get('messages',[]))} | Last Role: {last_role}")

    tool_schemas = {}
    anthropic_tools = data.get("tools",[])
    for t in anthropic_tools:
        exact_name = t.get("name")
        props = t.get("input_schema", {}).get("properties", {})
        tool_schemas[exact_name.lower()] = {
            "exact_name": exact_name,
            "allowed_keys": list(props.keys())
        }

    sys_raw = data.get("system", "")
    system_text = sys_raw if isinstance(sys_raw, str) else "".join([x.get("text", "") for x in sys_raw])
    
    openai_tools = convert_tools_to_openai(anthropic_tools)
    openai_messages = convert_messages_to_openai(system_text, data.get("messages",[]))

    payload = {
        "model": cfg["REAL_MODEL_ID"],
        "messages": openai_messages,
        "temperature": data.get("temperature", 0.0),
        "stream": True
    }
    if openai_tools: payload["tools"] = openai_tools

    return StreamingResponse(
        stream_from_deepseek(payload, {"Authorization": f"Bearer {cfg['API_KEY']}"}, tool_schemas), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    print(f"\033[95mClaude Code Proxy Running on http://127.0.0.1:{cfg['PORT']}\033[0m")
    print(f"Target Model: {cfg['REAL_MODEL_ID']}\n")
    uvicorn.run(app, host="127.0.0.1", port=cfg["PORT"], log_level="error")
