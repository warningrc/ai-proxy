import time
import json
import logging
from typing import Dict, Any, AsyncGenerator, Callable, Optional

logger = logging.getLogger("ai-proxy")

def claude_to_openai_request(claude_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Claude API request payload to OpenAI API request payload.
    Supports text, images, and tools.
    """
    openai_messages = []
    
    # Handle system prompt
    if "system" in claude_data:
        system_content = claude_data["system"]
        content = ""
        if isinstance(system_content, list):
             text_parts = []
             for part in system_content:
                 if isinstance(part, dict) and part.get("type") == "text":
                     text_parts.append(part.get("text", ""))
                 # Optionally handle other system block types or log warning
             content = "\n".join(text_parts)
        elif isinstance(system_content, str):
            content = system_content
            
        if content:
            openai_messages.append({"role": "system", "content": content})
    
    # Handle messages
    for msg in claude_data.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            new_content = []
            if isinstance(content, str):
                new_content = content # String content is fine for OpenAI user role
            elif isinstance(content, list):
                # Check for tool_result usage which maps to 'tool' role in OpenAI
                tool_results = [block for block in content if isinstance(block, dict) and block.get("type") == "tool_result"]
                regular_content = [block for block in content if isinstance(block, dict) and block.get("type") != "tool_result"]
                
                # Process regular content (text, image)
                new_regular_content = []
                for block in regular_content:
                    if block["type"] == "text":
                        new_regular_content.append({"type": "text", "text": block["text"]})
                    elif block["type"] == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            data = source.get("data")
                            media_type = source.get("media_type")
                            new_regular_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{data}"
                                }
                            })
                
                # If we have regular content, add a user message
                if new_regular_content:
                    openai_messages.append({"role": "user", "content": new_regular_content})
                
                # If we have tool results, add distinct tool messages
                for res in tool_results:
                    # Tool result content in Claude can be string or list of blocks (text/image)
                    # OpenAI expects simple string usually, but some models support content parts.
                    # We'll flatten to string for max compatibility or use content parts if supported.
                    # Standard OpenAI tool_role content is string.
                    
                    tc_content = res.get("content", "")
                    if isinstance(tc_content, list):
                         # Simplification: extract text parts
                         tc_text_parts = [b["text"] for b in tc_content if b.get("type") == "text"]
                         tc_content_str = "\n".join(tc_text_parts)
                    else:
                        tc_content_str = str(tc_content)
                        
                    openai_messages.append({
                        "role": "tool", 
                        "tool_call_id": res.get("tool_use_id"),
                        "content": tc_content_str
                    })
                
                continue # Skip the default append since we handled split roles

        elif role == "assistant":
            # Handle tool_use in assistant message -> tool_calls in OpenAI
            if isinstance(content, list):
                text_parts = []
                tool_calls = []
                
                for block in content:
                    if block["type"] == "text":
                        text_parts.append(block["text"])
                    elif block["type"] == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"])
                            }
                        })
                
                msg_obj = {"role": "assistant"}
                if text_parts:
                    msg_obj["content"] = "\n".join(text_parts)
                if tool_calls:
                    msg_obj["tool_calls"] = tool_calls
                    if "content" not in msg_obj:
                         msg_obj["content"] = None # Optional in OpenAI if tool_calls present
                
                openai_messages.append(msg_obj)
                continue # Skip default

        # Default fallback (simple text or simple structure)
        openai_messages.append({"role": role, "content": content})
        
    openai_req = {
        "model": claude_data.get("model"),
        "messages": openai_messages,
        "stream": claude_data.get("stream", False),
    }
    if claude_data.get("stream", False):
        openai_req["stream_options"] = {"include_usage": True}

    # Tools Mapping
    if "tools" in claude_data:
        openai_tools = []
        for tool in claude_data["tools"]:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description"),
                    "parameters": tool["input_schema"]
                }
            })
        openai_req["tools"] = openai_tools

    # Tool Choice Mapping
    if "tool_choice" in claude_data:
        tc = claude_data["tool_choice"]
        if isinstance(tc, dict):
            if tc["type"] == "any":
                 openai_req["tool_choice"] = "required"
            elif tc["type"] == "auto":
                 openai_req["tool_choice"] = "auto"
            elif tc["type"] == "tool":
                 openai_req["tool_choice"] = {
                     "type": "function",
                     "function": {"name": tc["name"]}
                 }
        else:
             openai_req["tool_choice"] = tc

    # Parameters
    if "max_tokens" in claude_data:
        openai_req["max_tokens"] = claude_data["max_tokens"]
        
    if "temperature" in claude_data:
        openai_req["temperature"] = claude_data["temperature"]
        
    if "top_p" in claude_data:
        openai_req["top_p"] = claude_data["top_p"]
        
    if "stop_sequences" in claude_data:
        openai_req["stop"] = claude_data["stop_sequences"]

    return openai_req


def openai_to_claude_response(openai_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenAI non-streaming response to Claude response.
    Robust handling for missing choices, tool calls, and usage.
    """
    if not openai_data.get("choices"):
        # Fallback or error if no choices
        return {
            "id": openai_data.get("id", "error_no_id"),
            "type": "error",
            "error": {"type": "upstream_error", "message": "No choices in upstream response"}
        }

    choice = openai_data["choices"][0]
    message = choice.get("message", {})
    
    stop_reason = "end_turn"
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "stop":
        stop_reason = "end_turn"
    
    content_blocks = []
    
    # Text content
    msg_content = message.get("content")
    if msg_content:
        # OpenAI content is usually string, but strictly it could be array in some multi-modal outputs (though less common for assistant response)
        # We assume string for standard LLMs, or handle simple list
        text_val = ""
        if isinstance(msg_content, str):
            text_val = msg_content
        elif isinstance(msg_content, list):
            # Concatenate text parts
            text_val = "".join([c.get("text", "") for c in msg_content if c.get("type") == "text"])
            
        if text_val:
            content_blocks.append({
                "type": "text",
                "text": text_val
            })
        
    # Tool calls
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            if tc.get("type") == "function":
                fname = tc.get("function", {}).get("name", "unknown")
                args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except:
                    args = {"error": "failed to parse arguments", "raw": args_str}
                
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", "call_unknown"),
                    "name": fname,
                    "input": args
                })

    return {
        "id": openai_data.get("id", f"msg_{int(time.time())}"),
        "type": "message",
        "role": "assistant",
        "model": openai_data.get("model", "unknown-model"),
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_data.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_data.get("usage", {}).get("completion_tokens", 0)
        }
    }

async def openai_to_claude_stream(
    openai_stream_generator,
    fallback_model: str = "claude-3-5-sonnet",
    on_usage_done: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert OpenAI streaming chunks to Claude SSE events.
    Handles Text, Tool Calls, and Usage with improved protocol fidelity.
    """
    
    # State tracking
    has_sent_start = False
    current_block_index = 0
    text_block_started = False
    
    tool_index_to_block_index = {}
    
    # We buffer the first chunk to get the ID and Model real info
    # But we must iterate.
    
    msg_id = None
    msg_model = None
    
    accumulated_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    
    try:
        async for chunk in openai_stream_generator:
            # Handle dict or Pydantic object
            data = chunk if isinstance(chunk, dict) else chunk.model_dump() if hasattr(chunk, 'model_dump') else {}
            
            # Capture metadata from first chunk(s)
            if not msg_id and data.get("id"):
                msg_id = data["id"]
            if not msg_model and data.get("model"):
                msg_model = data["model"]
                
            # If we haven't sent start yet, and we have enough info OR we have content, send it.
            # Ideally we send it immediately on first chunk.
            if not has_sent_start:
                # Fallbacks
                final_id = msg_id or f"msg_{int(time.time())}"
                final_model = msg_model or fallback_model
                
                yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': final_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': final_model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                has_sent_start = True
    
            # Check for usage at the end (stream_options: {include_usage: true})
            if data.get("usage"):
                # Update accumulated usage
                u = data["usage"]
                accumulated_usage["input_tokens"] = u.get("prompt_tokens", 0)
                accumulated_usage["output_tokens"] = u.get("completion_tokens", 0)
                
                cache_read = 0
                details = u.get("prompt_tokens_details") or {}
                if isinstance(details, dict):
                    cache_read = details.get("cached_tokens", 0)
                if cache_read == 0:
                    cache_read = u.get("prompt_cache_hit_tokens", 0)
                accumulated_usage["cache_read_tokens"] = cache_read
    
            choices = data.get("choices", [])
            if not choices:
                continue
                
            delta = choices[0].get("delta", {})
            
            # 1. Handle Content (Text)
            content_delta = delta.get("content")
            if content_delta:
                if not text_block_started:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': current_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    text_block_started = True
                
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_block_index, 'delta': {'type': 'text_delta', 'text': content_delta}})}\n\n"
    
            # 2. Handle Tool Calls
            tool_calls_delta = delta.get("tool_calls")
            if tool_calls_delta:
                if text_block_started:
                     yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_block_index})}\n\n"
                     text_block_started = False
                     current_block_index += 1
    
                for tc in tool_calls_delta:
                    idx = tc.get("index")
                    # OpenAI indices are reliable per choice
                    
                    # New tool call start
                    if idx not in tool_index_to_block_index:
                        tool_index_to_block_index[idx] = current_block_index
                        current_block_index += 1
                        
                        t_id = tc.get("id", "")
                        t_name = tc.get("function", {}).get("name", "")
                        
                        # Emit start
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': tool_index_to_block_index[idx], 'content_block': {'type': 'tool_use', 'id': t_id, 'name': t_name, 'input': {}}})}\n\n"
                    
                    b_idx = tool_index_to_block_index[idx]
                    
                    # Args delta
                    args = tc.get("function", {}).get("arguments")
                    if args:
                         yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': b_idx, 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n\n"
    
    
            # 3. Handle Finish
            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                # Close blocks
                if text_block_started:
                     yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_block_index})}\n\n"
                     text_block_started = False
    
                for idx in tool_index_to_block_index.values():
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
    
                stop_reason = "end_turn"
                if finish_reason == "length":
                    stop_reason = "max_tokens"
                elif finish_reason == "tool_calls":
                    stop_reason = "tool_use"
                    
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': accumulated_usage['output_tokens']}})}\n\n"
                
        # Final cleanup if loop finishes naturally
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    finally:
        if on_usage_done:
            try:
                import inspect
                if inspect.iscoroutinefunction(on_usage_done):
                    await on_usage_done(accumulated_usage)
                else:
                    on_usage_done(accumulated_usage)
            except Exception as e:
                logger.warning("on_usage_done callback failed: %s", e)
