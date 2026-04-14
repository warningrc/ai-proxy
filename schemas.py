from typing import List, Union, Optional, Any, Literal, Dict
from pydantic import BaseModel, Field, field_validator

class ClaudeContentBlock(BaseModel):
    type: Literal["text", "image", "tool_use", "tool_result"]
    text: Optional[str] = None
    # Image fields
    source: Optional[dict] = None 
    # Tool Use fields
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None
    # Tool Result fields
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None # Recursive definition simplified
    is_error: Optional[bool] = False

    @field_validator('source')
    def validate_source(cls, v):
        if v:
            if v.get('type') != 'base64':
                raise ValueError("Only base64 image source is supported")
            media_type = v.get('media_type')
            if media_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                raise ValueError(f"Unsupported media_type: {media_type}")
        return v

class ClaudeMessage(BaseModel):
    role: str
    content: Union[str, List[ClaudeContentBlock]]

class ClaudeTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ClaudeChatRequest(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[Union[dict, str]] = None
    metadata: Optional[dict] = None
