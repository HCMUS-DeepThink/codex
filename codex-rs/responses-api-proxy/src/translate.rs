use anyhow::Context;
use anyhow::Result;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;

const DEFAULT_CHAT_MODEL: &str = "gpt-4o-mini";
const DEFAULT_RESPONSE_ID: &str = "chatcmpl-proxy";
const DEFAULT_FUNCTION_NAME: &str = "tool_call";

pub struct MappedRequestBody {
    pub body: Vec<u8>,
    pub stream: bool,
}

pub fn responses_to_chat_completions_request(body: &[u8]) -> Result<MappedRequestBody> {
    let request: Value = serde_json::from_slice(body).context("parsing responses request json")?;
    let model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_CHAT_MODEL);
    let stream = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut messages = Vec::new();

    if let Some(instructions) = request.get("instructions").and_then(Value::as_str)
        && !instructions.is_empty()
    {
        messages.push(json!({
            "role": "system",
            "content": instructions,
        }));
    }

    for item in request
        .get("input")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(item_type) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        match item_type {
            "message" => {
                let role = item
                    .get("role")
                    .and_then(Value::as_str)
                    .map(|role| if role == "developer" { "system" } else { role })
                    .unwrap_or("user");
                let content = message_content_to_chat_content(item.get("content"));
                messages.push(json!({
                    "role": role,
                    "content": content,
                }));
            }
            "function_call_output" | "custom_tool_call_output" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let content = function_call_output_to_text(item.get("output"));
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }));
            }
            "mcp_tool_call_output" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let content = item
                    .get("output")
                    .cloned()
                    .unwrap_or(Value::Null)
                    .to_string();
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }));
            }
            "function_call" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(DEFAULT_FUNCTION_NAME);
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                messages.push(json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    }],
                }));
            }
            _ => {}
        }
    }

    let tools = request.get("tools").cloned().unwrap_or_else(|| json!([]));
    let tool_choice = request
        .get("tool_choice")
        .cloned()
        .unwrap_or_else(|| Value::String("auto".to_string()));
    let parallel_tool_calls = request
        .get("parallel_tool_calls")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let mut mapped = json!({
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "stream": stream,
    });

    if stream {
        mapped["stream_options"] = json!({ "include_usage": true });
    }

    if let Some(text) = request.get("text")
        && let Some(format) = text.get("format")
        && let Some(schema) = format.get("schema")
    {
        let name = format
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("response_format");
        let strict = format
            .get("strict")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        mapped["response_format"] = json!({
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
                "strict": strict,
            },
        });
    }

    Ok(MappedRequestBody {
        body: serde_json::to_vec(&mapped).context("serializing chat completions request")?,
        stream,
    })
}

pub fn chat_completions_json_to_responses_json(body: &[u8]) -> Result<Vec<u8>> {
    let chat_response: Value =
        serde_json::from_slice(body).context("parsing chat completions response json")?;
    let response_id = chat_response
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_RESPONSE_ID)
        .to_string();
    let model = chat_response
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let mut output_items = Vec::new();
    let message = chat_response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"));

    if let Some(message) = message {
        let text = chat_message_text(message.get("content"));
        if !text.is_empty() {
            output_items.push(json!({
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": text,
                }],
            }));
        }

        for tool_call in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let call_id = tool_call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let function = tool_call.get("function").unwrap_or(&Value::Null);
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(DEFAULT_FUNCTION_NAME);
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            output_items.push(json!({
                "type": "function_call",
                "name": name,
                "arguments": arguments,
                "call_id": call_id,
            }));
        }
    }

    let usage = chat_response.get("usage");
    let prompt_tokens = usage
        .and_then(|value| value.get("prompt_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let completion_tokens = usage
        .and_then(|value| value.get("completion_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let total_tokens = usage
        .and_then(|value| value.get("total_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(prompt_tokens + completion_tokens);

    let response = json!({
        "id": response_id,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": output_items,
        "usage": {
            "input_tokens": prompt_tokens,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": completion_tokens,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": total_tokens,
        },
    });
    serde_json::to_vec(&response).context("serializing mapped responses json")
}

pub fn chat_completions_sse_to_responses_sse(body: &[u8]) -> Result<Vec<u8>> {
    let text = std::str::from_utf8(body).context("reading chat completions sse as utf-8")?;
    let mut out = String::new();
    let mut emitted_created = false;
    let mut response_id = DEFAULT_RESPONSE_ID.to_string();
    let mut assistant_text = String::new();
    let mut usage = None::<Value>;
    let mut tool_calls: BTreeMap<usize, ToolCallState> = BTreeMap::new();

    for block in text.split("\n\n") {
        for line in block.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload.trim() == "[DONE]" {
                continue;
            }

            let chunk: Value = match serde_json::from_str(payload) {
                Ok(chunk) => chunk,
                Err(_) => continue,
            };

            if let Some(id) = chunk.get("id").and_then(Value::as_str) {
                response_id = id.to_string();
            }
            if !emitted_created {
                emitted_created = true;
                push_sse(
                    &mut out,
                    json!({
                        "type": "response.created",
                        "response": { "id": response_id },
                    }),
                )?;
            }

            if let Some(chunk_usage) = chunk.get("usage") {
                usage = Some(chunk_usage.clone());
            }

            let Some(choice) = chunk
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
            else {
                continue;
            };

            if let Some(content) = choice
                .get("delta")
                .and_then(|delta| delta.get("content"))
                .and_then(Value::as_str)
            {
                assistant_text.push_str(content);
                push_sse(
                    &mut out,
                    json!({
                        "type": "response.output_text.delta",
                        "delta": content,
                    }),
                )?;
            }

            for tool_delta in choice
                .get("delta")
                .and_then(|delta| delta.get("tool_calls"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let index = tool_delta
                    .get("index")
                    .and_then(Value::as_u64)
                    .map(|index| index as usize)
                    .unwrap_or(0);
                let state = tool_calls.entry(index).or_default();
                if let Some(id) = tool_delta.get("id").and_then(Value::as_str) {
                    state.id = id.to_string();
                }
                if let Some(name) = tool_delta
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                {
                    state.name = name.to_string();
                }
                if let Some(arguments) = tool_delta
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(Value::as_str)
                {
                    state.arguments.push_str(arguments);
                }
            }

            if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str)
                && (finish_reason == "stop"
                    || finish_reason == "length"
                    || finish_reason == "tool_calls")
            {
                if !assistant_text.is_empty() {
                    push_sse(
                        &mut out,
                        json!({
                            "type": "response.output_item.done",
                            "item": {
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": assistant_text,
                                }],
                            },
                        }),
                    )?;
                    assistant_text.clear();
                }

                for state in tool_calls.values() {
                    push_sse(
                        &mut out,
                        json!({
                            "type": "response.output_item.done",
                            "item": {
                                "type": "function_call",
                                "name": state.name,
                                "arguments": state.arguments,
                                "call_id": state.id,
                            },
                        }),
                    )?;
                }
                tool_calls.clear();
            }
        }
    }

    if !emitted_created {
        push_sse(
            &mut out,
            json!({
                "type": "response.created",
                "response": { "id": response_id },
            }),
        )?;
    }

    let prompt_tokens = usage
        .as_ref()
        .and_then(|value| value.get("prompt_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let completion_tokens = usage
        .as_ref()
        .and_then(|value| value.get("completion_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let total_tokens = usage
        .as_ref()
        .and_then(|value| value.get("total_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(prompt_tokens + completion_tokens);

    push_sse(
        &mut out,
        json!({
            "type": "response.completed",
            "response": {
                "id": response_id,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "input_tokens_details": { "cached_tokens": 0 },
                    "output_tokens": completion_tokens,
                    "output_tokens_details": { "reasoning_tokens": 0 },
                    "total_tokens": total_tokens,
                },
            },
        }),
    )?;

    Ok(out.into_bytes())
}

#[derive(Default)]
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
}

fn push_sse(out: &mut String, event: Value) -> Result<()> {
    out.push_str("data: ");
    out.push_str(&serde_json::to_string(&event).context("serializing mapped sse event")?);
    out.push_str("\n\n");
    Ok(())
}

fn function_call_output_to_text(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    match value {
        Value::String(content) => content.to_string(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => value.to_string(),
    }
}

fn chat_message_text(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        Value::String(text) => text.to_string(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                let content_type = part.get("type").and_then(Value::as_str);
                if content_type == Some("text") {
                    part.get("text").and_then(Value::as_str).map(str::to_string)
                } else {
                    part.get("text")
                        .and_then(|text| text.get("value"))
                        .and_then(Value::as_str)
                        .map(str::to_string)
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn message_content_to_chat_content(content: Option<&Value>) -> Value {
    let Some(items) = content.and_then(Value::as_array) else {
        return Value::String(String::new());
    };

    let mut parts = Vec::new();
    let mut has_image = false;
    for item in items {
        let Some(item_type) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        match item_type {
            "input_text" | "output_text" => {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    parts.push(json!({
                        "type": "text",
                        "text": text,
                    }));
                }
            }
            "input_image" => {
                if let Some(image_url) = item.get("image_url").and_then(Value::as_str) {
                    has_image = true;
                    parts.push(json!({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    }));
                }
            }
            _ => {}
        }
    }

    if parts.is_empty() {
        Value::String(String::new())
    } else if !has_image && parts.len() == 1 {
        let part = &parts[0];
        if part.get("type").and_then(Value::as_str) == Some("text") {
            Value::String(
                part.get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )
        } else {
            Value::Array(parts)
        }
    } else {
        Value::Array(parts)
    }
}

#[cfg(test)]
mod tests {
    use super::chat_completions_sse_to_responses_sse;
    use super::responses_to_chat_completions_request;
    use pretty_assertions::assert_eq;
    use serde_json::Value;
    use serde_json::json;

    #[test]
    fn map_responses_request_to_chat_completions() {
        let request = json!({
            "model": "gpt-4.1",
            "instructions": "Be brief",
            "stream": true,
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "foo",
                    "description": "bar",
                    "parameters": {"type":"object","properties":{}}
                }
            }],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type":"input_text","text":"Hello"}]
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "done"
                }
            ]
        });
        let mapped = responses_to_chat_completions_request(
            serde_json::to_string(&request).unwrap().as_bytes(),
        )
        .unwrap();

        let body: Value = serde_json::from_slice(&mapped.body).unwrap();
        assert!(mapped.stream);
        assert_eq!(body["model"], "gpt-4.1");
        assert!(body["stream"].as_bool().unwrap());
        assert!(body["stream_options"]["include_usage"].as_bool().unwrap());
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_1");
    }

    #[test]
    fn map_chat_stream_to_responses_stream() {
        let sse = concat!(
            "data: {\"id\":\"chatcmpl_1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"foo\",\"arguments\":\"{\\\"a\\\":\"}}]},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"index\":0,\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
            "data: [DONE]\n\n"
        );
        let mapped = chat_completions_sse_to_responses_sse(sse.as_bytes()).unwrap();
        let mapped_text = String::from_utf8(mapped).unwrap();
        assert!(mapped_text.contains("\"type\":\"response.created\""));
        assert!(mapped_text.contains("\"type\":\"response.output_text.delta\""));
        assert!(mapped_text.contains("\"type\":\"response.output_item.done\""));
        assert!(mapped_text.contains("\"type\":\"function_call\""));
        assert!(mapped_text.contains("\"call_id\":\"call_1\""));
        assert!(mapped_text.contains("\"type\":\"response.completed\""));
    }
}
