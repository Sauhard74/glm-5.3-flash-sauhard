import json

ARCHS = ("qwen", "glm")


def sse_events(text: str):
    events = []
    for chunk in text.split("\n\n"):
        data = [line[5:].strip() for line in chunk.splitlines() if line.startswith("data:")]
        if data and data[0] != "[DONE]":
            events.append(json.loads(data[0]))
    return events
