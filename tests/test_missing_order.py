
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mock_llm

messages = [
    {
        "role": "system",
        "content": "You are an AGENT. Help customers with refund requests."
    },
    {
        "role": "user",
        "content": "I need a refund. I don't have my order ID."
    }
]

response = mock_llm.chat(messages)

message = response.choices[0].message

print("\n========== MISSING ORDER ID TEST ==========")
print("Agent response:", message.content)
print("Finish reason:", response.choices[0].finish_reason)
