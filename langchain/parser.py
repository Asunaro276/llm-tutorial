from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

message = AIMessage(content="こんにちは")
s = "こんばんは"

parser = StrOutputParser()

print(parser.invoke(message))
print(parser.invoke(s))
