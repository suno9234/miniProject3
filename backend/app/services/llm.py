# LLM 설정 및 structured output 정의
import os
from langchain_openai import ChatOpenAI
from app.models import AgentRoute

# 기본 LLM
llm = ChatOpenAI(
    model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
    temperature=float(os.getenv('TEMPERATURE', '0.0')),
    api_key=os.getenv('OPENAI_API_KEY')
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=600)


# 플래너용 structured output LLM
llm_with_agent_route = llm.with_structured_output(AgentRoute)