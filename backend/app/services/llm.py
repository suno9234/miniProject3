# LLM 설정 및 structured output 정의
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from app.models import AgentRoute
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

INTERNAL_LLM_MODEL = os.getenv(INTERNAL_LLM_MODEL,"qwen/Qwen1.5-0.5B-Chat")

# 기본 LLM
llm = ChatOpenAI(
    model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
    temperature=float(os.getenv('TEMPERATURE', '0.0')),
    api_key=os.getenv('OPENAI_API_KEY')
)

tokenizer = AutoTokenizer.from_pretrained(INTERNAL_LLM_MODEL)
internal_llm = AutoModelForCausalLM.from_pretrained(INTERNAL_LLM_MODEL, device_map="auto")

internal_pipe = pipeline("text-generation", model=internal_llm, tokenizer=tokenizer, max_new_tokens=256)

# 플래너용 structured output LLM
llm_with_agent_route = llm.with_structured_output(AgentRoute)