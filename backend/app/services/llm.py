# LLM 설정 및 structured output 정의
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from app.models import AgentRoute
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
ENVIRONMENT = os.getenv("ENVIRONMENT", "test")

if ENVIRONMENT == "operate":
    # In production, use a different model as needed.
    INTERNAL_LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    device_map = "cuda"
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(INTERNAL_LLM_MODEL, token=HF_TOKEN)
    internal_llm = AutoModelForCausalLM.from_pretrained(
        INTERNAL_LLM_MODEL,
        device_map=device_map,
        quantization_config=quantization_config,
        token=HF_TOKEN
    )
else:
    INTERNAL_LLM_MODEL = "qwen/Qwen1.5-0.5B-Chat"
    device_map = "auto"
    tokenizer = AutoTokenizer.from_pretrained(INTERNAL_LLM_MODEL)
    internal_llm = AutoModelForCausalLM.from_pretrained(INTERNAL_LLM_MODEL, device_map=device_map)


load_dotenv()

# 기본 LLM
llm = ChatOpenAI(
    model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
    temperature=float(os.getenv('TEMPERATURE', '0.0')),
    api_key=os.getenv('OPENAI_API_KEY')
)


internal_pipe = pipeline("text-generation", model=internal_llm, tokenizer=tokenizer, max_new_tokens=256)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=600)

# 플래너용 structured output LLM
llm_with_agent_route = llm.with_structured_output(AgentRoute)