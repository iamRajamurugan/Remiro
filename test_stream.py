from dotenv import load_dotenv
load_dotenv()
from src.config import get_settings
from src.llm_service import CareerGuideLLM

settings = get_settings()
llm = CareerGuideLLM(
    google_api_key=settings.google_api_key,
    model_name=settings.gemini_model,
    serper_api_key=settings.serper_api_key,
)
history = [{'role': 'user', 'content': 'Hi, who are you? Just answer in 1 sentence.'}]
for chunk in llm.stream_reply(history):
    print(chunk, end='', flush=True)
print('\nDone')
