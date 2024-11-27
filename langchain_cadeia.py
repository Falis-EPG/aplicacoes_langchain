from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.globals import set_debug
import os
from dotenv import load_dotenv

load_dotenv()
set_debug(True)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

modelo_cidade = ChatPromptTemplate.from_template(
    "Sugira uma cidade dado meu interesse por {interesse}. A sua saída deve ser SOMENTE o nome da cidade. Cidade:"
)

modelo_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)

cadeia_cidade = LLMChain(prompt=modelo_cidade, llm=llm)
cadeia_restaurantes= LLMChain(prompt=modelo_restaurantes, llm=llm)
cadeia_cultural = LLMChain(prompt=modelo_cultural, llm=llm)

cadeia = SimpleSequentialChain(chains=[cadeia_cidade, cadeia_restaurantes, cadeia_cultural],
                               verbose=True)

resultado = cadeia.invoke("Praias")
print(resultado)