from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()
set_debug(True)

class Destino(BaseModel):
    cidade = Field("Cidade a visitar")
    motivo = Field("Motivo pelo qual é interessante")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

parseador = JsonOutputParser(pydantic_object=Destino)

modelo_cidade = PromptTemplate(
    template="""Sugira uma cidade dado meu interesse por {interesse}.
    {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parseador.get_format_instructions()}
)

modelo_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)

modelo_final = ChatPromptTemplate.from_messages(
    [
    ("ai", "Sugestão de viagem para a cidade: {cidade}"),
    ("ai", "Restaurantes que você não pode perder: {restaurantes}"),
    ("ai", "Atividades e locais recomendados: {locais_culturais}"),
    ("human", "Por Favor, Combine as informações das cadeias anteriores em 2 parágrafos coerentes.")
    ]
)

parte_1 = modelo_cidade | llm | parseador
parte_2 = modelo_restaurantes | llm | StrOutputParser()
parte_3 = modelo_cultural | llm | StrOutputParser()
parte_4 = modelo_final | llm | StrOutputParser()

cadeia = (parte_1 | {
    "restaurantes": parte_2,
    "locais_culturais": parte_3,
    "cidade": itemgetter("cidade")
} | parte_4)

resultado = cadeia.invoke({"interesse": "praias"})
print(resultado)