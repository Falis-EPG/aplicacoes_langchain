from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

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

mensagens = [
        "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
        "Qual é o melhor período do ano para visitar em termos de clima?",
        "Quais tipos de atividades ao ar livre estão disponíveis?",
        "Alguma sugestão de acomodação eco-friendly por lá?",
        "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
        "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

longa_conversa = ""
for mensagem in mensagens:
    longa_conversa = f"Usuário: {mensagem}\n"
    longa_conversa += f"AI: "

    modelo = PromptTemplate(template=longa_conversa, 
                            input_variables=[""])
    cadeia = modelo | llm | StrOutputParser()
    resposta = cadeia.invoke(input={})

    longa_conversa += resposta + "\n"
    print(longa_conversa)