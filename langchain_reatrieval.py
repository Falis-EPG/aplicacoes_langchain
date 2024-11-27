from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.globals import set_debug
from pydantic import Field, BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()
set_debug(True)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    convert_system_message_to_human=True,
    timeout=None,
    max_retries=2,
    GOOGLE_API_KEY=os.getenv("GEMINI_API_KEY")
)

carregador = TextLoader("file.txt", encoding="utf-8")
documento = carregador.load()

quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
textos = quebrador.split_documents(documento)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", GOOGLE_API_KEY=os.getenv("GEMINI_API_KEY"))
db = FAISS.from_documents(textos, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

question = "Quem irá preparar minha assinatura de e-mail e me entregar na empresa?"
resultado = qa_chain.invoke({"query":question})
print(resultado)