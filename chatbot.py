from langchain_ollama import ChatOllama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import load_and_split_pdf, format_docs
from config import MODEL_NAME, EMBEDDING_DB_PATH, MBTI_FEATURES, PDF_PATHS

class MBTIChatBot:
    def __init__(self, mbti, temperature=0):
        self.mbti = mbti
        self.feature = MBTI_FEATURES[mbti]
        self.pdf_path = PDF_PATHS[mbti]
        self.embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        self.db = Chroma(
            persist_directory=f"{EMBEDDING_DB_PATH}{mbti}_chroma_db",
            embedding_function=self.embeddings
        )
        self.llm = ChatOllama(model="anpigon/eeve-korean-10.8b", temperature=temperature)
        self.chat_template = ChatPromptTemplate.from_template(
            """
            '''
            mbti
            {mbti}
            '''
            '''
            특징
            {feature}
            '''
            '''
            대화 예시
            {context}
            '''
            '''
            질문
            {query}
            '''

            너는 {mbti} 성격을 가진 사람처럼 행동해야 해.
            아래 {feature}를 참고해서 {mbti}의 성격을 바탕으로 자연스럽고 일상적인 대화를 이어가줘.
            대답은 반드시 가볍고 편안한 반말로 하고, 필요하면 상대방의 질문을 확장해서 더 친근하게 답변해줘.
            """
        )

    def initialize_db(self):
        docs = load_and_split_pdf(self.pdf_path)
        self.db.add_documents(docs)

    def get_response(self, query):
        retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8}
        )
        relevant_docs = retriever.get_relevant_documents(query)
        context = format_docs(relevant_docs)
        input_data = {
            "mbti": self.mbti,
            "feature": self.feature,
            "context": context,
            "query": query
        }
        chain = self.chat_template | self.llm | StrOutputParser()
        return chain.invoke(input_data)