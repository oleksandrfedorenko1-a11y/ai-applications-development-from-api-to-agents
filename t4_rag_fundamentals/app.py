import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from commons.constants import OPENAI_API_KEY

_SYSTEM_PROMPT = """You are an expert assistant specializing in microwave oven usage and troubleshooting.

Each user message contains two blocks:
- ##RAG CONTEXT: relevant excerpts retrieved from the microwave manual based on the user's question
- ##USER QUESTION: the user's actual question

Instructions:
- Answer using ONLY the information provided in ##RAG CONTEXT and the current conversation.
- If the answer cannot be found in ##RAG CONTEXT, respond with: "I don't have information about that in the microwave manual."
- Never answer questions unrelated to the microwave manual content.
"""

_USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """
        Load existing FAISS index from disk or create a new one.
        Returns:
              VectorStore: Initialized FAISS vectorstore.
        """
        print("Setting up vectorstore...")
        index_path = os.path.join(os.path.dirname(__file__), "microwave_faiss_index")
        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            vectorstore = self._create_new_index()
        return vectorstore

    def _create_new_index(self) -> VectorStore:
        """
        Load the manual, split into chunks, embed, and save a new FAISS index.
        Returns:
              VectorStore: Newly created and saved FAISS vectorstore.
        """
        manual_path = os.path.join(os.path.dirname(__file__), "microwave_manual.txt")
        docs = TextLoader(manual_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "."])
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        index_path = os.path.join(os.path.dirname(__file__), "microwave_faiss_index")
        vectorstore.save_local(index_path)
        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3):
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        chunks = []
        for doc, relevance_score in results:
            print(f"Relevance score: {relevance_score:.4f}")
            chunks.append(doc.page_content)
        return "\n\n".join(chunks)

    def augment_prompt(self, query: str, context: str):
        """
        Inject retrieved context and user query into the prompt template.
        Args:
              query (str): The user's question.
              context (str): Retrieved context from the vectorstore.
        Returns:
              str: Formatted prompt ready for the LLM.
        """
        augmented = _USER_PROMPT.format(context=context, query=query)
        print(augmented)
        return augmented

    def generate_answer(self, augmented_prompt: str):
        """
        Send the augmented prompt to the LLM and return its response.
        Args:
              augmented_prompt (str): The prompt with injected context and query.
        Returns:
              str: The LLM-generated answer.
        """
        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=augmented_prompt)]
        response = self.llm_client.invoke(messages)
        print(response.content)
        return response.content


def main(rag: MicrowaveRAG):
    print("Welcome to the Microwave Manual RAG assistant! Ask any question about your microwave.")
    while True:
        query = input("You: ")
        context = rag.retrieve_context(query)
        augmented_prompt = rag.augment_prompt(query, context)
        rag.generate_answer(augmented_prompt)


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=SecretStr(OPENAI_API_KEY))
    llm_client = ChatOpenAI(temperature=0.0, model='gpt-5.2', api_key=SecretStr(OPENAI_API_KEY))
    rag = MicrowaveRAG(embeddings=embeddings, llm_client=llm_client)
    main(rag)