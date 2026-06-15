import os

from commons.constants import OPENAI_API_KEY, OPENAI_EMBEDDINGS_ENDPOINT, OPENAI_CHAT_COMPLETIONS_ENDPOINT
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t5_rag_advanced.chat.chat_completion_client import ChatCompletionClient
from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.embeddings.text_processor import TextProcessor, SearchMode

SYSTEM_PROMPT = """You are a RAG-powered assistant for microwave oven usage questions.
Each user message contains two sections: RAG Context (retrieved from the knowledge base) and User Question.
Answer the User Question using ONLY the provided RAG Context and conversation history.
Do not answer questions that are unrelated to microwave usage, not covered by the RAG Context, or outside the scope of the conversation history.
"""

USER_PROMPT = """## RAG Context
{context}

## User Question
{question}
"""

if __name__ == "__main__":
    embeddings_client = EmbeddingsClient(
        endpoint=OPENAI_EMBEDDINGS_ENDPOINT,
        model_name="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    chat_completion_client = ChatCompletionClient(
        endpoint=OPENAI_CHAT_COMPLETIONS_ENDPOINT,
        model_name="gpt-5.2",
        api_key=OPENAI_API_KEY,
    )
    text_processor = TextProcessor(
        embeddings_client=embeddings_client,
        db_config={
            "host": "localhost",
            "port": 5433,
            "database": "vectordb",
            "user": "postgres",
            "password": "postgres",
        },
    )

    manual_path = os.path.join(os.path.dirname(__file__), "embeddings", "microwave_manual.txt")
    print("Loading document into vector DB...")
    text_processor.process_text_file(
        manual_path,
        chunk_size=300,
        overlap=50,
        dimensions=384,
        truncate=True,
    )
    print("Document loaded. Starting chat (Ctrl+C to exit).\n")

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    while True:
        user_input = input("You: ")
        context_chunks = text_processor.search(
            SearchMode.COSINE_DISTANCE,
            user_input,
            top_k=4,
            min_score=0.5,
            dimensions=384,
        )
        context = "\n\n".join(context_chunks)
        augmented = USER_PROMPT.format(context=context, question=user_input)
        conversation.add_message(Message(Role.USER, augmented))
        response = chat_completion_client.get_completion(conversation.get_messages())
        conversation.add_message(response)
        print(f"Assistant: {response.content}\n")
