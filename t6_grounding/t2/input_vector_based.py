import asyncio
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from commons.constants import OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient

SYSTEM_PROMPT = (
    "You are a RAG-powered assistant. The user message contains two sections: RAG CONTEXT and USER QUESTION. "
    "Answer ONLY based on the provided RAG CONTEXT. "
    "If no relevant information exists in the RAG CONTEXT, state that the question cannot be answered."
)

USER_PROMPT = "RAG CONTEXT:\n{context}\n\nUSER QUESTION: {query}"


def format_user_document(user: dict[str, Any]) -> str:
    result = "User:\n"
    for key, value in user.items():
        result += f"  {key}: {value}\n"
    result += "\n"
    return result


class UserRAG:
    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self._llm_client = OpenAI(api_key=OPENAI_API_KEY)
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        users = UserServiceClient().get_all_users()
        print(f"Formatting {len(users)} user documents...")
        documents = [Document(page_content=format_user_document(user)) for user in users]
        print(f"↗️ Creating embeddings and vectorstore for {len(documents)} documents...")
        self.vectorstore = await self._create_vectorstore_with_batching(documents, batch_size=100)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        coroutines = [FAISS.afrom_documents(batch, self.embeddings) for batch in batches]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        final_vectorstore = None
        for result in results:
            if isinstance(result, Exception):
                continue
            if final_vectorstore is None:
                final_vectorstore = result
            else:
                final_vectorstore.merge_from(result)
        if final_vectorstore is None:
            raise Exception("All batches failed to process")
        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        print("Retrieving context...")
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        context_parts = []
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")
        print("=" * 100 + "\n")
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": augmented_prompt},
        ]
        response = self._llm_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=messages,
        )
        return response.choices[0].message.content or ""


async def main():
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small',
        api_key=OPENAI_API_KEY,
        dimensions=384,
    )

    async with UserRAG(embeddings) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break

            context = await rag.retrieve_context(user_question)
            augmented_prompt = rag.augment_prompt(user_question, context)
            answer = rag.generate_answer(augmented_prompt)
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce