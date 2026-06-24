import asyncio
import json
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pydantic import BaseModel, Field

from commons.constants import OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)


class HobbyEntry(BaseModel):
    hobby: str = Field(description="Hobby name, e.g. 'hiking'")
    user_ids: list[int] = Field(description="List of user IDs that mention this hobby")


class HobbyMatches(BaseModel):
    matches: list[HobbyEntry] = Field(
        description="List of hobby entries, each with a hobby name and matching user IDs"
    )


NEE_SYSTEM_PROMPT = (
    "You are a Named Entity Extraction assistant specializing in hobby detection. "
    "You will receive user profile data (each user has an ID and an about_me field) and a question. "
    "Extract hobbies relevant to the question from the about_me fields and map each hobby to the user IDs that mention it. "
    "Return structured JSON only. Do not include users who do not match."
)

NEE_USER_PROMPT = "User data:\n{context}\n\nQuestion: {query}"


class HobbySearchWizard:

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self._llm = OpenAI(api_key=OPENAI_API_KEY)
        self._user_client = UserServiceClient()
        self._vectorstore: Optional[Chroma] = None
        self._stored_ids: set[int] = set()

    async def __aenter__(self):
        print("🔎 Cold start: loading all users...")
        users = self._user_client.get_all_users()
        print(f"↗️ Creating embeddings for {len(users)} users (id + about_me only)...")
        documents = [
            Document(page_content=self._format_embed_doc(user), id=str(user["id"]))
            for user in users
        ]
        self._vectorstore = Chroma.from_documents(documents, self.embeddings)
        self._stored_ids = {user["id"] for user in users}
        print("✅ Vectorstore ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def _format_embed_doc(self, user: dict[str, Any]) -> str:
        return f"User ID: {user['id']}\nAbout: {user.get('about_me', '')}"

    def _sync_vectorstore(self, current_users: list[dict[str, Any]]):
        current_ids = {user["id"] for user in current_users}

        to_delete = self._stored_ids - current_ids
        if to_delete:
            print(f"🗑️ Removing {len(to_delete)} deleted users from vectorstore...")
            self._vectorstore.delete(ids=[str(i) for i in to_delete])

        to_add = current_ids - self._stored_ids
        if to_add:
            print(f"➕ Adding {len(to_add)} new users to vectorstore...")
            new_users = [u for u in current_users if u["id"] in to_add]
            new_docs = [
                Document(page_content=self._format_embed_doc(user), id=str(user["id"]))
                for user in new_users
            ]
            self._vectorstore.add_documents(new_docs)

        self._stored_ids = current_ids

    async def retrieve_context(self, query: str, k: int = 20, score: float = 0.1) -> str:
        print("🔄 Syncing vectorstore with live user data...")
        current_users = self._user_client.get_all_users()
        self._sync_vectorstore(current_users)

        print("Retrieving context...")
        results = self._vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        context_parts = []
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content[:80]}...")
        print("=" * 100 + "\n")
        return "\n\n".join(context_parts)

    def extract_entities(self, query: str, context: str) -> HobbyMatches:
        print("🧠 Extracting named entities (hobbies → user IDs)...")
        messages = [
            {"role": "system", "content": NEE_SYSTEM_PROMPT},
            {"role": "user", "content": NEE_USER_PROMPT.format(context=context, query=query)},
        ]
        response = self._llm.beta.chat.completions.parse(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=messages,
            response_format=HobbyMatches,
        )
        return response.choices[0].message.parsed

    async def ground_output(self, hobby_matches: HobbyMatches) -> dict[str, list[dict[str, Any]]]:
        print("✅ Output grounding: verifying user IDs and fetching full profiles...")
        result: dict[str, list[dict[str, Any]]] = {}
        for entry in hobby_matches.matches:
            verified_users = []
            for user_id in entry.user_ids:
                try:
                    full_user = await self._user_client.get_user(user_id)
                    verified_users.append(full_user)
                except Exception:
                    print(f"  ⚠️ User ID {user_id} not found — skipping")
            if verified_users:
                result[entry.hobby] = verified_users
        return result


async def main():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
        dimensions=384,
    )

    async with HobbySearchWizard(embeddings) as wizard:
        print("\nQuery samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find users interested in painting and photography")

        while True:
            user_question = input("> ").strip()
            if not user_question:
                continue
            if user_question.lower() in ["quit", "exit"]:
                break

            context = await wizard.retrieve_context(user_question)
            if not context:
                print("\n--- No relevant users found ---\n")
                continue

            hobby_matches = wizard.extract_entities(user_question, context)
            result = await wizard.ground_output(hobby_matches)
            print("\n=== RESULTS ===")
            print(json.dumps(result, indent=2))
            print()


if __name__ == "__main__":
    asyncio.run(main())
