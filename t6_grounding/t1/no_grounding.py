import asyncio
from typing import Any

from openai import AsyncOpenAI

from commons.constants import OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient

BATCH_SYSTEM_PROMPT = (
    "You are a user search assistant. Analyze the search criteria in the user question. "
    "Examine each user in the provided list and determine if they match. "
    "Return the full details of matching users in their original format. "
    "If no users match, return exactly: NO_MATCHES_FOUND"
)

FINAL_SYSTEM_PROMPT = (
    "You are a search results compiler. Review all batch search results provided. "
    "Combine and deduplicate matching users found across batches. "
    "Present the final results in a clear, organized manner."
)

USER_PROMPT = "Users:\n{context}\n\nQuestion: {query}"


class TokenTracker:

    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "batch_count": len(self.batch_tokens),
            "batch_tokens": self.batch_tokens,
        }


llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    result = ""
    for user in context:
        result += "User:\n"
        for key, value in user.items():
            result += f"  {key}: {value}\n"
        result += "\n"
    return result


async def generate_response(system_prompt: str, user_message: str, model: str = "gpt-4o-mini") -> str:
    print("Processing...")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    response = await llm_client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=messages,
    )
    total_tokens = response.usage.total_tokens if response.usage else 0
    token_tracker.add_tokens(total_tokens)
    content = response.choices[0].message.content or ""
    print(f"{content}\n[tokens: {total_tokens}]")
    return content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()

    if user_question:
        print("\n--- Searching user database ---")
        users = UserServiceClient().get_all_users()
        batches = [users[i:i + 100] for i in range(0, len(users), 100)]

        coroutines = [
            generate_response(BATCH_SYSTEM_PROMPT, USER_PROMPT.format(context=join_context(batch), query=user_question))
            for batch in batches
        ]
        batch_results = await asyncio.gather(*coroutines)

        print("\n--- Compiling results ---")
        relevant_results = [r for r in batch_results if r.strip() != "NO_MATCHES_FOUND"]

        print("\n=== SEARCH RESULTS ===")
        if relevant_results:
            combined_results = "\n\n".join(relevant_results)
            await generate_response(
                FINAL_SYSTEM_PROMPT,
                f"{combined_results}\n\nOriginal question: {user_question}",
                model="gpt-4o",
            )
        else:
            print("No users found matching your criteria.")
            print("Try refining your search with different keywords.")

        summary = token_tracker.get_summary()
        print(f"\n=== Performance ===")
        print(f"Total API calls: {summary['batch_count']}")
        print(f"Total tokens:    {summary['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation