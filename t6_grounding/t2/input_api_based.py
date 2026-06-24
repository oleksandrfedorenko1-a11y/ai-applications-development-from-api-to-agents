from enum import StrEnum
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from commons.constants import OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient

QUERY_ANALYSIS_PROMPT = (
    "You are a query analysis system. Available search fields: name, surname, email. "
    "Analyze the user question and extract explicit search values. "
    "Map extracted values to the appropriate search fields. "
    "Only extract values that are clearly stated — do not infer or assume. "
    'Examples: "Who is John?" → name: "John"; "Find John Smith" → name: "John", surname: "Smith"; '
    '"Find user with email john@example.com" → email: "john@example.com".'
)

SYSTEM_PROMPT = (
    "You are a RAG-powered assistant. The user message contains two sections: RAG CONTEXT and USER QUESTION. "
    "Answer ONLY based on the provided RAG CONTEXT. "
    "If no relevant information exists in the RAG CONTEXT, state that the question cannot be answered. "
    "Format user information clearly when presenting it."
)

USER_PROMPT = "RAG CONTEXT:\n{context}\n\nUSER QUESTION: {query}"


class SearchField(StrEnum):
    NAME = "name"
    SURNAME = "surname"
    EMAIL = "email"


class SearchRequest(BaseModel):
    search_field: SearchField = Field(description="Search field")
    search_value: str = Field(description="Search value. Sample: Adam.")


class SearchRequests(BaseModel):
    search_request_parameters: list[SearchRequest] = Field(
        description="List of search parameters to execute",
        default_factory=list
    )


llm_client = OpenAI(api_key=OPENAI_API_KEY)

user_client = UserServiceClient()


def retrieve_context(user_question: str) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": QUERY_ANALYSIS_PROMPT},
        {"role": "user", "content": user_question},
    ]
    response = llm_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=messages,
        response_format=SearchRequests,
    )
    params = response.choices[0].message.parsed.search_request_parameters
    if params:
        params_dict = {p.search_field.value: p.search_value for p in params}
        print(f"Searching with parameters: {params_dict}")
        return user_client.search_users(**params_dict)
    print("No specific search parameters found!")
    return []


def augment_prompt(user_question: str, context: list[dict[str, Any]]) -> str:
    formatted = ""
    for user in context:
        formatted += "User:\n"
        for key, value in user.items():
            formatted += f"  {key}: {value}\n"
        formatted += "\n"
    augmented = USER_PROMPT.format(context=formatted, query=user_question)
    print(augmented)
    return augmented


def generate_answer(augmented_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": augmented_prompt},
    ]
    response = llm_client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def main():
    print("Query samples:")
    print(" - I need user emails that filled with hiking and psychology")
    print(" - Who is John?")
    print(" - Find users with surname Adams")
    print(" - Do we have smbd with name John that love painting?")

    while True:
        user_question = input("> ").strip()
        if user_question:
            if user_question.lower() in ['quit', 'exit']:
                break

            print("\n--- Retrieving context ---")
            context = retrieve_context(user_question)
            if context:
                print("\n--- Augmenting prompt ---")
                augmented_prompt = augment_prompt(user_question, context)
                print("\n--- Generating answer ---")
                answer = generate_answer(augmented_prompt)
                print(f"\nAnswer: {answer}\n")
            else:
                print("\n--- No relevant information found ---")


if __name__ == "__main__":
    main()


# The problems with API based Grounding approach are:
#   - We need a Pre-Step to figure out what field should be used for search (Takes time)
#   - Values for search should be correct (✅ John -> ❌ Jonh)
#   - Is not so flexible
# Benefits are:
#   - We fetch actual data (new users added and deleted every 5 minutes)
#   - Costs reduce