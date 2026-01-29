# Add references
import asyncio
from typing import cast
import os
from dotenv import load_dotenv
from agent_framework import ChatMessage, Role, ConcurrentBuilder, WorkflowOutputEvent
from agent_framework_azure_ai import AzureAIAgentsProvider
from azure.identity.aio import AzureCliCredential
from agent_framework import AgentExecutor

# Load environment variables from .env file
load_dotenv()
project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
model_deployment = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")

async def main():

    # 1) Create three domain agents using AzureChatClient
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentsProvider(credential=credential) as client,
    ):


        researcher = await client.create_agent(
            instructions=(
                "You're an expert market and product researcher. Given a prompt, provide concise, factual insights,"
                " opportunities, and risks."
            ),
            name="researcher",
        )

        marketer = await client.create_agent(
            instructions=(
                "You're a creative marketing strategist. Craft compelling value propositions and target messaging"
                " aligned to the prompt."
            ),
            name="marketer",
        )

        legal = await client.create_agent(
            instructions=(
                "You're a cautious legal/compliance reviewer. Highlight constraints, disclaimers, and policy concerns"
                " based on the prompt."
            ),
            name="legal",
        )

        workflow = ConcurrentBuilder().participants([researcher, marketer, legal]).build()

        output_evt: WorkflowOutputEvent  | None = None
        async for event in workflow.run_stream("We are launching a new budget-friendly electric bike for urban commuters."):
            if isinstance(event, WorkflowOutputEvent):
                output_evt = event

        if output_evt:
            print("===== Final Aggregated Conversation (messages) =====")
            messages: list[ChatMessage] | Any = output_evt.data
            for i, msg in enumerate(messages, start=1):
                name = msg.author_name if msg.author_name else "user"
                print(f"{'-' * 60}\n\n{i:02d} [{name}]:\n{msg.text}")
    
    
    
if __name__ == "__main__":
    asyncio.run(main())