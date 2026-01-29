# Add references
import asyncio
from typing import cast
import os
from dotenv import load_dotenv
from agent_framework import ChatMessage, Role, GroupChatBuilder, GroupChatState, WorkflowOutputEvent,GroupChatResponseReceivedEvent
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
                "Gather concise facts that help answer the question. Be brief and factual."
            ),
            name="Researcher",
            description= "Collects relevant background information."
        )

        writer = await client.create_agent(
            instructions=(
                "Compose clear, structured answers using any notes provided. Be comprehensive."
            ),
            name="Writer",
            description= "Synthesizes polished answers using gathered information."
        )

        orchestrator_agent  = await client.create_agent(
            instructions="""
            You coordinate a team conversation to solve the user's task.

            Guidelines:
            - Start with Researcher to gather information
            - Then have Writer synthesize the final answer
            - Only finish after both have contributed meaningfully
            """,
            name="Orchestrator",
            description="Coordinates multi-agent collaboration by selecting speakers",

        )

        workflow = (
            GroupChatBuilder()
            .with_agent_orchestrator(agent=orchestrator_agent)
            # Set a hard termination condition: stop after 4 assistant messages
            # The agent orchestrator will intelligently decide when to end before this limit but just in case
            .with_termination_condition(lambda messages: sum(1 for msg in messages if msg.role == Role.ASSISTANT) >= 4)
            .participants([researcher, writer])
            .build()
        )
        

        task = "What are the key benefits of async/await in Python?"

        print(f"Task: {task}\n")
        print("=" * 80)

        final_conversation: list[ChatMessage] = []
        last_executor_id: str | None = None

        # Run the workflow
        async for event in workflow.run_stream(task):
            if isinstance(event, WorkflowOutputEvent):
                # Workflow completed - data is a list of ChatMessage
                final_conversation = cast(list[ChatMessage], event.data)

        if final_conversation:
            print("\n\n" + "=" * 80)
            print("Final Conversation:")
            for msg in final_conversation:
                author = getattr(msg, "author_name", "Unknown")
                text = getattr(msg, "text", str(msg))
                print(f"\n[{author}]\n{text}")
                print("-" * 80)

        print("\nWorkflow completed.")

    
    
    
if __name__ == "__main__":
    asyncio.run(main())