# Add references
import asyncio
import json
from typing import cast
import os
from dotenv import load_dotenv
from agent_framework import (
    ChatMessage, 
    Role,  
    WorkflowOutputEvent,
    RequestInfoEvent,
    HostedCodeInterpreterTool,
    MagenticBuilder,
    AgentRunUpdateEvent,
    MagenticOrchestratorEvent,
    MagenticProgressLedger,
    MagenticPlanReviewRequest,
    MagenticPlanReviewResponse
)
from agent_framework_azure_ai import AzureAIProjectAgentProvider
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
        AzureAIProjectAgentProvider(credential=credential) as client,
    ):


        researcher_agent  = await client.create_agent(
            instructions=(
                "You are a Researcher. You find information without additional computation or quantitative analysis."
            ),
            name="ResearcherAgent",
            description= "Specialist in research and information gathering"
        )

        coder_agent = await client.create_agent(
            instructions=(
                "You solve questions using code. Please provide detailed analysis and computation process."
            ),
            name="CoderAgent",
            description= "A helpful assistant that writes and executes code to process and analyze data.",
            tools=HostedCodeInterpreterTool()
        )

        manager_agent   = await client.create_agent(
            instructions="You coordinate a team to complete complex tasks efficiently.",
            name="MagenticManager",
            description="Orchestrator that coordinates the research and coding workflow"
        )

        workflow = (
            MagenticBuilder()
            .participants([researcher_agent, coder_agent])
            .with_standard_manager (
                agent=manager_agent,
                max_round_count=10,
                max_stall_count=3,
                max_reset_count=2,
            )
            .with_plan_review() 
            .build()
        )
        

        task = (
            "I am preparing a report on the energy efficiency of different machine learning model architectures. "
            "Compare the estimated training and inference energy consumption of ResNet-50, BERT-base, and GPT-2 "
            "on standard datasets (for example, ImageNet for ResNet, GLUE for BERT, WebText for GPT-2). "
            "Then, estimate the CO2 emissions associated with each, assuming training on an Azure Standard_NC6s_v3 "
            "VM for 24 hours. Provide tables for clarity, and recommend the most energy-efficient model "
            "per task type (image classification, text classification, and text generation)."
        )


        pending_request: RequestInfoEvent | None = None
        pending_responses: dict[str, MagenticPlanReviewResponse] | None = None
        output_event: WorkflowOutputEvent | None = None

        while not output_event:
            if pending_responses is not None:
                stream = workflow.send_responses_streaming(pending_responses)
            else:
                stream = workflow.run_stream(task)

            last_message_id: str | None = None
            async for event in stream:
                if isinstance(event, AgentRunUpdateEvent):
                    message_id = event.data.message_id
                    if message_id != last_message_id:
                        if last_message_id is not None:
                            print("\n")
                        print(f"- {event.executor_id}:", end=" ", flush=True)
                        last_message_id = message_id
                    print(event.data, end="", flush=True)

                elif isinstance(event, RequestInfoEvent) and event.request_type is MagenticPlanReviewRequest:
                    pending_request = event

                elif isinstance(event, WorkflowOutputEvent):
                    output_event = event

            pending_responses = None

            # Handle plan review request if any
            if pending_request is not None:
                event_data = cast(MagenticPlanReviewRequest, pending_request.data)

                print("\n\n[Magentic Plan Review Request]")
                if event_data.current_progress is not None:
                    print("Current Progress Ledger:")
                    print(json.dumps(event_data.current_progress.to_dict(), indent=2))
                    print()
                print(f"Proposed Plan:\n{event_data.plan.text}\n")
                print("Please provide your feedback (press Enter to approve):")

                reply = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
                if reply.strip() == "":
                    print("Plan approved.\n")
                    pending_responses = {pending_request.request_id: event_data.approve()}
                else:
                    print("Plan revised by human.\n")
                    pending_responses = {pending_request.request_id: event_data.revise(reply)}
                pending_request = None

    # The output of the Magentic workflow is a list of ChatMessages with only one final message
    # generated by the orchestrator.
    output_messages = cast(list[ChatMessage], output_event.data)
    output = output_messages[-1].text
    print(output)

    
    
    
if __name__ == "__main__":
    asyncio.run(main())