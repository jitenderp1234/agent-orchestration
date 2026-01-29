# Add references
import asyncio
import json
from typing import cast
import os
from dotenv import load_dotenv
from typing import Annotated
from agent_framework import (
    ChatMessage, 
    Role,  
    WorkflowOutputEvent,
    RequestInfoEvent,
    HandoffBuilder,
    AgentRunUpdateEvent,
    RequestInfoEvent,
    HandoffAgentUserRequest,
)
from agent_framework_azure_ai import AzureAIProjectAgentProvider
from azure.identity.aio import AzureCliCredential
from agent_framework import AgentExecutor

# Load environment variables from .env file
load_dotenv()
project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
model_deployment = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")


def process_refund(order_number: Annotated[str, "Order number to process refund for"]) -> str:
    """Simulated function to process a refund for a given order number."""
    return f"Refund processed successfully for order {order_number}."


def check_order_status(order_number: Annotated[str, "Order number to check status for"]) -> str:
    """Simulated function to check the status of a given order number."""
    return f"Order {order_number} is currently being processed and will ship in 2 business days."

def process_return(order_number: Annotated[str, "Order number to process return for"]) -> str:
    """Simulated function to process a return for a given order number."""
    return f"Return initiated successfully for order {order_number}. You will receive return instructions via email."

async def main():

    # 1) Create three domain agents using AzureChatClient
    async with (
        AzureCliCredential() as credential,
        AzureAIProjectAgentProvider(credential=credential) as client,
    ):


        triage_agent   = await client.create_agent(
            instructions=(
                "You are frontline support triage. Route customer issues to the appropriate specialist agents "
                "based on the problem described."
            ),
            name="triageAgent",
            description= "Triage agent that handles general inquiries."
        )

        refund_agent  = await client.create_agent(
            instructions=(
                "You process refund requests."
            ),
            name="refundAgent",
            description= "Agent that handles refund requests.",
            tools=[process_refund]
        )

        order_agent = await client.create_agent(
            instructions="You handle order and shipping inquiries.",
            name="orderAgent",
            description="Agent that handles order tracking and shipping issues.",
            tools=[check_order_status],
        )

        return_agent  = await client.create_agent(
            instructions="You manage product return requests.",
            name="returnAgent",
            description="Agent that handles return processing.",
            tools=[process_return],
        )

        workflow = (
            HandoffBuilder(
                name="support_with_approvals",
                participants=[triage_agent, refund_agent, order_agent, return_agent],
            )
            .with_start_agent(triage_agent) # Triage receives initial user input
            .with_termination_condition(
                # Custom termination: Check if one of the agents has provided a closing message.
                # This looks for the last message containing "welcome", which indicates the
                # conversation has concluded naturally.
                lambda conversation: len(conversation) > 0 and "welcome" in conversation[-1].text.lower()
            )
            .with_autonomous_mode(
                agents=[triage_agent],
                turn_limits={triage_agent.name: 3},
                prompts={triage_agent.name: "Continue with your best judgment as the user is unavailable."},
            )
            # Triage cannot route directly to refund agent
            .add_handoff(triage_agent, [order_agent, return_agent])
            # Only the return agent can handoff to refund agent - users wanting refunds after returns
            .add_handoff(return_agent, [refund_agent])
            # All specialists can handoff back to triage for further routing
            .add_handoff(order_agent, [triage_agent])
            .add_handoff(return_agent, [triage_agent])
            .add_handoff(refund_agent, [triage_agent])
            .build()
        )
        
        pending_requests: list[RequestInfoEvent] = []

        # Start workflow
        async for event in workflow.run_stream("I need help with my order."):
            if isinstance(event, RequestInfoEvent):
                pending_requests.append(event)


        while pending_requests:
            responses: dict[str, object] = {}

            for request in pending_requests:
                if isinstance(request.data, HandoffAgentUserRequest):
                    # Agent needs user input
                    print(f"Agent {request.source_executor_id} asks:")
                    for msg in request.data.agent_response.messages[-2:]:
                        print(f"  {msg.author_name}: {msg.text}")

                    user_input = input("You: ")
                    responses[request.request_id] = HandoffAgentUserRequest.create_response(user_input)
               

            # Send all responses and collect new requests
            pending_requests = []
            async for event in workflow.send_responses_streaming(responses):
                if isinstance(event, RequestInfoEvent):
                    pending_requests.append(event)
                elif isinstance(event, WorkflowOutputEvent):
                    print("\nWorkflow completed!")
  
if __name__ == "__main__":
    asyncio.run(main())