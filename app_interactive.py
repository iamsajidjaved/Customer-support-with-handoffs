"""
Customer Support Chatbot with State Machine Pattern
====================================================

This application demonstrates an AI-powered customer support chatbot that uses
a state machine pattern to guide conversations through different stages:
1. Warranty Collection
2. Issue Classification
3. Resolution

Key Concepts:
- State Machine: The conversation flows through predefined steps
- Dynamic Prompts: Agent behavior changes based on current step
- Tool Calling: AI can invoke functions to record information and transition states
- Memory: Conversation history is preserved across turns

Author: Your Name
Date: December 25, 2025
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================

import os
import sys
from typing import Literal, Callable
from typing_extensions import NotRequired
import uuid

# LangChain Core Imports
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime

# LangGraph Imports (for state management and memory)
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

# Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================================
# SECTION 2: CONFIGURATION & SETUP
# ============================================================================

# Check if Google API key is set in environment variables
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
    print("=" * 60)
    print("ERROR: Google API Key not found!")
    print("=" * 60)
    print("\nPlease set your Google API key using one of these methods:")
    print("\n1. In PowerShell (temporary):")
    print('   $env:GOOGLE_API_KEY = "your-api-key-here"')
    print("\n2. In the code:")
    print('   os.environ["GOOGLE_API_KEY"] = "your-api-key-here"')
    print("\nGet your API key from: https://makersuite.google.com/app/apikey")
    print("=" * 60)
    sys.exit(1)

# Initialize the Gemini model
# This will be used to generate AI responses throughout the conversation
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ============================================================================
# SECTION 3: STATE DEFINITION
# ============================================================================

# Define the possible steps in our support workflow
# Using Literal type ensures only these exact values can be used
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]


class SupportState(AgentState):
    """
    State schema that defines what information we track during the conversation.
    
    Attributes:
        current_step: Which stage of the workflow we're in
        warranty_status: Whether the device is under warranty or not
        issue_type: Whether it's a hardware or software issue
    
    Note: NotRequired means these fields are optional initially
    """
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


# ============================================================================
# SECTION 4: TOOL DEFINITIONS
# ============================================================================
# Tools are functions the AI can call to perform actions or record information

@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """
    Record the customer's warranty status and transition to issue classification.
    
    This tool is called by the AI when it determines the warranty status.
    It updates the state and moves to the next step automatically.
    
    Args:
        status: Whether device is "in_warranty" or "out_of_warranty"
        runtime: Access to current state and metadata
    
    Returns:
        Command object that updates the state
    """
    return Command(
        update={
            # Add a message to the conversation history
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            # Update state variables
            "warranty_status": status,
            "current_step": "issue_classifier",  # Move to next step
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """
    Record the type of issue and transition to resolution specialist.
    
    This tool classifies the customer's problem as either hardware or software,
    then moves to the final resolution step.
    
    Args:
        issue_type: "hardware" for physical issues, "software" for digital issues
        runtime: Access to current state and metadata
    
    Returns:
        Command object that updates the state
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",  # Move to final step
        }
    )


@tool
def escalate_to_human(reason: str) -> str:
    """
    Escalate the case to a human support specialist.
    
    Used when the AI cannot resolve the issue automatically,
    such as out-of-warranty hardware issues requiring paid repairs.
    
    Args:
        reason: Explanation of why escalation is needed
    
    Returns:
        Confirmation message
    """
    # In a production system, this would:
    # - Create a support ticket
    # - Notify human agents
    # - Collect contact information
    return f"Escalating to human support. Reason: {reason}"


@tool
def provide_solution(solution: str) -> str:
    """
    Provide a solution to the customer's issue.
    
    Used to deliver troubleshooting steps or warranty information.
    
    Args:
        solution: The detailed solution or instructions for the customer
    
    Returns:
        Formatted solution message
    """
    return f"Solution provided: {solution}"


# ============================================================================
# SECTION 5: PROMPTS FOR EACH STEP
# ============================================================================
# These prompts define how the AI behaves at each stage of the workflow

WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Warranty verification

At this step, you need to:
1. Greet the customer warmly
2. Ask if their device is under warranty
3. Use record_warranty_status to record their response and move to the next step

Be conversational and friendly. Don't ask multiple questions at once."""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Issue classification
CUSTOMER INFO: Warranty status is {warranty_status}

At this step, you need to:
1. Ask the customer to describe their issue
2. Determine if it's a hardware issue (physical damage, broken parts) or software issue (app crashes, performance)
3. Use record_issue_type to record the classification and move to the next step

If unclear, ask clarifying questions before classifying."""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this step, you need to:
1. For SOFTWARE issues: provide troubleshooting steps using provide_solution
2. For HARDWARE issues:
   - If IN WARRANTY: explain warranty repair process using provide_solution
   - If OUT OF WARRANTY: escalate_to_human for paid repair options

Be specific and helpful in your solutions."""


# ============================================================================
# SECTION 6: STEP CONFIGURATION
# ============================================================================
# Maps each step to its prompt, available tools, and required state

STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],  # Only this tool available in this step
        "requires": [],  # No prerequisites for first step
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],  # Different tool for this step
        "requires": ["warranty_status"],  # Need warranty info before this step
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],  # Multiple tools available
        "requires": ["warranty_status", "issue_type"],  # Need both pieces of info
    },
}


# ============================================================================
# SECTION 7: MIDDLEWARE - DYNAMIC BEHAVIOR INJECTION
# ============================================================================

@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    Middleware that configures agent behavior based on the current workflow step.
    
    This is the "magic" that makes the state machine work. Before each AI response,
    this function:
    1. Checks what step we're in
    2. Loads the appropriate prompt and tools for that step
    3. Validates we have the required information
    4. Injects everything into the AI's context
    
    Args:
        request: The incoming request to the AI model
        handler: The function that actually calls the AI
    
    Returns:
        The AI's response configured for the current step
    """
    # Get current step from state (default to first step if not set)
    current_step = request.state.get("current_step", "warranty_collector")

    # Look up configuration for this step
    stage_config = STEP_CONFIG[current_step]

    # Validate that we have all required information for this step
    # This prevents jumping to later steps without necessary data
    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # Format the prompt with current state values
    # This replaces {warranty_status} and {issue_type} with actual values
    system_prompt = stage_config["prompt"].format(**request.state)

    # Inject the customized prompt and step-specific tools into the request
    request = request.override(
        system_prompt=system_prompt,
        tools=stage_config["tools"],
    )

    # Call the AI with the modified request
    return handler(request)


# ============================================================================
# SECTION 8: AGENT CREATION
# ============================================================================

# Collect all tools (needed for agent initialization, even though only
# specific tools are available at each step due to middleware)
all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]

# Create the agent with all components
agent = create_agent(
    model,                          # The AI model to use
    tools=all_tools,                # All available tools
    state_schema=SupportState,      # Schema defining our state structure
    middleware=[apply_step_config], # Middleware to inject step-specific behavior
    checkpointer=InMemorySaver(),   # Saves conversation history in memory
)


# ============================================================================
# SECTION 9: INTERACTIVE TERMINAL INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Generate a unique thread ID for this conversation session
    # This allows the checkpointer to track this specific conversation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Display welcome message
    print("=" * 60)
    print("Customer Support Chatbot")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the conversation\n")

    # Main conversation loop
    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for contacting support. Goodbye!")
            break

        # Skip empty inputs
        if not user_input:
            continue

        try:
            # Send message to agent and get response
            result = agent.invoke(
                {"messages": [HumanMessage(user_input)]},
                config
            )

            # Extract the last message (AI's response)
            last_message = result['messages'][-1]

            # Extract clean text content from the message
            if hasattr(last_message, 'content'):
                content = last_message.content
                
                # Handle structured output (list of content parts)
                if isinstance(content, list):
                    text_parts = [
                        part.get('text', '') 
                        for part in content 
                        if part.get('type') == 'text'
                    ]
                    agent_response = ' '.join(text_parts)
                else:
                    # Handle simple string content
                    agent_response = str(content)
            else:
                agent_response = str(last_message)

            # Display the agent's response
            print(f"\nAgent: {agent_response}\n")

            # Optional: Show current step for debugging
            # Uncomment the next line to see which step you're in
            # print(f"[Debug - Current step: {result.get('current_step')}]\n")

        except Exception as e:
            # Handle and display any errors
            print(f"\nError: {e}\n")
