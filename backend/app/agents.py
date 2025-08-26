"""
OpenAI Agents SDK integration for CV Engine.
Provides a simple interface to the official OpenAI Agents SDK.
"""

import logging
from typing import Dict, Any, Optional
from agents import Agent, Runner
from app.config import settings

logger = logging.getLogger(__name__)

def create_agent(
    name: str,
    instructions: str,
    model: Optional[str] = None,
    tools: Optional[list] = None
) -> Agent:
    """
    Create an OpenAI Agent with the specified configuration.
    
    Args:
        name: Name of the agent
        instructions: Instructions for the agent
        model: Model to use (defaults to settings)
        tools: List of tools for the agent
    
    Returns:
        Configured Agent instance
    """
    try:
        # Use the specified model or fall back to default
        agent_model = model or settings.cv_generator.cv_generation_model
        
        # Create the agent
        agent = Agent(
            name=name,
            instructions=instructions,
            model=agent_model,
            tools=tools or []
        )
        
        logger.info(f"Created agent '{name}' with model '{agent_model}'")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create agent '{name}': {e}")
        raise

async def run_agent(
    agent: Agent,
    input_data: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Run an agent with the given input data.
    
    Args:
        agent: The agent to run
        input_data: Input data for the agent
        **kwargs: Additional arguments for the agent run
    
    Returns:
        Agent response as a dictionary
    """
    try:
        # Use the Runner to execute the agent
        runner = Runner()
        result = await runner.run(agent, input_data, **kwargs)
        
        logger.info(f"Agent '{agent.name}' completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Failed to run agent '{agent.name}': {e}")
        raise

def get_agent_response_text(result: Dict[str, Any]) -> str:
    """
    Extract the text response from an agent result.
    
    Args:
        result: Agent result dictionary
    
    Returns:
        Extracted text response
    """
    try:
        # The structure depends on the agent type and response format
        if isinstance(result, dict):
            # Try different possible response formats
            if 'output' in result:
                return str(result['output'])
            elif 'content' in result:
                return str(result['content'])
            elif 'response' in result:
                return str(result['response'])
            elif 'result' in result:
                return str(result['result'])
            else:
                # Return the entire result as string
                return str(result)
        else:
            return str(result)
            
    except Exception as e:
        logger.error(f"Failed to extract response text: {e}")
        return str(result)

# Convenience functions for common agent types
def create_face_analyzer_agent() -> Agent:
    """Create an agent for analyzing face characteristics."""
    instructions = """
    You are a face analysis expert. Analyze the provided face image and extract:
    - Estimated age range
    - Gender
    - Whether the face is appropriate for a professional CV
    
    Return your analysis as a JSON object with the following structure:
    {
        "estimated_age": "age range (e.g., 'mid-20s', 'early 30s')",
        "gender": "male/female/unknown",
        "experience_appropriate": true/false,
        "reasoning": "brief explanation"
    }
    
    Note: This agent is designed for text-based analysis. For image analysis,
    use the dedicated image validation service.
    """
    return create_agent(
        name="Face Age Analyzer",
        instructions=instructions,
        model=settings.cv_generator.face_analysis_model
    )

def create_cv_generator_agent() -> Agent:
    """Create an agent for generating CV content."""
    instructions = """
    You are a CV generation expert. Generate professional CV content based on:
    - The provided CV template
    - Person characteristics from face analysis
    - Professional requirements
    
    Return your response as a JSON object with:
    {
        "html_content": "complete HTML CV content",
        "person_characteristics": {
            "name": "generated name",
            "age": "age range",
            "gender": "gender",
            "profession": "job title"
        }
    }
    """
    return create_agent(
        name="Integrated CV Generator",
        instructions=instructions,
        model=settings.cv_generator.cv_generation_model
    )

def create_reranking_agent() -> Agent:
    """Create an agent for reranking search results."""
    instructions = """
    You are a search result reranking expert. Reorder the provided search results
    based on relevance to the query. Return the results in order of relevance.
    """
    return create_agent(
        name="Search Reranker",
        instructions=instructions,
        model=settings.rag.reranking_model
    )
