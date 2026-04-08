from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List

class MeetingAction(Action):
    """Action taken by the agent - extracted action items from meeting notes."""
    
    action_items: List[str] = Field(..., description="List of action items extracted")
    assignees: List[str] = Field(..., description="Who is responsible for each item")
    deadlines: List[str] = Field(..., description="Deadline for each item, or 'unspecified'")

class MeetingObservation(Observation):
    """Observation given to the agent - the meeting transcript to analyze."""
    
    transcript: str = Field(..., description="The meeting transcript to analyze")
    task_name: str = Field(..., description="Current task difficulty level")
    hint: str = Field(default="", description="Optional hint for the agent")