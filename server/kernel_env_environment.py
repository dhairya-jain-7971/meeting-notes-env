from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MeetingAction, MeetingObservation
except ImportError:
    from models import MeetingAction, MeetingObservation

TASKS = {
    "easy": {
        "transcript": """
John: Alright team, quick sync. Sarah, can you send the project report to the client by Friday?
Sarah: Sure, I'll get that done.
John: Great. Mike, please book the conference room for next Monday's presentation.
Mike: Got it.
John: That's all for today.
        """,
        "expected_items": ["send project report to client", "book conference room"],
        "expected_assignees": ["sarah", "mike"],
        "expected_deadlines": ["friday", "monday"],
    },
    "medium": {
        "transcript": """
Lisa: Sprint planning time. Dev team needs to finish the login feature before the release on the 15th.
Tom: I'll handle the backend. Should be done by Wednesday.
Lisa: Good. Anna, can you write tests for the payment module? No hard deadline but ideally before Tom finishes.
Anna: I'll try to get it done by Tuesday then.
Lisa: Also someone needs to update the docs. 
Tom: I can do that after the backend work.
Lisa: Perfect. Let's also schedule a review meeting - can you set that up Anna?
Anna: Sure, I'll send invites for Thursday.
        """,
        "expected_items": ["finish login feature", "write tests for payment module", "update docs", "schedule review meeting"],
        "expected_assignees": ["tom", "anna", "tom", "anna"],
        "expected_deadlines": ["15th", "tuesday", "unspecified", "thursday"],
    },
    "hard": {
        "transcript": """
CEO: Q3 planning. We have a lot to cover.
VP Eng: The infrastructure migration needs to wrap up. James, where are we?
James: 60% done. We need design approval before we proceed with the database layer.
VP Eng: Mark, can you get design to sign off this week?
Mark: I'll chase them. If not this week, early next.
CEO: Make it this week. Also the mobile app - we promised the board a demo by end of month.
VP Product: Priya's team is on it but they need the API endpoints from James first.
James: I'll prioritize and have endpoints ready by Thursday.
VP Product: Then Priya can have the demo ready by the 28th.
CEO: Good. Marketing needs campaign assets too - Rachel?
Rachel: We need the product screenshots from Priya's team first, then we can turn around assets in 48 hours.
CEO: So Rachel delivers assets two days after Priya shares screenshots. Make sure that's tracked.
VP Eng: Also we need a security audit before launch. I'll find a vendor this week.
        """,
        "expected_items": [
            "get design approval for database layer",
            "prepare api endpoints",
            "build mobile app demo",
            "deliver campaign assets",
            "find security audit vendor",
        ],
        "expected_assignees": ["mark", "james", "priya", "rachel", "vp eng"],
        "expected_deadlines": ["this week", "thursday", "28th", "48 hours after screenshots", "this week"],
    },
}


def grade(action: MeetingAction, task_key: str) -> float:
    """Score the agent's extraction against expected values."""
    expected = TASKS[task_key]
    n = len(expected["expected_items"])
    if n == 0:
        return 0.0

    score = 0.0
    transcript_lower = expected["transcript"].lower()

    for i, item in enumerate(action.action_items[:n]):
        item_lower = item.lower()
        # Check action item makes sense (key words appear in transcript)
        words = [w for w in item_lower.split() if len(w) > 3]
        item_score = sum(1 for w in words if w in transcript_lower) / max(len(words), 1)

        # Check assignee
        assignee_score = 0.0
        if i < len(action.assignees):
            expected_assignee = expected["expected_assignees"][i].lower()
            if expected_assignee in action.assignees[i].lower():
                assignee_score = 1.0

        # Check deadline
        deadline_score = 0.0
        if i < len(action.deadlines):
            expected_deadline = expected["expected_deadlines"][i].lower()
            agent_deadline = action.deadlines[i].lower()
            if expected_deadline in agent_deadline or agent_deadline in expected_deadline:
                deadline_score = 1.0
            elif expected_deadline == "unspecified" and agent_deadline in ["unspecified", "none", "not specified", ""]:
                deadline_score = 1.0

        score += (item_score * 0.4 + assignee_score * 0.35 + deadline_score * 0.25)

    return min(score / n, 1.0)


class KernelEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_key = "easy"
        self._done = False

    def reset(self, task: str = "easy") -> MeetingObservation:
        self._task_key = task if task in TASKS else "easy"
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False

        return MeetingObservation(
            transcript=TASKS[self._task_key]["transcript"],
            task_name=self._task_key,
            hint="Extract all action items, who is responsible, and any deadlines mentioned.",
            done=False,
            reward=0.0,
        )

    def step(self, action: MeetingAction) -> MeetingObservation:
        self._state.step_count += 1
        self._done = True

        reward = grade(action, self._task_key)

        return MeetingObservation(
            transcript=TASKS[self._task_key]["transcript"],
            task_name=self._task_key,
            hint="",
            done=True,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state