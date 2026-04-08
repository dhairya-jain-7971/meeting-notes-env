from openenv.core.env_server.http_server import create_app

try:
    from models import MeetingAction, MeetingObservation
    from server.kernel_env_environment import KernelEnvironment
except ModuleNotFoundError:
    from ..models import MeetingAction, MeetingObservation
    from .kernel_env_environment import KernelEnvironment


app = create_app(
    KernelEnvironment,
    MeetingAction,
    MeetingObservation,
    env_name="meeting_notes_env",
    max_concurrent_envs=10,
)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
