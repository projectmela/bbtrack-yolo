from loguru import logger
import os
from comet_ml import Experiment


def setup_comet(project_name: str, workspace: str) -> Experiment | None:
    if api_key := os.getenv("COMET_API_KEY"):
        return Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace
        )
    logger.info("Comet API key not found. Comet logger not registered.")
    return None
