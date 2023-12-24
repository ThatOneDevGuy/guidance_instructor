import os

import guidance
import pytest


@pytest.fixture(scope="session", autouse=True)
def model():
    print("Loading the model")

    if "MODEL_TYPE" not in os.environ or "MODEL_PATH" not in os.environ:
        raise ValueError(
            "You must set MODEL_TYPE and MODEL_PATH environment variables."
        )

    model_type = os.environ.get("MODEL_TYPE", "").lower()
    model_path = os.environ.get("MODEL_PATH")
    model_context_len = int(os.environ.get("MODEL_CONTEXT_LEN", 2048))

    if model_type == "llamacppchat":
        model = guidance.models.LlamaCppChat(model_path, n_ctx=model_context_len)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    yield model
