from enum import Enum

from bot.model.settings.llama import Llama31Settings, Llama31ToolSettings, Llama32OneSettings, Llama32ThreeSettings


class Model(Enum):
    LLAMA_3_1 = "llama-3.1"
    LLAMA_3_1_tool = "llama-3.1-tool"
    LLAMA_3_2_one = "llama-3.2:1b"
    LLAMA_3_2_three = "llama-3.2"


SUPPORTED_MODELS = {
    Model.LLAMA_3_1.value: Llama31Settings,
    Model.LLAMA_3_1_tool.value: Llama31ToolSettings,
    Model.LLAMA_3_2_one.value: Llama32OneSettings,
    Model.LLAMA_3_2_three.value: Llama32ThreeSettings,
}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_settings(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings
