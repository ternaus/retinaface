from collections import namedtuple

from torch.utils import model_zoo

from retinaface.predict_single import Model

model = namedtuple("model", ["url", "model"])

models = {
    "resnet50_2020-07-20": model(
        url="https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip",  # noqa: E501 pylint: disable=C0301
        model=Model,
    )
}


def get_model(model_name: str, max_size: int, device: str = "cpu") -> Model:
    model = models[model_name].model(max_size=max_size, device=device)
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")

    model.load_state_dict(state_dict)

    return model
