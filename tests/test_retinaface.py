import pytest

from retinaface.pre_trained_models import get_model
from tests.conftest import images

max_size = 1280


@pytest.mark.parametrize(
    ["image", "faces"],
    [
        (images["with_faces"]["image"], images["with_faces"]["faces"]),
        (images["with_no_faces"]["image"], images["with_no_faces"]["faces"]),
    ],
)
def test_predict_jsons(image, faces):
    model = get_model("resnet50_2020-07-20", max_size=max_size)
    model.eval()

    result = model.predict_jsons(image)

    assert result == faces
