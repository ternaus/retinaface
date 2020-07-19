import re
from pathlib import Path
from typing import Union, Optional, Any, Dict, Tuple

import numpy as np
import torch


def load_checkpoint(file_path: Union[Path, str], rename_in_layers: Optional[dict] = None) -> Dict[str, Any]:
    """Loads PyTorch checkpoint, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if rename_in_layers is not None:
        model_state_dict = checkpoint["state_dict"]

        result = {}
        for key, value in model_state_dict.items():
            for key_r, value_r in rename_in_layers.items():
                key = re.sub(key_r, value_r, key)

            result[key] = value

        checkpoint["state_dict"] = result

    return checkpoint


def random_color() -> Tuple[int, ...]:
    return tuple(int(x) for x in np.random.randint(0, 255, size=3))
