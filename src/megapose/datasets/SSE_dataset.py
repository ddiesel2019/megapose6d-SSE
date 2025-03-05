"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Standard Library
import json
from pathlib import Path
from typing import List

# MegaPose
from megapose.config import MEMORY

# Local Folder
from .object_dataset import RigidObject, RigidObjectDataset


@MEMORY.cache
def make_sse_infos(sse_dir: Path, model_name: str = "model.obj") -> List[str]:
    """Parses SSE dataset directory to retrieve valid object IDs."""
    sse_dir = Path(sse_dir)
    models_dir = sse_dir.iterdir()
    invalid_ids_path = sse_dir.parent / "invalid_objects.json"

    invalid_ids = set()
    if invalid_ids_path.exists():
        invalid_ids = set(json.loads(invalid_ids_path.read_text()))

    object_ids = []
    for model_dir in models_dir:
        if (model_dir / "meshes" / model_name).exists():
            object_id = model_dir.name
            if object_id not in invalid_ids:
                object_ids.append(object_id)

    object_ids.sort()
    return object_ids


class SSEObjectDataset(RigidObjectDataset):
    def __init__(self, sse_root: Path, split: str = "default"):
        """Initializes the SSE dataset with a specific split (default, normalized, etc.)."""
        self.sse_dir = sse_root / f"models_{split}"

        # Define a scaling factor based on the dataset split (if necessary)
        if split == "default":
            scaling_factor = 1.0
        elif split == "scaled":
            scaling_factor = 0.1  # Example: adjust scaling for normalized versions

        object_ids = make_sse_infos(self.sse_dir)
        objects = []
        for object_id in object_ids:
            model_path = self.sse_dir / object_id / "meshes" / "model.obj"
            label = f"sse_{object_id}"
            obj = RigidObject(
                label=label,
                mesh_path=model_path,
                scaling_factor=scaling_factor,
            )
            objects.append(obj)

        super().__init__(objects)
