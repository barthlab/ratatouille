from typing import Optional
import os.path as path
import json
import logging

from kitchen.configs.routing import default_recipe_path
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.loader import ephys_loader, two_photon_loader


logger = logging.getLogger(__name__)


def load_dataset(template_id: str, cohort_id: str, recipe: str, name: Optional[str] = None) -> DataSet:
    """
    Load a dataset for a given template and cohort using a specific recipe.
    """
    recipe_path = default_recipe_path(recipe)
    assert path.exists(recipe_path), f"Cannot find recipe path: {recipe_path}"
    with open(recipe_path, 'r') as f:
        recipe_dict = json.load(f)
    logger.info(f"Loading dataset with recipe: {recipe_dict} for template {template_id} and cohort {cohort_id}")
    logger.debug(f"Recipe name: {recipe_dict['name']}")
    logger.debug(f"Recipe description: {recipe_dict['description']}")

    data_type = recipe_dict['data_type']
    name = name if name is not None else template_id + "_" + cohort_id
    if data_type == "ephys":
        loaded_dataset = ephys_loader.cohort_loader(template_id, cohort_id, recipe_dict, name)
    elif data_type == "two_photon":
        loaded_dataset = two_photon_loader.cohort_loader(template_id, cohort_id, recipe_dict, name)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    logger.info(str(loaded_dataset))
    return loaded_dataset