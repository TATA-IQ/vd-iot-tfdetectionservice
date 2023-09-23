from typing import Union

from pydantic import BaseModel


class Image_Model(BaseModel):
    """
    Model generated for FastApi Query
    """

    image_name: Union[str, None] = None
    image: Union[str, None] = None
    model_config: Union[dict, None] = None
