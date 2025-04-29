from pathlib import Path
# from pydantic import (
#     FilePath as PydanticFilePath,
#     DirectoryPath as PydanticDirectoryPath,
# )

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic.typing import CallableGenerator

from ._utils import strict_set

from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler

class FilePath(Path):
    """
    Coerce into a `pathlib.Path` type. If strict mode is enabled (see _utils.use_strict), will raise
    if file does not exist.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Start from the built-in Path schema
        path_schema = handler(Path)

        def validate(value: Path) -> Path:
            value = Path(value)
            if strict_set():
                # This is essentially the FilePath validation from Pydantic v2
                if not value.is_file():
                    raise ValueError(f"Path '{value}' is not a valid file")
            return value

        return core_schema.no_info_after_validator_function(
            validate,
            path_schema
        )

class DirectoryPath(Path):
    """
    Coerce into a `pathlib.Path` type. If strict mode is enabled, will raise
    if directory does not exist.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Base schema from Path
        path_schema = handler(Path)

        def validate(value: Path) -> Path:
            value = Path(value)
            if strict_set():
                if not value.is_dir():
                    raise ValueError(f"Path '{value}' is not a valid directory")
            return value

        return core_schema.no_info_after_validator_function(
            validate,
            path_schema
        )
