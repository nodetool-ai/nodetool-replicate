import os
from typing import Any, Type, Union
from time import sleep
import black
import httpx
from openapi_pydantic.v3.parser import OpenAPIv3
from pydantic import BaseModel
from nodetool.common.environment import Environment
from openapi_pydantic.v3 import DataType, Reference, Schema, parse_obj
from .replicate_node import (
    capitalize,
    log,
    parse_model_info,
    sanitize_enum,
)
from nodetool.dsl.codegen import field_default, type_to_string
from nodetool.metadata.utils import is_enum_type
import asyncio


def format_code(code: str) -> str:
    """Formats the provided Python code using the Black formatter."""
    mode = black.FileMode()
    return black.format_file_contents(code, fast=False, mode=mode)


async def retry_http_request(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], retries: int = 3
) -> Any:
    """
    Retry an HTTP request with exponential backoff.

    Args:
        url (str): The URL to make the request to.
        headers (dict[str, str]): The headers to include in the request.
        retries (int, optional): The number of retries. Defaults to 3.

    Returns:
        Any: The response data.

    Raises:
        httpx.RequestError: If the request fails after all retries.
    """
    for attempt in range(retries + 1):
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < retries:
                delay = 2**attempt
                await asyncio.sleep(delay)
            else:
                raise e


class ModelMetadata(BaseModel):
    model_id: str
    model_version: str
    api: OpenAPIv3
    model_info: dict[str, Any]


async def get_model_api(
    client: httpx.AsyncClient,
    model_id: str,
):
    """
    Retrieves the API specification for a specific model and version from the Replicate API.

    Args:
        model_id (str): The replicate model ID.
        model_version (str, optional): The model version. Defaults to None.
    """
    if "REPLICATE_API_TOKEN" not in os.environ:
        raise ValueError("REPLICATE_API_TOKEN environment variable is not set")

    headers = {
        "Authorization": "Token " + Environment.get("REPLICATE_API_TOKEN"),
    }

    model_info = await retry_http_request(
        client,
        f"https://api.replicate.com/v1/models/{model_id}",
        headers=headers,
    )
    model_info.update(
        await parse_model_info(client, f"https://replicate.com/{model_id}")
    )
    latest_version = model_info.get("latest_version", None)
    if latest_version is None:
        raise ValueError(f"No latest version found for model {model_id}")
    api = parse_obj(latest_version["openapi_schema"])
    model_version = latest_version.get("id", "")

    assert api.components
    assert api.components.schemas

    return ModelMetadata(
        model_id=model_id,
        model_version=model_version,
        api=api,
        model_info=model_info,
    )


def convert_datatype_to_type(
    datatype: DataType | list[DataType], is_optional: bool
) -> type:
    """
    Converts an OpenAPI property schema to a Pydantic field.

    Args:
        datatype (DataType | list[DataType]): The datatype or list of datatypes to convert.
        is_optional (bool): Whether the field is optional.

    Returns:
        type: The type of the field.

    Raises:
        ValueError: If the datatype is a list.

    """
    if isinstance(datatype, list):
        raise ValueError("Cannot convert list of datatypes to type")
    t = None
    if datatype == DataType.STRING:
        t = str
    elif datatype == DataType.NUMBER:
        t = float
    elif datatype == DataType.INTEGER:
        t = int
    elif datatype == DataType.BOOLEAN:
        t = bool
    elif datatype == DataType.ARRAY:
        t = list
    elif datatype == DataType.OBJECT:
        t = dict
    else:
        raise ValueError(f"Unknown type: {datatype}")
    if is_optional:
        return t | None  # type: ignore
    return t


def return_type_repr(return_type: Any) -> str:
    if isinstance(return_type, dict):
        return (
            "{"
            + ", ".join(f"'{k}': {type_to_string(v)}" for k, v in return_type.items())
            + "}\n"
        )
    elif isinstance(return_type, type):
        return type_to_string(return_type)
    else:
        return str(return_type)


def generate_model_source_code(
    model_name: str,
    model_info: dict[str, Any],
    enums: list[str],
    description: str,
    schema: Schema,
    type_lookup: dict[str, type],
    return_type: type,
    hardware: str,
    replicate_model_id: str,
    overrides: dict[str, type] = {},
    output_index: int = 0,
    output_key: str = "output",
) -> str:
    """
    Generate source code for a Pydantic model from an OpenAPI schema.

    Args:
        model_name (str): The class name of the model.
        model_info (dict[str, Any]): The model information.
        description (str): Description of the model.
        enums (List[str]): A list of enum classes.
        schema (Schema): The schema object.
        type_lookup (Dict[str, Type]): Mapping of type names to Python types.
        return_type (Type): The return type of the model.
        hardware (str): The hardware information.
        replicate_model_id (str): The replicate model ID.
        overrides (Dict[str, Type], optional): Field type overrides. Defaults to {}.
        output_index (int, optional): The index for list outputs to use. Defaults to 0.
        output_key (str, optional): The key for dict outputs to use. Defaults to "output".

    Returns:
        str: The source code of the model.

    Raises:
        ValueError: If the schema has no properties or if the 'allOf' type cannot be handled.
    """

    if not schema.properties:
        raise ValueError("Schema has no properties")

    basic_fields = [p for p in schema.properties.keys()][:3]

    imports = ""

    lines = [
        f"class {model_name}(ReplicateNode):",
        f'    """{description}"""',
        *enums,
        "",
        "    @classmethod",
        f"    def get_basic_fields(cls): return {repr(basic_fields)}",
        "    @classmethod",
        f"    def replicate_model_id(cls): return '{replicate_model_id}'",
        "    @classmethod",
        f"    def get_hardware(cls): return '{hardware}'",
        "    @classmethod",
        f"    def get_model_info(cls): return {repr(model_info)}",
    ]
    if return_type:
        lines += [
            "    @classmethod",
            f"    def return_type(cls): return {return_type_repr(return_type)}",
        ]

    if output_index > 0:
        lines.append(f"   def output_index(self): return {output_index}")

    if output_key != "output":
        lines.append(f"    def output_key(self): return '{output_key}'")

    lines.append("\n")

    for name, prop in schema.properties.items():
        if name == "api_key" or isinstance(prop, Reference):
            continue

        is_optional = prop.default is None

        field_type, field_args = Any, ""
        if name in overrides:
            field_type = overrides[name]
            field_args = f" = Field(default=types.{field_type.__name__}(), description={repr(prop.description)})"
        else:
            enum_name = None
            if prop.type:
                field_type = convert_datatype_to_type(prop.type, is_optional)
            elif prop.allOf:
                ref_type = prop.allOf[0]
                if hasattr(ref_type, "ref"):
                    ref = ref_type.ref  # type: ignore
                    field_type = type_lookup[ref]
                    enum_name = field_type.__name__
                    if is_optional:
                        field_type = Union[field_type, None]

            field_args = " = Field("
            if prop.title:
                field_args += f"title='{prop.title}', "
            if prop.description:
                field_args += f"description={repr(prop.description)}, "
            if prop.minimum is not None:
                field_args += f"ge={prop.minimum}, "
            if prop.maximum is not None:
                field_args += f"le={prop.maximum}, "

            field_args += f"default={field_default(prop.default, enum_name)})"

        if is_enum_type(field_type):
            imports += f"from {field_type.__module__} import {field_type.__name__}\n"

        if isinstance(field_type, TypeName):
            field_type = field_type.__name__

        if isinstance(field_type, type):
            lines.append(f"    {name}: {type_to_string(field_type)}{field_args}")  # type: ignore
        else:
            lines.append(f"    {name}: {field_type}{field_args}")  # type: ignore

    code = "\n".join(lines)
    return imports + "\n\n" + format_code(code)


class TypeName:
    """
    A class representing a type name.

    Attributes:
        __name__ (str): The name of the type.
    """

    def __init__(self, name: str):
        self.__name__ = name

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


def create_enum_from_schema(name: str, schema: Schema):
    """
    Create source code for an enum model.

    Args:
        name (str): The name of the enum model.
        schema (Schema): The OpenAPI schema containing the enum field.

    Returns:
        str: The source code of the enum model.

    Raises:
        ValueError: If the schema has no enum field.
    """
    if schema.enum is not None:
        enum_type = type(schema.enum[0])
        values = "".join(
            [
                f"        {sanitize_enum(str(value))} = {repr(value)}\n"
                for value in schema.enum
            ]
        )
        return f"    class {name}({enum_type.__name__}, Enum):\n" + values
    else:
        raise ValueError("Schema has no enum field")


def create_replicate_node(
    node_name: str,
    model_id: str,
    return_type: Type,
    metadata: ModelMetadata,
    overrides: dict[str, type] = {},
    output_index: int = 0,
    output_key: str = "output",
    namespace: str = "",
):
    """
    Creates a node from a model ID and version.
    Gets the model from the replicate API and creates a node class from it.
    The node class is then added to the NODE_TYPES list.

    Args:
        node_name (str): The name of the node.
        enums (set[str]): A set of enum names.
        model_id (str): The ID of the model on replicate.
        return_type (Type): The return type of the node.
        metadata (ModelMetadata): The metadata of the model.
        model_version (str, optional): The version of the model. Defaults to None.
        overrides (dict[str, type]): A dictionary of overrides for the node fields.
        output_index (int): The index for list outputs to use. Defaults to 0.
        output_key (str): The key for dict outputs to use. Defaults to "output".
        namespace (str): The namespace of the node.
    Returns:
        Replicate: The generated node class.

    Raises:
        ValueError: If the schema has no properties or enum.
    """

    type_lookup = {}
    source_code = ""
    enums = []

    print("Creating node for", metadata.model_id)

    assert metadata.api.components
    assert metadata.api.components.schemas

    for name, schema in metadata.api.components.schemas.items():
        if name in (
            "Input",
            "Output",
            "Request",
            "Response",
            "PredictionRequest",
            "PredictionResponse",
            "Status",
            "WebhookEvent",
            "ValidationError",
            "HTTPValidationError",
        ):
            continue

        if hasattr(schema, "enum") and schema.enum is not None:  # type: ignore
            capitalized_name = capitalize(name)
            type_lookup[f"#/components/schemas/{name}"] = TypeName(capitalized_name)
            enums += [
                "\n\n",
                create_enum_from_schema(capitalized_name, schema),  # type: ignore
            ]

    del metadata.model_info["default_example"]
    del metadata.model_info["latest_version"]

    source_code += generate_model_source_code(
        model_name=node_name,
        model_info=metadata.model_info,
        enums=enums,
        description=metadata.model_info.get("description", ""),
        schema=metadata.api.components.schemas["Input"],  # type: ignore
        type_lookup=type_lookup,
        overrides=overrides,
        return_type=return_type,
        output_index=output_index,
        output_key=output_key,
        hardware=metadata.model_info.get("hardware", None),
        replicate_model_id=f"{model_id}:{metadata.model_version}",
    )

    return source_code


async def create_replicate_namespace(
    folder: str, namespace: str, nodes: list[dict[str, Any]]
):
    imports = (
        "from pydantic import BaseModel, Field\n"
        "import typing\n"
        "import nodetool.metadata.types as types\n"
        "from nodetool.dsl.graph import GraphNode\n"
        "from nodetool.nodes.replicate.replicate_node import ReplicateNode\n"
        "from enum import Enum\n"
    )
    namespace_path = os.path.join(folder, namespace.replace(".", "/"))
    namespace_folder = os.path.dirname(namespace_path)
    os.makedirs(namespace_folder, exist_ok=True)

    async with httpx.AsyncClient(
        timeout=60, follow_redirects=True, limits=httpx.Limits(max_connections=5)
    ) as client:

        async def get(node: dict[str, Any]):
            print("Getting metadata for", node["model_id"])
            return await get_model_api(client, node["model_id"])

        metadata = await asyncio.gather(*[get(node) for node in nodes])

    with open(namespace_path + ".py", "w") as f:
        f.write(imports)
        for node, metadata in zip(nodes, metadata):
            print("Creating node for", node["model_id"])
            source_code = create_replicate_node(**node, metadata=metadata)
            f.write(source_code)


def default_for(datatype: DataType | list[DataType] | None) -> Any:
    if datatype == DataType.STRING:
        return ""
    elif datatype == DataType.NUMBER:
        return 0.0
    elif datatype == DataType.INTEGER:
        return 0
    elif datatype == DataType.BOOLEAN:
        return False
    elif datatype == DataType.ARRAY:
        return []
    elif datatype == DataType.OBJECT:
        return {}
    else:
        return None
