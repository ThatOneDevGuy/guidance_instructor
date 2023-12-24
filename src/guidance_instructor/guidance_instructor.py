import re
from enum import Enum
from types import GenericAlias, NoneType
from typing import Tuple, Type, Union, _UnionGenericAlias

import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo

import guidance
from guidance.models import Model


UNESCAPED_STRING_CHARS = (
    r"abcdefghijklmnopqrstuvwxyz"
    r"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    r"0123456789"
    r"!@#\$%\^&\*\(\)_\+{}\|:<>\?\[\];',\./`~ "
)
ALL_STRING_CHARS = UNESCAPED_STRING_CHARS + r"\\\""
YAML_START_MARKER = "\n```yaml\n"
YAML_END_MARKER = "\n```"


def set_allowed_chars(chars: str) -> None:
    """
    Sets the allowed characters for string generation.

    Args:
    - chars: A string containing the allowed characters.
    """
    global UNESCAPED_STRING_CHARS, ALL_STRING_CHARS
    UNESCAPED_STRING_CHARS = "".join(f"\\{x}" for x in chars)
    ALL_STRING_CHARS = UNESCAPED_STRING_CHARS + r"\\\""


@guidance(stateless=True)
def generate_str(
    lm: Model, field_info: Union[FieldInfo, Type[str]], depth: int = 0
) -> Model:
    """
    Generates a string based on a regular expression pattern.

    Args:
    - language_model: The language model used for generating strings.
    - field_info: Optional field information for additional context.
    - depth: The current indentation level for formatting.

    Returns:
    - Model after generating the string
    """
    return guidance.gen(
        regex=rf'"([{UNESCAPED_STRING_CHARS}]|(\\[{ALL_STRING_CHARS}]))*"'
    )


def _escaped(s: str) -> str:
    """
    Escapes a string for use in a regex.

    Args:
    - s: The string to escape.

    Returns:
    - The escaped string.
    """
    return yaml.dump(s, explicit_end=None).strip()


@guidance(stateless=True)
def generate_enum(lm: Model, enum_type: Type[Enum], depth: int = 0) -> Model:
    """
    Selects a value from an enumeration.

    Args:
    - language_model: The language model used for selection.
    - enum_type: The enumeration type from which to select a value.
    - depth: The current indentation level for formatting.

    Returns:
    - Model after generating a value from the enumeration.
    """
    choices = [_escaped(x.value) for x in enum_type.__members__.values()]
    return guidance.select(choices)


@guidance(stateless=True)
def generate_bool(
    lm: Model, field_info: Union[FieldInfo, Type[bool]], depth: int = 0
) -> Model:
    """
    Generates a boolean.

    Args:
    - language_model: The language model used for generating booleans.
    - field_info: Optional field information for additional context.
    - depth: The current indentation level for formatting.

    Returns:
    - Model after generating the boolean.
    """
    return guidance.select(
        [
            "true",
            "false",
        ]
    )


@guidance(stateless=True)
def generate_int(
    lm: Model, field_info: Union[FieldInfo, Type[int]] = None, depth: int = 0
) -> Model:
    """
    Generates an integer.

    Args:
    - language_model: The language model used for generating integers.
    - field_info: Optional field information for additional context.
    - depth: The current indentation level for formatting.

    Returns:
    - Model after generating the integer.
    """
    return guidance.gen(regex=r"\d+")


@guidance(stateless=True)
def generate_float(
    lm: Model, field_info: Union[FieldInfo, Type[float]] = None, depth: int = 0
) -> Model:
    """
    Generates a floating-point number.

    Args:
    - language_model: The language model used for generating floats.
    - field_info: Optional field information for additional context.
    - depth: The current indentation level for formatting.

    Returns:
    - Model after generating the float.
    """
    return guidance.gen(regex=r"\-?\d+\.\d+")


@guidance(stateless=True)
def generate_key_value_pair(
    lm: Model, value_type=Union[FieldInfo, Type], depth: int = 0
) -> Model:
    """
    Generates a key-value pair with the specified value type.

    Args:
    - language_model: The language model used for generating the pair.
    - value_type: The type of the value in the pair.
    - depth: The current indentation level for formatting.

    Returns:
    - Model after generating the pair.
    """
    key = generate_str(None, depth)
    value = generate_field_by_type(value_type, depth + 1)
    return key + ": " + value


# Cache for key-values generation functions
_keyvals_cache = {}


def generate_dict_items(value_type: Union[FieldInfo, Type], depth: int = 0) -> Model:
    """
    Generates a sequence of key-value pairs.

    Args:
    - language_model: The language model used for generation.
    - value_type: The type of the values in the key-value pairs.
    - depth: The current indentation level for formatting.

    Returns:
    - Model after generating the sequence.
    """
    global _keyvals_cache
    key = (value_type, depth)
    if key in _keyvals_cache:
        return _keyvals_cache[key]

    @guidance(stateless=True, dedent=False)
    def result(lm: Model, prefix: str = "") -> Model:
        if prefix:
            indentation = prefix
        else:
            indentation = "\n" + "  " * depth
        return guidance.select(
            [
                indentation + generate_key_value_pair(value_type, depth),
                indentation + generate_key_value_pair(value_type, depth) + result(),
            ]
        )

    _keyvals_cache[key] = result
    return result


@guidance(stateless=True)
def generate_dict(
    lm: Model,
    field_info: Union[FieldInfo, GenericAlias] = None,
    depth: int = 0,
    prefix: str = "",
) -> Model:
    """
    Generates a dictionary representation.

    Args:
    - language_model: The language model used for generation.
    - field_info: Field information containing key and value types.
    - depth: The current indentation level for formatting.

    Returns:
    - Model after generating the dictionary.
    """
    if isinstance(field_info, FieldInfo):
        field_info = field_info.annotation

    key_type, value_type = field_info.__args__

    return guidance.select(
        [
            "{}",
            generate_dict_items(value_type, depth)(prefix),
        ]
    )


# Cache for items generation functions
_items_cache = {}


def generate_list_items(
    field_info: Union[FieldInfo, Type] = None, depth: int = 0
) -> Model:
    """
    Generates a sequence of items.

    Args:
    - language_model: The language model used for generation.
    - field_info: Field information for the items.
    - depth: The current indentation level for formatting.

    Returns:
    - Model after generating the sequence.
    """
    global _items_cache
    key = (field_info, depth)
    if key in _items_cache:
        return _items_cache[key]

    @guidance(stateless=True, dedent=False)
    def result(lm: Model, prefix: str = "") -> Model:
        if prefix:
            indentation = prefix + "- "
        else:
            indentation = "  " * (depth - 1) + "- "
        return guidance.select(
            [
                generate_field_by_type(field_info, depth, prefix=indentation),
                generate_field_by_type(field_info, depth, prefix=indentation)
                + "\n"
                + result(),
            ]
        )

    _items_cache[key] = result
    return result


@guidance(stateless=True)
def generate_list(
    lm: Model,
    field_info: Union[FieldInfo, GenericAlias] = None,
    depth=0,
    prefix: str = "",
) -> Model:
    """
    Generates a list representation.

    Args:
    - language_model: The language model used for generation.
    - field_info: Field information containing the item type.
    - depth: The current indentation level for formatting.

    Returns:
    - Model after generating the list.
    """
    if isinstance(field_info, FieldInfo):
        field_info = field_info.annotation

    item_type = field_info.__args__[0]

    return guidance.select(
        [
            "[]",
            "\n" + generate_list_items(item_type, depth)(prefix),
        ]
    )


def _compile_context(field_info, depth: int, prefix: str = "") -> str:
    """
    Compiles the context or metadata into a formatted string.

    Args:
    - field_info: Field information containing the metadata.
    - indentation_level: The current indentation level for formatting.
    - prefix: An optional prefix to prepend to each line.

    Returns:
    - A formatted string containing the context.
    """
    indentation = "  " * depth
    return "\n".join(f"{indentation}# %s" % line for line in field_info.metadata)


@guidance(stateless=True)
def generate_field_by_type(
    lm: Model,
    field_type: Union[FieldInfo, Type],
    depth: int,
    prefix: str = "",
    skip_keys={},
) -> Model:
    """
    Generates a field based on its type.

    Args:
    - language_model: The language model used for parsing.
    - field_type: The type of the field, either as FieldInfo or a Python type.
    - nesting_level: The current nesting level for indentation purposes.

    Returns:
    - Model after generating the field.
    """
    is_required = True
    parsed_result = None

    if isinstance(field_type, FieldInfo):
        is_required = field_type.is_required()
        field_type = field_type.annotation

    if isinstance(field_type, GenericAlias):
        if field_type.__origin__ == dict:
            parsed_result = generate_dict(field_type, depth, prefix)
        elif field_type.__origin__ == list:
            parsed_result = generate_list(field_type, depth, prefix)
    elif isinstance(field_type, _UnionGenericAlias):
        union_options = []
        for union_arg in field_type.__args__:
            if union_arg == NoneType:
                is_required = False
            else:
                union_options.append(generate_field_by_type(union_arg, depth))
        parsed_result = guidance.select(union_options)
    elif isinstance(type(field_type), type):
        if field_type == int:
            parsed_result = prefix + generate_int(field_type, depth)
        elif field_type == float:
            parsed_result = prefix + generate_float(field_type, depth)
        elif field_type == str:
            parsed_result = prefix + generate_str(field_type, depth)
        elif field_type == bool:
            parsed_result = prefix + generate_bool(field_type, depth)
        elif issubclass(field_type, BaseModel):
            parsed_result = generate_object(field_type, depth, prefix, skip_keys)
        elif issubclass(field_type, Enum):
            parsed_result = prefix + generate_enum(field_type, depth)
    else:
        raise Exception(f"Unsupported type: {field_type}")

    if is_required:
        return parsed_result
    else:
        return guidance.select(
            [
                parsed_result,
                "null",
            ]
        )


@guidance(stateless=True)
def generate_object(
    lm: Model,
    pydantic_class: Type[BaseModel],
    depth: int = 0,
    prefix: str = "",
    skip_keys={},
) -> Model:
    """
    Generates a yaml config of an object based on a pydantic class.

    Args:
    - language_model: The language model used for parsing.
    - pydantic_class: The Pydantic class to be parsed.
    - nesting_level: The current nesting level for indentation purposes.

    Returns:
    - Model after generating the object.
    """
    if isinstance(pydantic_class, FieldInfo):
        pydantic_class = pydantic_class.annotation

    if prefix:
        indentation = prefix
    elif depth == 0:
        indentation = "  " * depth
    else:
        indentation = "\n" + "  " * depth

    parsed_result = ""
    trailing_newline = False

    for field_name, field_info in pydantic_class.model_fields.items():
        if field_name in skip_keys:
            continue
        comment = _compile_context(field_info, depth)
        if comment:
            if not trailing_newline:
                parsed_result += "\n"
            parsed_result += comment + "\n"
            trailing_newline = True

        if trailing_newline:
            indentation = indentation.lstrip("\n")

        parsed_result += f"{indentation}{field_name}: "
        parsed_result += generate_field_by_type(field_info, depth + 1)
        trailing_newline = False

        indentation = "\n" + "  " * depth

    return parsed_result


def generate_pydantic_object(
    lm: Model, pydantic_class: Type[BaseModel], **kwargs
) -> Tuple[Model, BaseModel]:
    """
    Generates a pydantic object.

    Args:
    - lm: The language model used to generate the instructions.
    - pydantic_class: The pydantic class defining the structure of the object to be extracted.

    Returns:
    - Model after generating a yaml config of the object.
    - The object generated from the specified Pydantic class.
    """
    lm += YAML_START_MARKER

    # Fill in the pre-filled fields
    if kwargs:
        kwargs = kwargs.copy()
        # Convert any BaseModel objects to a dict
        for key, value in kwargs.items():
            if isinstance(value, BaseModel):
                kwargs[key] = value.model_dump()

        # Dump the kwargs into yaml
        yaml_content = yaml.dump(kwargs, explicit_end=None)

        # Add the yaml content to the generation result
        lm += f"{yaml_content}"

    # Generate the rest of the fields
    lm += generate_field_by_type(pydantic_class, 0, skip_keys=kwargs) + YAML_END_MARKER

    # Extract the yaml content
    generation_output = str(lm)
    start_idx = generation_output.rfind(YAML_START_MARKER) + len(YAML_START_MARKER)
    end_idx = generation_output.rfind(YAML_END_MARKER)
    yaml_content = generation_output[start_idx:end_idx]
    dict_content = yaml.safe_load(yaml_content)

    # Create the pydantic object
    if issubclass(pydantic_class, BaseModel):
        pydantic_object = pydantic_class(**dict_content)
    else:
        pydantic_object = dict_content

    return lm, pydantic_object
