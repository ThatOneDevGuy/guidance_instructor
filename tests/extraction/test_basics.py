from enum import Enum
from typing import Optional
from typing_extensions import Annotated

import guidance
from guidance_instructor import generate_object
from pydantic import BaseModel


def test_direct_extraction(model):
    class FruitEnum(str, Enum):
        pear = "pear"
        banana = "banana"
        apple = "apple"

    class SimpleClass(BaseModel):
        name: Annotated[str, "Provide a name."]
        age: Annotated[int, "Provide an age in years."]
        favorite_fruit: Optional[FruitEnum]

    with guidance.user():
        lm = (
            model
            + "Extract the following into an object: Jack is a 30 year old male that loves apples."
        )

    with guidance.assistant():
        lm, jack = generate_object(lm, SimpleClass)

    assert jack.name.lower() == "jack"
    assert jack.age == 30
    assert jack.favorite_fruit == FruitEnum.apple


def test_direct_list_extraction(model):
    class NameList(BaseModel):
        names: list[str]

    with guidance.user():
        lm = model + "Repeat as a bulleted list: Jack, Jill."

    with guidance.assistant():
        lm, obj = generate_object(lm, NameList)

    assert [x.lower() for x in obj.names] == ["jack", "jill"]


def test_field_instructions(model):
    class FruitList(BaseModel):
        fruits: Annotated[list[str], "Bulleted list of fruits in all uppercase:"]

    with guidance.user():
        lm = model + "Say apples, oranges."

    with guidance.assistant():
        lm, obj = generate_object(lm, FruitList)

    assert obj.fruits == ["APPLES", "ORANGES"]


def test_extract_complex_object(model):
    class Story(BaseModel):
        summary: str
        events: Annotated[list[str], "Bulleted list of events that occurred:"]

    class Role(BaseModel):
        name: str
        description: str

    class Person(BaseModel):
        name: str
        attributes: Annotated[list[str], "Bulleted list of attributes:"]
        backstories: Annotated[list[Story], "Bulleted list of backstories:"]
        tendencies: Annotated[list[Role], "Bulleted list of typical roles:"]

    with guidance.user():
        story = (
            "Jackie is a tough woman with a pure heart. She was born on an apple "
            "farm adjacent to a small village. When she was younger, she ran away "
            "from home to live in the big city. Jackie tends to act as the moderator "
            "of the group, and is always trying to keep the peace. Sometimes she'll "
            "get caught up playing the part of the rival. She has a competitive streak."
        )

        lm = model + f"Extract the following into an object: {story}"

    with guidance.assistant():
        lm, obj = generate_object(lm, Person)

    assert obj.name.lower() == "jackie"
    assert len(obj.attributes) > 0, "Failed to extract any of Jackie's attributes."
    assert len(obj.backstories) > 0, "Failed to extract any of Jackie's backstory."
    assert len(obj.tendencies) > 0, "Failed to extract any of Jackie's tendencies."

    for story in obj.backstories:
        assert len(story.events) > 0, "Failed to extract any events from a backstory."
