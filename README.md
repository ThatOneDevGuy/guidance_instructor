# Installation
`pip install --upgrade git+https://github.com/ThatOneDevGuy/guidance_instructor`

# Overview
This lets you reliably get structured outputs from open source LLMs using guidance + pydantic.
- guidance is a library for restricting LLM outputs based on regular expressions and context free grammars.
- pydantic is an OpenAPI-friendly library for defining data classes with strong typing and validation.

# Example
```python
from enum import Enum
from typing import Optional
from typing_extensions import Annotated

import guidance
from guidance_instructor import generate_pydantic_object
from pydantic import BaseModel

# Load a chat-finetuned mistral model
model = guidance.models.LlamaCppChat("openhermes-2.5-mistral-7b.Q5_K_M.gguf")

# Create a sample pydantic class. This nests a FruitEnum inside a SimpleClass.
class FruitEnum(str, Enum):
    pear = "pear"
    banana = "banana"
    apple = "apple"

class SimpleClass(BaseModel):
    name: Annotated[str, "Provide a name."]
    age: Annotated[int, "Provide an age in years."]
    favorite_fruit: Optional[FruitEnum]

# Send the 'user' message to the LLM. In guidance, messages and generations get appended
# to some pseudo-string that begins with the underlying model. The `with guidance.user()`
# notation is how guidance abstracts different roles that the LLM recognizes.
with guidance.user():
    lm = model + "Extract the following into an object: Jack is a 30 year old dude that loves apples."

# Read the 'assistant' message from the LLM
with guidance.assistant():
    # This returns an "lm" object, which can be continued for further generations, plus a
    # "jack" object, which contains a SimpleClass representation of Jack's information as
    # described in the user instruction above.
    lm, jack = generate_pydantic_object(lm, SimpleClass)

print(jack)
# prints {'name': 'Jack', 'age': 30, 'favorite_fruit': 'apple'}
```
