import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
import asyncio
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.contents.chat_history import ChatHistory

# Function to get completion using Semantic Kernel
async def get_completion_skernel(prompt, model="gpt-3.5-turbo"):
    kernel = Kernel()

    # Add OpenAI Chat Completion Service
    openai_service = OpenAIChatCompletion(
        api_key= os.getenv('OPENAI_API_KEY'),
        ai_model_id=model
    )
    kernel.add_service( openai_service)

    # Prepare the chat execution settings
    chat_execution_settings = OpenAIChatPromptExecutionSettings(
        ai_model_id=model,
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9
    )
    chat_completion_service = kernel.get_service(type=ChatCompletionClientBase)  
    chat_history = ChatHistory()

    chat_prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="grounded_response",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="query_term", description="The user input", is_required=True),
    ],
    execution_settings=chat_execution_settings,
)

    chat_function = kernel.add_function(
        function_name="ChatGPTFunc",
        plugin_name="chatGPTPlugin",
        prompt_template_config=chat_prompt_template_config
    )
    arguments = KernelArguments( query_term=prompt)

    # Invoke the function with the prompt
    result = await kernel.invoke(
                chat_function,arguments
            )

    return result

async def main(prompt):
    result = await get_completion_skernel(prompt)
    print(result)

# Run the main function
#asyncio.run(main(prompt))

