#MistralAI

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING, Iterator
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from llama_cpp import Llama
import os
from llama_index.core.utils import get_cache_dir
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
DEFAULT_LLAMA_CPP_MODEL_VERBOSITY = True

from pydantic import BaseModel
# class ChatCompletionResponseChoice(BaseModel):
#     index: int
#     message: ChatMessage
#     finish_reason: Optional[FinishReason]

from mistralai.models.chat_completion import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionResponseChoice,
    ResponseFormat,
    ToolChoice,
    FinishReason
)
from mistralai.models.common import UsageInfo

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo




#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)

from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    get_from_param_or_env,
    stream_chat_to_completion_decorator,
)
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.llms.mistralai.utils import (
    is_mistralai_function_calling_model,
    mistralai_modelname_to_contextsize,
)

from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ToolCall

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool
    from llama_index.core.chat_engine.types import AgentChatResponse

# DEFAULT_MISTRALAI_MODEL = "mistral-tiny"
# DEFAULT_MISTRALAI_ENDPOINT = "https://api.mistral.ai"
# DEFAULT_MISTRALAI_MAX_TOKENS = 512

from mistralai.models.chat_completion import ChatMessage as mistral_chatmessage


def to_mistral_chatmessage(
    messages: Sequence[ChatMessage],
) -> List[mistral_chatmessage]:
    # print("to mistral rec messages:\n\n",messages)
    new_messages = []
    for m in messages:
        tool_calls = m.additional_kwargs.get("tool_calls")
        # print("m:\n\n",m)
        # print("to mistral tool_calls:\n\n",tool_calls)
        new_messages.append(
            mistral_chatmessage(
                role=m.role.value, content=m.content, tool_calls=tool_calls
            )
        )
        # print("To Mistral Chatmessage output:","\n",new_messages)
    # print(new_messages)
    return new_messages
    


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


class LlamaCppFunctionCalling(FunctionCallingLLM):
    """MistralAI LLM.

    Examples:
        `pip install llama-index-llms-mistralai`

        ```python
        from llama_index.llms.mistralai import MistralAI

        # To customize your API key, do this
        # otherwise it will lookup MISTRAL_API_KEY from your env variable
        # llm = MistralAI(api_key="<api_key>")

        llm = MistralAI()

        resp = llm.complete("Paul Graham is ")

        print(resp)
        ```
    """

    model_path: str = Field(
        description="The path to the llama-cpp model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    timeout: float = Field(
        default=120, description="The timeout to use in seconds.", gte=0
    )
    max_retries: int = Field(
        default=5, description="The maximum number of API retries.", gte=0
    )
    safe_mode: bool = Field(
        default=False,
        description="The parameter to enforce guardrails in chat generations.",
    )
    seed: str = Field(
        default=None, description="The random seed to use for sampling."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the MistralAI API."
    )
    verbose: bool = Field(
    default=DEFAULT_LLAMA_CPP_MODEL_VERBOSITY,
    description="Whether to print verbose output.",
    )
    
    _model: Any = PrivateAttr()

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        model_path: str = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        timeout: int = 120,
        max_retries: int = 5,
        safe_mode: bool = False,
        seed: Optional[int] = None,
        api_key: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        verbose: bool = DEFAULT_LLAMA_CPP_MODEL_VERBOSITY,
        endpoint: Optional[str] = None,

    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 # check if model is cached
        if model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(
                    "Provided model path does not exist. "
                    "Please check the path or provide a model_url to download."
                )
            else:
                self._model = Llama(model_path=model_path, **additional_kwargs)
        else:
            cache_dir = get_cache_dir()
            model_url = model_url or self._get_model_path_for_version()
            model_name = os.path.basename(model_url)
            model_path = os.path.join(cache_dir, "models", model_name)
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self._download_url(model_url, model_path)
                assert os.path.exists(model_path)

            self._model = Llama(model_path=model_path, **additional_kwargs)
            model_path = model_path
            generate_kwargs = generate_kwargs or {}
            generate_kwargs.update(
                {"temperature": temperature, "max_tokens":  max_tokens}
            )
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # api_key = get_from_param_or_env("api_key", api_key, "MISTRAL_API_KEY", "")

        # if not api_key:
        #     raise ValueError(
        #         "You must provide an API key to use mistralai. "
        #         "You can either pass it in as an argument or set it `MISTRAL_API_KEY`."
        #     )

        # # Use the custom endpoint if provided, otherwise default to DEFAULT_MISTRALAI_ENDPOINT
        # endpoint = endpoint or DEFAULT_MISTRALAI_ENDPOINT

        # self._client = MistralClient(
        #     api_key=api_key,
        #     endpoint=endpoint,
        #     timeout=timeout,
        #     max_retries=max_retries,
        # )
        # self._aclient = MistralAsyncClient(
        #     api_key=api_key,
        #     endpoint=endpoint,
        #     timeout=timeout,
        #     max_retries=max_retries,
        # )

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            safe_mode=safe_mode,
            seed=seed,
            model_path=model_path,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "LlamaCPP_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._model.context_params.n_ctx,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model_path,
            safe_mode=self.safe_mode,
            seed=self.seed,
            is_function_calling_model=True,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model_path,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            # "safe_mode": self.safe_mode,
        }
        return {
            **base_kwargs,
            # **self.additional_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # convert messages to mistral ChatMessage
        print("Kwargs:","\n\n",kwargs)
        # messages = to_mistral_chatmessage(messages)
        # all_kwargs = self._get_all_kwargs(**kwargs)
        # response = self._client.chat(messages=messages, **all_kwargs)

        # prompt = self.messages_to_prompt(messages)
        
        # messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        # print("Messages: \n \n",messages, "\n \n")
        # response = self.complete(messages, formatted=True, **kwargs)
        simple_response = self._model.create_chat_completion(messages=messages, **kwargs)
        print("simple_response: \n \n",simple_response, "\n \n")
        print("Type of simple_response: \n \n",type(simple_response), "\n \n")
        
        response = ChatCompletionResponse(**simple_response)
        # print("Type: ", type(response))
        # print("Response:","\n",response)
        # return completion_response_to_chat_response(completion_response)
        tool_calls = response.choices[0].message.tool_calls

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.choices[0].message.content,
                additional_kwargs={"tool_calls": tool_calls}
                if tool_calls is not None
                else {},
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        # completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        # response = self._client.chat_stream(messages=messages, **all_kwargs)
        response = self.stream_complete(messages, formatted=True, **kwargs)
# Could give issues
        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for chunk in response:
                content_delta = chunk.choices[0].delta.content
                if content_delta is None:
                    continue
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=chunk,
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = await self._aclient.chat(messages=messages, **all_kwargs)
        tool_calls = response.choices[0].message.tool_calls
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.choices[0].message.content,
                additional_kwargs={"tool_calls": tool_calls}
                if tool_calls is not None
                else {},
            ),
            raw=dict(response),
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        # convert messages to mistral ChatMessage

        messages = to_mistral_chatmessage(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._aclient.chat_stream(messages=messages, **all_kwargs)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            role = MessageRole.ASSISTANT
            async for chunk in response:
                content_delta = chunk.choices[0].delta.content
                if content_delta is None:
                    continue
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=chunk,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Predict and call the tool."""
        # misralai uses the same openai tool format
        tool_specs = [tool.metadata.to_openai_tool() for tool in tools]
        print("tool_specs ",tool_specs)
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)
        
        response = self.chat(
            messages,
            tools=tool_specs,
            **kwargs,
        )
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    async def achat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Predict and call the tool."""
        # misralai uses the same openai tool format
        tool_specs = [tool.metadata.to_openai_tool() for tool in tools]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        response = await self.achat(
            messages,
            tools=tool_specs,
            **kwargs,
        )
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: "AgentChatResponse",
        error_on_no_tool_call: bool = True,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, ToolCall):
                raise ValueError("Invalid tool_call object")
            if tool_call.type != "function":
                raise ValueError("Invalid tool type. Unsupported by Mistralai.")
            argument_dict = json.loads(tool_call.function.arguments)

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections
