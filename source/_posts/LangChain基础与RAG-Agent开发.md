---
title: LangChain基础与RAG Agent开发
date: 2025-01-26 23:01:13
mathjax: true
tags: 
  - 大语言模型
  - LangChain
  - RAG
  - LLM
  - 人工智能
  - AI
  - Agent
categories:
  - 技术
---
# LangChain与RAG完整开发指南

## 一、LangChain核心组件
### 1.1 模型实例化

```python
# 模型调用示例
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    openai_api_key="sk-xxx",
    openai_api_base="https://api.moonshot.cn/v1",  # 支持国内模型
    model_name="moonshot-v1-8k",
    temperature=0.7,  # 控制生成随机性（0-1）
    streaming=True    # 启用流式响应
)
```
关键参数说明：
- `model_name`: 支持主流模型如gpt-4、moonshot系列等
- `temperature`: 值越大生成结果越随机
- `max_tokens`: 控制生成内容的最大长度

---

### 1.2 提示词模板
#### PromptTemplate 最基础的模板
```python
from langchain_core.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template(
    "给我把{text}翻译成{language}"
)
result = prompt_template.format(text='你好', language='英语')
print(result)
```
输出
```
给我把你好翻译成英语
```
#### ChatPromptTemplate 聊天消息模板
```python
from langchain_core.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个{language}语言助手'),
    ('user', '{text}')
])
```
ChatPromptTemplate其实是将systemmessage和humanmessage等封装成一个消息列表
示例
```python
chat_templates = ChatPromptTemplate.from_messages([
    ('system', '你是一位智能助手，你的名字是{name}'),
    ('human', '你好'),
    ('ai', '我很好，谢谢！'),
    ('human', '{user_input}')
])

chat_templates = chat_templates.format_messages(name='SSAI', user_input='你的名字是什么？')
print(chat_templates)
```
等价于 (建议用法)
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
chat_templates = ChatPromptTemplate.from_messages([
    SystemMessage(content='你是一位智能助手，你的名字是SSAI'),
    HumanMessage(content='你好'),
    AIMessage(content='我很好，谢谢！'),
    HumanMessage(content='你的名字是什么？')
])
chat_templates = chat_templates.format_messages(name='SSAI', user_input='你的名字是什么？')
print(chat_templates)
```

输出
```
[SystemMessage(content='你是一位智能助手，你的名字是SSAI', additional_kwargs={}, response_metadata={}), HumanMessage(content='你好', additional_kwargs={}, response_metadata={}), AIMessage(content='我很好，谢谢！', additional_kwargs={}, response_metadata={}), HumanMessage(content='你的名字是什么？', additional_kwargs={}, response_metadata={})] 
```
#### MessagesPlaceholder 占位符 可以传入一组消息 使得聊天前具有上下文
```python
from langchain_core.prompts import MessagesPlaceholder
chat_templates = ChatPromptTemplate.from_messages([
    ('system', '你是一位智能助手，你的名字是{name}'),
    MessagesPlaceholder("msg") # msg占位 可以传入一组消息
])
chat_templates = chat_templates.format_messages(name='SSAI', msg=[AIMessage(content='很高兴遇到你，请问有什么可以服务的？'), HumanMessage(content='你的名字是什么？')])
print(chat_templates)
```

输出
```
[SystemMessage(content='你是一位智能助手，你的名字是SSAI', additional_kwargs={}, response_metadata={}), AIMessage(content='很高兴遇到你，请问有什么可以服务的？', additional_kwargs={}, response_metadata={}), HumanMessage(content=' 
你的名字是什么？', additional_kwargs={}, response_metadata={})]
```

### 1.3 chain链式 组装提示词模板与模型 调用模型
```python
chain = chat_templates | model 
print(chain.invoke({'name': 'SSAI', 'user_input': '你的名字是什么？'}))
```
输出
```
content='你好！我是你的人工智能助手，你可以叫我SSAI。有什么可以帮助你的吗？' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 40, 'total_tokens': 59, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'moonshot-v1-8k', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-301d3f19-9dbe-4e86-8aa4-5810cd6730a0-0' usage_metadata={'input_tokens': 40, 'output_tokens': 19, 'total_tokens': 59, 'input_token_details': {}, 'output_token_details': {}}
```

### 1.4 输出解析器
```python
parser = StrOutputParser() # 字符串输出解析器
chain = chat_templates | model | parser
print(chain.invoke({'name': 'SSAI', 'user_input': '你的名字是什么？'}))
```
输出
```
你好！我是你的人工智能助手，你可以叫我SSAI。有什么可以帮助你的吗？
```

### 1.5 提示词工程Prompt示例
#### 动态模板示例
```python
from langchain_core.prompts import ChatPromptTemplate

# 多消息类型模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个{domain}领域专家"), 
    ("human", "{query}"),
    ("ai", "好的，我将按照以下步骤分析：\n1."),
    MessagesPlaceholder("history")  # 历史消息占位符
])

# 使用示例
messages = prompt_template.format_messages(
    domain="人工智能",
    query="解释Transformer架构",
    history=[AIMessage(content="欢迎咨询AI问题")]
)
```
#### FewShow
```python
from langchain.prompts.few_shot import FewShotPromptTemplate
examples = [
    {
        'question': '谁的寿命更长，阿尔伯特·爱因斯坦还是斯蒂芬·威廉·霍金？',
        'answer':
            """
            这里需要跟进问题吗：是的。
            跟进：阿尔伯特·爱因斯坦去世时多大？
            中间答案：阿尔伯特·爱因斯坦去世时80岁。
            跟进：斯蒂芬·威廉·霍金去世时多大？
            中间答案：斯蒂芬·威廉·霍金去世时76岁。
            所以最终答案是：阿尔伯特·爱因斯坦的寿命更长。
            """
    },
    {
        'question': '特斯拉电动车的创始人是什么时候出生的？',
        'answer':
            """
            这里需要跟进问题吗：是的。
            跟进：特斯拉电动车的创始人是谁？
            中间答案：特斯拉电动车的创始人是埃隆·马斯克。
            跟进：埃隆·马斯克是哪一年出生的？
            中间答案：埃隆·马斯克出生于1971年6月28日。
            所以最终答案是：特斯拉电动车的创始人是埃隆·马斯克，他出生于1971年6月28日。
            """
    },
    {
        'question': '发明电脑的人与发明互联网的人是诞生于同一个国家吗？',
        'answer':
            """
            这里需要跟进问题吗：是的。
            跟进：发明电脑的人是谁？
            中间答案：发明电脑的人是艾伦·凯。
            跟进：艾伦·凯是哪个国家的人？
            中间答案：艾伦·凯是美国人。
            跟进：发明互联网的人是谁？
            中间答案：发明互联网的人是蒂姆·伯纳斯-李。
            跟进：蒂姆·伯纳斯-李是哪个国家的人？
            中间答案：蒂姆·伯纳斯-李是英国人。
            所以最终答案是：不是。
            """
    }
]
example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题：{question}\\n{answer}")


prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    example_separator="\\n\\n",
    prefix="你是一位智能助手，你的名字是{name}",
    suffix="问题：{input}",
    input_variables=["input", "name"]
)

chain = prompt | model | StrOutputParser()

print(chain.invoke({"name": "SSAI", "input": "发明手机的人现在还活着吗？"}))
```
输出
```
这里需要跟进问题吗：是的。

跟进：发明手机的人是谁？
中间答案：通常认为马丁·库珀（Martin Cooper）是现代手机的发明者，他在1973年展示了第一部手持移动电话。

跟进：马丁·库珀现在是否还活着？
中间答案：截至我的知识更新日期（2023年），马丁·库珀仍然健在。

所以最终答案是：是的，发明手机的人马丁·库珀还活着。
```
#### ExampleSelector 示例选择器 找出与问题最接近的示例
```python
from langchain_core.example_selectors.semantic_similarity import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese"
)

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题：{question}\\n{answer}")

# 语义相似性示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embed_model,
    Chroma,
    k=1,
)

question = "谷歌的创始人是什么时候出生的？"

selected_examples = example_selector.select_examples({"question": question})
print(f"最相似的示例: {question}")
for example in selected_examples:
    print("\\n")
    for k, v in example.items():
        print(f"{k}: {v}")
```
输出
```
最相似的示例: 谷歌的创始人是什么时候出生的？
\n
answer:
            这里需要跟进问题吗：是的。
            跟进：特斯拉电动车的创始人是谁？
            中间答案：特斯拉电动车的创始人是埃隆·马斯克。
            跟进：埃隆·马斯克是哪一年出生的？
            中间答案：埃隆·马斯克出生于1971年6月28日。
            所以最终答案是：特斯拉电动车的创始人是埃隆·马斯克，他出生于1971年6月28日。

question: 特斯拉电动车的创始人是什么时候出生的？
```
#### FewShot与ExampleSelector结合使用
```python
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embed_model,
    Chroma,
    k=1,
)

question = "谷歌的创始人是什么时候出生的？"

selected_examples = example_selector.select_examples({"question": question})
example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题：{question}\\n{answer}")

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=selected_examples,
    example_separator="\\n\\n",
    prefix="你是一位智能助手，你的名字是{name}",
    suffix="问题：{input}",
    input_variables=["input", "name"]
)

chain = prompt | model | StrOutputParser()
print(chain.invoke({"name": "SSAI", "input": question}))
```
输出
```
谷歌的创始人是拉里·佩奇（Larry Page）和谢尔盖·布林（Sergey Brin）。拉里·佩奇出生于1973年3月26日，谢尔盖·布林出生于1973年8月21日。
```




## 二. LangChain核心语法详解
### 2.1 消息类型与处理
```python
from langchain_core.messages import (
    SystemMessage,  # 系统指令
    HumanMessage,   # 用户输入
    AIMessage,      # AI响应
    FunctionMessage # 函数调用结果
)

# 典型消息序列示例
messages = [
    SystemMessage(content="你是一个翻译助手"),
    HumanMessage(content="请翻译：Hello World"),
    AIMessage(content="你好，世界"),
    FunctionMessage(name="search", content="查询结果...")
]
```

### 2.2 输出解析器

```python
from langchain_core.output_parsers import (
    StrOutputParser,      # 字符串输出
    JSONOutputParser,     # JSON格式
    CommaSeparatedListOutputParser # 逗号分隔列表
)

# 结构化输出示例
json_parser = JSONOutputParser()
response = model.invoke("列出三个颜色，返回JSON数组")
parsed = json_parser.invoke(response)

# 输出: ["red", "green", "blue"]
```

### 2.3 文档处理
```python
from langchain_core.documents import Document

# 文档元数据示例
doc = Document(
    page_content="LangChain核心概念...",
    metadata={
        "source": "内部文档",
        "page": 23,
        "category": "技术文档"
    }
)
```

## 三、核心API深度解析
### 3.1 流式输出
同步流式输出
```python
chunks = []

for chunk in model.stream("你好"):
    chunks.append(chunk)
    print(chunk.content, end="", flush=True)
```

异步流式输出
```python
prompt = ChatPromptTemplate.from_template("给我讲一下{topic}的原理")

chain = prompt | model | StrOutputParser()

async def async_stream():
    async for chunk in chain.astream({"topic": "人会饿"}):
        print(chunk, end="", flush=True)

asyncio.run(async_stream())
```
### 3.2 事件流
```python
async def async_stream():
    events = []
    async for event in model.astream_events("你好", version="v2"):
        events.append(event)
    print(events)

asyncio.run(async_stream())
```
输出 从调用模型开始打印事件内容
```
[{'event': 'on_chat_model_start', 'data': {'input': '你好'}, 'name': 'ChatOpenAI', 'tags': [], 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='你好', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='！', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='很高兴', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='为你', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='服务', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='。', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': 
{'chunk': AIMessageChunk(content='有什么', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='可以帮助', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': 
{'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='你的', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='吗', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='？', additional_kwargs={}, response_metadata={}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_stream', 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'data': {'chunk': AIMessageChunk(content='', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'moonshot-v1-8k', 'system_fingerprint': 'fpv0_ca1d2527'}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'parent_ids': []}, {'event': 'on_chat_model_end', 'data': {'output': 
AIMessageChunk(content='你好！很高兴为你服务。有什么可以帮助你的吗？', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'moonshot-v1-8k', 'system_fingerprint': 'fpv0_ca1d2527'}, id='run-bf032d74-a54d-4ccd-8254-13dd0b4a495e')}, 'run_id': 'bf032d74-a54d-4ccd-8254-13dd0b4a495e', 'name': 'ChatOpenAI', 'tags': [], 'metadata': {'ls_provider': 'openai', 'ls_model_name': 'moonshot-v1-8k', 'ls_model_type': 'chat', 'ls_temperature': 0.0}, 'parent_ids': []}]
```

### 3.3 Runnable协议
```python
from langchain_core.runnables import (
    RunnableLambda,    # Lambda函数封装
    RunnableParallel,   # 并行执行
    RunnablePassthrough # 数据透传
)

# 复杂链示例
chain = (
    RunnableParallel({
        "text": RunnablePassthrough(),
        "length": RunnableLambda(lambda x: len(x))
    })
    | RunnableLambda(lambda d: f"{d['text']} (长度:{d['length']})")
)

```

### 3.4 自定义工具創建及工具绑定
#### 自定义工具創建方法一
```python
from langchain_core.tools import tool

# 自定义工具
@tool
def multiply(a:int, b:int) -> int:
    '''Multiply two numbers.'''
    return a * b
```
#### 自定义工具創建方法二
```python
from langchain_core.tools import StructuredTool
def multiply1(a:int, b:int) -> int:
    '''Multiply two numbers.'''
    return a * b

async def multiply2(a:int, b:int) -> int:
    '''Multiply two numbers.'''
    return a * b
```
#### 工具調用
```python
# 同步调用
calculator = StructuredTool.from_function(func=multiply1)
print(calculator.invoke({"a": 2, "b": 3}))

# 异步调用
import asyncio

async def main():
    calculator = StructuredTool.from_function(coroutine=multiply2)
    print(await calculator.ainvoke({"a": 2, "b": 3}))

asyncio.run(main())
```

#### 工具异常捕获
```python
from langchain_core.tools import ToolException

def get_weather(city: str) -> str:
    raise ToolException("City not found")

get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=True
    # handle_tool_error="没找到这个城市"
)

resp = get_weather_tool.invoke({"city": "New York"})
```

#### 工具綁定
```python
# 工具绑定
model_with_tools = model.bind_tools(
    [get_weather_tool, multiply1],
    tool_choice="auto"
)
```


### 3.5 聊天历史管理，使用ChatMessageHistory存储
```python
# 引入聊天历史记录
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 创建聊天提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个智能聊天助手擅长{ability}，请尽你所能回答用户问题"
        ),
        # 历史聊天占位符
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

store = {}

# 定义一个获取会话历史的函数，传入session_id，返回会话历史记录
def get_session_history(user_id: str, session_id: str)-> BaseChatMessageHistory:
    if (user_id, session_id) not in store:
        store[(user_id, session_id)] = ChatMessageHistory()
    return store[(user_id, session_id)]

# 创建一个带会话历史消息记录的Runnable 链
with_message_history = RunnableWithMessageHistory(
    runnable, 
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="用户的唯一标识符",
            default='',
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="会话的唯一标识符",
            default='',
            is_shared=True,
        )
    ]
)    
```
测试输出
```python
response = with_message_history.invoke(
    {"ability": "math", "input": "余弦相似性是什么？"},
    config={"configurable": {"user_id": "123456", "session_id": "123456"}},
)

print(response.content)

response = with_message_history.invoke(
    {"ability": "math", "input": "请再讲一遍？"},
    config={"configurable": {"user_id": "123456", "session_id": "123456789"}},
)

print(response.content)
```
输出
```bash
余弦相似性（Cosine Similarity）是一种衡量两个向量之间相似性的方法，它通过计算两个非零向量的夹角的余弦值来 确定它们之间的相似度。余弦相似性经常用于文本处理、机器学习和信息检索等领域，尤其是在比较文本文档或用户偏好时。在文本分析中，每个单词可以被看作一个维度，词频（TF）或者TF-IDF值可以作为该维度的向量分量，通过计算文档向量之间的余 弦相似性来衡量文档间的相似度。

您好！请告诉我您需要我讲什么内容，或者有什么问题需要我帮助解答的？我会尽我所能为您提供信息和帮助。
```

### 3.6 聊天历史管理，使用RedisChatMessageHistory存储
```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

REDIS_URL = "redis://:password@localhost:6379/0"  # 添加密码

# 定义一个获取会话历史的函数，传入session_id，返回会话历史记录
def get_session_history(session_id: str)-> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id,
        url=REDIS_URL,  # 使用包含密码的URL
    )

# 创建一个带会话历史消息记录的Runnable
with_message_history = RunnableWithMessageHistory(
    runnable, 
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

### 3.7 消息裁剪 避免历史消息过多
```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# 历史消息
temp_chat_history = ChatMessageHistory()
temp_chat_history.add_ai_message("你好，我是你的智能助手。有什么问题可以问我。")
temp_chat_history.add_user_message("你好，我叫张三，我今年20岁。")
temp_chat_history.add_ai_message("很高兴认识你张三")
temp_chat_history.add_user_message("我今天很开心")
temp_chat_history.add_ai_message("那非常棒")
temp_chat_history.add_user_message("我下午要去打球")
temp_chat_history.add_ai_message("那祝你玩的开心")
temp_chat_history.messages

# 裁剪历史消息函数
def trim_messages(chain_input):
    stored_messages = temp_chat_history.messages
    if len(stored_messages) <= 2:
        return False
    temp_chat_history.clear()
    for message in stored_messages[-2:]:
        temp_chat_history.add_message(message)
    return True

# 创建一个带会话历史消息记录的Runnable
with_message_history = RunnableWithMessageHistory(
    chain, 
    lambda session_id: temp_chat_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 合并裁剪的历史消息记录到runnable
chain_with_trimming = (
    RunnablePassthrough.assign(messages_trimmed=trim_messages) | with_message_history
)
```

测试输出
```python
response = chain_with_trimming.invoke(
    {"input": "我今年几岁？"},
    {"configurable": {"session_id": "123456"}}
)

print(response.content)
```
输出
```bash
哈哈，这个问题可真把我难住了，我并没有关于你年龄的数据。不过，如果你愿意的话，我们可以聊聊别的，比如你打球的爱好或 
者你最近遇到的有趣的事情。
```

### 3.8 总结历史消息
```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
# 创建一个带会话历史消息记录的Runnable
with_message_history = RunnableWithMessageHistory(
    chain, 
    lambda session_id: temp_chat_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 总结历史消息函数
def summarize_messages(chain_input):
    stored_messages = temp_chat_history.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
            ("human", "请总结一下我们之前的对话内容,尽可能包含多个细节。")
        ]
    )
    summarization_chain = summarization_prompt | model
    summarize_message = summarization_chain.invoke({"history": stored_messages})
    temp_chat_history.clear()
    temp_chat_history.add_message(summarize_message)
    return True

# 总结消息
chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages) | with_message_history
)
```
测试输出
```python
response = chain_with_summarization.invoke(
    {"input": "名字，年龄，下午干嘛，心情"},
    {"configurable": {"session_id": "123456"}}
)

print(response.content)
```
输出
```bash
根据我们的对话历史，以下是你提供的信息：

- **名字**：张三
- **年龄**：20岁
- **下午干嘛**：要去打球
- **心情**：今天很开心
```

### 3.9 网页搜索
```python
from langchain_community.document_loaders import WebBaseLoader

# 傳入網頁
loader = WebBaseLoader("https://zh.wikipedia.org/wiki/通用人工智慧")

# 加載網頁内容
docs = loader.load()
```

### 3.10 分割documents
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

```



## 四、关键组件详解
### 4.1 向量数据库操作
#### 创建embedding
```python
from langchain_huggingface import HuggingFaceEmbeddings

# 使用huggingface的嵌入模型
embed_model = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese"
)
```

####  创建向量数据库
```python
from langchain_community.vectorstores import Chroma

vector_store = Chroma.from_documents(documents, embedding=embed_model)
```

#### 查询向量
```python
from langchain_community.vectorstores import Chroma

# 查询示例
results = vector_store.similarity_search(
    query="机器学习应用",
    k=5,                  # 返回5个结果
    filter={"year": 2023}, # 元数据过滤
    score_threshold=0.7   # 相似度阈值
)

# 分数查询
results = vector_store.similarity_search_with_score(
    query="机器学习应用",
    k=5,                  # 返回5个结果
    filter={"year": 2023}, # 元数据过滤
    score_threshold=0.7   # 相似度阈值
)

# 混合搜索（BM25 + 向量）
from langchain_community.retrievers import BM25Retriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        BM25Retriever.from_documents(docs),
        vector_store.as_retriever()
    ],
    weights=[0.4, 0.6]
)
```
### 4.2 检索器
#### 创建检索器
```python
retriever = vector.as_retriever()
```

#### 创建检索器工具
```python
retriever_tool = create_retriever_tool(
    retriever,
    "search",
    "搜索网页内容"
)
```

#### 整合模型与检索器
```python
from langchain.chains.retrieval import create_retrieval_chain

chain2 = create_retrieval_chain(retriever, chain1)
```

### 4.2 SQL数据库集成
```python
# 高级查询配置
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model,
    custom_tools=[custom_sql_tool],  # 自定义工具
    max_string_length=500           # 结果截断长度
)

# 安全配置
db = SQLDatabase.from_uri(
    MYSQL_URI,
    include_tables=['drug_table'],  # 限制访问表
    sample_rows_in_table_info=2      # 表信息采样行数
)

```

### 4.3 Agent基础
#### Tavily搜索工具
```python
from langchain_community.tools.tavily_search import TavilySearchResults

os.environ["TAVILY_API_KEY"] = 'tvly-xxxxx'
search = TavilySearchResults(max_results=2) # 返回两个搜索结果
```

#### 封裝工具列表
```python
tools = [search, retriever_tool]
```

#### 导入prompt
```python
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")
```

#### 创建agent 传入model、工具和prompt
```python
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(model, tools, prompt)
```

#### 创建agent执行器
```python
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

#### 创建带记忆的agent
```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

from langchain_core.runnables import RunnableWithMessageHistory

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
```
测试输出
```python
resp = agent_with_history.invoke(
    {
        "input": "通用人工智能是什麽？杭州天氣今天怎麽樣？"
    },
    config={"configurable": {"session_id": "123"}}   
)

print(resp)
```
输出
```bash
{'input': '通用人工智能是什麽？杭州天氣今天怎麽樣？', 'chat_history': [], 'output': '通用人工 
智能（Artificial General Intelligence，AGI）是指具有一般人类智慧，可以执行人类能够执行的任何智
力任务的机器智能。它具有高效的学习和泛化能力、能够根据所处的复杂动态环境自主产生并完成任务的通
用人工智能体，具备自主的感知、认知、决策等能力。\n\n至于杭州的天气，今天的天气状况如下：白天最
高气温2℃，最低气温-4℃，北风4~5级；夜间东北风3~4级，气温6℃。杭州市临平区、余杭区、临安区气象台 
都发布了低温橙色预警，提醒市民注意防寒保暖。天气寒冷，建议您穿着保暖衣物，尽量减少户外活动。'}
```


### 4.4 复杂代理配置
```python
from langgraph.prebuilt import chat_agent_executor
# 多工具代理配置
tools = [
    TavilySearchResults(max_results=2),
    SQLDatabaseToolkit(db=db).get_tools()[0],
    currency_converter
]

# 带记忆的代理
agent = chat_agent_executor.create_tool_calling_executor(
    model=model,
    tools=tools,
    checkpointer=RedisCheckpointer()  # 持久化记忆
)
```

## 五、RAG核心实现
准备embedding模型 -> 准备私有化数据加载数据 -> 分块chunks -> embedding嵌入数据到向量数据库 -> 生成检索器 -> llm整合prompt检索数据 -> 输出

### 5.1 数据处理流程
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 智能分块策略
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # 块大小
    chunk_overlap=200,     # 重叠区
    length_function=len,
    is_separator_regex=False
)

# 中文嵌入模型配置
embed_model = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': 'cuda'},  # GPU加速
    encode_kwargs={'normalize_embeddings': True}
)
```
### 5.2 向量存储
```python
# 使用HuggingFace嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embed_model,
    persist_directory="./chroma_db"
)
```
### 5.3 检索增强
```python
from langchain_core.runnables import RunnablePassthrough

# 文档格式化函数
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# RAG完整链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} 
    | ChatPromptTemplate.from_template("基于以下上下文：\n{context}\n回答：{question}")
    | model
    | StrOutputParser()
)

# 调用示例
response = rag_chain.invoke("什么是注意力机制？")
```

## 六、进阶功能实现
### 6.1 数据库集成
```python
# MySQL数据库连接
from langchain_community.utilities import SQLDatabase
MYSQL_URI = 'mysql+mysqldb://user:pass@host:port/db'
db = SQLDatabase.from_uri(MYSQL_URI)
# SQL Agent配置
from langchain_community.agent_toolkits import SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
```
### 6.2 网络搜索集成
```python
from langchain_community.tools.tavily_search import TavilySearchResults
os.environ["TAVILY_API_KEY"] = 'tvly-xxxxx'
search = TavilySearchResults(max_results=3) # 返回三个搜索结果
tools = [search]
# Agent执行器配置
from langgraph.prebuilt import chat_agent_executor
agent_executor = chat_agent_executor.create_tool_calling_executor(
    model=model,
    tools=tools
)
```

### 6.3 多模态输入(需要用多模态模型)
#### 图片描述 访问url识别
```python
import base64
# 用httpx 获取图片的二进制数据
import httpx
from langchain_core.messages import HumanMessage

# 获取图片并转换为 base64
image_url = "http://47.99.123.130/assets/blog-2.CBbwvyj6.png"

# 将图片的二进制数据转换为base64编码
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

# 使用正确的格式构造消息
message = HumanMessage(
    content=[
        {'type': 'text', 'text': "用中文描述这张图片"},
        {'type': "image_url", "image_url": {"url": image_url}},
    ]
)

response = model.invoke([message])
print(response.content)
```
#### 图片描述 转码base64下载识别
```python
message = HumanMessage(
    content=[
        {'type': 'text', 'text': "用中文描述这张图片"},
        {'type': "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
    ]
)
```

#### 多图片识别
```python
# 获取图片并转换为 base64
image_url1 = "http://47.99.123.130/assets/blog-2.CBbwvyj6.png"
image_url2 = "http://47.99.123.130/assets/blog-1.Bhl1hejx.png"

# 使用正确的格式构造消息
message = HumanMessage(
    content=[
        {'type': 'text', 'text': "这两张图片是一样的吗？"},
        {'type': "image_url", "image_url": {"url": image_url1}},
        {'type': "image_url", "image_url": {"url": image_url2}},
    ]
)

response = model.invoke([message])
print(response.content)
```

###





## 七、最佳实践
### 7.1 部署服务
```python
from fastapi import FastAPI
from langserve import add_routes
app = FastAPI()
add_routes(app, rag_chain, path="/rag-api")
if name == "main":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```
或 创建脚手架项目
```bash
pip install --upgrade "langserve[all]"
pip install -U langchain-cli
pip install pipx
pipx ensurepath 
pip install poetry
# 安装依赖
poetry add langchain

# 创建项目
langchain app new langserve 

# 启动项目
poetry run langchain serve --port=8000

```


### 7.2 Langsmith监控与追踪
windows导入环境变量
```bash
setx LANGCHAIN_TRACING_V2 "true" # 开启监控
setx LANGCHAIN_API_KEY "lsv2_xxx" # 设置smith的apikey
setx TAVILY_API_KEY "tvly-xxxxx" # 设置tavily的apikey
```

```python
# LangSmith配置
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_xxx"
os.environ["LANGCHAIN_PROJECT"] = "rag-project"
os.environ["TAVILY_API_KEY"] = "tvly-xxxxx"
```
在langsmith可以看到历史链路记录


```python
from langchain.globals import set_verbose, set_debug
# 打印详细日志
set_verbose(True)
# 打印调试详细日志
set_debug(True)
```

## 八、Agent开发
### LangGraph

## 九、性能优化
1. 分块策略优化：根据内容类型调整chunk_size
2. 多路召回：结合关键词检索和向量检索
3. 缓存机制：对常见查询结果缓存
4. 异步处理：使用async/await提升并发性能

## 参考资源
- [LangChain官方文档](https://python.langchain.com/docs)
- [RAG技术白皮书](https://arxiv.org/abs/2005.11401)
- [示例代码仓库](https://github.com/pixegami/rag-tutorial-v2)

