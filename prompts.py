from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an geographics assistant for question-answering tasks about countries."
)
human_prompt = (
    "Use the following pieces of retrieved context to answer this question {input}."
    "If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "\n"
    "this the context:"
    "\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)