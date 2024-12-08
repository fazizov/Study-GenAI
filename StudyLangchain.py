from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings  
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import YoutubeLoader,PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter 
from langchain.vectorstores import  FAISS
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from datasets import load_dataset


import os

from langchain.agents.react.agent import create_react_agent

load_dotenv()

model = AzureChatOpenAI(azure_deployment=  os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],temperature=0)

emb_model=AzureOpenAIEmbeddings(azure_endpoint=os.environ["AZURE_OPENAI_EMB_ENDPOINT"],api_key=os.environ["AZURE_OPENAI_EMB_API_KEY"],
                                api_version=os.environ["AZURE_OPENAI_EMB_API_VERSION"])

#1. Simple LLM call

def simpleLLMCall():
    resp=model.invoke("Tell me dads joke")
    print(resp.content)
    return (resp)

#2. Prompt template usage
def petNamer(p_pet_type,p_pet_breed):
    temp="Suggest 3 cool names for my {pet_type} of following breed {pet_breed}"
    pt=PromptTemplate(input_variables=['pet_type','pet_color'],template= temp)
    lc=pt|model|StrOutputParser()
    resp=lc.invoke({'pet_type':p_pet_type,'pet_breed':p_pet_breed})
    return (resp)

#3. Multi-step Chains
def multiStepChains(p_season):
    # prompt = ChatPromptTemplate.from_template("Suggest one cool place to visit in {season} time of the year")
    prompt = PromptTemplate(template="Suggest one cool place to visit in {season} time of the year")
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"season": p_season}))
    analysis_prompt = PromptTemplate.from_template(template="Translate this text to Turkish {output_text}")
    composed_chain = {"output_text": chain} | analysis_prompt | model | StrOutputParser()
    print(composed_chain.invoke({"season": p_season}))
    return 

#4. Vector index
def vectorSearchFn(my_question):
    loader=PyPDFLoader("Langchain\\nvc.pdf")
    pdf_docs=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pdf_docs)
    db=FAISS.from_documents(documents=docs,embedding=emb_model)
    ind_docs=db.similarity_search(query=my_question,k=4)
    docs_page_content = " ".join([d.page_content for d in ind_docs])
    pt=PromptTemplate(template=""" You are a helpful assistant that can answer questions about documents.
            Answer the following question: {question}
            By searching the following document: {docs}
            List 3 most relevant answers.                  
            Only use the factual information from the document to answer the question.
            If you feel like you don't have enough information to answer the question, say "I don't know".
            Your answers should be verbose and detailed.
            """, input_variables=["question", "docs"])

    chain=pt|model|StrOutputParser()
    resp=chain.invoke({"question":my_question,"docs":docs_page_content})
    print(resp)

#5 Retrievers
def wikiRetriever(topic):
    retriever = WikipediaRetriever()
    docs = retriever.invoke(input=topic)
    return (docs[0])


# Tools and chains
def learnTools(question):
    @tool
    def multiply(first_int: int, second_int: int) -> int:
        """Multiply two integers together."""
        return first_int * second_int

    @tool
    def summarize(first_int: int, second_int: int) -> int:
        """Summarize two integers together."""
        return first_int + second_int
    print(multiply.invoke({"first_int": 4, "second_int": 5}))
    print(summarize.invoke({"first_int": 4, "second_int": 5}))
    llm_with_tools = model.bind_tools([multiply,summarize])
    msg = llm_with_tools.invoke("whats 5 times forty two")
    msg2 = llm_with_tools.invoke("whats 5 added to forty")
    print(msg.tool_calls,msg2.tool_calls)
    lc=llm_with_tools | (lambda x: x.tool_calls[0]["args"])|multiply
    print(lc.invoke(question))

 # Agents
def learnToolsAgent():
    @tool
    def multiply(first_int: int, second_int: int) -> int:
        """Multiply two integers together."""
        return first_int * second_int

    @tool
    def add(first_int: int, second_int: int) -> int:
        "Add two integers."
        return first_int + second_int


    @tool
    def exponentiate(base: int, exponent: int) -> int:
        "Exponentiate the base to the exponent power."
        return base**exponent
    prompt = hub.pull("hwchase17/openai-tools-agent")
    prompt.pretty_print()

    tools = [add, exponentiate]
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke(
    {
        "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
    }
)

def learnReactAgent():

    vtools = [TavilySearchResults(max_results=1)]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(model, vtools, prompt)
    agent_executor  = AgentExecutor(agent=agent, tools=vtools, verbose=True, handle_parsing_errors=True)
    agent_executor.invoke({"input": "What is average price of detached houses in London Ontario?What was the trend for last 2 years and what would be price prediction for 2025?"})

def pandasDFAgent():
    dataset = load_dataset("maharshipandya/spotify-tracks-dataset")
    df = dataset["train"].to_pandas()
    prefix = """ Input should be sanitized by removing any leading or trailing backticks. if the input starts with ”python”, remove that word as well. Use the dataset provided. The output must start with a new line."""

    dataqio = create_pandas_dataframe_agent(
    model,
    df,
    verbose=True,
    max_iterations=3,
    prefix=prefix,
    agent_executor_kwargs={
        "handle_parsing_errors": True
    }
    )
    dataqio.invoke("What is the artist name of most popular song based on popularity?")
    dataqio.invoke("What is the total number of rows?")

#Memory
# from langchain.chains.conversation.memory import ConversationBufferMemory
# from  langchain.chains import Convers

# def  simpleConvMemory():
#     mem=ConversationBufferMemory()
#     model = AzureChatOpenAI(azure_deployment=  os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],temperature=0,mem)

#1. print(petNamer('dog','Husky'))
#3. multiStepChains('Summer')
#4 vectorSearchFn("Why moralistic judgements are not good?")
#5 print(wikiRetriever("What are top reasons for climate change"))
#6 learnTools("whats 15 times two")
#7 learnToolsAgent()
#8 learnReactAgent()
pandasDFAgent()
# prompt = hub.pull("hwchase17/react")
# print (prompt)