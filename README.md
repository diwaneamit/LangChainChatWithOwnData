<h1 align="center">LLM LangChain Chatbot POC With Your Own Data</h1>

<div align="center">
 

[![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/apps)
[![LangChain](https://dcbadge.vercel.app/api/server/KgmN4FPxxT?compact=true&style=flat)](https://python.langchain.com/docs/get_started/introduction.html)

![Logo](https://github.com/diwaneamit/LangChainChatWithYourData/blob/main/Images/LangChain_Logo.jpg)
</div>
 
### POC to create a chatbot using LangChain LLM
 
This repository serves as a comprehensive summary of the knowledge acquired during my participation in the <b>LLM with LangChain Courses</b> offered by the [DeepLearning.AI](https://www.deeplearning.ai/short-courses/) platform, combined with my hands-on experience. It showcases a small Proof of Concept (POC) where I have mentioned applying each step involved in developing a chatbot using the Language Learning Model (LLM) on a personalized dataset.

The purpose of this repository is to provide learners with a valuable reference tool, enabling them to explore chatbot development using the LLM methodology. The POC implementation serves as a practical example, demonstrating the various stages involved in constructing a chatbot from scratch. By studying the code and accompanying documentation, learners can gain a deeper understanding of the underlying principles and techniques used in this domain.

A handpicked selection of outside references and links that support the ideas discussed in the LLM with LangChain Course is also included in this repository. These sources provide additional ways for learners to explore specific aspects of deep learning, chatbot development, and related subjects.

This repository seeks to offer a useful and accessible resource to assist you traverse the exciting world of deep learning-based chatbot development, whether you are a beginner looking for an introduction to chatbot development or an experienced practitioner who wants to improve your expertise. To start a rewarding learning trip, explore the code, go through the explanations, and use the resources offered.

### Old Chatbot Framework vs LLM

Traditional systems like Dialogflow have led the way for interactions with computers in the world of natural language processing (NLP) and chatbot development. However, a new platform called LangChain is altering the game with its powerful Language Learning Models (LLMs). 

<b>Traditional Chatbot Limitations:</b> Traditional chatbot platforms, such as Dialogflow, rely on pre-defined rules and templates, making them less flexible and incapable of understanding natural language. They have difficulty generating human-like replies and adapting to varied user inputs.
<br>Reference for POC using DialogFlow in SAP UI5 <a href='https://blogs.sap.com/2019/04/05/integration-of-speech-enabled-chatbot-with-sap-fiori/'> Integration of Speech Enabled Chatbot with SAP Fiori</a>
<br>
<br><b>The Power of LLMs:</b> Using deep learning techniques to understand the complexity of language, LangChain's LLMs revolutionize chatbot development. LLMs can handle complicated language tasks including translation, summarization, and sentiment analysis while delivering context-appropriate replies. This adaptability distinguishes LLMs from traditional chatbots.

Additional Learning <a href='https://cobusgreyling.medium.com/conversational-ai-explained-in-the-simplest-terms-b714662ef960'> Conversational AI Explained in the Simplest term</a>.

### LangChain

<a href='https://python.langchain.com/docs/get_started/introduction.html'>LangChain</a> is a framework designed to simplify the creation of applications using large language models. It is an open-source development framework with Python and Javascript(Typescript) Packages. LangChain was launched in October 2022 as an open-source project by Harrison Chase.

LangChain provides various Modules which make Chatbot Development easy for developers.
<br><br>
<div align='center'>
<img src='https://github.com/diwaneamit/LangChainChatWithYourData/blob/main/Images/LangChain%20Modules.png'/>
<br>Source: DeepLearnig.ai</center><br><br>
</div>

### Development of Chatbot with Own Data

Usually, the development of a Chatbot using LangChain with its Own Data Includes the Following Steps:<br><br>
<br><br>
<div align='center'>
<img src='https://github.com/diwaneamit/LangChainChatWithYourData/blob/main/Images/LangChainProcess.png'/>
<br>Source: DeepLearnig.ai</center><br><br>
</div>
<b>a. Importing Data from Diverse Sources:</b> <br>
LangChain provides over 80 document loaders, allowing developers to effortlessly import data from diverse sources. This feature offers versatility and makes it simple to access a variety of databases.<br>
The Details about all the loaders for Data Connection can be found <a href='https://python.langchain.com/docs/modules/data_connection/'> here</a>
<br><br>
<div align='center'>
<img src='https://github.com/diwaneamit/LangChainChatWithYourData/blob/main/Images/LangChainDocumentLoaders.png'/>
<br>Source: DeepLearnig.ai</center><br><br>
</div>
In my POC I used Data Loaders for PDF, TXT File, and WebURL. <br>

```python
from langchain.document_loaders import PyPDFLoader

loaders = []
loaders.append(PyPDFLoader(file_path))
```
<br><br>

<b>b. Chunking and Complexity Handling: </b><br>
Once the documents are loaded, LangChain allows developers to divide them into smaller, more manageable chunks. This stage is critical for effective processing and addressing the complications that develop when working with large and complicated documents. This helps create pieces of text that are smaller than the original document, which is useful because we may not be able to pass the whole document to the language model. So we want to create these small chunks so we can only pass the most relevant ones to the language model. If our chunks are too small or too large, it may lead to imprecise search results or missed opportunities to surface relevant content.LangChain provides multiple types of Splitters. The most common ones are ChracterTextSplitter, TokenTextSplitter, and RecursiveCharacterTextSplitter. The details about other splitter types can be found <a href='https://python.langchain.com/docs/modules/data_connection/document_transformers/'>here</a>
<br><br>
<div align='center'>
<img src='https://github.com/diwaneamit/LangChainChatWithYourData/blob/main/Images/SplitterTypes.png'/>
<br>Source: DeepLearnig.ai</center><br><br>
</div>
In my POC I used CharacterTextSplitter to split the document in a chunk_size=1000

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loaders = []
loaders.append(PyPDFLoader(file_path))
if loaders:
   pages = []
   for loader in loaders:
        # split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        #pages.extend(loader.load_and_split())
        pages.extend(text_splitter.split_documents(loader.load()))
```
<br><br>

<b>c. Embeddings for Semantic Search:</b><br>
LangChain assists in the generation of embeddings, which are vector representations of text. These embeddings are saved in vector storage, allowing for efficient semantic search. Developers can use semantic similarities to run powerful searches and retrieve relevant documents. LangChain provides many embedding model providers(OpenAI,Cohere,Hugging Face,etc) more details can be found <a href='https://python.langchain.com/docs/modules/data_connection/text_embedding/'>here.</a>.
Details about Vector Stores can be found <a href='https://python.langchain.com/docs/modules/data_connection/vectorstores/'>here.</a>
<br><br>
In my POC I used OpenAI Embeddings and store it in the chroma vector store as its lightweight and in memory which makes it easy to get up and started with

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

#Large Language Models (LLMs) encode words and other terms into vectors based on their context 
# in sentences, based on training from a massive corpus. 
# We are using OpenAIEmbeddings API for doing it  
embeddings = OpenAIEmbeddings()
#Create Memory for Vector Store
persist_directory = "docs/chroma/"
store = Chroma.from_documents(pages, embeddings, collection_name='TrainingData',persist_directory=persist_directory)
```
<br><br>

<b>d. Semantic Search Limitations: </b><br>
Semantic search offers numerous advantages, but it also has limitations. It may struggle to accurately capture the necessary context in some circumstances. It is critical to be aware of these limits and to investigate advanced retrieval methods in order to overcome such issues.
A few limitations discussed in the course are:<br>
In the case of Duplicate entries, it will fetch both similar results, and that way we will not get correct results.<br>
Another problem is it will not check for metadata for the results

In POC in case if you want to try a semantic search without any retrieval technique you can directly call the model to get the response,
Code Part which you need to modify in QA_Bot.py
```python
# Run chain with Prompt
#qa_chain = RetrievalQA.from_chain_type(
#                       llm,
#                       retriever=store.as_retriever(),
#                       return_source_documents=True,
#                       #From below 3 Chain Type Select any one
#                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#                       #chain_type="refine"      
#                       #chain_type="map_reduce"
#                       )
            #Run chain with a direct question without any chain technique
            response = agent_executor.run(prompt)

            #Query to Model
            #response = qa_chain(prompt)
```
<br><br>
<b>e. Integration of Retrieval:</b> <br>
Advanced Retrieval Techniques: LangChain presents advanced retrieval techniques to improve the effectiveness of search results. These algorithms target edge scenarios and give more accurate retrieval, thereby improving the user experience. The course talks about below retrieval techniques<br>
Vector-Store-Based Retrieval Technique
<ol>
  <li>Basic Semantic Similarity</li>
  <li>Maximum Marginal Relevance</li>
  <li>LLM Aided Retrieval</li>
  <li>Compression</li>
</ol><br>
There are other non Vector Based Retrieval Techniques too such as
<ol>
 <li>SVM</li>
 <li>TF-IDF</li>
</ol>
Each of them has there pros and cons. Details about Vector store backed retriever can be found <a href='https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/vectorstore'> here</a>
<br><br>
In my POC I am using the default retriever of vector-store.<br>

```python
 retriever=store.as_retriever(),
 ```
<br><br>
<b>f. Using Language Learning Models:</b><br>
In addition to retrieval, LLMs play an important role in creating exact answers to user queries. LLMs produce context-aware responses by merging retrieved documents with the user's question, allowing chatbots to participate in meaningful dialogues.LangChain provides integration to various different LLM models, details about it can be found <a href='https://python.langchain.com/docs/modules/model_io/models/llms/'>here</a>
<br><br>
<div align='center'>
<img src='https://github.com/diwaneamit/LangChainChatWithYourData/blob/main/Images/RetrievalQAChain.png'/>
<br>Source: DeepLearnig.ai</center><br><br>
</div>
The course also discusses various qa chain techniques such as Map_Reduce, Refine, and Map_rerank to improve the response. So here we can create a retrieval QA chain. This does retrieval and then does question answering over the retrieved documents.<br><br>
1. Map_reduce-<br> This basically takes all the chunks, passes them along with the question to a language model, gets back a response, and then uses another language model call to summarize all of the individual responses into a final answer. This is really powerful because it can operate over any number of documents and it treats all documents independently. But it does take a lot more calls.<br><br>
2. Refine-<br> This is another method, that is again used to loop over many documents. But it actually does it iteratively. It builds upon the answer from the previous document. So this is really good for combining information and building up an answer over time. It will generally lead to longer answers. And it's also not as fast because now the calls aren't independent. They depend on the result of previous calls.<br><br>
3. Map_rerank-<br>It is a pretty interesting and a bit more experimental one where you do a single call to the language model for each document. And you also ask it to return a score.  And then you select the highest score. This relies on the language model to know what the score should be. This is also expensive as we are making a bunch of language model calls.<br><br>
In my POC I tried with Map_reduce and Refine. The code line referring to it in QA_bot.py is as below:<br>

```python
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Build prompt
template = """Use the following pieces of context to answer the question at the end within 4096 tokens limit. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain with Prompt
qa_chain = RetrievalQA.from_chain_type(
                       llm,
                       retriever=store.as_retriever(),
                       return_source_documents=True,
                       #From below 3 Chain Type Select any one
                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                       #chain_type="refine"      
                       #chain_type="map_reduce"
                       )
 ```
<br><br>
<b>f. Building a Fully Functional Chatbot:</b><br>
To complete the Chatbot experience, we must add conversational aspects to the bot. Developers can design a fully working chatbot tailored to their individual needs by exploiting LangChain's Conversational Retrieval Chain features. This end-to-end solution enables users to have dynamic and engaging discussions with personalized and contextually relevant responses. More details about it can be found <a href='https://python.langchain.com/docs/modules/chains/popular/chat_vector_db'>here</a>
<br><br>
<div align='center'>
<img src='https://github.com/diwaneamit/LangChainChatWithYourData/blob/main/Images/ConversationRetrievalChain.png'/>
<br>Source: DeepLearnig.ai</center><br><br>
</div>

In the POC this has been implemented using the below code line which is a part of ChatBot.py file:<br>

```python
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

 #Memory
 memory = ConversationBufferMemory(
          memory_key="chat_history",
          return_messages=True
          )
# create a chatbot chain. Memory is managed externally.
qa_chain = ConversationalRetrievalChain.from_llm(
                 llm, 
                 #chain_type="refine",
                 memory = memory,
                 #chain_type="map_reduce"
                 chain_type="stuff",
                 verbose=True,
                 retriever=store.as_retriever(), 
                 #return_source_documents=True,
                 #return_generated_question=True,
                 combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
                )
```

<br><br>

## Demo
<br>

[![Demo](https://github.com/diwaneamit/LangChainChatWithYourData/blob/main/Images/Bot.png)](https://www.youtube.com/watch?v=6XaJBEKPKso)

## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Start the server

```bash
  streamlit run <filename>.py
```
## Lessons Learned

I was able to better understand the ideas because of this POC. The chatbot was unable to respond to the data when it was first loaded, and it returned an error stating that the response was too long and exceeded the recommended token limitations. I tried lowering the size of the data chunks to get around the issue. I was able to get the answers I needed with it, but at first, I had trouble because I had placed all of the data directly in VectorStore. 

Another problem I noticed was with chain type refine, which used to reply in detail and also used to add sentences on its own in response even though they weren't part of the training data I used. Although I was unable to entirely eliminate this problem, lowering the temperature value somewhat helped. With a different chain type of map_reduce, the outcome was enhanced.

A different issue I saw with the vector store's persistent memory was that the chatbot query call used to fail as it grew bigger. I tried erasing the persistent memory to solve this problem, and it worked.

I tried using ChatGPT to get assistance for a few coding errors, but it typically returned code that used a deprecated method. I had to clarify to the ChatGPT  which method I wanted to use by referring to https://api.python.langchain.com/en/latest/api_reference.html, that way it helped.

Also agents tool is one of the important features of the LangChain LLM, try exploring more around it. It will help to improve your chatbot.

There are many ways to debug the LLM Model, one easy way is by setting verbose=True, Another tool to explore is https://github.com/amosjyng/langchain-visualizer

## Acknowledgements

 - [DeepLearning.ai](https://learn.deeplearning.ai/)

## License

[MIT](https://choosealicense.com/licenses/mit/)


