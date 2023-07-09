import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def save_file(file):
    file_path = os.path.join("./TrainingData", file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    else:
        return False

def delete_files_in_directory():
    for filename in os.listdir("./TrainingData"):
        file_path = os.path.join("./TrainingData", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def main():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get your OpenAI API Key](https://platform.openai.com/account/api-keys)"
        # URL Uploader
        url_path = st.text_input("Input the Web URL Path", key="url_key")

        folder_path = "./TrainingData"

        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # File uploader
        file = st.file_uploader("Upload a PDF or TXT file or enter a web link", type=["pdf", "txt"])

        if file is not None:
            # Save the file to the specified folder
            file_path = save_file(file)
            st.success(f"File saved successfully at: {file_path}")
        
        # File deleter
        st.subheader("Delete Files")
        files_in_folder = os.listdir(folder_path)
        if len(files_in_folder) == 0:
            st.info("No files found in the folder.")
        else:
            selected_file = st.selectbox("Select a file to delete:", files_in_folder)
            delete_button = st.button("Delete File")            
            if delete_button:
                file_path = os.path.join(folder_path, selected_file)
                if delete_file(file_path):
                    st.success("File deleted successfully.")
                else:
                    st.error("File not found or unable to delete.")

        #Clear Directory
        st.subheader("Clear Model Data")
        delete_model_data = st.button("Clear Model Data")
        if delete_model_data:
            delete_files_in_directory()

    with st.container():
        st.write('<h2 style="display: inline-block; padding: 0px; margin: 0px;">🦜🔗 QnA Bot using LLM on Own Dataset</h2>', unsafe_allow_html=True)
        st.write('<br><br>', unsafe_allow_html=True)

    prompt = st.text_input('Input your prompt here')

    if prompt:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        os.environ['OPENAI_API_KEY'] = openai_api_key

        #If Temprature is set low then it will try to give only factual answer, 
        #for high it will try to be creative
        #Setting verbose to True will print out some internal states of the Chain object while it is being ran
        llm = OpenAI(temperature=0.1, verbose=True) 

        #Large Language Models (LLMs) encode words and other terms into vectors based on their context 
        # in sentences, based on training from a massive corpus. 
        # We are using OpenAIEmbeddings API for doing it          
        embeddings = OpenAIEmbeddings()

        #Loaders will load the data from the files into a list[Document] 
        loaders = []

        for filename in os.listdir("./TrainingData"):
            file_path = os.path.join("./TrainingData", filename)
            if os.path.isfile(file_path):
                if filename.endswith(".pdf"):
                    loaders.append(PyPDFLoader(file_path))
                elif filename.endswith(".txt"):
                    loaders.append(TextLoader(file_path))
                # Add support for other file types here
        if url_path:
            loaders.append(WebBaseLoader(url_path))
            

        if loaders:
            pages = []
            for loader in loaders:
                # split the documents into chunks
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                #pages.extend(loader.load_and_split())
                pages.extend(text_splitter.split_documents(loader.load()))

            #Create Memory for Vector Store
            persist_directory = "docs/chroma/"
            store = Chroma.from_documents(pages, embeddings, collection_name='TrainingData',persist_directory=persist_directory)

            for loader in loaders:
                documents = loader.load()
                store.add_documents(documents)

            vectorstore_info = VectorStoreInfo(
                name="Model_Data",
                description="Training Data",
                vectorstore=store
            )

            toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

            #Create VectorStore Agent 
            agent_executor = create_vectorstore_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True
            )


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
            
            #Run chain with direct question without any chain technique
            #response = agent_executor.run(prompt)

            #Query to Model
            response = qa_chain(prompt)
            
            st.write(response["result"])
            
            #Document Similarity Search
            with st.expander('Document Similarity Search'):
                search = store.similarity_search_with_score(prompt)
                #search = store.max_marginal_relevance_search(prompt,k=2,fetch_k=3)
                st.markdown('<h4>Reference 1</h4>', unsafe_allow_html=True)
                st.write(search[0][0].page_content) #For similarity_search_with_score
                #st.write(search[0].page_content) #For max_marginal_relevance_search
                st.markdown('<h4>Reference 2</h4>', unsafe_allow_html=True)
                st.write(search[1][0].page_content) #For similarity_search_with_score
                #st.write(search[1].page_content) #For max_marginal_relevance_search
                #print( search)     #To debug result
        else:
                st.info("Please Load the Data to Train the Model")

if __name__ == "__main__":
    main()
