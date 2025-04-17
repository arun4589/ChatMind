from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnableLambda
import tempfile
# file = '1706.03762v7.pdf'
def get_embed(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    loader = PyMuPDFLoader(tmp_path)
    doc = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        separators=' ',
        chunk_overlap = 5
    )
    docs = splitter.split_documents(doc)
    embed_model = OllamaEmbeddings(model = 'nomic-embed-text')
    store = Chroma(
        embedding_function=embed_model,
        persist_directory="data",
        collection_name='doc_collection'
    )
    store.add_documents(docs)
    return store

def get_ans(query,v_store):
    parser = StrOutputParser()
    prompt = PromptTemplate(template="""On the basis of given context : \n {context} \n
                             answer query if required also use your information and query is : {query}""" , 
                             input_variables=['context','query'])
    llm = ChatOllama(model='mistral')
    retriver = v_store.as_retriever(
        search_type = 'mmr',
        search_kwargs = {'k':3,'lambda_mult':0.5}
    ) 
    retrive = MultiQueryRetriever.from_llm(
    retriever=v_store.as_retriever(search_kwargs={'k':3}),
    llm=llm
    )

    
    chain = RunnableLambda(lambda q : retrive.invoke(q)) | RunnableLambda(lambda docus: {
            "context": "\n\n".join([doc.page_content for doc in docus]),
            "query": query
        }) | prompt | llm | parser

    return chain.invoke(query)  

    
# print(get_ans('how many encoders and decoders are there in transformer dicussed in this paper', doc_loader(file)))


