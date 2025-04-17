import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from src.qa_agent import get_ans,get_embed
from src.general_tool_agent import answer_with_tools

st.set_page_config(page_title="LangChain Chatbot", layout="wide")
st.title("🧠 LangChain Chatbot with Document Q&A")

# Tabs for sections
tab1, tab2 = st.tabs(["💬 General Chat", "📄 Document Q&A"])

# ========== TAB 1: General Chat ==========
with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "conversation" not in st.session_state:
        llm = ChatOllama(model="mistral")
        memory = ConversationBufferMemory()
        st.session_state.conversation = ConversationChain(llm=llm, memory=memory)

    st.subheader("💬 Chat with Memory")

    if st.button("🧹 New Conversation"):
        st.session_state.chat_history = []
        st.session_state.conversation.memory.clear()

    user_input = st.chat_input("Say something...")

    if user_input:
        with st.spinner("Thinking..."):
            # if you want to use tool agent like for code creatiion or searching then enable below code and also remove first line 
            # just after commented code -> response = st.session_state.conversation.run(user_input)
            # try:
            #     result = answer_with_tools.invoke({"input": user_input})
            #     intermediate_steps = result.get("intermediate_steps", [])
            #     response = result["output"]

                
            #     for step in intermediate_steps:
            #         action = step[0]
            #         observation = step[1]
            #         st.session_state.chat_history.append(("Bot", f"🔧 **Tool Used:** `{action.tool}`"))
            #         st.session_state.chat_history.append(("Bot", f"🛠️ **Input:** `{action.tool_input}`"))
            #         st.session_state.chat_history.append(("Bot", f"📤 **Output:**\n```\n{observation}\n```"))
            # except Exception as e:
            #     response = st.session_state.conversation.run(user_input)
            response = st.session_state.conversation.run(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)
with tab2:
    st.subheader("📄 Ask Questions Based on Document")

    uploaded_file = st.file_uploader("Upload a PDF or text document", type=["pdf", "txt"])

    if uploaded_file:
        # Cache or track embedding only once
        if 'vec_store' not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
            st.session_state.vec_store = get_embed(uploaded_file)
            st.session_state.last_file = uploaded_file.name

        query = st.text_input("Ask a question about the document:")

        if query:
            with st.spinner("Searching the document..."):
                try:
                    vec_store = st.session_state.vec_store  
                    answer = get_ans(query, vec_store)  
                    st.success(answer)
                except Exception as e:
                    st.error(f"Error: {e}")


            
