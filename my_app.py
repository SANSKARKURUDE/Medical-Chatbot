import streamlit as st
from med_chatbot import get_vectorstore,load_llm, get_prompt
from langchain.chains import RetrievalQA 

st.title("Ask Medico Chatbot!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt=st.chat_input("How can I assist you with your health today?")

if prompt and prompt.strip():
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content': prompt})

    prompt_template = """
         Use the pieces of information provided in the context to answer user's question.
         If you dont know the answer, just say that you dont know, dont try to make up an answer. 
         Dont provide anything out of the given context

         Context: {context}
         Question: {question}

         Start the answer directly. No small talk please.
         """

    try: 
        vectorstore=get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the vector store")

        qa_chain=RetrievalQA.from_chain_type(
             llm=load_llm(),
             chain_type="stuff",
             retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
             return_source_documents=True,
             chain_type_kwargs={'prompt':get_prompt(prompt_template, input_variables=["context", "question"])}
         )

        response=qa_chain.invoke({'query':prompt})

        result=response["result"]
        #source_documents=response["source_documents"]
        #result_to_show=result+"\n\nSource Docs:"+str(source_documents)
        st.chat_message('assistant').markdown(result)
        st.session_state.messages.append({'role':'assistant', 'content': result})

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    except Exception as e:
        import traceback
        st.error(f"Error: {str(e)}")
        st.text(traceback.format_exc())


