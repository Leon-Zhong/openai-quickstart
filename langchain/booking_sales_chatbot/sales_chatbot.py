import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    
    global SALES_BOT    
    # SALES_BOT = RetrievalQA.from_chain_type(ChatOpenAI(),
    #                                        retriever=db.as_retriever(search_type="similarity_score_threshold",
    #                                                                  search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果

    # SALES_BOT.return_source_documents = True

    SALES_BOT = db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.8})

    return SALES_BOT

def sales_chat(message, history):
    enable_chat = True
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    ans = SALES_BOT.get_relevant_documents(message)
    ans_list = [doc.page_content.split("[销售回答] ")[-1] for doc in ans]
    
    if len(ans_list) > 0:
        return ans_list[0]
    elif enable_chat:
        template = """
            以下是之前的对话：
            {history}
            客戶的最新问题是：{question}
            你需要给一个专业连贯的回复，回复应尽可能贴近订舱销售员的回答
            """
        
        formatted_prompt = template.format(history=history, question=message)

        return llm.invoke(formatted_prompt).content
    else:
        return "稍等，这个需要和领导请示一下."
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="订舱销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
