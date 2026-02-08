import pandas as pd
from baseline import baseline_llm
from agent import agent_generate
from evaluation import evaluate
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

file_name = "data/Aircraft_Annotation_DataFile.csv"
df = pd.read_csv(file_name)
df = df[['PROBLEM', 'ACTION']].dropna().reset_index(drop=True)

print(f"Successfully loaded {file_name} with {len(df)} rows. First 5 rows:")
print(df.head())

docs = [
    Document(
        page_content=row["PROBLEM"],
        metadata={"ACTION": row["ACTION"]}
    )
    for _, row in df.iterrows()
]

vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

print("Vector DB 구축 완료!")


# 샘플 평가용 데이터
eval_df = df.sample(10, random_state=42)

questions = eval_df["PROBLEM"].tolist()
ground_truth = eval_df["ACTION"].tolist()

baseline_answers = []
agent_answers = []
contexts = []

for q in questions:
    # 🔹 Baseline
    base_ans = baseline_llm(q)
    baseline_answers.append(base_ans)

    # 🔹 Agent
    agent_ans = agent_app.invoke({"PROBLEM": q})["final_action"]
    agent_answers.append(agent_ans)

    # 🔹 Context (RAG용)
    docs = retriever.invoke(q)
    ctx = [d.page_content for d in docs]
    contexts.append(ctx)

baseline_dataset = Dataset.from_dict({
    "question": questions,
    "answer": baseline_answers,
    "contexts": contexts,
    "ground_truth": ground_truth
})

agent_dataset = Dataset.from_dict({
    "question": questions,
    "answer": agent_answers,
    "contexts": contexts,
    "ground_truth": ground_truth
})

print("Baseline / Agent 평가 데이터셋 생성 완료!")


ragas_llm = LangchainLLMWrapper(llm)
ragas_embed = LangchainEmbeddingsWrapper(embeddings)

print("▶ Baseline 평가 중...")
baseline_results = evaluate(
    dataset=baseline_dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_embed
)

print("▶ Agent 평가 중...")
agent_results = evaluate(
    dataset=agent_dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_embed
)

baseline_df = baseline_results.to_pandas()
agent_df = agent_results.to_pandas()

result_df = pd.DataFrame({
    "Fault": eval_df["PROBLEM"],
    "Ground Truth": eval_df["ACTION"],
    "Baseline Answer": baseline_answers,
    "Agent Answer": agent_answers
})

result_df.to_csv('data/result_df.csv')

print(baseline_df['answer_relevancy'].mean(), agent_df['answer_relevancy'].mean())