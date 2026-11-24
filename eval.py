import os
import time
import pandas as pd
from rag import answer, ask_llm

# ---- Change these for each experiment ----
RUN_NAME = "k8_strict"
K = 8
RERANK = False
PAUSE = 15   # seconds between questions
# ------------------------------------------

def judge(question, correct, predicted):
    prompt = (
        "You are grading an answer.\n"
        f"Question: {question}\n"
        f"Correct answer: {correct}\n"
        f"Student answer: {predicted}\n\n"
        "Does the student answer give the same key information as the correct answer? "
        "Ignore wording differences. Reply with ONLY the digit 1 (yes) or 0 (no)."
    )
    out = ask_llm(prompt).strip()
    return 1 if out.startswith("1") else 0

df = pd.read_csv("eval.csv")
rows = []

for i, row in df.iterrows():
    q, correct, src = row["question"], row["correct_answer"], row["source"]
    print(f"[{i+1}/{len(df)}] {q}")

    text, sources = answer(q, k=K, rerank=RERANK)
    off_topic = (src == "none")

    if off_topic:
        hit = None
        correct_flag = 1 if "not found" in text.lower() else 0
    else:
        hit = int(any(os.path.basename(s["source"]).startswith(src) for s in sources))
        correct_flag = judge(q, correct, text)

    rows.append({
        "question": q, "expected": correct, "got": text,
        "off_topic": off_topic, "retrieval_hit": hit, "correct": correct_flag,
    })
    time.sleep(PAUSE)

res = pd.DataFrame(rows)
res.to_csv(f"results_{RUN_NAME}.csv", index=False)

answerable = res[~res["off_topic"]]
offtopic = res[res["off_topic"]]

hit_rate = answerable["retrieval_hit"].mean() * 100
accuracy = answerable["correct"].mean() * 100
refusal = offtopic["correct"].mean() * 100 if len(offtopic) else float("nan")

print("\n===== RESULTS:", RUN_NAME, "=====")
print(f"Retrieval hit rate : {hit_rate:.1f}%  ({len(answerable)} questions)")
print(f"Answer accuracy    : {accuracy:.1f}%")
print(f"Correct refusals   : {refusal:.1f}%  ({len(offtopic)} off-topic questions)")

summary = pd.DataFrame([{
    "version": RUN_NAME, "hit_rate": round(hit_rate, 1),
    "accuracy": round(accuracy, 1), "refusal_rate": round(refusal, 1),
}])
summary.to_csv("results_summary.csv", mode="a", index=False,
               header=not os.path.exists("results_summary.csv"))