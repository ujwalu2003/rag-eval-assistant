import sys
import pandas as pd

name = sys.argv[1] if len(sys.argv) > 1 else "rerank_k4"
df = pd.read_csv(f"results_{name}.csv")
df["got"] = df["got"].fillna("")

bad = df[(df["off_topic"] == False) & (df["correct"] == 0)]
print("Failed:", len(bad))
for _, r in bad.iterrows():
    print("\nQ:", r["question"])
    print("Expected:", r["expected"])
    print("Got:", r["got"][:200])