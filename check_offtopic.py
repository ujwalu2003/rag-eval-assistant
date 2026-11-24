import pandas as pd

print(pd.read_csv("results_summary.csv"))

df = pd.read_csv("results_k8.csv")
df["got"] = df["got"].fillna("")
for _, r in df[df["off_topic"] == True].iterrows():
    print("\nQ:", r["question"])
    print("Got:", r["got"][:300])
    print("Marked correct:", r["correct"])