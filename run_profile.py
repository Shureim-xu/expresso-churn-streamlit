# run_profile.py
from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("Expresso_churn_dataset.csv")
profile = ProfileReport(df, title="Expresso Churn Dataset Profile", explorative=True)
profile.to_file("expresso_profile.html")
print("Saved expresso_profile.html")
