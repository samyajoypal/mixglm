import os
import numpy as np
import pandas as pd
import urllib.request
import zipfile
from sklearn.preprocessing import StandardScaler

def prepare_communities_and_crime():
    print("Fetching Communities and Crime Dataset (Continuous)...")
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
    df = pd.read_csv(url, header=None)

    y = df.iloc[:, 127].values
    X = df.iloc[:, 5:127] # 122 predictive features

    X = X.replace('?', np.nan).astype(float)
    X = X.fillna(X.mean()).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

    import urllib.request
    import re
    import json

    # Fetch names
    names_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names"
    names_req = urllib.request.urlopen(names_url)
    names_text = names_req.read().decode('utf-8')
    feature_names = []
    for line in names_text.split('\n'):
        if line.startswith('@attribute'):
            parts = line.split()
            if len(parts) >= 2:
                feature_names.append(parts[1])

    # The dataframe has 128 columns. Y is index 127. X is 5 to 126 (122 features).
    # feature_names has 128 items.
    if len(feature_names) >= 127:
        X_names = feature_names[5:127]
    else:
        X_names = [f"Crime_Feat_{i}" for i in range(X_scaled.shape[1]-1)]

    X_names = ["Intercept"] + X_names

    os.makedirs("data", exist_ok=True)
    np.save("data/crime_y.npy", y)
    np.save("data/crime_X.npy", X_scaled)
    with open("data/crime_features.json", "w") as f:
        json.dump(X_names, f)
    print(f"Communities and Crime Saved. Shape X: {X_scaled.shape}, y: {y.shape}")

def prepare_blog_feedback():
    print("Fetching BlogFeedback Dataset (Discrete Count)...")
    os.makedirs("scratch/blog", exist_ok=True)
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip"

    if not os.path.exists("scratch/blog/blogData_train.csv"):
        urllib.request.urlretrieve(url, "scratch/blog/blog.zip")
        with zipfile.ZipFile("scratch/blog/blog.zip", 'r') as zip_ref:
            zip_ref.extractall("scratch/blog/")

    df = pd.read_csv("scratch/blog/blogData_train.csv", header=None)

    # Subsample to 2000 points so the EM algorithm converges quickly
    df_sub = df.sample(n=2000, random_state=42)
    y = df_sub.iloc[:, -1].values
    X = df_sub.iloc[:, :-1].values # 280 raw features

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    X_names = ["Intercept"] + [f"Blog_Feat_{i+1}" for i in range(X.shape[1])]

    os.makedirs("data", exist_ok=True)
    np.save("data/blog_y.npy", y)
    np.save("data/blog_X.npy", X_scaled)
    import json
    with open("data/blog_features.json", "w") as f:
        json.dump(X_names, f)
    print(f"BlogFeedback Saved. Shape X: {X_scaled.shape}, y: {y.shape}")

if __name__ == "__main__":
    prepare_communities_and_crime()
    prepare_blog_feedback()
