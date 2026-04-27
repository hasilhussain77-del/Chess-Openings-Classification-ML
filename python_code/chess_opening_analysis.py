# ============================================
# CHESS OPENINGS CLASSIFICATION USING DECISION TREE
# ============================================

# ---------- 1. INSTALL REQUIRED PACKAGES ----------
# Run this only once (if needed)
# !pip install python-chess graphviz


# ---------- 2. IMPORT LIBRARIES ----------
import chess.pgn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import graphviz


# ---------- 3. LOAD AND PARSE PGN DATA ----------
def load_pgn_data(file_path, max_games=None):
    games_data = []
    count = 0

    with open(file_path, errors="ignore") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None or (max_games and count >= max_games):
                break
            try:
                opening = game.headers["Opening"]
                result = game.headers["Result"]
                games_data.append((opening, result))
                count += 1
            except KeyError:
                continue

    return pd.DataFrame(games_data, columns=["Opening", "Result"])


# ---------- 4. DATA PREPROCESSING ----------
def preprocess_data(df, selected_openings=None, exclude_openings=None, remove_draws=False):
    
    # Map result
    df["Winner"] = df["Result"].map({
        "1-0": "White",
        "0-1": "Black",
        "1/2-1/2": "Draw"
    })

    # Filter specific openings
    if selected_openings:
        df = df[df["Opening"].isin(selected_openings)]

    # Exclude certain openings
    if exclude_openings:
        df = df[~df["Opening"].isin(exclude_openings)]

    # Remove draws if needed
    if remove_draws:
        df = df[df["Winner"] != "Draw"]

    return df


# ---------- 5. ENCODING ----------
def encode_data(df):
    label_encoder = LabelEncoder()
    df["Winner_encoded"] = label_encoder.fit_transform(df["Winner"])

    X = pd.get_dummies(df["Opening"])
    y = df["Winner_encoded"]

    return X, y, label_encoder


# ---------- 6. TRAIN MODEL ----------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return clf, accuracy


# ---------- 7. VISUALIZE TREE ----------
def visualize_tree(clf, X, label_encoder, filename):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=X.columns,
        class_names=label_encoder.classes_,
        filled=True,
        rounded=True,
        special_characters=True
    )

    graph = graphviz.Source(dot_data)
    graph.render(filename, format="png", cleanup=False)
    return graph


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":

    # Load dataset
    df = load_pgn_data("lichess_db_standard_rated_2013-01.pgn", max_games=5000)

    # -------- CASE 1: SELECTED OPENINGS --------
    selected_openings = ["Caro-Kann Defense", "Vienna Game", "London System"]

    df_selected = preprocess_data(df, selected_openings=selected_openings)
    X, y, le = encode_data(df_selected)

    model, acc = train_model(X, y)
    print("Selected Openings Accuracy:", acc)

    visualize_tree(model, X, le, "selected_openings_tree")


    # -------- CASE 2: OTHER OPENINGS --------
    excluded_openings = ["Caro-Kann Defense", "Vienna Game", "London System"]

    df_other = preprocess_data(df, exclude_openings=excluded_openings)

    # Optional: limit to top 5 openings
    top_openings = df_other["Opening"].value_counts().head(5).index.tolist()
    df_other = df_other[df_other["Opening"].isin(top_openings)]

    X2, y2, le2 = encode_data(df_other)

    model2, acc2 = train_model(X2, y2)
    print("Other Openings Accuracy:", acc2)

    visualize_tree(model2, X2, le2, "other_openings_tree")


    # -------- CASE 3: OVERALL MODEL --------
    selected_openings = ["Indian Defense", "Scandinavian Defense", "Fried Liver Attack"]

    df_overall = preprocess_data(df, selected_openings=selected_openings, remove_draws=True)

    X3 = pd.get_dummies(df_overall["Opening"])
    y3 = df_overall["Winner"].map({"White": 1, "Black": 0})

    model3, acc3 = train_model(X3, y3)
    print("Overall Model Accuracy:", acc3)

    # Plot tree using matplotlib
    plt.figure(figsize=(12, 7))
    plot_tree(
        model3,
        feature_names=X3.columns,
        class_names=["Black Win", "White Win"],
        filled=True,
        rounded=True
    )
    plt.title("Decision Tree - Overall Model")
    plt.show()
