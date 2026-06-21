"""Karrierút-előrejelző rendszer — Random Forest modell Google Forms kérdőíves adatok alapján

Bemeneti fájl:
    Career Survey Responses — AI Model-datas.xlsx

Kimenet:
    A modell eredményei, teljesítménymutatói és ábrái az outputs/ mappába kerülnek.

Futtatás:
    python career_model_survey_data.py
"""

# Python könyvtárak importálása
# os: mappák és fájlútvonalak kezelése
# re: szövegek tisztítása reguláris kifejezésekkel
# numpy: numerikus számítások és tömbműveletek
# pandas: Excel-adatok beolvasása és táblázatos adatkezelés
# matplotlib: grafikonok és ábrák létrehozása
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# A scikit-learn könyvtár gépi tanulási eszközeinek importálása
# ColumnTransformer: numerikus és szöveges változók külön kezelése
# RandomForestClassifier: Random Forest osztályozó modell
# accuracy_score, classification_report, confusion_matrix: teljesítménymutatók
# train_test_split, cross_val_score: tanító-teszt állományokra osztás és keresztvalidáció
# Pipeline: adat-előkészítés és modell egy folyamatba rendezése
# OneHotEncoder: szöveges változók numerikus formára alakítása
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

#Adat előkészítés, változók létrehozása és értékeadás
DATA_FILE = "Career Survey Responses — AI Model-datas.xlsx" #itt vannak a valaszok
SHEET_NAME = "Form Responses 1 Másolat" #Excel munkalap neve
OUTPUT_DIR = "outputs" #itt lesznek a kimeneti fajlok es abrak
RANDOM_STATE = 42

# A  vizsgált kompetenciák definiálása
# A kulcsok a programban használt rövid változónevek, az értékek pedig az Excel-oszlopokban szereplő kompetencianevek
COMPETENCIES = {
    "communication": "Communication",
    "problem_solving": "Problem Solving",
    "teamwork": "Teamwork",
    "adaptability": "Adaptability",
    "independent_work": "Independent Work",
    "digital_skills": "Digital Skills",
    "data_analysis": "Data Analysis",
    "project_management": "Project Management",
    "business_knowledge": "Business Knowledge",
    "foreign_language": "Foreign Language Skills",
}

#Demográfiai és szakmai jellemzők listája, bemeneti változok
DEMOGRAPHIC_FEATURES = [
    "age",
    "gender",
    "education",
    "experience_years",
    "professional_field",
    "job_level",
]

# A kategóriaváltozók tisztítása és egységesítése
def clean_label(value):
   
    if pd.isna(value):
        return value
    value = str(value).strip()
    value = re.sub(r"^\d+\s+[–-]\s+", "", value).strip()

    # A karrierút-kategóriák egységesítése, egyszerűsítés céljából
    career_prefixes = [
        "Technical / Specialist", "Analytical", "Management / Leadership",
        "Entrepreneurial", "Creative", "Consulting / Advisory",
        "Administrative / Operational", "Research / Academic"
    ]
    for prefix in career_prefixes:
        if value.startswith(prefix):
            return prefix

    return value

# Az Excel-fájl megfelelő oszlopainak azonosítása
def find_column(df, contains_all):
    
    for col in df.columns:
        col_lower = str(col).lower()
        if all(fragment.lower() in col_lower for fragment in contains_all):
            return col
    raise KeyError(f"Column not found for fragments: {contains_all}")

#Adat beolvasás Excel állományból
def load_and_prepare_data(file_path=DATA_FILE, sheet_name=SHEET_NAME):
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name)
    df_raw = df_raw.dropna(how="all").copy()

    eligibility_col = find_column(df_raw, ["5 years"])
    df_raw = df_raw[df_raw[eligibility_col].astype(str).str.lower().eq("yes")].copy()

    rename_map = {
        find_column(df_raw, ["age"]): "age",
        find_column(df_raw, ["gender"]): "gender",
        find_column(df_raw, ["education"]): "education",
        find_column(df_raw, ["work experience"]): "experience_years",
        find_column(df_raw, ["professional field"]): "professional_field",
        find_column(df_raw, ["job level"]): "job_level",
        find_column(df_raw, ["career path category"]): "career_path",
        find_column(df_raw, ["satisfied"]): "career_satisfaction",
    }

    # A kezdeti és jelenlegi kompetenciaszinthez tartozó oszlopok azonosítása és átnevezése
    for key, label in COMPETENCIES.items():
        rename_map[find_column(df_raw, ["early career competency", label])] = f"early_{key}"
        rename_map[find_column(df_raw, ["current competency", label])] = f"current_{key}"

    df = df_raw.rename(columns=rename_map)
# Csak azok az oszlopok maradnak meg, amelyek a modellhez és az elemzéshez szükségesek
    selected_cols = (
        DEMOGRAPHIC_FEATURES
        + [f"early_{key}" for key in COMPETENCIES]
        + [f"current_{key}" for key in COMPETENCIES]
        + ["career_path", "career_satisfaction"]
    )
    df = df[selected_cols].copy()
    # A szöveges változók tisztítása

    for col in DEMOGRAPHIC_FEATURES + ["career_path", "career_satisfaction"]:
        df[col] = df[col].apply(clean_label)
        
# A kompetenciaértékek numerikus adattípussá alakítása
    numeric_cols = [c for c in df.columns if c.startswith("early_") or c.startswith("current_")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # A hiányos kompetenciaértékeket vagy célváltozót tartalmazó sorok eltávolítása
    df = df.dropna(subset=numeric_cols + ["career_path"]).reset_index(drop=True)
    return df

#A Random Foret modell létrehozása
def build_model(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300, #döntési fák száma
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Top-3 pontosság számítása
# Ez azt vizsgálja, hogy a helyes karrierút szerepel-e a modell által javasolt három legvalószínűbb kategória között
def top_k_accuracy(model, X_test, y_test, k=3):
    probabilities = model.predict_proba(X_test)
    classes = model.classes_
    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    correct = [y_test.iloc[i] in classes[top_k[i]] for i in range(len(y_test))]
    return np.mean(correct)

# A változók fontosságának meghatározása
# A Random Forest modell feature_importances_ mutatója alapján látható,
# hogy mely kompetenciák és háttérváltozók járultak hozzá leginkább az előrejelzéshez
def get_feature_importance(model, numeric_features, categorical_features):
    preprocessor = model.named_steps["preprocessor"]
    rf = model.named_steps["model"]

    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_features) + list(cat_names)

    importance = pd.DataFrame({
        "feature": all_feature_names,
        "importance": rf.feature_importances_,
    })

    
    def group_name(feature):
        for cat in categorical_features:
            if feature.startswith(cat + "_"):
                return cat
        return feature

    importance["feature_group"] = importance["feature"].apply(group_name)
    grouped = importance.groupby("feature_group", as_index=False)["importance"].sum()
    return grouped.sort_values("importance", ascending=False)

# A legfontosabb változók ábrázolása oszlopdiagramon
# Az ábra az outputs mappába lesz mentve 
def plot_feature_importance(importance_df, save_path):
    top = importance_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature_group"], top["importance"])
    plt.title("Feature importance in the Random Forest model")
    plt.xlabel("Importance score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

#Konfúziós mátrix elkészítése és mentése
# A mátrix megmutatja, hogy a modell mely karrierúttípusokat sorolta be helyesen vagy hibásan
def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("Actual career path")
    plt.xlabel("Predicted career path")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# A három legvalószínűbb karrierút előrejelzése egy adott válaszadó alapján, valószínűségi sorrendben
def predict_top3(model, input_row):
    probabilities = model.predict_proba(input_row)[0]
    classes = model.classes_
    order = np.argsort(probabilities)[::-1][:3]
    return pd.DataFrame({
        "Rank": [1, 2, 3],
        "Career Path": classes[order],
        "Probability": probabilities[order],
    })


# A program fő futási folyamata
# Itt történik az adatok beolvasása, a modell betanítása, tesztelése,
# a teljesítménymutatók kiszámítása, valamint a kimeneti fájlok és ábrák mentése
def main():
    # Kimeneti mappa létrehozása, ha még nem létezik
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Adatok beolvasása és előkészítése
    df = load_and_prepare_data()
    print("=" * 70)
    print("Career Path Prediction — Random Forest on survey data")
    print("=" * 70)
    print(f"Valid responses used: {len(df)}")
    print("Career path distribution:")
    print(df["career_path"].value_counts().to_string())
    print()

    # Numerikus és szöveges bemeneti változók meghatározása
    numeric_features = [f"early_{key}" for key in COMPETENCIES] + [f"current_{key}" for key in COMPETENCIES]
    categorical_features = DEMOGRAPHIC_FEATURES

    # X tartalmazza a bemeneti változókat, y pedig a célváltozót, vagyis a karrierúttípust
    X = df[numeric_features + categorical_features]
    y = df["career_path"]

    # Az adatállomány felosztása tanító és tesztelő részre
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=stratify
    )

    # Modell létrehozása és betanítása
    model = build_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    # Előrejelzések készítése a tesztadatokra
    y_pred = model.predict(X_test)

    # A modell teljesítménymutatóinak kiszámítása
    accuracy = accuracy_score(y_test, y_pred)
    top3_acc = top_k_accuracy(model, X_test, y_test, k=3)

    # Keresztvalidáció a modell stabilitásának vizsgálatához
    min_class_count = y.value_counts().min()
    cv_folds = min(5, min_class_count)
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy") if cv_folds >= 2 else None

    # Eredmények kiírása
    print("MODEL PERFORMANCE")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Top-3 accuracy: {top3_acc:.2%}")
    if cv_scores is not None:
        print(f"Cross-validation accuracy ({cv_folds}-fold): {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # A modell teljesítményjelentésének mentése szöveges fájlba
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Valid responses used: {len(df)}\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n")
        f.write(f"Top-3 accuracy: {top3_acc:.4f}\n")
        if cv_scores is not None:
            f.write(f"Cross-validation accuracy ({cv_folds}-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write(classification_report(y_test, y_pred, digits=3))

    # Konfúziós mátrix kiszámítása és ábraként mentése
    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plot_confusion_matrix(cm, labels, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # Változófontosság számítása, Excel-fájlba mentése és ábrázolása
    importance_df = get_feature_importance(model, numeric_features, categorical_features)
    importance_df.to_excel(os.path.join(OUTPUT_DIR, "feature_importance.xlsx"), index=False)
    plot_feature_importance(importance_df, os.path.join(OUTPUT_DIR, "feature_importance.png"))

    # Példa-előrejelzés készítése az első válaszadó alapján
    example = X.iloc[[0]]
    top3 = predict_top3(model, example)
    top3.to_excel(os.path.join(OUTPUT_DIR, "example_top3_prediction.xlsx"), index=False)
    print("Example top-3 prediction:")
    print(top3.to_string(index=False, formatters={"Probability": "{:.2%}".format}))

    print()
    print(f"Outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
