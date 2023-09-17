from OmdenaKenyaRoadAccidents.pipeline.train_pipeline import train_pipeline

# from sklearn.tree import DecisionTreeClassifier

def main() -> None:
    # https://drive.google.com/file/d/1ScumN0PnQDPMl_ElMCySKOg2syceIkZW/view?usp=drive_link
    id = "1ScumN0PnQDPMl_ElMCySKOg2syceIkZW"

    # model = DecisionTreeClassifier(class_weight="balanced")

    # results = train_pipeline(id=id, model=model)

    results = train_pipeline(id=id)
    print(results)


if __name__ == "__main__":
    main()
