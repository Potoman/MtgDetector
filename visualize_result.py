import json

if __name__ == '__main__':
    model_names = {}
    model_names["model1"] = "training_nn32_256_512_8_do0.0_s90_ds200_e10.json"
    model_names["model2"] = "training_nnmodel2_do0.0_s90_ds200_e10.json"

    import matplotlib.pyplot as plt
    for title, file_name in model_names.items():
        with open(file_name, "r", encoding="utf-8") as file:
            data = json.load(file)
            plt.plot(data['coord_mae_when_present'], label=title)

    plt.legend()
    plt.show()


