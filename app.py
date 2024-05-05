import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template


try:
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
except Exception as e:
    BASEDIR = os.getcwd()


model = pickle.load(open(os.path.join(BASEDIR, "artifacts", "model_lg.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASEDIR, "artifacts", "scaler.pkl"), "rb"))

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        english = float(request.form.get("nota_exame_ingles", 0.0))
        qi = int(request.form.get("qi", 0))
        psico = int(request.form.get("nota_teste_psico", 0))

        if not (0 <= english <= 10 and 0 <= qi <= 200 and 0 <= psico <= 100):
            raise ValueError("Valores de entrada inválidos")

        predict = model.predict(
            scaler.transform(np.array([english, qi, psico]).reshape(1, -1))
        )[0]

        message = (
            "O aluno poderá ser inscrito no curso"
            if predict == 1
            else "O aluno não poderá ser inscrito no curso"
        )
    except Exception as e:
        message = f"Erro: {e}"

    return render_template("index.html", message=message)


if __name__ == "__main__":
    app.run()
