from data_cleaning import clean_defender_data
from flask import *
import pandas as pd
app = Flask(__name__)


@app.route("/predictions")
def show_tables():
    DefenderDF = clean_defender_data()
    # return render_template('defenderDF.html',tables=[DefenderDF.to_html(classes='defenders')],
    #                        titles = ['na', 'Defenders'])
    return DefenderDF.to_json()

# @app.route("/")
# def hello():
#     return "<h1 style='color:blue'>Hello There!</h1>"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
