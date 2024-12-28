from flask import Flask, request
from flask_cors import CORS
import importlib.util
app = Flask(__name__)
cors = CORS(app, resources={r"/post": {"origins": "http://localhost:3000"}})
cors = CORS(app, resources={r"/postgurobi": {"origins": "http://localhost:3000"}})



@app.route("/post", methods=['POST'])
def process_data():
    # Get the JSON data from the request body
    data = request.get_json()

    # Extract the file content from the data
    file_content = data['fileContent']
    
    # Pass the file content to your coordinates function
    print(file_content)
    text_file = open("output.txt", "wt")
    n = text_file.write(file_content)
    text_file.close()
    spec = importlib.util.spec_from_file_location("tabu","./tabu.py")        
    tabu = spec.loader.load_module()
    return {"results": tabu.coordinates}, 200

@app.route("/postgurobi", methods=['POST'])
def process_datagurobi():
    # Get the JSON data from the request body
    data = request.get_json()

    # Extract the file content from the data
    file_content = data['fileContent']
    
    # Pass the file content to your coordinates function
    print(file_content)
    text_file = open("output.txt", "wt")
    n = text_file.write(file_content)
    text_file.close()
    spec = importlib.util.spec_from_file_location("calc","./calc.py")        
    calc = spec.loader.load_module()
    return {"results": calc.routes_with_coords[1:]}, 200




@app.route("/members")
def members():
    spec = importlib.util.spec_from_file_location("tabu","./tabu.py")        
    tabu = spec.loader.load_module()
    return{"route": tabu.coordinates}

if __name__== "__main__":
    app.run(debug=True, port=5001)