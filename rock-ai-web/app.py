from flask import Flask, request, render_template, redirect, url_for
import os
import subprocess
from datetime import datetime
import csv

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
LOG_PATH = "classification_log.csv"

# Update these to match your model setup
NET = "/home/nvidia/jetson-inference/python/training/classification/models/ezras_rocks2"
DATASET = "/home/nvidia/jetson-inference/python/training/classification/data/ezras_rocks"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = datetime.now().strftime("%Y%m%d-%H%M%S_") + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Output file (just to avoid errors)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")

            # Run imagenet.py
            command = [
                "imagenet.py",
                f"--model={NET}/resnet18.onnx",
                f"--labels={DATASET}/labels.txt",
                "--input_blob=input_0",
                "--output_blob=output_0",
                filepath,
                output_path
            ]

            try:
                output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
                # Look for the classification output in stdout
                for line in output.splitlines():
                    if "class" in line.lower() and "%" in line:
                        result = line
                        break
                else:
                    result = "✅ Image processed, but no prediction found."

                # Log the result
                if result and filename:
                    try:
                        if '(' in result:
                            prediction = result.split('-')[-1].split('(')[0].strip()
                            confidence = result.split('(')[1].replace('%)', '').replace('%', '').strip()
                        else:
                            prediction = result.strip()
                            confidence = "N/A"

                        with open(LOG_PATH, "a", newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                filename,
                                prediction,
                                confidence
                            ])
                    except Exception as log_error:
                        print("❌ Failed to log classification:", log_error)

            except subprocess.CalledProcessError as e:
                result = f"❌ Error:\n{e.output}"

    # Read existing log data
    log_entries = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, newline='') as csvfile:
                reader = csv.reader(csvfile)
                log_entries = list(reader)
        except Exception as read_err:
            print("❌ Failed to read log:", read_err)

    return render_template("index.html", result=result, filename=filename, log_entries=log_entries)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)