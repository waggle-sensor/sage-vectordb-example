import os
import weaviate
import argparse
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import weaviate
from test import testImage, testText
from data import load_data, clear_data
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','jfif'])

# Initialize Weaviate client
def initialize_weaviate_client():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weaviate",
        default=os.getenv("WEAVIATE_API"),
        help="Weaviate REST endpoint.",
    )
    args = parser.parse_args()
    return weaviate.Client(args.weaviate)


client = initialize_weaviate_client()

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
    try:
        # Check if the directory contains any files
        if os.listdir('static/Images'):
            data_loaded = True
        else:
            data_loaded = False
    except FileNotFoundError:
        # Handle the FileNotFoundError by setting data_loaded to False
        data_loaded = False
    return render_template('upload.html', data_loaded=data_loaded)

@ app.route('/text_description', methods=['POST'])
def text_description():
	# This function uses the testText function to get results for a text query.
	# This function has been designed taking into consideration that some users might
	# also add text data to weaviate and then the results would contain text as well as images.

	text = request.form.get("description")
	
	dic = testText({"concepts":[text]}, client)
	text_results = dic['objects']
	certainty = dic['scores']
	# Using two lists to store image result and text result
	images = []
	texts = []
	sym = ['.jfif','.jpg','.jpeg','.png']
	for i in text_results:
		add = 0
		for s in sym:
			if s in i:
				images.append(i)
				add = 1
				break
		if add==0:
			texts.append(i)
	# Passing text result and image names to upload.html page
	return render_template('upload.html', description=text,images=images,texts=texts,certainty=certainty)

@app.route('/', methods=['POST','GET'])
def upload_image():
    # Function to upload an image. You can upload images from test folder or from the internet and use them.
	# This function also uses the testImage function to get the top 3 similar image names from weaviate.
	# These are then passed to the upload.html page so as to display them to the user.
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)

	file = request.files['file']

	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)

	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		
		flash('Image successfully uploaded and displayed below')
		print(" ==========\n",'File saved\n',"==========\n")
		
		# Using the testImage in the line below.
		dic = testImage({"image":"static/uploads/{}".format(filename)}, client)
		imagePaths = dic['objects']
		certainty = dic['scores']

		return render_template('upload.html', filename=filename,imagePath=imagePaths,certainty=certainty)

	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif, jfif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename,uploaded=True):
    # Function to display uploaded image
    print("Display image called")

    if uploaded:
	    print("Uploaded")

    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/set_query', methods=['POST'])
def set_query():
	# Retrieve the values from the form data
	username = request.form.get("username")
	token = request.form.get("token")
	query = request.form.get("query")
	render_template('upload.html', show_loading=True)
	load_data(username, token, query, client)
	render_template('upload.html', show_loading=False, data_loaded=True)
	return redirect(url_for('upload_form'))

# Route to handle clearing data
@app.route('/clear_data', methods=['POST'])
def clear_data_route():
    clear_data()
    return render_template('upload.html')
	
if __name__ == "__main__":
    if os.getenv("CLUSTER_FLAG"):
        # Script is running in a cluster
        app.run(host="0.0.0.0", port=int("5000"), debug=True)
    else:
        # Script is running locally
        app.run(debug=True)
