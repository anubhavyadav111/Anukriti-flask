from flask import Flask, render_template, request, send_file, redirect, url_for
from generator import generate_video_from_pdf

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/templates/generate-video-page.html')
def generate_video_page():
    return render_template('generate-video-page.html')

@app.route('/generate-video', methods=['POST'])
def handle_generate_video():
    if request.method == 'POST':
        # Access form data
        gender = request.form.get('gender')
        language = request.form.get('language')
        pdf_file = request.files['pdf']

        # Process the form data and uploaded file (generate the video)
        # Add your video generation logic here
        
        # Redirect to a success page or render a template
        return redirect(url_for('success'))

@app.route('/success')
def success():
    return "Video generated successfully!"

if __name__ == '__main__':
    app.run(debug=True)
