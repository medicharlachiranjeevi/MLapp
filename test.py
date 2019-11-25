from flask import Flask, render_template, request
from werkzeug import secure_filename
from  model import predict
app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def student():
   if request.method == 'POST':
      result = request.files['file']
      result.save(secure_filename(result.filename))
      img=predict(result.filename)
      return render_template('student.html',result = img)
   else: 
      return render_template('student.html')


if __name__ == '__main__':
   app.run(debug = True)