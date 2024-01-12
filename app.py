from flask import Flask, request, render_template, send_from_directory, flash
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import pywt
import pywt.data
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

classes = ["Depressed Face", "No Depressed"]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'andhhgdfkbl'


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/upload", methods=["POST", "GET"])
def upload():
    print('a')
    if request.method == 'POST':
        m = int(request.form["alg"])
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('images/', fn)
        myfile.save(mypath)
        acc = pd.read_csv("Accurary.csv")
        accepted_formated = ['jpg', 'png', 'jpeg', 'jfif']
        if fn.split('.')[-1] not in accepted_formated:
            flash("Image formats only Accepted", "Danger")

        if m == 1:
            new_model = load_model(r"alg/FinalModel.h5",compile=False)
            test_image = image.load_img(mypath, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image /= 255
            a = acc.iloc[m-1, 1]

        elif m == 2:
            new_model = load_model(r"alg/svm.h5",compile=False)
            test_image = image.load_img(mypath, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image /= 255
            a = acc.iloc[m-1, 1]

        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        print(result)
        print(np.argmax(result))
        prediction = classes[np.argmax(result)]

        m = mpimg.imread(myfile)
        c = pywt.wavedec2(m, 'db5', mode='periodization', level=2)
        imgr = pywt.waverec2(c, 'db5', mode='periodization')
        imgr = np.uint8(imgr)
        cA2 = c[0]
        (cH1, cV1, cD1) = c[-1]
        (cH2, cV2, cD2) = c[-2]
        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(cA2, cmap=plt.cm.gray)
        plt.title('cA2: Approximation Coeff.', fontsize=30)

        plt.subplot(2, 2, 2)
        plt.imshow(cH2, cmap=plt.cm.gray)
        plt.title('cA2: Horizontal Detailed Coeff.', fontsize=30)

        plt.subplot(2, 2, 3)
        plt.imshow(cV2, cmap=plt.cm.gray)
        plt.title('cV2: Vertical Detailed Coeff.', fontsize=30)

        plt.subplot(2, 2, 4)
        plt.imshow(cD2, cmap=plt.cm.gray)
        plt.title('cD2: Diagonal Detailed Coeff.', fontsize=30)

        plt.figure()
        plt.imshow(imgr, cmap=plt.cm.gray)
        plt.title('Reconstructed Image', fontsize=10)
        plt.show()
    return render_template("template.html", image_name=fn, text=prediction, a=round(a*100, 3))


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


if __name__ == '__main__':
    app.run(debug=True)


