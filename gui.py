from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import numpy as np
import cv2
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, img_to_array
import dlib
import gc
from functools import partial
import multiprocessing

# bu satir modellerin process olarak cagrilmasi icin gerekli
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# vgg-face base modeli
def baseModel():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model


# race modeli, agirligi yuklenerek cagriliyor
def raceModel_load():
    model = baseModel()

    # --------------------------

    classes = 6
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    # --------------------------

    race_model = Model(inputs=model.input, outputs=base_model_output)
    race_model.load_weights("models/race_model.h5")
    return race_model


# yas tahmini olasiliklarini sayiya ceviren fonksiyon
def findApparentAge(age_predictions):
    output_indexes = np.array([i for i in range(0, 101)])

    apparent_age = np.sum(age_predictions * output_indexes)

    return apparent_age


# tespit edilen yuzu modeller icin uygun hale getiren fonksiyon
def preprocess_pixels(detected_face):
    img_pixels = img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    return img_pixels


# yuz tespit eden fonksyion,
# dikkat edilmesi gereken tek nokta, fotografin pathi 'foto 1' gibi bosluk vs. icermemeli
def detect_face(img_path):
    exact_image = False
    if type(img_path).__module__ == np.__name__:
        exact_image = True

    base64_img = False
    if len(img_path) > 11 and img_path[0:11] == "data:image/":
        base64_img = True


    elif not exact_image:
        if not os.path.isfile(img_path):
            raise ValueError("Confirm that ", img_path, " exists")

        # img_path ozel karakter icerme ihtimaline karsi decode edilerek okunuyor
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # rects: yuz tespiti yapilan cerceve
    # scores: yuz tespiti skorlari
    # best_score_index: en yuksek skora sahip yuzun indisi
    rects, scores, idx = face_detector.run(gray, 1, 0)
    faces = []
    if len(rects) == 0:
        return faces
    else:
        # en yuksek skor ile tespit edilen yuz seciliyor
        best_score_index = np.argmax(scores)

        # tespit edilen yuzler fotograftan kirpiliyor
        for rect in rects:
            detected_face = img[max(0, rect.top()): min(rect.bottom(), img.shape[0]),
                            max(0, rect.left()): min(rect.right(), img.shape[1])]

            # fotograflar 224, 224 hale getiriliyor
            detected_face = cv2.resize(detected_face, (224, 224))
            faces.append(detected_face)

        # en yuksek skora sahip yuz donduruluyor
        return faces[best_score_index]

# butun tahminleri baslatan fonksiyon
def make_analysis(img_path):

    detected_img_label.image = None
    age_str.set('')
    gender_str.set('')
    race_str.set('')
    beauty_str.set('')

    detected_face = detect_face(img_path)

    # yuz tespit edilemediyse arayuzde belirtiliyor
    if len(detected_face) == 0:
        detect_str.set('No Face Detected')
        root.update()

    else:
        # yuz tespit edildiyse arayuzde gosteriliyor
        im = Image.fromarray(detected_face[:, :, ::-1])
        img = ImageTk.PhotoImage(image=im)
        detect_str.set('Detected Face (Analysing)')

        detected_img_label.image = img
        panel = Label(root, image=img)
        panel.grid(row=2, column=50)
        root.update()

        img_pixels = preprocess_pixels(detected_face)

        race_labels = ['Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino-Hispanic']

        beauty_model_names = ['asian-female', 'asian-male', 'indian-female', 'indian-male',
                              'black-female', 'black-male', 'white-female', 'white-male',
                              'middle-eastern-female', 'middle-eastern-male',
                              'latino-hispanic-female', 'latino-hispanic-male']

        # yas tahmini yapiliyor
        age_predictions = age_model.predict(img_pixels)[0, :]
        apparent_age = findApparentAge(age_predictions)

        # cinsiyet tahmini yapiliyor
        gender_prediction = gender_model.predict(img_pixels)[0, :]
        if np.argmax(gender_prediction) == 0:
            beauty_model_index = 0
            gender = "Woman"
        elif np.argmax(gender_prediction) == 1:
            beauty_model_index = 1
            gender = "Man"

        # etnik koken tahmini yapiliyor
        prediction_proba = race_model.predict(img_pixels)[0, :]
        prediction = np.argmax(prediction_proba)
        race = race_labels[prediction]

        # cinsiyet ve etnik koken tahminine gore uygun guzellik modelinin indisi belirleniyor
        beauty_model_index += prediction * 2
        # print('beauty_model:', beauty_model_names[beauty_model_index], beauty_model_index)

        # tahminler arayuzde gosteriliyor, fotojeniklik tahminine gecilecek
        age_str.set('Age: ' + str(round(apparent_age)))
        gender_str.set('Gender: ' + gender)
        race_str.set('Race: ' + race)
        beauty_str.set('Photogenicity: Predicting...')
        root.update()

        print('fotojeniklik processi basliyor')

        # arayuzle model cagirirken gpu nun dolmamasini garanti eden yontem, process olusturmak
        p = multiprocessing.Process(
            target=run_beauty_process,
            args=(queue, beauty_model_names, beauty_model_index, img_pixels)
        )
        p.start()
        p.join()
        print('fotojeniklik processi sonlandi')

        # processte yapilan tahmin aliniyor
        beauty = queue.get()

        gc.collect()

        # tahminler hem arayuzde hem konsolda gosteriliyor
        print(f'Age:{round(apparent_age)}\nGender:{gender}\nRace:{race}\nBeauty:{beauty}')

        detect_str.set('Detected Face')
        beauty_str.set('Photogenicity: ' + str(beauty))

        root.update()


# fotograf secme islemini gerceklestiren ve secilen fotografi arayuzde gosteren fonksiyon
def open_img():

    # Fotograf secmek ekrani aciliyor
    x = openfilename()
    if x != '':
        img = Image.open(x)

        # fotograf 224,224 boyutuna getirildi
        img = img.resize((224, 224), Image.ANTIALIAS)

        # arayuzde gosterebilmek icin gerekli islemler
        img = ImageTk.PhotoImage(img)

        text_label = Label(root, text='Original Image')
        text_label.grid(row=3)
        panel = Label(root, image=img)

        panel.image = img
        panel.grid(row=2)

        # fotograf secimi yaptiktan sonra analiz butonunun gozukmesi icin burda olusturuluyor
        analyse_btn = Button(root, text='analyse', command=partial(make_analysis, x)).grid(
            row=25)


# arayuzde fotograf secme ekranini acan fonksiyon
def openfilename():
    filename = filedialog.askopenfilename(title='Open')
    print(filename)
    return filename


# fotojeniklik analizi icin process ile model olusumu ve tahminini yapan fonksiyon
def run_beauty_process(queue, beauty_model_names, beauty_model_index, img_pixels):
    print('fotojeniklik processi devam ediyor')
    beauty_model = load_model('models/' + beauty_model_names[beauty_model_index] + '.hdf5')
    prediction = beauty_model.predict(img_pixels)[0, :]
    print(f'beauty predictions:{prediction}')
    beauty = True if np.argmax(prediction) == 1 else False

    # tahmin, process bitisinde alinmak uzere kaydediliyor
    queue.put(beauty)


# modellerin kabul ettigi fotograf boyutu
target_size = (224, 224)

# yuz detektoru
face_detector = dlib.get_frontal_face_detector()

# gerekli modeller ayaga kaldiriliyor,
# fotojeniklik modelleri cok fazla olduklari icin GPU'ya sigmalari mumkun degil, process olarak her seferinde bastan olusturulacaklar
# diger modelleri bu sekilde dogrudan cagirmamiz, yas, cinsiyet ve etnik koken tahminlerinin cok hizli yapilmasini saglayacak.

race_model = raceModel_load()
age_model = load_model('models/age.hdf5')
gender_model = load_model("models/gender.hdf5")
queue = multiprocessing.Queue()


if __name__ == '__main__':
    # arayuz icin ana window
    root = Tk()

    # Arayuz basligi
    root.title("Demo Interface")

    # Arayuz penceresinin boyutu
    root.geometry("500x400")

    # Arayuz resize edilebilir
    root.resizable(width=True, height=True)

    # fotograf yuklemek icin gerekli pencereyi acan buton
    btn = Button(root, text='open image', command=open_img).grid(
        row=1, columnspan=5)

    # tespit edilen yuzun yerlestirilecegi label
    detected_img_label = Label(root, image='')
    detected_img_label.grid(row=2, column=50)

    # arayuz icin gerekli degiskenler
    detect_str = StringVar()
    age_str = StringVar()
    gender_str = StringVar()
    race_str = StringVar()
    beauty_str = StringVar()

    detect_label = Label(root, textvariable=detect_str)
    detect_label.grid(row=3, column=50)

    age_label = Label(root, textvariable=age_str)
    age_label.grid(row=25, column=50)

    gender_label = Label(root, textvariable=gender_str)
    gender_label.grid(row=35, column=50)

    race_label = Label(root, textvariable=race_str)
    race_label.grid(row=45, column=50)

    beauty_label = Label(root, textvariable=beauty_str)
    beauty_label.grid(row=55, column=50)

    root.mainloop()
