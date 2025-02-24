from django.shortcuts import render,redirect
from django.http import JsonResponse
from .models import UserRegistrationForm,UserLogdet
from django.contrib.auth.models import User
import cv2
import numpy as np
import os
from PIL import Image
import base64
import pickle
from sklearn.preprocessing import LabelEncoder
from .prepro import Prepro
from .models import Product
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from .models import Add_chart




# Create your views here
def home(req):
    return render(req,'index.html')


def detect_face(image_data):
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Decode the base64 image data
    face_image = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
    face_image = cv2.imdecode(face_image, cv2.IMREAD_COLOR)

    # Define a fixed region of interest (ROI) size
    fixed_roi_size = (100, 100)  # Change this to your desired size

    # Perform face detection using OpenCV
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Now, you can perform face recognition on the fixed ROI size
    for (x, y, w, h) in faces:
        # Extract the face region with the fixed size
        face_roi = gray[y:y + fixed_roi_size[1], x:x + fixed_roi_size[0]]

        # Print the dimensions (shape) of the fixed-size detected face region
        print("Detected face dimensions (height, width):", face_roi.shape)

    return face_roi




def login(request):
    le = LabelEncoder()
    faces, Id = getImagesAndLabels("TrainingImage")
    Id = le.fit_transform(Id)
    output = open('label_encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(Id))
    recognizer.save(r"Trained_Model\Trainner.yml")
    recognizer1 = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer1.read(r"Trained_Model\Trainner.yml")
    harcascadePath = r"Haarcascade\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    global cam,det
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pkl_file = open('label_encoder.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    global tt, unknown
    det=0
    unknown=0
    while True:
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer1.predict(gray[y:y + h, x:x + w])
            print(conf)
            if (conf < 124):
                det+=1
                tt = le.inverse_transform([Id])
                tt=str(tt[0])
                print("...........",tt)
                username="User Id: "+tt
                # if det ==10:
                #      return render(request,'register.html')
                user = UserRegistrationForm.objects.filter(id=tt).exists()
                
                print('------------------------------')
                print(user)
                if det==10:
                    if user:
                        try:
                            id1 = request.session['id']=tt
                            print(id1)
                            cam.release()
                            cv2.destroyAllWindows()
                            return redirect(buy_pro)
                        except ValueError:
                            return render(request,'uhome.html',{'msg1':'kk'})
                    det=0
                    
            else:
                unknown+=1
                username='Unknown'
                if unknown==10:
                    cam.release()
                    
                    cv2.destroyAllWindows()  
                    return redirect(home)
            cv2.putText(frame,str(username), (x, y + h),font, 1, (255, 255, 255), 2)
        
        cv2.imshow('im', frame)
        if (cv2.waitKey(1) == ord('q')):
            break

    cam.release()
    cv2.destroyAllWindows()
    return render(request,'login.html')


def register(request):
    user = user_registration_forms = UserRegistrationForm.objects.all()
    # Extract only the 'id' field from the objects
    ids = [user.id for user in user_registration_forms]
    id = len(ids)+1
    if request.method == 'POST':
        id = request.POST['id']
        print(id)
        name1 = request.POST['name']
        address1 = request.POST['address']
        phone_number1 = request.POST['phone_number']
        
        if name1 == '' or address1 == '' or phone_number1 == '':
            return render(request,'register.html',{'id':id})
        
        else:
            cam = cv2.VideoCapture(0)
            harcascadePath = "Haarcascade/haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(harcascadePath)
            sampleNum = 0
            while (True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            # incrementing sample number
                    sampleNum = sampleNum + 1
                            # saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage/ " + id + "." + str(
                                sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                            # display the frame

                else:
                    cv2.imshow('frame', img)
                        # wait for 100 miliseconds
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                        # break if the sample number is morethan 100
                elif sampleNum > 150:
                        break

            cam.release()
            cv2.destroyAllWindows()
            UserRegistrationForm.objects.create(name=name1,address=address1,phone_number=phone_number1)

            return render(request,'register.html')
        
    return render(request,'register.html',{'id':id})

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = str(os.path.split(imagePath)[-1].split(".")[0])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids
print('_____________________________________________________________________')





def uhome(req):
    global name, price, image
    if req.method == 'POST':
        inp = req.POST['text']
        tdata = 'combined_dataset.csv'
       
        df = Prepro.pre(tdata)
        # Find unique names
        data = df
        unique_name = data['name'].unique()

        # Create a dictionary to map author names to numerical values
        name_to_numeric = {name: idx for idx, name in enumerate(unique_name)}
        # Convert author names to numerical values
        data['name_numeric'] = data['name'].map(name_to_numeric)

        k = 11  # Number of neighbors, including the item itself
        knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn_model.fit(data[['name_numeric', 'actual_price']])

        # User input
        user_name = inp

        # Filter products containing the user input in their name
        filtered_products = data[data['name'].str.contains(user_name, case=False)]

        if filtered_products.empty:
            print("No products with the specified keyword found.")
        else:
            # Sort the filtered products by a relevant criterion (e.g., cost) and get the top 15
            top_15_products = filtered_products.sort_values(by='actual_price')[:15]

            # Get recommended product names, costs, and images from the top 15 products
            name = top_15_products['name'].tolist()
            price = top_15_products['actual_price'].tolist()
            image = top_15_products['image'].tolist()
            data = name,price,image
            print(f"data : {data}")
            return render(req,'uhome.html',{'data':data,'name':name,'price':price,'image':image,'mm':'valid'})
    return render(req,'uhome.html')


def uhome1(req):
    return render(req,'uhome1.html')


def refer(req):
    if req.method == 'POST':
        na = req.POST['na']
        im = req.POST['im']
        pr = req.POST['pr']
        if na == '' or im == '' or pr == '':
            return render(req,'uhome.html',{'mhh':'not'})
        print('_____________________________________')
        Product.objects.create(name1 = na,image1=im,price1 = pr,name_id=req.session['id'])
        return render(req,'uhome1.html',{'ms':'sucess'})
    return render(req,'uhome.html')


def buy_pro(req):
    try:
        import pandas as pd
        tdata = 'combined_dataset.csv'
        df = Prepro.pre(tdata)

        # Create a LabelEncoder to encode product names
        label_encoder = LabelEncoder()
        df['name_numeric'] = label_encoder.fit_transform(df['name'])
        
        
        prod = Add_chart.objects.filter(b_id=req.session['id']).order_by('-id').first()
        
        print(prod)
        name45 = prod.name2

        # Assuming 'name45' is the product name you want to find recommendations for
        # Replace with the actual product name

        # Check if the product exists in the dataset
        if name45 not in df['name'].values:
            print(f"Product '{name45}' not found in the dataset.")
            # Handle the case where the product is not found
            # You can choose to exit the program or provide a default recommendation

        # Encode the user input into the feature space
        user_input_name_numeric = label_encoder.transform([name45])

        # Create a DataFrame for the user input
        user_input_features = pd.DataFrame([[user_input_name_numeric[0], 0]], columns=['name_numeric', 'actual_price'])

        # Compute cosine similarity between user input and all products
        cosine_sim = cosine_similarity(user_input_features, df[['name_numeric', 'actual_price']])

        # Get indices of the top K similar products
        k = 11  # Number of neighbors, including the item itself
        indices = cosine_sim.argsort()[0][-k-1:-1][::-1]

        # Get recommended product names, costs, and images based on the nearest neighbors
        recommended_product_names = df.iloc[indices]['name'].tolist()
        recommended_product_costs = df.iloc[indices]['actual_price'].tolist()
        recommended_product_images = df.iloc[indices]['image'].tolist()


        name = recommended_product_names
        price = recommended_product_costs
        image = recommended_product_images
        data = Add_chart.objects.filter(b_id=req.session['id'])
        if req.method == "POST":
            p_name = req.POST['id']
            print('__________________________________________')
            print(p_name)
            print('__________________________________________')
            ud = Add_chart.objects.get(id=p_name)
            ud.delete()
        return render(req,'buy.html',{'name':name,'price':price,'image':image,'data':data})
    except ValueError:
        user = UserRegistrationForm.objects.get(id=req.session['id'])
        name = user.name
        print(name)
        return render(req,'fuhome.html',{'name':name})
    

def add_cart(req):
    data = Product.objects.filter(name_id=req.session['id'])
    # Extract the 'name1' field and store it in a list
    name = [item.name1 for item in data]
    image = [item.image1 for item in data]
    price = [item.price1 for item in data]
    return render(req,'add_cart1.html',{'name':name,'price':price,'image':image})


def buy(req):
    if req.method == 'POST':
        from datetime import date,timedelta
        dic = req.POST.dict()
        del dic['csrfmiddlewaretoken']
        print(dic)
        det = []
        for i in dic.keys():
            det.append(dic[i])
        if det[0] == '' or det[1] == '' or det[2] == '' or det[3] == '' or det[4] == '' or det[5] == '':
            return render(req,'add_cart1.html')
        else:
            name3=req.session['na']
            print(name3)
            today = date.today()
            del_date = today+timedelta(days=6)
            Add_chart.objects.create(u_name=det[0],card_num=det[1],cvv=det[2],ba_name=det[3],
                                    p_pin=det[3],m_num=det[5],de_date=del_date,name2=req.session['na'],
                                    image2=req.session['im'],price2=req.session['pr'],b_id=req.session['id'])
            return render(req,'uhome1.html',{'ms':'sucess1'})
    return render(req,'buy.html')


def cancel_order(req):
    if req.method == 'POST':
        val = req.POST['id']
        ud = Add_chart.objects.get(id=val)
        ud.delete()
        return redirect(delinfo)
    return render(req,'view_chart1.html')



def view_chart(req):
    global data
    data = Product.objects.filter(name_id=req.session['id'])
    try:
        if req.method == 'POST':
            val = req.POST['id1']
            idr  = req.POST['id1']
            print(idr)
            prod = Product.objects.get(id = val)
            naa = req.session['na']=prod.name1
            imm = req.session['im']=prod.image1
            prr = req.session['pr']=prod.price1
            print(prod.name1)
            return redirect(add_cart)
    except:
        val = req.POST['id2']
        ud = Product.objects.get(id=val)
        print(ud)
        ud.delete()
        return render(req,'view_chart1.html',{'data':data,'m':'suc'})
    return render(req,'view_chart1.html',{'data':data})


def delinfo(req):
    data = Add_chart.objects.filter(b_id=req.session['id'])
    return render(req,'view_chart.html',{'data':data})

def orderdet(req):
    data = Add_chart.objects.all()
    return render(req,'orderdet.html',{'data':data})


def changedate(req):
    if req.method == 'POST':
        p_id = req.POST['p_id']
        id = req.session['id'] = p_id
        print(id)
        print('_____________________')
        return render(req,'adddate.html')
    data = Add_chart.objects.filter(b_id=req.session['id'])
    return render(req,'orderdet.html',{'data':data})

def adddate(request):
    if request.method == 'POST':
        date = request.POST.get('date')
        print(date)
        dat = Add_chart.objects.get(id=request.session['id'])
        dat.de_date = date
        dat.save()
        data = Add_chart.objects.all()
        return render(request,'orderdet.html',{'data':data,'msg12':'suc'})
