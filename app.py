import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
# request for getting dat from Form

df = pickle.load(open('dataframe.pkl','rb'))
pipe = pickle.load(open('pipe.pkl', 'rb')) # I have loaded DataFrame
# pipe = pickle.load(open('pipe.pkl', 'rb'))  # I have loaded model

app = Flask(__name__)  # important


@app.route('/')  # for pointing homepage
def index():

    
    # Company = sorted(df['Company'].unique())
    # TypeName = sorted(df['TypeName'].unique())
    # Inches = sorted(df['Inches'].unique())
    # Ram = sorted(df['Ram'].unique())
    # OpSys = sorted(df['OpSys'].unique())
    # Weight = sorted(df['Weight'].unique())
    # IPS = sorted(df['IPS'].unique())
    # touch_screen = sorted(df['touch_screen'].unique())
    # PPI = sorted(df['PPI'].unique())
    # Brand = sorted(df['Brand'].unique())
    # SDD = sorted(df['SDD'].unique())
    # HDD = sorted(df['HDD'].unique())
    # GPU = df['GPU'].unique()

    return render_template('index.html', company=Company, typename=TypeName, inches=Inches, ram=Ram, opsys=OpSys, weight=Weight, ips=IPS, touchscreen=touch_screen, ppi=PPI, brand=Brand, sdd=SDD, hdd=HDD, gpu=GPU)


@app.route('/predict', methods=['post','get'])
def predict():
    Company = request.form.get('company')
    TypeName = request.form.get('typename')
    Inches = request.form.get('inches')
    Ram = request.form.get('ram')
    OpSys = request.form.get('opsys')
    Weight = request.form.get('weight')
    IPS = request.form.get('ips')
    touch_screen = request.form.get('touchscreen')
    PPI = request.form.get('ppi')
    Brand = request.form.get('brand')
    SDD = request.form.get('sdd')
    HDD = request.form.get('hdd')
    GPU = request.form.get('gpu')

# let's make and Dataframe for prediction as our model takes just dataframe 
    
    
    prediction = pipe.predict(input)[0] # it will give list which 0 element would be our prediction


    return str(prediction)

    # array.append(Inches = request.form.get('Inches'))
    # array.append(TypeName = request.form.get('TypeName'))
    # array.append(Ram = request.form.get('Ram'))
    # array.append(OpSys = request.form.get('OpSys'))
    # array.append(Weight = request.form.get('Weight'))
    # array.append(IPS = request.form.get('IPS'))
    # array.append(touch_screen = request.form.get('touch_screen'))
    # array.append(PPI = request.form.get('PPI'))
    # array.append(Brand = request.form.get('Brand'))
    # array.append(SDD = request.form.get('SDD'))
    # array.append(HDD = request.form.get('HDD'))
    # array.append(GPU = request.form.get('GPU'))

    # now here I will send each columns of DF as a list to the index.html


# @app.route('/recommend')
# def recommend_ui():
#     return render_template('recommend.html')

# @app.route('/recommend_books', methods =['post']) ## for recommending books method is post, cause we are getting data from user Forms

# def recommend():
#     input_data = request.form.get('user_input').strip()  ## got title from user
#     # now I will show result

#     index = np.where(pt.index == input_data)[0][0]
#     if  index >=0 :
#         similar_item=sorted(list(enumerate(cs[index])), key=lambda x: x[1], reverse=True)[1:9]

#         data= []
#         book['Book-Title'] = book['Book-Title'].drop_duplicates() # removing duplicates

#         for i in similar_item:
#             # print(i) # we got index, need index[i[0]]
#             ## now I will match this title with each title of Books and will fetch book author name and Image
#             # print(pt.index[i[0]])
#             item= []
#             # here book must bee chosen by top-50 books so that he/she can see recommendation

#             item.extend(list(book[book['Book-Title']==pt.index[i[0]]]['Book-Title'].values))
#             item.extend(list(book[book['Book-Title']==pt.index[i[0]]]['Book-Author'].values))
#             item.extend(list(book[book['Book-Title']==pt.index[i[0]]]['Image-URL-M'].values))
#             data.append(item)  ## we will get the list of list
#         # print(data)

#         return render_template('recommend.html', data = data) #passing to recommend.html
#     else:
#         return "Book is out of stock"
if __name__ == "__main__":
    app.run(debug=True)
