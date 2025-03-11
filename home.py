import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


st.title("🐷🐷🐷การพยากรณ์โรคหัวใจล้มเหลวด้วยเทคนิคเหมืองข้อมูล🐷🐷")
st.header("🍖🍖การพยากรณ์โรคหัวใจล้มเหลวด้วยเทคนิคเหมืองข้อมูล🍖🍖")

st.image('./img/h1.jpg')
st.subheader("โรคหัวใจล้มเหลว")

dt = pd.read_csv("./data/Heart.csv")
st.write(dt.head(10))
st.image('./img/h5.jpg')
st.subheader("โรคหัวใจล้มเหลว")


st.subheader("่สถิติข้อมูลโรคหัวใจ")
st.write(dt.describe())
st.write("สถิติจำนวนเพศหญิง = 0 ชาย = 1")


count_male = dt.groupby('Sex').size()[1]
count_female = dt.groupby('Sex').size()[0]
dx = [count_male, count_female]
dx2 = pd.DataFrame(dx, index=["Male", "Female"])
st.bar_chart(dx2)


st.subheader("ข้อมูลค่าเฉลี่ยอายุแยกตามเพศ")
average_male_age = dt[dt['Sex'] == 1]['Age'].mean()
average_female_age = dt[dt['Sex'] == 0]['Age'].mean()

dxage = [average_male_age, average_female_age]
dxage2 = pd.DataFrame(dxage, index=["Male", "Female"])
st.bar_chart(dxage2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")
A1 = st.number_input("กรุณาเลือกข้อมูลอายุ")
A2 = st.number_input("กรุณาเลือกเพศชาย=1 หญิง=0")
A3 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก")
A4 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก1")
A5 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก2")
A6 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก3")
A7 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก4")
A8 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก5")
A9 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก6")
A10 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก7")
A11 = st.number_input("กรุณาเลือกข้อมูลของเจ็บหน้าอก8")

pt_len = st.slider("กรุณาเลือกข้อมูล petal.length")
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width")

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length")
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

if st.button("ทำนายผล"):
    #st.write("ทำนาย")
   dt = pd.read_csv("./data/Heart.csv") 
   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == 1:
    st.image("./img/h4.jpg")
   elif out[0] == 'Versicolor':       
    st.image("./img/h2.jpg")
else:
    st.write("ไม่ทำนาย")