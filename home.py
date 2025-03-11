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
st.write('ผลรวม')
