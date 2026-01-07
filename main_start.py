import streamlit as st
st.title("Hello World")
userName = st.text_input("Enter a your name:")
submit_button = st.button("Submit your name")
 
if submit_button:
    greet = "Hello " + userName + "!"
    st.write(greet)
else:
    st.write("")
 