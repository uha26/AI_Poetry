import streamlit as st
from poetry_generator import generate_poem

# Streamlit User Interface
st.title("Poetry Generator with GPT-2")
st.write("Generate poems based on your theme using GPT-2!")

# Input for the user to type in the theme
theme = st.text_input("Enter the theme for the poem:")

# Generate and display the poem when the user inputs a theme
if theme:
    poem = generate_poem(theme)
    st.subheader("Generated Poem:")
    formatted_poem = "\n".join([f"    {line.strip()}" for line in poem.split(". ")])
    st.write(poem)
