from flask import Flask, request, render_template
import pandas as pd
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

### FOR CSV Generation
import csv
import random
from faker import Faker
#########################

app = Flask(__name__)


######CSV GENERATION#####
# fake = Faker()

# def generate_book():
#     title = fake.sentence()
#     author = fake.name()
#     publication_year = random.randint(1900, 2023)
#     genre = random.choice(['Fiction', 'Non-Fiction', 'Mystery', 'Science Fiction', 'Fantasy', 'Romance', 'Thriller'])
#     text = fake.paragraph()

#     return title, author, publication_year, genre, text

# # Generate 100 books
# num_books = 100
# books_data = [['Title', 'Author', 'Publication_Year', 'Genre', 'Text']]

# for _ in range(num_books):
#     books_data.append(generate_book())

# # Write to CSV file
# csv_file_path = 'books_data.csv'
# with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerows(books_data)

# print(f'CSV file generated: {csv_file_path}')

###############################################33

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Stores the search history globally
search_history = []

def get_embedding(text, model="text-embedding-ada-002"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_recommendations(user_vector, df, num_recommendations=1):
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, user_vector))
    sorted_by_similarity = df.sort_values("similarities", ascending=False).head(num_recommendations)
    recommendations = sorted_by_similarity['Text'].values.tolist()
    
    # Drops the temporary "similarities" column
    df.drop("similarities", axis=1, inplace=True)
    
    return recommendations

@app.route('/')
def search_form():
    return render_template('search_form.html')

@app.route('/search')
def search():
    global search_history

    query = request.args.get('query')

    df = pd.read_csv("books_data.csv")
    df["embedding"] = df['Text'].apply(lambda x: get_embedding(x))  # Generate embeddings dynamically

    search_term_embedding = get_embedding(query)

    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_embedding))
    sorted_by_similarity = df.sort_values("similarities", ascending=False).head(1)
    results = sorted_by_similarity['Text'].values.tolist()

    # Adds the current query to the search history
    search_history.append(query)
    print(search_history)
    # Uses the second-to-last search query for recommendations
    if len(search_history) > 1:
        print(search_history[-2])
        user_vector = get_embedding(search_history[-2])
        recommendations = get_recommendations(user_vector, df)
    else:
        recommendations = []

    return render_template('search_results.html', query=query, results=results, recommendations=recommendations)

@app.route('/book_details/<title>')
def book_details(title):
    df = pd.read_csv("books_data.csv")
    
    # Finds the book details based on the title
    book_details = df[df['Text'] == title].to_dict(orient='records')[0]

    return render_template('book_details.html', book_details=book_details)


if __name__ == '__main__':
    app.run(debug=True)
