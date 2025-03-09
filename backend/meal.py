import numpy as np
from scikit_learn.feature_extraction.text import TfidfVectorizer
from scikit_learn.metrics.pairwise import cosine_similarity

items = [
    "North Indian Thali", "Phulkha (3 pcs)", "Chapathi (2 pcs)", "Roti", "Naan", 
    "Butter Roti", "Butter Naan", "Uthapam", "Plain Dosa", "Onion Dosa", "Onion Uthapam", 
    "Masala Dosa", "Plain Rice", "Jeera Rice", "Veg Noodles", "Egg Noodles", 
    "Chicken Noodles", "Moong Dhal", "Aloo Pakke Subji", "Black Channa Masala", 
    "Chilly Baby Corn", "Papedi Chat", "Pani Poori", "Dahi Poori", "Veg Rice", 
    "Egg Rice", "Chicken Rice", "Mushroom Rice", "Egg Biryani", "Chicken Biryani", 
    "Chicken 65", "Egg Curry", "Single Egg Curry", "Boiled Eggs", "Scrambled Eggs", 
    "Pepper Chicken", "Chicken Masala", "Chicken Manchurian", "Lemon Juice", "Watermelon Juice"
]

costs = [
    68, 16, 14, 9, 9, 14, 14, 24, 21, 27, 27, 32, 21, 34, 45, 63, 66, 27, 32, 39, 
    43, 21, 17, 21, 47, 61, 76, 56, 63, 83, 78, 23, 13, 9, 13, 70, 68, 72, 13, 19
]

Meals = {}
for item, cost in zip(items, costs):
    Meals[item] = {"cost": cost, "rating": 5}

# Learning rate
alpha = 0.1
#Exploration-Exploitation tradeoff
epsilon = 0.1

q_table = {meal: 5 for meal in Meals}

meal_texts = ["".joins(meal.split()) for meal in Meals]
vectorizer = TfidfVectorizer()
meal_vectors = vectorizer.fit_transform(meal_texts)

def RAG_similar_meals(preferred_meal: str, top_n: int = 3) -> List[str] :
    # process preferred meal
    text = " ".join(preferred_meal.split())
    
    #transform into vector
    vector = vectorizer.transform([text])
    
    # calculate similarity
    similarities = cosine_similarity(vector, meal_vectors).flatten()
    
    # sort based on similarity score
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Get the top_n meals
    similar_indices = sorted_indices[-(top_n+1):-1]
    similar_indices = similar_indices[::-1] # reverse
    
    # convert dict to list for index access
    similar_meals = []
    meal_keys = list(Meals.keys())
    
    for index in similar_indices:
        similar_meals.append(meal_keys[index])
        
    return similar_meals