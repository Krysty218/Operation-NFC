from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

Q_table = {meal: 5 for meal in Meals}

meal_texts = [" ".join(meal.split()) for meal in Meals]
vectorizer = TfidfVectorizer()
meal_vectors = vectorizer.fit_transform(meal_texts)


def RAG_similar_meals(preferred_meal: str, top_n: int = 3) -> List[str]:
    "Get similar meals using TFID and cosine similarity"
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

def knapsack_dp(meals_list : List[str], budget : int) -> List[str]:
    "Select the best combination of meals using Knapsack DP"
    
    n = len(meals_list)
    # Initialize DP table with 0s in a matrix of size (n+1)*(budget+1)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]  

    # Select meals
    for i in range(1,n+1):
        current_meal = meals_list[i-1]
        cost = Meals[current_meal]["cost"]
        rating = Q_table[current_meal]
        
        for j in range(budget+1):
            if cost <= j:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-cost] + rating) 
            else:
                dp[i][j] = dp[i-1][j]
                
    # Backtrack to find the selected meals
    selected_meals = []
    j = budget
    for i in range(n, 0, -1):   
        if dp[i][j] != dp[i-1][j]:
            selected_meals.append(meals_list[i-1])
            j -= Meals[meals_list[i-1]]["cost"]
            
    return selected_meals    
    
def Q_learning_update(meal: str, rating: int):
    "Update the Q-table based on user feedback"
    Q_table[meal] = Q_table[meal] + alpha * (rating - Q_table[meal]) # rating = reward

def hybrid_recommend(preferences: List[str], dislikes: List[str], budget: int) -> List[str]:
    "Combination of the 3 methods"
    # Step 1 - Exploration vs Exploitation
    if np.random.rand() < epsilon:
        # Explore: Randomly select meals not in disliked list
        candidate_meals = [meal for meal in Meals if meal not in dislikes]
    else:
        # Exploit: Select meals with highest Q_values
        candidate_meals = sorted(Q_table, key=Q_table.get, reverse=True)
        candidate_meals = [meal for meal in candidate_meals if meal not in dislikes]
    
    # Step 2 - Knapsack budget constraint
    selected_meals = knapsack_dp(candidate_meals, budget)
    
    # Step 3 - RAG enhanced
    for meal in selected_meals:
        similar_meals = RAG_similar_meals(meal)
        print(f"Since you liked {meal}, you might also like: {', '.join(similar_meals)}")
    
    return selected_meals