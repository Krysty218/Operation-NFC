from typing import List
import numpy as np

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
# Exploration-Exploitation tradeoff
epsilon = 0.1

Q_table = {meal: 5 for meal in Meals}

def knapsack_dp(meals_list: List[str], budget: int) -> List[str]:
    """Select the best combination of meals using Knapsack DP"""
    n = len(meals_list)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        current_meal = meals_list[i-1]
        cost = Meals[current_meal]["cost"]
        rating = Q_table[current_meal]
        
        for j in range(budget + 1):
            if cost <= j:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j - cost] + rating)
            else:
                dp[i][j] = dp[i-1][j]

    selected_meals = []
    j = budget
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i-1][j]:
            selected_meals.append(meals_list[i-1])
            j -= Meals[meals_list[i-1]]["cost"]
            
    return selected_meals

def Q_learning_update(meal: str, rating: int):
    """Update the Q-table based on user feedback"""
    Q_table[meal] = Q_table[meal] + alpha * (rating - Q_table[meal])

def hybrid_recommend(preferences: List[str], dislikes: List[str], budget: int) -> List[str]:
    """Combination of Q-learning and Knapsack DP"""
    if np.random.rand() < epsilon:
        candidate_meals = [meal for meal in Meals if meal not in dislikes]
    else:
        candidate_meals = sorted(Q_table, key=Q_table.get, reverse=True)
        candidate_meals = [meal for meal in candidate_meals if meal not in dislikes]
    
    selected_meals = knapsack_dp(candidate_meals, budget)
    return selected_meals

def test_harness():
    """Test the recommendation system with simulated user interactions"""
    test_cases = [
        {"preferences": ["Chicken Biryani"], "dislikes": ["Boiled Eggs"], "budget": 200},
        {"preferences": ["Veg Noodles"], "dislikes": ["Egg Curry"], "budget": 150},
        {"preferences": ["Masala Dosa"], "dislikes": ["Chicken 65"], "budget": 100}
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*40}\n Test Case {i}: Budget ₹{test['budget']}, Likes: {test['preferences']}, Dislikes: {test['dislikes']}\n{'='*40}")

        recommendations = hybrid_recommend(
            preferences=test['preferences'],
            dislikes=test['dislikes'],
            budget=test['budget']
        )

        total_cost = sum(Meals[meal]['cost'] for meal in recommendations)
        print(f"\nRecommended meals: {recommendations}")
        print(f"Total cost: ₹{total_cost}/₹{test['budget']}")

        if recommendations:
            liked_meal = np.random.choice(recommendations)
            Q_learning_update(liked_meal, rating=9)
            print(f"\nUser liked '{liked_meal}' → Updated Q-values:")
            print({k: round(v, 2) for k, v in Q_table.items() if k in recommendations})

def main():
    """Main execution block"""
    np.random.seed(42)
    print("🍽️ Meal Recommendation System Test Harness 🍴\n")
    test_harness()
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()
