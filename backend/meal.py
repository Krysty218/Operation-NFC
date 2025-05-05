from typing import List
import numpy as np
from collections import Counter

# --- MEAL DATA ---
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

# --- MEAL DATA STRUCTURE ---
Meals = {item: {"cost": cost, "rating": 5} for item, cost in zip(items, costs)}

# --- DIETARY CATEGORIES ---
veg_meals = [
    "North Indian Thali", "Phulkha (3 pcs)", "Chapathi (2 pcs)", "Roti", "Naan",
    "Butter Roti", "Butter Naan", "Uthapam", "Plain Dosa", "Onion Dosa",
    "Onion Uthapam", "Masala Dosa", "Moong Dhal", "Aloo Pakke Subji",
    "Black Channa Masala", "Chilly Baby Corn", "Papedi Chat", "Pani Poori",
    "Dahi Poori", "Lemon Juice", "Watermelon Juice", "Plain Rice", "Jeera Rice",
    "Veg Rice", "Mushroom Rice", "Veg Noodles"
]

nonveg_meals = [item for item in items if item not in veg_meals]

# --- SIMILARITY MATRIX INITIALIZATION ---
n_meals = len(items)
meal_to_idx = {meal: i for i, meal in enumerate(items)}
similarity_matrix = np.zeros((n_meals, n_meals))

def set_relationship(meal_a: str, meal_b: str, value: float):
    """Set relationship value between two meals"""
    i, j = meal_to_idx[meal_a], meal_to_idx[meal_b]
    value = max(min(value, 1.0), -1.0)  # Clamp between -1 and 1
    similarity_matrix[i][j] = value
    similarity_matrix[j][i] = value  # Ensure symmetry

def initialize_relationships():
    """Automatically set initial relationships based on meal groups"""
    groups = {
        "standalone": [
            "North Indian Thali", "Chicken Biryani", "Egg Biryani"
        ],
        "snacks_juices": [
            "Papedi Chat", "Pani Poori", "Dahi Poori", 
            "Lemon Juice", "Watermelon Juice"
        ],
        "breads": [
            "Phulkha (3 pcs)", "Chapathi (2 pcs)", "Roti", "Naan",
            "Butter Roti", "Butter Naan", "Uthapam", "Plain Dosa",
            "Onion Dosa", "Onion Uthapam", "Masala Dosa"
        ],
        "side_dishes": [
            "Moong Dhal", "Aloo Pakke Subji", "Black Channa Masala",
            "Chilly Baby Corn", "Chicken 65", "Egg Curry",
            "Single Egg Curry", "Pepper Chicken", "Chicken Masala",
            "Chicken Manchurian"
        ],
        "rice_noodles": [
            "Plain Rice", "Jeera Rice", "Veg Rice", "Mushroom Rice",
            "Egg Rice", "Chicken Rice", "Veg Noodles", "Egg Noodles",
            "Chicken Noodles"
        ]
    }

    # Set alternatives within same group
    for group_meals in groups.values():
        for i in range(len(group_meals)):
            for j in range(i+1, len(group_meals)):
                set_relationship(group_meals[i], group_meals[j], -0.7)

    # Set complements between groups
    complementary_groups = [
        ("standalone", "snacks_juices"),
        ("breads", "side_dishes"),
        ("rice_noodles", "side_dishes")
    ]
    
    for group1, group2 in complementary_groups:
        for meal1 in groups[group1]:
            for meal2 in groups[group2]:
                set_relationship(meal1, meal2, 0.6)

    # Set standalone incompatibility with other main components
    for standalone in groups["standalone"]:
        for meal in groups["breads"] + groups["rice_noodles"]:
            set_relationship(standalone, meal, -0.8)

    # Set veg-nonveg incompatibility
    for veg in veg_meals:
        for nonveg in nonveg_meals:
            if veg != nonveg:
                set_relationship(veg, nonveg, -0.4)

# Initialize the relationships
initialize_relationships()

# --- LEARNING PARAMETERS ---
alpha = 0.1  # Learning rate
epsilon = 0.3  # Exploration rate (increased to encourage more exploration)
Q_table = {meal: 5 for meal in Meals}  # Initial Q-values
meal_history = []  # Track meal history for variety

# --- KNAPSACK DP IMPLEMENTATION ---
def knapsack_dp(meals_list: List[str], budget: int, max_quantity: int = 3) -> List[str]:
    """Select optimal meals using similarity-aware Knapsack DP with quantity limits"""
    n = len(meals_list)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    # Use separate arrays for current and previous selections
    prev_selected = [[] for _ in range(budget + 1)]
    
    for i in range(1, n + 1):
        current_meal = meals_list[i-1]
        cost = Meals[current_meal]["cost"]
        current_selected = [[] for _ in range(budget + 1)]
        
        for j in range(budget + 1):
            if cost <= j:
                # Check current meal quantity and similarity with selected meals
                current_quantity = prev_selected[j - cost].count(current_meal)
                if current_quantity >= max_quantity:
                    dp[i][j] = dp[i-1][j]
                    current_selected[j] = prev_selected[j].copy()
                    continue

                # Check if current meal has negative similarity with any selected meal
                has_negative_similarity = any(
                    similarity_matrix[meal_to_idx[current_meal]][meal_to_idx[m]] < 0
                    for m in prev_selected[j - cost]
                )
                
                if has_negative_similarity:
                    # Skip this meal if it has negative similarity with any selected meal
                    dp[i][j] = dp[i-1][j]
                    current_selected[j] = prev_selected[j].copy()
                    continue

                similarity_penalty = sum(
                    similarity_matrix[meal_to_idx[current_meal]][meal_to_idx[m]]
                    for m in set(prev_selected[j - cost])
                    if similarity_matrix[meal_to_idx[current_meal]][meal_to_idx[m]] < 0
                )
                
                adjusted_rating = Q_table[current_meal] + similarity_penalty
                
                if (dp[i-1][j - cost] + adjusted_rating) > dp[i-1][j]:
                    dp[i][j] = dp[i-1][j - cost] + adjusted_rating
                    current_selected[j] = prev_selected[j - cost] + [current_meal]
                else:
                    dp[i][j] = dp[i-1][j]
                    current_selected[j] = prev_selected[j].copy()
            else:
                dp[i][j] = dp[i-1][j]
                current_selected[j] = prev_selected[j].copy()
        
        prev_selected = current_selected  # Update previous selections for next iteration

    return prev_selected[budget]

# --- Q-LEARNING UPDATE ---
def Q_learning_update(meal: str, rating: int):
    """Update Q-values based on user feedback"""
    Q_table[meal] += alpha * (rating - Q_table[meal])

# --- HYBRID RECOMMENDATION SYSTEM ---
def hybrid_recommend(preferences: List[str], dislikes: List[str], 
                    budget: int, dietary: str) -> List[str]:
    """Combination of Q-learning and Knapsack DP with dietary filtering"""
    global meal_history
    available_meals = veg_meals if dietary == "veg" else items
    
    valid_prefs = [m for m in preferences if m in available_meals]
    valid_dislikes = [m for m in dislikes if m in available_meals]
    
    # Exploration vs Exploitation
    if np.random.rand() < epsilon:
        # Explore: Randomly select meals not in disliked list and not in recent history
        candidate_meals = [m for m in available_meals 
                          if m not in valid_dislikes and m not in meal_history]
    else:
        # Exploit: Select meals with highest Q-values, excluding disliked meals
        candidate_meals = sorted(
            [m for m in available_meals if m not in valid_dislikes],
            key=lambda x: Q_table[x], 
            reverse=True
        )[:len(available_meals)]  # Consider all available meals during exploitation
    
    # If no candidate meals are left, reset meal history
    if not candidate_meals:
        meal_history = []
        candidate_meals = [m for m in available_meals if m not in valid_dislikes]
    
    # Use Knapsack DP to select meals within budget
    selected_meals = knapsack_dp(candidate_meals, budget)
    
    # Update meal history
    meal_history.extend(selected_meals)
    if len(meal_history) > 10:  # Keep only the last 10 meals in history
        meal_history = meal_history[-10:]
    
    return selected_meals

# --- TEST HARNESS WITH DIETARY SELECTION ---
def test_harness():
    """Interactive test harness with dietary preference selection"""
    while True:
        dietary_choice = input("\nChoose dietary preference (veg/nonveg): ").lower()
        if dietary_choice in ["veg", "nonveg"]:
            break
        print("Invalid choice! Please enter 'veg' or 'nonveg'.")
    
    test_cases = [
        {"preferences": ["Chapathi (2 pcs)", "Naan"], 
         "dislikes": [], "budget": 50},
        {"preferences": ["Jeera Rice", "Veg Noodles"], 
         "dislikes": [], "budget": 100},
        {"preferences": ["Masala Dosa", "Uthapam"], 
         "dislikes": [], "budget": 100}
    ]
    
    filtered_cases = []
    for case in test_cases:
        filtered_prefs = [m for m in case["preferences"] 
                         if m in (veg_meals if dietary_choice == "veg" else items)]
        filtered_dislikes = [m for m in case["dislikes"] 
                            if m in (veg_meals if dietary_choice == "veg" else items)]
        filtered_cases.append({
            "preferences": filtered_prefs,
            "dislikes": filtered_dislikes,
            "budget": case["budget"]
        })

    print(f"\n{'='*40}\n Testing {dietary_choice.capitalize()} Recommendations\n{'='*40}")
    
    for i, test in enumerate(filtered_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Budget: ₹{test['budget']}")
        print(f"Preferences: {test['preferences'] or 'None'}")
        print(f"Dislikes: {test['dislikes'] or 'None'}")
        
        recommendations = hybrid_recommend(
            preferences=test['preferences'],
            dislikes=test['dislikes'],
            budget=test['budget'],
            dietary=dietary_choice
        )
        
        meal_counts = Counter(recommendations)
        formatted_recommendations = [f"{count}x {meal}" if count > 1 else meal 
                                     for meal, count in meal_counts.items()]
        
        total_cost = sum(Meals[meal]['cost'] * count for meal, count in meal_counts.items())
        print(f"\nRecommended meals: {formatted_recommendations}")
        print(f"Total Cost: ₹{total_cost}/₹{test['budget']}")
        print(f"Budget Utilization: {(total_cost/test['budget'])*100:.1f}%")
        
        if recommendations:
            liked_meal = np.random.choice(recommendations)
            Q_learning_update(liked_meal, 9)
            print(f"\nUser liked '{liked_meal}' → Updated Q-value: {Q_table[liked_meal]:.2f}")

def main():
    np.random.seed(42)  # For reproducible results
    print("🍽️ Smart Meal Recommendation System 🧠")
    print(" With Dietary Preference Filtering \n")
    test_harness()
    print("\n✅ Test Completed!")

if __name__ == "__main__":
    main()