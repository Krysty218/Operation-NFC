import random

def knapsack_meal_selection(items, costs, days, daily_budget):
    total_budget = days * daily_budget
    remaining_budget = total_budget
    daily_meals = []
    min_cost = int(0.6 * daily_budget)  # Minimum total cost per day
    menu = list(zip(items, costs))
    
    for day in range(1, days + 1):
        # Adjust daily budget dynamically to ensure full usage
        avg_remaining = remaining_budget / (days - day + 1)
        max_day_budget = min(remaining_budget, max(avg_remaining * 1.2, daily_budget * 1.5))
        available_meals = [meal for meal in menu if meal[1] <= max_day_budget]
        
        if not available_meals:
            break  # No affordable meals left
        
        selected_meals = []
        day_spent = 0
        random.shuffle(available_meals)
        
        # Try to fill the budget using multiple meals
        for meal, cost in sorted(available_meals, key=lambda x: -x[1]):  # Pick expensive meals first
            if day_spent + cost <= max_day_budget:
                selected_meals.append((meal, cost))
                day_spent += cost
                if day_spent >= min_cost:
                    break  # Stop once minimum spend requirement is met
        
        if selected_meals:
            daily_meals.append((day, selected_meals, day_spent))
            remaining_budget -= day_spent
        
        if remaining_budget <= 0:
            break  # Stop if budget is exhausted
    
    return daily_meals, remaining_budget

# Sample Data
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

# User Input
num_days = int(input("Enter number of days: "))
daily_budget = int(input("Enter daily budget (\u20B9): "))

# Run the algorithm
daily_meals, remaining_budget = knapsack_meal_selection(items, costs, num_days, daily_budget)

# Output results
for day, meals, spent in daily_meals:
    meal_names = ", ".join([f"{meal} (\u20B9{cost})" for meal, cost in meals])
    print(f"\nDay {day}: You will have {meal_names}. Total spent today: \u20B9{spent}\n")

print(f"\nAfter {num_days} days, you have \u20B9{remaining_budget} remaining from your total budget.\n")
