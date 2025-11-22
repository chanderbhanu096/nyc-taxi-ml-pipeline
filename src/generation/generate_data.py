import json
import random
import time
from datetime import datetime, timedelta
from faker import Faker
import os

fake = Faker()

DATA_DIR = "data/landing"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_users(n=100):
    users = []
    for _ in range(n):
        users.append({
            "user_id": fake.uuid4(),
            "name": fake.name(),
            "email": fake.email(),
            "address": fake.address(),
            "signup_date": fake.date_between(start_date='-1y', end_date='today').isoformat()
        })
    return users

def generate_products(n=20):
    products = []
    categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Toys']
    for _ in range(n):
        products.append({
            "product_id": fake.uuid4(),
            "name": fake.word().capitalize() + " " + fake.word().capitalize(),
            "category": random.choice(categories),
            "price": round(random.uniform(10.0, 500.0), 2)
        })
    return products

def generate_orders(users, products, n=500):
    orders = []
    for _ in range(n):
        user = random.choice(users)
        product = random.choice(products)
        quantity = random.randint(1, 5)
        order_date = fake.date_time_between(start_date='-1y', end_date='now')
        
        orders.append({
            "order_id": fake.uuid4(),
            "user_id": user['user_id'],
            "product_id": product['product_id'],
            "quantity": quantity,
            "total_amount": round(product['price'] * quantity, 2),
            "order_date": order_date.isoformat(),
            "status": random.choice(['Completed', 'Pending', 'Cancelled', 'Shipped'])
        })
    return orders

def save_to_json(data, filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Generated {len(data)} records in {filepath}")

if __name__ == "__main__":
    print("Generating synthetic data...")
    
    # Generate static dimensions
    users = generate_users(200)
    products = generate_products(50)
    
    # Generate transactional data
    orders = generate_orders(users, products, 1000)
    
    # Save to landing zone
    save_to_json(users, "users.json")
    save_to_json(products, "products.json")
    save_to_json(orders, "orders.json")
    
    print("Data generation complete.")
