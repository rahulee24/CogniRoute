import os
import sqlite3

def init_database():
    db_dir = "data"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "company_sales.db")
    
    print(f"Initializing database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT,
        price REAL,
        stock_quantity INTEGER
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT,
        country TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER,
        product_id INTEGER,
        order_date TEXT,
        quantity INTEGER,
        total_amount REAL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    """)
    
    # Seed data if empty
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        print("Seeding database...")
        
        # Products
        products = [
            ("Enterprise Plan License", "Software", 4999.00, 9999),
            ("Team Plan Annual License", "Software", 1200.00, 9999),
            ("Pro Plan Monthly License", "Software", 49.00, 9999),
            ("Premium Support Addon", "Support", 250.00, 9999),
            ("AI Assistant Credits Package", "AI Addon", 99.00, 9999),
        ]
        cursor.executemany("INSERT INTO products (name, category, price, stock_quantity) VALUES (?, ?, ?, ?)", products)
        
        # Customers
        customers = [
            ("Alice Smith", "alice@enterprise.com", "USA"),
            ("Bob Jones", "bob@startup.io", "Canada"),
            ("Charlie Brown", "charlie@smb.co.uk", "UK"),
            ("Diana Prince", "diana@amazon.corp", "USA"),
            ("Evan Wright", "evan@agency.de", "Germany"),
        ]
        cursor.executemany("INSERT INTO customers (name, email, country) VALUES (?, ?, ?)", customers)
        
        # Orders
        orders = [
            (1, 1, "2026-05-01", 2, 9998.00),  # Alice bought 2 Enterprise plans
            (2, 2, "2026-05-12", 1, 1200.00),  # Bob bought 1 Team Plan
            (3, 3, "2026-05-15", 3, 147.00),   # Charlie bought 3 Pro Plans
            (4, 4, "2026-05-20", 1, 4999.00),  # Diana bought 1 Enterprise Plan
            (4, 4, "2026-05-20", 1, 250.00),   # Diana bought Premium Support
            (5, 5, "2026-05-25", 5, 495.00),   # Evan bought 5 AI Credit packages
        ]
        cursor.executemany("INSERT INTO orders (customer_id, product_id, order_date, quantity, total_amount) VALUES (?, ?, ?, ?, ?)", orders)
        
        conn.commit()
        print("Database seeded successfully.")
    else:
        print("Database already contains data. Skipping seeding.")
        
    conn.close()

if __name__ == "__main__":
    init_database()
